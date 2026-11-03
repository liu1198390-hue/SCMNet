"""SCMNet: structure-aware encoding, feature fusion and instance refinement."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

from .layers import PPM, ConvGNAct, _gn_groups
from .vmamba import PatchEmbed2D, PatchMerging2D, VSSBlock


class RelationPredictor(nn.Module):
    """Left/top same-instance affinities before a high-resolution VSS stage."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(16, channels // 4)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_gn_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(_gn_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class RelationEncoder(nn.Module):
    """VMamba-Tiny-compatible VSS encoder with relation-aware first two stages."""

    def __init__(
        self,
        dims=(96, 192, 384, 768),
        depths=(2, 2, 9, 2),
        patch_size=4,
        d_state=16,
        d_conv=3,
        expand=2,
        drop_path_rate=0.2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dims = tuple(int(v) for v in dims)
        self.depths = tuple(int(v) for v in depths)
        if len(self.dims) != 4 or len(self.depths) != 4:
            raise ValueError("RelationEncoder requires four stages")
        self.use_checkpoint = bool(use_checkpoint)
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed2D(
            patch_size=int(patch_size), in_chans=3, embed_dim=self.dims[0], norm_layer=norm
        )
        self.pos_drop = nn.Identity()
        drop_paths = torch.linspace(0, float(drop_path_rate), sum(self.depths)).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for stage_index, depth in enumerate(self.depths):
            stage = nn.Module()
            stage.blocks = nn.ModuleList(
                [
                    VSSBlock(
                        hidden_dim=self.dims[stage_index],
                        drop_path=float(drop_paths[offset + block_index]),
                        norm_layer=norm,
                        d_state=int(d_state),
                        d_conv=int(d_conv),
                        expand=int(expand),
                    )
                    for block_index in range(depth)
                ]
            )
            stage.downsample = (
                PatchMerging2D(dim=self.dims[stage_index], norm_layer=norm)
                if stage_index < 3
                else None
            )
            self.stages.append(stage)
            offset += depth
        self.relation_heads = nn.ModuleList(
            [RelationPredictor(self.dims[0]), RelationPredictor(self.dims[1])]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor):
        x = self.pos_drop(self.patch_embed(image))
        features, relation_logits = [], []
        for index, stage in enumerate(self.stages):
            scan_relation = None
            if index < 2:
                logits = self.relation_heads[index](x.permute(0, 3, 1, 2).contiguous())
                relation_logits.append(logits)
                scan_relation = logits.sigmoid()
            for block in stage.blocks:
                if self.use_checkpoint and self.training:
                    x = checkpoint.checkpoint(
                        lambda t, b=block, r=scan_relation: b(t, scan_relation=r),
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, scan_relation=scan_relation)
            features.append(x.permute(0, 3, 1, 2).contiguous())
            if stage.downsample is not None:
                x = stage.downsample(x)
        return features, relation_logits


class EvidenceFusion(nn.Module):
    """Directly select shallow localization or deep context at each location."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(24, channels // 4)
        self.selector = nn.Sequential(
            ConvGNAct(channels * 3 + 2, hidden, kernel_size=3),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)

    def forward(self, lateral, topdown, relation):
        topdown = F.interpolate(
            topdown, size=lateral.shape[-2:], mode="bilinear", align_corners=False
        )
        if relation is None:
            relation = lateral.new_full((lateral.shape[0], 2, *lateral.shape[-2:]), 0.5)
        else:
            relation = F.interpolate(
                relation.sigmoid(), size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
        discrepancy = (F.normalize(lateral, dim=1) - F.normalize(topdown, dim=1)).abs()
        alpha = torch.tanh(
            self.selector(torch.cat([lateral, topdown, discrepancy, relation], dim=1))
        )
        return (1.0 + alpha) * lateral + (1.0 - alpha) * topdown, alpha


class RelationFPN(nn.Module):
    def __init__(self, in_channels: Sequence[int], channels=128, ppm_channels=128):
        super().__init__()
        self.ppm = PPM(int(in_channels[-1]), int(ppm_channels))
        self.lateral = nn.ModuleList(
            [ConvGNAct(int(c), channels, kernel_size=1, padding=0) for c in in_channels[:-1]]
        )
        self.deep_lateral = ConvGNAct(int(ppm_channels), channels, kernel_size=1, padding=0)
        self.fusion = nn.ModuleList([EvidenceFusion(channels) for _ in in_channels[:-1]])
        self.output = nn.ModuleList([ConvGNAct(channels, channels) for _ in in_channels[:-1]])
        self.bottleneck = ConvGNAct(channels * len(in_channels), channels)

    def forward(self, features, relation_logits):
        lateral = [proj(feat) for proj, feat in zip(self.lateral, features[:-1])]
        lateral.append(self.deep_lateral(self.ppm(features[-1])))
        alpha_maps = [None] * (len(lateral) - 1)
        for index in range(len(lateral) - 2, -1, -1):
            hint = relation_logits[index] if index < len(relation_logits) else None
            lateral[index], alpha_maps[index] = self.fusion[index](
                lateral[index], lateral[index + 1], hint
            )
        pyramid = [conv(tensor) for conv, tensor in zip(self.output, lateral[:-1])]
        pyramid.append(lateral[-1])
        target_size = pyramid[0].shape[-2:]
        pyramid = [
            F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
            if tensor.shape[-2:] != target_size
            else tensor
            for tensor in pyramid
        ]
        return self.bottleneck(torch.cat(pyramid, dim=1)), alpha_maps


class HighResolutionRefiner(nn.Module):
    """High-resolution joint readout of instance occupancy and local edges."""

    def __init__(
        self, decoder_channels=128, detail_channels=32, hidden_channels=64, embedding_dim=8
    ):
        super().__init__()
        self.detail = nn.Sequential(
            nn.Conv2d(3, detail_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(detail_channels), detail_channels),
            nn.SiLU(inplace=True),
            ConvGNAct(detail_channels, detail_channels),
        )
        self.half_refine = nn.Sequential(
            ConvGNAct(
                decoder_channels + detail_channels + 2, hidden_channels, kernel_size=1, padding=0
            ),
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False
            ),
            nn.GroupNorm(_gn_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.full_detail = ConvGNAct(3, 16)
        self.full_refine = ConvGNAct(
            hidden_channels + 16, hidden_channels, kernel_size=1, padding=0
        )
        self.region_head = nn.Conv2d(hidden_channels, 1, 1)
        # First four edges are E, S, SE, SW; remaining four span four pixels.
        self.relation_head = nn.Conv2d(hidden_channels, 8, 1)

        self.contact_head = nn.Conv2d(hidden_channels, 1, 1)
        self.embedding_head = nn.Conv2d(hidden_channels, embedding_dim, 1)

    def forward(self, image, decoded, relation_hint):
        detail = self.detail(image)
        size = detail.shape[-2:]
        decoded = F.interpolate(decoded, size=size, mode="bilinear", align_corners=False)
        hint = F.interpolate(
            relation_hint.sigmoid(), size=size, mode="bilinear", align_corners=False
        )
        refined = self.half_refine(torch.cat([decoded, detail, hint], dim=1))
        refined = F.interpolate(
            refined, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        refined = self.full_refine(torch.cat([refined, self.full_detail(image)], dim=1))
        embedding = F.normalize(self.embedding_head(refined).float(), dim=1, eps=1e-6)
        return (
            self.region_head(refined),
            self.relation_head(refined),
            self.contact_head(refined),
            embedding,
        )


class SCMNet(nn.Module):
    """Predict foreground, local affinities, contact gaps and instance embeddings.

    The default configuration has 25,505,688 trainable parameters. State-dict
    names and tensor shapes match the original training implementation.
    """

    model_version = "SCMNet"
    relation_offsets = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 4), (4, 0), (4, 4), (4, -4))

    def __init__(
        self,
        dims=(96, 192, 384, 768),
        depths=(2, 2, 9, 2),
        patch_size=4,
        d_state=16,
        d_conv=3,
        expand=2,
        drop_path_rate=0.2,
        decoder_channels=128,
        ppm_channels=128,
        detail_channels=32,
        embedding_dim=8,
        refinement_channels=64,
        use_checkpoint=False,
    ):
        super().__init__()
        self.model_config = dict(
            model_version=self.model_version,
            dims=tuple(dims),
            depths=tuple(depths),
            patch_size=int(patch_size),
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            drop_path_rate=float(drop_path_rate),
            decoder_channels=int(decoder_channels),
            ppm_channels=int(ppm_channels),
            detail_channels=int(detail_channels),
            embedding_dim=int(embedding_dim),
            refinement_channels=int(refinement_channels),
            use_checkpoint=bool(use_checkpoint),
        )
        self.encoder = RelationEncoder(
            dims=dims,
            depths=depths,
            patch_size=patch_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )
        self.decode_head = RelationFPN(dims, channels=decoder_channels, ppm_channels=ppm_channels)
        self.hr_mdr = HighResolutionRefiner(
            decoder_channels,
            detail_channels,
            refinement_channels,
            embedding_dim,
        )

    def load_vmamba_pretrained(self, checkpoint_path: str | Path):
        raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        source = (
            raw.get("model", raw.get("state_dict", raw.get("model_state_dict", raw)))
            if isinstance(raw, dict)
            else raw
        )
        own = self.state_dict()
        loaded = {}
        for name, tensor in source.items():
            if not torch.is_tensor(tensor):
                continue
            key = name
            prefixes = ("module.", "model.", "net.", "backbone.", "encoder.")
            while any(key.startswith(prefix) for prefix in prefixes):
                key = next(key[len(prefix) :] for prefix in prefixes if key.startswith(prefix))
            if key.startswith("layers."):
                key = key.replace("layers.", "stages.", 1)
            candidate = "encoder." + key
            if candidate in own and own[candidate].shape == tensor.shape:
                loaded[candidate] = tensor
        self.load_state_dict(loaded, strict=False)
        return {
            "loaded_tensors": len(loaded),
            "encoder_tensors": sum(key.startswith("encoder.") for key in own),
        }

    def forward(self, image: torch.Tensor, return_aux=False, aux_mode=None):
        features, encoder_relation = self.encoder(image)
        decoded, fusion_alpha = self.decode_head(features, encoder_relation)
        region, relation, contact, embedding = self.hr_mdr(image, decoded, encoder_relation[0])
        if not return_aux:
            return region
        aux = {
            "relation_logits": relation,
            "contact_logits": contact,
            "instance_embedding": embedding,
            "encoder_relation_logits": encoder_relation,
            "fusion_alpha": fusion_alpha,
        }
        return region, aux
