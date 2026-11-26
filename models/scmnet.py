# coding=utf-8
"""SGMNet: SGV encoder, structure-calibrated CAPC-GSFPN and Unified HR-MDR."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:  # Package import when running from scmnet.
    from .vmamba import PatchEmbed2D, PatchMerging2D, VSSBlock
except Exception:  # Script-style import fallback.
    from vmamba import PatchEmbed2D, PatchMerging2D, VSSBlock


def is_mamba_available() -> bool:
    try:
        from . import vmamba as _vmamba
    except Exception:
        try:
            import vmamba as _vmamba
        except Exception:
            return False
    return callable(getattr(_vmamba, "selective_scan_fn", None)) or callable(getattr(_vmamba, "selective_scan_fn_v1", None))


def _gn_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _continuous_contact_evidence(logits: torch.Tensor, power: float) -> torch.Tensor:
    """Compute sigmoid(logits)**power without a singular half-precision backward.

    Under fp16 autocast, sigmoid can round to exactly zero. Applying a
    fractional power afterwards gives an infinite local derivative (and can
    produce a NaN gradient). The log-domain identity keeps the calculation
    in float32 until its bounded result is converted back to the cue dtype.
    """
    evidence = torch.exp(-float(power) * F.softplus(-logits.float()))
    return evidence.to(dtype=logits.dtype)


def _contact_excess_evidence(evidence: torch.Tensor, power: float) -> torch.Tensor:
    """Map only contact evidence above an uninformative zero logit to WHERE.

    A zero-initialized contact head gives sigmoid(0)**power, not zero. Using
    that raw value as spatial permission would open MDR everywhere before the
    contact head has learned any instance-gap evidence.
    """
    neutral = 0.5 ** float(power)
    excess = F.relu((evidence.float() - neutral) / (1.0 - neutral))
    return excess.to(dtype=evidence.dtype)


def _relative_contact_evidence(
    logits: torch.Tensor, power: float, gain: float
) -> torch.Tensor:
    """Use contact evidence above its own image-patch background.

    Instance gaps are sparse, so a calibrated contact probability can stay
    below 0.5 even at a true gap. The old zero-logit cutoff then erases it.
    Spatial centering preserves such local peaks while a constant, untrained
    contact map still contributes exactly zero permission. All reductions run
    in float32 so the fp16 training path remains finite.
    """
    evidence = _continuous_contact_evidence(logits, power).float()
    background = evidence.mean(dim=(-2, -1), keepdim=True)
    contrast = F.relu(evidence - background - 1e-3)
    permission = float(gain) * contrast / (1.0 - background).clamp_min(0.1)
    return permission.clamp(0.0, 1.0).to(dtype=logits.dtype)


def _relative_contact_content(logits: torch.Tensor) -> torch.Tensor:
    """Signed contact content with zero response from a constant cue map."""
    values = logits.float()
    centered = values - values.mean(dim=(-2, -1), keepdim=True)
    return torch.tanh(0.5 * centered).to(dtype=logits.dtype)


try:
    from .evidence import calibrated_contact_evidence, joint_instance_logits
except ImportError:
    from evidence import calibrated_contact_evidence, joint_instance_logits


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, *, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GlandDirectionalScale(nn.Module):
    """One gland-aware directional scale used around VSS stage outputs."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        channels = int(channels)
        kernel_size = max(3, int(kernel_size) | 1)
        pad = kernel_size // 2
        self.h_scan = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            groups=channels,
            bias=False,
        )
        self.v_scan = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            groups=channels,
            bias=False,
        )
        self.local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.axis_gate = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1), nn.Sigmoid())
        self.fuse = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, g_h: torch.Tensor | None = None, g_v: torch.Tensor | None = None) -> torch.Tensor:
        h = self.h_scan(x)
        v = self.v_scan(x)
        if g_h is not None:
            h = h * g_h
        if g_v is not None:
            v = v * g_v
        axis = self.axis_gate(torch.cat([h, v], dim=1))
        directional = h * axis + v * (1.0 - axis)
        return self.fuse(torch.cat([directional, self.local(x)], dim=1))


class StructureGuidedPVSSAdapter(nn.Module):
    """SGV: structure-guided adapter around a pretrained VSS block.

    The original VMamba/VSS block remains the core `ssm` path. Structure cues are
    injected only through small, identity-centered residual paths so pretrained
    VMamba weights remain loadable and stable.
    """

    def __init__(
        self,
        channels: int,
        drop_path: float,
        *,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        attn_drop_rate: float = 0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        scales: Sequence[int] = (3, 5, 9),
        compact_ratio: float = 1.0,
        scan_guidance: bool = False,
        residual_scale_init: float = 5e-3,
    ):
        super().__init__()
        channels = int(channels)
        scales = tuple(max(3, int(v) | 1) for v in scales)
        compact_ratio = float(compact_ratio)
        if not 0.0 < compact_ratio <= 1.0:
            raise ValueError("SGV compact_ratio must be in (0, 1].")
        branch_channels = channels if compact_ratio == 1.0 else max(16, int(round(channels * compact_ratio)))
        self.compact_ratio = compact_ratio
        self.scan_guidance = bool(scan_guidance)
        hidden = max(8, channels // 4)
        self.ssm = VSSBlock(
            hidden_dim=channels,
            drop_path=float(drop_path),
            norm_layer=norm_layer,
            attn_drop_rate=float(attn_drop_rate),
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.norm = nn.GroupNorm(_gn_groups(channels), channels)
        self.input_proj = ConvGNAct(channels, channels, kernel_size=1, padding=0)
        self.structure_gate = nn.Sequential(
            ConvGNAct(channels + 2, hidden, kernel_size=1, padding=0),
            ConvGNAct(hidden, hidden, kernel_size=3),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )
        nn.init.zeros_(self.structure_gate[-1].weight)
        nn.init.zeros_(self.structure_gate[-1].bias)
        if compact_ratio == 1.0:
            self.pre_condition = nn.Sequential(
                ConvGNAct(channels, channels, kernel_size=3),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
            self.branch_input_proj = nn.Identity()
        else:
            self.pre_condition = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.GroupNorm(_gn_groups(channels), channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
            self.branch_input_proj = ConvGNAct(channels, branch_channels, kernel_size=1, padding=0)
        nn.init.zeros_(self.pre_condition[-1].weight)
        nn.init.zeros_(self.pre_condition[-1].bias)
        self.branches = nn.ModuleList([GlandDirectionalScale(branch_channels, k) for k in scales])
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, len(scales), kernel_size=1),
        )
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)
        self.output_proj = ConvGNAct(branch_channels, channels, kernel_size=1, padding=0)
        self.pre_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.post_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.structure_prior_channels = 2 + len(scales)

    @staticmethod
    def _feature_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
        gy = x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
        gx[:, :, :, 1:] = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        gy[:, :, 1:, :] = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
        return gx, gy

    def _forward_impl(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        base = self.input_proj(self.norm(x_nchw))
        gx, gy = self._feature_gradients(base)
        structure_logits = self.structure_gate(torch.cat([base, gx, gy], dim=1))
        g_h, g_v = structure_logits.chunk(2, dim=1)
        g_h = 1.0 + 0.15 * torch.tanh(g_h)
        g_v = 1.0 + 0.15 * torch.tanh(g_v)

        x_in = x_nchw + self.pre_scale * self.pre_condition(base)
        ssm_input = x_in.permute(0, 2, 3, 1).contiguous()
        if self.scan_guidance:
            # SS2D orders its aligned outputs as horizontal forward/reverse,
            # then vertical forward/reverse. Zero-initialized structure logits
            # make all four gates exactly one at initialization.
            scan_gate = torch.cat([g_h, g_h, g_v, g_v], dim=1)
            y = self.ssm(ssm_input, scan_gate=scan_gate)
        else:
            y = self.ssm(ssm_input)
        y_nchw = y.permute(0, 3, 1, 2).contiguous()
        morph_base = self.input_proj(self.norm(y_nchw))

        weights = torch.softmax(self.scale_gate(morph_base), dim=1)
        branch_base = self.branch_input_proj(morph_base)
        fused = branch_base.new_zeros(branch_base.shape)
        for i, branch in enumerate(self.branches):
            fused = fused + weights[:, i:i + 1] * branch(branch_base, g_h, g_v)
        out = self.output_proj(fused)
        z = y_nchw + self.post_scale * out
        scale_prior = weights.expand(-1, -1, morph_base.shape[-2], morph_base.shape[-1])
        scale_prior = scale_prior - (1.0 / float(len(self.branches)))
        structure_prior = torch.cat([torch.tanh(structure_logits), scale_prior], dim=1)
        return z.permute(0, 2, 3, 1).contiguous(), structure_prior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)[0]

    def forward_with_prior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_impl(x)


GlandMorphologyAdapter = StructureGuidedPVSSAdapter
GlandSSMBlock = StructureGuidedPVSSAdapter


class VSSStage(nn.Module):
    """One VMamba/VSS stage that exposes the pre-downsample feature as a skip."""

    def __init__(
        self,
        dim: int,
        depth: int,
        drop_paths: Sequence[float],
        *,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        attn_drop_rate: float = 0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        downsample: bool = True,
        use_checkpoint: bool = False,
        use_gland_ssm_block: bool = False,
        gland_ssm_scales: Sequence[int] = (3, 5, 7),
        sgv_compact_ratio: float = 1.0,
        sgv_scan_guidance: bool = False,
        sgv_adapter_placement: str = "all",
        residual_scale_init: float = 5e-3,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        depth = int(depth)
        if sgv_adapter_placement not in {"all", "last"}:
            raise ValueError("sgv_adapter_placement must be 'all' or 'last'.")
        blocks = []
        for i in range(depth):
            drop_path = float(drop_paths[i]) if i < len(drop_paths) else 0.0
            if use_gland_ssm_block and (sgv_adapter_placement == "all" or i == depth - 1):
                blocks.append(
                    StructureGuidedPVSSAdapter(
                        dim,
                        drop_path,
                        d_state=int(d_state),
                        d_conv=int(d_conv),
                        expand=int(expand),
                        attn_drop_rate=float(attn_drop_rate),
                        norm_layer=norm_layer,
                        scales=tuple(int(v) for v in gland_ssm_scales),
                        compact_ratio=float(sgv_compact_ratio),
                        scan_guidance=bool(sgv_scan_guidance),
                        residual_scale_init=float(residual_scale_init),
                    )
                )
            else:
                blocks.append(
                    VSSBlock(
                        hidden_dim=dim,
                        drop_path=drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop_rate,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.downsample = PatchMerging2D(dim=dim, norm_layer=norm_layer) if downsample else None
        self.structure_prior_channels = 2 + len(tuple(int(v) for v in gland_ssm_scales))

    def forward(
        self,
        x: torch.Tensor,
        return_structure_prior: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        structure_prior = None
        for blk in self.blocks:
            if isinstance(blk, StructureGuidedPVSSAdapter) and return_structure_prior:
                if self.use_checkpoint and self.training:
                    x, structure_prior = checkpoint.checkpoint(blk.forward_with_prior, x, use_reentrant=False)
                else:
                    x, structure_prior = blk.forward_with_prior(x)
            else:
                if self.use_checkpoint and self.training:
                    x = checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        if return_structure_prior:
            if structure_prior is None:
                structure_prior = skip.new_zeros(
                    (skip.shape[0], self.structure_prior_channels, skip.shape[1], skip.shape[2])
                )
            return skip, x, structure_prior
        return skip, x, None


class VSSEncoder(nn.Module):
    """Hierarchical SS2D encoder with VMamba-compatible stage names."""

    def __init__(
        self,
        in_chans: int = 3,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 9, 2),
        dims: Sequence[int] | None = None,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        use_gland_ssm_block: bool = True,
        gland_ssm_block_stages: Sequence[int] = (0, 1, 2, 3),
        gland_ssm_scales: Sequence[int] = (3, 5, 7),
        sgv_compact_ratio: float = 1.0,
        sgv_scan_guidance: bool = False,
        sgv_adapter_placement: str = "all",
        residual_scale_init: float = 5e-3,
    ):
        super().__init__()
        if dims is None:
            dims = tuple(int(embed_dim * (2 ** i)) for i in range(len(depths)))
        self.dims = tuple(int(v) for v in dims)
        self.structure_prior_channels = 2 + len(tuple(int(v) for v in gland_ssm_scales))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.dims[0],
            norm_layer=norm_layer if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=float(drop_rate))
        dpr = [x.item() for x in torch.linspace(0, float(drop_path_rate), sum(depths))]

        self.stages = nn.ModuleList()
        gland_stages = {int(v) for v in gland_ssm_block_stages}
        offset = 0
        for i, depth in enumerate(depths):
            stage_paths = dpr[offset:offset + int(depth)]
            offset += int(depth)
            stage_enabled = bool(use_gland_ssm_block and i in gland_stages)
            self.stages.append(
                VSSStage(
                    self.dims[i],
                    int(depth),
                    stage_paths,
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=int(expand),
                    attn_drop_rate=float(attn_drop_rate),
                    norm_layer=norm_layer,
                    downsample=i < len(depths) - 1,
                    use_checkpoint=bool(use_checkpoint),
                    use_gland_ssm_block=stage_enabled,
                    gland_ssm_scales=tuple(int(v) for v in gland_ssm_scales),
                    sgv_compact_ratio=float(sgv_compact_ratio),
                    sgv_scan_guidance=bool(sgv_scan_guidance),
                    sgv_adapter_placement=str(sgv_adapter_placement),
                    residual_scale_init=float(residual_scale_init),
                )
            )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_structure_priors: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        feats = []
        structure_priors = []
        for stage in self.stages:
            skip, x, prior = stage(x, return_structure_prior=return_structure_priors)
            feats.append(skip.permute(0, 3, 1, 2).contiguous())
            if return_structure_priors:
                structure_priors.append(prior)
        if return_structure_priors:
            return feats, structure_priors
        return feats


class PPM(nn.Module):
    def __init__(self, in_channels: int, channels: int, pool_scales: Iterable[int] = (1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(int(scale)),
                    ConvGNAct(in_channels, channels, kernel_size=1, padding=0),
                )
                for scale in pool_scales
            ]
        )
        self.bottleneck = ConvGNAct(in_channels + len(self.branches) * channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        priors = [x]
        for branch in self.branches:
            priors.append(F.interpolate(branch(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(priors, dim=1))


class ContactAwarePyramidCorrection(nn.Module):
    """Correct one GSFPN top-down fusion at boundary/contact-sensitive sites.

    The block first embeds the lateral and top-down features into a common
    comparison space. Their absolute difference and correlation form a
    cross-scale discrepancy cue. Two supervised gates estimate boundary
    proximity and gland-contact risk; these gates restrict a bounded residual
    correction without spatial warping. A small learnable scale keeps the
    initial path close to the original additive GSFPN fusion.
    """

    def __init__(
        self,
        channels: int,
        cue_channels: int = 16,
        prior_channels: int = 5,
        residual_scale_init: float = 5e-3,
        contact_floor: float = 0.25,
        prior_scale_init: float = 0.10,
        use_discrepancy: bool = True,
        use_contact_cue: bool = True,
        qc_evidence_mode: str = "legacy",
        contact_evidence_power: float = 0.5,
        contact_contrast_gain: float = 2.0,
    ):
        super().__init__()
        channels = int(channels)
        cue_channels = int(cue_channels)
        if cue_channels <= 0:
            raise ValueError("cue_channels must be positive.")
        if not 0.0 <= float(contact_floor) <= 1.0:
            raise ValueError("contact_floor must be in [0, 1].")
        self.contact_floor = float(contact_floor)
        self.use_discrepancy = bool(use_discrepancy)
        self.use_contact_cue = bool(use_contact_cue)
        self.qc_evidence_mode = str(qc_evidence_mode)
        self.contact_evidence_power = float(contact_evidence_power)
        self.contact_contrast_gain = float(contact_contrast_gain)
        if self.qc_evidence_mode not in {"legacy", "continuous", "relative", "calibrated"}:
            raise ValueError("Unsupported qc_evidence_mode")
        if not 0.0 < self.contact_evidence_power <= 1.0:
            raise ValueError("contact_evidence_power must be in (0, 1].")
        if self.contact_contrast_gain <= 0.0:
            raise ValueError("contact_contrast_gain must be positive.")
        self.lateral_embedding = ConvGNAct(channels, cue_channels, kernel_size=1, padding=0)
        self.topdown_embedding = ConvGNAct(channels, cue_channels, kernel_size=1, padding=0)
        self.discrepancy_encoder = ConvGNAct(cue_channels * 2, cue_channels, kernel_size=3)
        predictor_channels = channels * 2 + cue_channels
        hidden_channels = max(16, channels // 2)
        self.gate_head = nn.Sequential(
            ConvGNAct(predictor_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        self.prior_to_boundary = nn.Conv2d(int(prior_channels), 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.prior_to_boundary.weight)
        nn.init.zeros_(self.prior_to_boundary.bias)
        self.prior_scale = nn.Parameter(torch.tensor(float(prior_scale_init)))
        self.correction_head = nn.Sequential(
            ConvGNAct(predictor_channels + 2, channels, kernel_size=3),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(
        self,
        lateral: torch.Tensor,
        topdown: torch.Tensor,
        structure_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if topdown.shape[-2:] != lateral.shape[-2:]:
            topdown = F.interpolate(topdown, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        lateral_embedding = F.normalize(self.lateral_embedding(lateral), p=2, dim=1, eps=1e-6)
        topdown_embedding = F.normalize(self.topdown_embedding(topdown), p=2, dim=1, eps=1e-6)
        discrepancy = self.discrepancy_encoder(
            torch.cat(
                [
                    torch.abs(lateral_embedding - topdown_embedding),
                    lateral_embedding * topdown_embedding,
                ],
                dim=1,
            )
        )
        if not self.use_discrepancy:
            # Equal-capacity ablation: preserve the branch while removing its evidence.
            discrepancy = torch.zeros_like(discrepancy)
        shared = torch.cat([lateral, topdown, discrepancy], dim=1)
        gate_logits = self.gate_head(shared)
        boundary_logits = gate_logits[:, 0:1]
        raw_contact_logits = gate_logits[:, 1:2]
        contact_logits = raw_contact_logits if self.use_contact_cue else torch.zeros_like(raw_contact_logits)
        prior_boundary_bias = boundary_logits.new_zeros(boundary_logits.shape)
        if structure_prior is not None:
            if structure_prior.shape[-2:] != lateral.shape[-2:]:
                structure_prior = F.interpolate(
                    structure_prior,
                    size=lateral.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            prior_boundary_bias = torch.tanh(self.prior_to_boundary(structure_prior))
            boundary_logits = boundary_logits + self.prior_scale * prior_boundary_bias
        boundary_gate = torch.sigmoid(boundary_logits)
        if self.use_contact_cue:
            contact_gate = torch.sigmoid(contact_logits)
            if self.qc_evidence_mode == "continuous":
                contact_gate = _continuous_contact_evidence(
                    contact_logits, self.contact_evidence_power
                )
            elif self.qc_evidence_mode == "relative":
                contact_gate = _relative_contact_evidence(
                    contact_logits, self.contact_evidence_power, self.contact_contrast_gain
                )
            elif self.qc_evidence_mode == "calibrated":
                contact_gate = calibrated_contact_evidence(
                    contact_logits, self.contact_evidence_power, self.contact_contrast_gain
                )
        else:
            # In relative mode, disabling C removes its extra correction
            # permission. Keep the historical neutral gate for old runs.
            contact_gate = (
                torch.zeros_like(contact_logits)
                if self.qc_evidence_mode in {"relative", "calibrated"}
                else torch.ones_like(contact_logits)
            )
        correction_gate = boundary_gate * (
            self.contact_floor + (1.0 - self.contact_floor) * contact_gate
        )
        correction = torch.tanh(
            self.correction_head(torch.cat([shared, boundary_gate, contact_gate], dim=1))
        )
        fused = lateral + topdown + self.residual_scale * correction_gate * correction
        diagnostics = {
            'capc_prior_boundary_abs_mean': prior_boundary_bias.abs().mean(),
            'capc_prior_scale': self.prior_scale,
            "capc_gate_mean": correction_gate.mean(),
            "capc_contact_evidence_mean": contact_gate.mean(),
            "capc_gate_map": correction_gate.detach(),
            "capc_contact_evidence_map": contact_gate.detach(),
            "capc_correction_abs_mean": correction.abs().mean(),
            "capc_discrepancy_mean": discrepancy.abs().mean(),
        }
        return fused, boundary_logits, contact_logits, diagnostics


class GlandSelectiveFPNHead(nn.Module):
    """Original GSFPN topology enhanced by decoder-internal CAPC nodes.

    PPM, top-down FPN and BSSF-style scale selection are treated as one decoder
    module rather than separate blocks.  The decoder keeps UPerNet's proven
    multi-scale skip topology, then learns spatial/channel selection for gland
    boundaries, touching regions and shallow texture noise.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 128,
        ppm_channels: int = 128,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        use_bssf_lite: bool = True,
        use_capc: bool = True,
        capc_cue_channels: int = 16,
        capc_contact_floor: float = 0.25,
        capc_use_discrepancy: bool = True,
        capc_use_contact_cue: bool = True,
        structure_prior_channels: int = 5,
        prior_scale_init: float = 0.10,
        residual_scale_init: float = 5e-3,
        capc_residual_scale_init: float = 1e-2,
        qc_evidence_mode: str = "legacy",
        contact_evidence_power: float = 0.5,
        contact_contrast_gain: float = 2.0,
    ):
        super().__init__()
        self.in_channels = tuple(int(c) for c in in_channels)
        self.ppm = PPM(self.in_channels[-1], int(ppm_channels), pool_scales=pool_scales)
        self.lateral_convs = nn.ModuleList(
            [ConvGNAct(c, channels, kernel_size=1, padding=0) for c in self.in_channels[:-1]]
        )
        self.ppm_lateral = ConvGNAct(int(ppm_channels), channels, kernel_size=1, padding=0)
        self.use_capc = bool(use_capc)
        self.capc_blocks = nn.ModuleList(
            [
                ContactAwarePyramidCorrection(
                    channels=int(channels),
                    cue_channels=int(capc_cue_channels),
                    prior_channels=int(structure_prior_channels),
                    residual_scale_init=float(capc_residual_scale_init),
                    contact_floor=float(capc_contact_floor),
                    prior_scale_init=float(prior_scale_init),
                    use_discrepancy=bool(capc_use_discrepancy),
                    use_contact_cue=bool(capc_use_contact_cue),
                    qc_evidence_mode=qc_evidence_mode,
                    contact_evidence_power=contact_evidence_power,
                    contact_contrast_gain=contact_contrast_gain,
                )
                for _ in self.in_channels[:-1]
            ]
        ) if self.use_capc else nn.ModuleList()
        self.fpn_convs = nn.ModuleList([ConvGNAct(channels, channels, kernel_size=3) for _ in self.in_channels[:-1]])
        self.skip_selector = BSSFScaleSelectiveFusion(
            num_scales=len(self.in_channels),
            channels=int(channels),
            residual_scale_init=float(residual_scale_init),
        ) if bool(use_bssf_lite) else nn.Identity()
        self.fpn_bottleneck = ConvGNAct(len(self.in_channels) * channels, channels, kernel_size=3)

    def forward(
        self,
        feats: Sequence[torch.Tensor],
        structure_priors: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if len(feats) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} encoder features, got {len(feats)}.")
        if structure_priors is not None and len(structure_priors) != len(feats):
            raise ValueError(f"Expected {len(feats)} structure priors, got {len(structure_priors)}.")
        laterals = [conv(feats[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(self.ppm_lateral(self.ppm(feats[-1])))

        boundary_logits = []
        contact_logits = []
        capc_gate_maps = []
        capc_contact_evidence_maps = []
        diagnostic_lists: dict[str, list[torch.Tensor]] = {
            "capc_gate_mean": [],
            "capc_correction_abs_mean": [],
            "capc_discrepancy_mean": [],
        }
        for i in range(len(laterals) - 1, 0, -1):
            topdown = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False
            )
            if self.use_capc:
                stage_prior = structure_priors[i - 1] if structure_priors is not None else None
                fused, boundary_logit, contact_logit, diagnostics = self.capc_blocks[i - 1](
                    laterals[i - 1], topdown, stage_prior
                )
                laterals[i - 1] = fused
                boundary_logits.append(boundary_logit)
                contact_logits.append(contact_logit)
                capc_gate_maps.append(diagnostics.pop("capc_gate_map"))
                capc_contact_evidence_maps.append(diagnostics.pop("capc_contact_evidence_map"))
                for key, value in diagnostics.items():
                    diagnostic_lists.setdefault(key, []).append(value)
            else:
                laterals[i - 1] = laterals[i - 1] + topdown

        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(laterals[-1])
        out_size = fpn_outs[0].shape[-2:]
        fpn_outs = [
            F.interpolate(x, size=out_size, mode="bilinear", align_corners=False) if x.shape[-2:] != out_size else x
            for x in fpn_outs
        ]
        fpn_outs = self.skip_selector(fpn_outs)
        diagnostics: dict[str, object] = {
            "capc_boundary_logits": list(reversed(boundary_logits)),
            "capc_contact_logits": list(reversed(contact_logits)),
            "capc_gate_maps": list(reversed(capc_gate_maps)),
            "capc_contact_evidence_maps": list(reversed(capc_contact_evidence_maps)),
        }
        for key, values in diagnostic_lists.items():
            if values:
                diagnostics[key] = torch.stack(values).mean()
        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1)), diagnostics


class BSSFScaleSelectiveFusion(nn.Module):
    """BSSF-Lite: scale-selective FPN fusion with spatial and channel gates.

    The module keeps the proven UPer-FPN topology and only reweights each
    upsampled scale by a tiny residual gate. This makes the initial behavior
    close to the baseline while giving training a learnable multi-scale
    selection mechanism.
    """

    def __init__(self, num_scales: int, channels: int, residual_scale_init: float = 5e-3):
        super().__init__()
        self.spatial_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, max(1, channels // 4), kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_gn_groups(max(1, channels // 4)), max(1, channels // 4)),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(max(1, channels // 4), 1, kernel_size=1),
                    nn.Sigmoid(),
                )
                for _ in range(int(num_scales))
            ]
        )
        self.channel_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels, max(1, channels // 8), kernel_size=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(max(1, channels // 8), channels, kernel_size=1),
                    nn.Sigmoid(),
                )
                for _ in range(int(num_scales))
            ]
        )
        self.residual_scale = nn.Parameter(torch.full((int(num_scales),), float(residual_scale_init)))

    def forward(self, feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(feats) != len(self.spatial_gates):
            raise ValueError(f"Expected {len(self.spatial_gates)} FPN scales, got {len(feats)}.")
        selected = []
        for i, feat in enumerate(feats):
            gate = self.spatial_gates[i](feat) * self.channel_gates[i](feat)
            gate = 2.0 * gate - 1.0
            selected.append(feat * (1.0 + self.residual_scale[i].view(1, 1, 1, 1) * gate))
        return selected


class DFSLiteBlock(nn.Module):
    """DFS-Lite: lightweight horizontal/vertical directional fusion."""

    def __init__(self, channels: int, kernel_size: int = 7, residual_scale_init: float = 5e-3):
        super().__init__()
        channels = int(channels)
        padding = int(kernel_size) // 2
        self.norm = nn.GroupNorm(_gn_groups(channels), channels)
        self.h_scan = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, int(kernel_size)),
            padding=(0, padding),
            groups=channels,
            bias=False,
        )
        self.v_scan = nn.Conv2d(
            channels,
            channels,
            kernel_size=(int(kernel_size), 1),
            padding=(padding, 0),
            groups=channels,
            bias=False,
        )
        self.axis_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mix = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.norm(x)
        h = self.h_scan(base)
        v = self.v_scan(base)
        gate = self.axis_gate(torch.cat([h, v], dim=1))
        fused = self.mix(torch.cat([h * gate, v * (1.0 - gate)], dim=1))
        return x + self.residual_scale * fused


class UnifiedHRMDR(nn.Module):
    """One H/2 feature repair using morphology, image detail, and CAPC cues."""

    def __init__(
        self,
        image_channels: int,
        decoder_channels: int,
        context_channels: int,
        dilations: Sequence[int] = (1, 3, 5),
        min_gate: float = 0.05,
        what_gradient_scale: float = 0.50,
        image_feature_channels: int = 32,
        cue_feature_channels: int = 16,
        num_cue_scales: int = 3,
        use_cue_guidance: bool = True,
        use_contact_cue: bool = True,
        qc_evidence_mode: str = "legacy",
        contact_evidence_power: float = 0.5,
        contact_contrast_gain: float = 2.0,
    ):
        super().__init__()
        decoder_channels = int(decoder_channels)
        image_feature_channels = int(image_feature_channels)
        cue_feature_channels = int(cue_feature_channels)
        num_cue_scales = int(num_cue_scales)
        dilations = tuple(int(value) for value in dilations)
        if not dilations:
            raise ValueError("Unified HR-MDR requires at least one morphology dilation.")
        if num_cue_scales < 1:
            raise ValueError("num_cue_scales must be positive.")
        if not 0.0 <= float(min_gate) <= 1.0:
            raise ValueError("min_gate must be in [0, 1].")
        if not 0.0 <= float(what_gradient_scale) <= 1.0:
            raise ValueError("what_gradient_scale must be in [0, 1].")
        self.min_gate = float(min_gate)
        self.what_gradient_scale = float(what_gradient_scale)
        self.dilations = dilations
        self.num_cue_scales = num_cue_scales
        self.use_cue_guidance = bool(use_cue_guidance)
        self.use_contact_cue = bool(use_contact_cue)
        self.qc_evidence_mode = str(qc_evidence_mode)
        self.contact_evidence_power = float(contact_evidence_power)
        self.contact_contrast_gain = float(contact_contrast_gain)
        self.context_proj = ConvGNAct(int(context_channels), decoder_channels, kernel_size=1, padding=0)
        self.context_fuse = ConvGNAct(decoder_channels * 2, decoder_channels, kernel_size=1, padding=0)
        self.morphology_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    decoder_channels,
                    decoder_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=decoder_channels,
                    bias=False,
                ),
                nn.GroupNorm(_gn_groups(decoder_channels), decoder_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1, bias=False),
                nn.GroupNorm(_gn_groups(decoder_channels), decoder_channels),
                nn.SiLU(inplace=True),
            )
            for dilation in dilations
        ])
        morphology_channels = max(32, decoder_channels // 2)
        self.morphology_fuse = ConvGNAct(
            decoder_channels * len(self.morphology_branches), morphology_channels, kernel_size=1, padding=0
        )
        self.base_projection = ConvGNAct(decoder_channels, decoder_channels, kernel_size=1, padding=0)
        image_stem_mid_channels = max(16, image_feature_channels * 3 // 4)
        self.image_stem = nn.Sequential(
            nn.Conv2d(int(image_channels), image_stem_mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(image_stem_mid_channels), image_stem_mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(image_stem_mid_channels, image_feature_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(image_feature_channels), image_feature_channels),
            nn.SiLU(inplace=True),
        )
        self.cue_proj = nn.Sequential(
            nn.Conv2d(2, cue_feature_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(cue_feature_channels), cue_feature_channels, affine=False),
            nn.SiLU(inplace=True),
        )
        fine_biased_logits = torch.zeros(num_cue_scales)
        fine_biased_logits[0] = 1.38629436112
        self.what_q_scale_logits = nn.Parameter(fine_biased_logits.clone())
        self.what_c_scale_logits = nn.Parameter(fine_biased_logits.clone())
        self.where_q_scale_logits = nn.Parameter(fine_biased_logits.clone())
        self.where_c_scale_logits = nn.Parameter(fine_biased_logits.clone())
        self.permission_alpha_logit = nn.Parameter(torch.tensor(0.8472978604))
        self.contact_beta_logit = nn.Parameter(torch.tensor(0.6190392084))
        repair_in_channels = decoder_channels + morphology_channels + image_feature_channels + cue_feature_channels
        self.repair_generator = nn.Sequential(
            ConvGNAct(repair_in_channels, decoder_channels, kernel_size=3),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, groups=decoder_channels, bias=False),
            nn.GroupNorm(_gn_groups(decoder_channels), decoder_channels),
            nn.SiLU(inplace=True),
            ConvGNAct(decoder_channels, decoder_channels, kernel_size=1, padding=0),
        )
        self.repair_out = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1)
        nn.init.zeros_(self.repair_out.weight)
        nn.init.zeros_(self.repair_out.bias)

    @staticmethod
    def _resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if value.shape[-2:] == size:
            return value
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(value.dtype)
        return (value * weights).sum() / weights.expand_as(value).sum().clamp_min(1.0)

    @staticmethod
    def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
        return value.detach() + float(scale) * (value - value.detach())

    def _validate_cue_scales(self, cues: Sequence[torch.Tensor], name: str) -> list[torch.Tensor]:
        if len(cues) != self.num_cue_scales:
            raise RuntimeError(f"Unified HR-MDR expects {self.num_cue_scales} {name} cue scales, got {len(cues)}.")
        return list(cues)

    def _fuse_cues(
        self,
        boundary_logits: Sequence[torch.Tensor],
        contact_logits: Sequence[torch.Tensor],
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        boundary_logits = self._validate_cue_scales(boundary_logits, "boundary")
        contact_logits = self._validate_cue_scales(contact_logits, "contact")
        q_items = [torch.tanh(0.5 * self._resize(item, size)) for item in boundary_logits]
        if self.qc_evidence_mode == "calibrated":
            # WHAT retains the calibrated signed semantics of the original C.
            # WHERE adds local contrast without subtracting a whole-tile mean.
            c_items = [torch.tanh(0.5*self._resize(item, size)) for item in contact_logits]
            c_permission = [calibrated_contact_evidence(
                self._resize(item, size), self.contact_evidence_power, self.contact_contrast_gain
            ) for item in contact_logits]
        elif self.qc_evidence_mode == "relative":
            c_items = [
                _relative_contact_content(self._resize(item, size))
                for item in contact_logits
            ]
            c_permission = [
                _relative_contact_evidence(
                    self._resize(item, size),
                    self.contact_evidence_power,
                    self.contact_contrast_gain,
                )
                for item in contact_logits
            ]
        elif self.qc_evidence_mode == "continuous":
            c_evidence = [
                _continuous_contact_evidence(self._resize(item, size), self.contact_evidence_power)
                for item in contact_logits
            ]
            c_items = [torch.tanh(0.5 * self._resize(item, size)) for item in contact_logits]
            c_permission = [
                _contact_excess_evidence(item, self.contact_evidence_power)
                for item in c_evidence
            ]
        else:
            c_items = [torch.tanh(0.5 * self._resize(item, size)) for item in contact_logits]
            c_evidence = [F.relu(item) for item in c_items]
            c_permission = c_evidence
        what_q_weights = torch.softmax(self.what_q_scale_logits, dim=0)
        what_c_weights = torch.softmax(self.what_c_scale_logits, dim=0)
        where_q_weights = torch.softmax(self.where_q_scale_logits, dim=0)
        where_c_weights = torch.softmax(self.where_c_scale_logits, dim=0)
        q_sem = sum(weight * self._scale_gradient(item, self.what_gradient_scale) for weight, item in zip(what_q_weights, q_items))
        c_sem = sum(weight * self._scale_gradient(item, self.what_gradient_scale) for weight, item in zip(what_c_weights, c_items))
        q_where = sum(weight * F.relu(item.detach()) for weight, item in zip(where_q_weights, q_items))
        c_where = sum(weight * item.detach() for weight, item in zip(where_c_weights, c_permission))
        if not self.use_contact_cue:
            c_sem = torch.zeros_like(c_sem)
            c_where = torch.zeros_like(c_where)
        diagnostics = {}
        for index, scale_name in enumerate(("fine", "mid", "coarse")):
            if index >= self.num_cue_scales:
                break
            diagnostics[f"hr_what_q_weight_{scale_name}"] = what_q_weights[index]
            diagnostics[f"hr_what_c_weight_{scale_name}"] = what_c_weights[index]
            diagnostics[f"hr_where_q_weight_{scale_name}"] = where_q_weights[index]
            diagnostics[f"hr_where_c_weight_{scale_name}"] = where_c_weights[index]
        return q_sem, c_sem, q_where, c_where, diagnostics

    def forward(
        self,
        image: torch.Tensor,
        decoder_feature: torch.Tensor,
        deep_context: torch.Tensor,
        capc_boundary_logits: Sequence[torch.Tensor],
        capc_contact_logits: Sequence[torch.Tensor],
        return_visual_maps: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_detail = self.image_stem(image)
        repair_size = image_detail.shape[-2:]
        base_hr = self._resize(self.base_projection(decoder_feature), repair_size)
        context = self._resize(deep_context, decoder_feature.shape[-2:])
        morphology_base = self.context_fuse(torch.cat([decoder_feature, self.context_proj(context)], dim=1))
        morphology = self.morphology_fuse(torch.cat([branch(morphology_base) for branch in self.morphology_branches], dim=1))
        morphology = self._resize(morphology, repair_size)
        q_sem, c_sem, q_where, c_where, cue_diagnostics = self._fuse_cues(
            capc_boundary_logits, capc_contact_logits, repair_size
        )
        if not self.use_cue_guidance:
            # Equal-capacity ablation: retain HR repair capacity but remove Q/C guidance.
            q_sem = torch.zeros_like(q_sem)
            c_sem = torch.zeros_like(c_sem)
            q_where = torch.zeros_like(q_where)
            c_where = torch.zeros_like(c_where)
        cue_feature = self.cue_proj(torch.cat([q_sem, c_sem], dim=1))
        alpha = torch.sigmoid(self.permission_alpha_logit)
        beta = torch.sigmoid(self.contact_beta_logit)
        boundary_evidence = alpha * q_where + (1.0 - alpha) * F.max_pool2d(q_where, kernel_size=5, stride=1, padding=2)
        boundary_evidence = boundary_evidence.clamp(0.0, 1.0)
        permission = 1.0 - (1.0 - boundary_evidence) * (1.0 - beta * c_where)
        permission = permission.clamp(0.0, 1.0)
        gate = self.min_gate + (1.0 - self.min_gate) * permission
        if not self.use_cue_guidance:
            permission = torch.ones_like(permission)
            gate = torch.ones_like(gate)
        delta = torch.tanh(self.repair_out(self.repair_generator(torch.cat([
            base_hr, morphology, image_detail, cue_feature
        ], dim=1))))
        gated_delta = gate * delta
        refined = base_hr + gated_delta
        emphasis = permission
        active = emphasis > 0.10
        sampled_emphasis = emphasis.detach().float().flatten()[::64]
        emphasis_p95 = torch.quantile(sampled_emphasis, 0.95) if sampled_emphasis.numel() else emphasis.new_zeros(())
        contact_only = (boundary_evidence < 0.10) & (c_where > 0.10)
        contact_extra_permission = (permission - boundary_evidence).clamp_min(0.0)
        contact_extra_gate = (1.0 - self.min_gate) * contact_extra_permission
        repair_ratio = gated_delta.abs().mean() / base_hr.abs().mean().clamp_min(1e-6)
        contact_repair_ratio = self._masked_mean(gated_delta.abs(), contact_only) / self._masked_mean(
            base_hr.abs(), contact_only
        ).clamp_min(1e-6)
        diagnostics = {
            "hr_gate_map": gate.detach(),
            "hr_gate_mean": gate.mean(),
            "hr_gate_emphasis_mean": emphasis.mean(),
            "hr_gate_emphasis_active_ratio": active.float().mean(),
            "hr_gate_emphasis_p95": emphasis_p95,
            "hr_boundary_evidence_mean": boundary_evidence.mean(),
            "hr_contact_evidence_mean": c_where.mean(),
            "hr_signed_cue_abs_mean": 0.5 * (q_sem.abs().mean() + c_sem.abs().mean()),
            "hr_delta_abs_mean": delta.abs().mean(),
            "hr_gated_delta_abs_mean": gated_delta.abs().mean(),
            "hr_repair_ratio": repair_ratio,
            "hr_contact_only_ratio": contact_only.float().mean(),
            "hr_contact_only_emphasis_mean": self._masked_mean(emphasis, contact_only),
            "hr_contact_only_delta_abs_mean": self._masked_mean(delta.abs(), contact_only),
            "hr_contact_only_gated_delta_abs_mean": self._masked_mean(gated_delta.abs(), contact_only),
            "hr_contact_only_repair_ratio": contact_repair_ratio,
            "hr_contact_extra_permission_mean": contact_extra_permission.mean(),
            "hr_contact_extra_gate_mean": contact_extra_gate.mean(),
            "hr_permission_alpha": alpha,
            "hr_contact_beta": beta,
            "hr_what_gradient_scale": gate.new_tensor(self.what_gradient_scale),
        }
        diagnostics.update(cue_diagnostics)
        if return_visual_maps:
            diagnostics.update({
                "hr_boundary_evidence_map": boundary_evidence.detach(),
                "hr_contact_evidence_map": c_where.detach(),
                "hr_contact_extra_gate_map": contact_extra_gate.detach(),
                "hr_gated_delta_abs_map": gated_delta.detach().abs().mean(dim=1, keepdim=True),
            })
        return refined, diagnostics


class SegHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.head = nn.Sequential(
            ConvGNAct(in_ch, in_ch, kernel_size=3),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DSMNetRes(nn.Module):
    """SGV encoder with CAPC-enhanced GSFPN and unified HR-MDR repair."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 9, 2),
        dims: Sequence[int] | None = None,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        decoder_channels: int = 128,
        ppm_channels: int = 128,
        ppm_pool_scales: Sequence[int] = (1, 2, 3, 6),
        decoder_variant: str = "gsfpn",
        use_checkpoint: bool = False,
        use_separation_head: bool = True,
        use_gland_ssm_block: bool = True,
        gland_ssm_block_stages: Sequence[int] = (0, 1, 2, 3),
        gland_ssm_scales: Sequence[int] = (3, 5, 7),
        sgv_compact_ratio: float = 1.0,
        sgv_scan_guidance: bool = False,
        sgv_adapter_placement: str = "all",
        use_bssf_lite: bool = False,
        repair_variant: str = "unified_hrmdr",
        residual_scale_init: float = 5e-3,
        capc_residual_scale_init: float = 1e-2,
        capc_prior_scale_init: float = 0.10,
        use_capc: bool = True,
        hr_min_gate: float = 0.05,
        hr_what_gradient_scale: float = 0.50,
        hr_morphology_dilations: Sequence[int] = (1, 3, 5),
        hr_image_feature_channels: int = 32,
        hr_cue_feature_channels: int = 16,
        capc_cue_channels: int = 16,
        capc_contact_floor: float = 0.25,
        capc_use_discrepancy: bool = True,
        capc_use_contact_cue: bool = True,
        capc_use_sgv_prior: bool = True,
        hr_use_cue_guidance: bool = True,
        qc_evidence_mode: str = "legacy",
        contact_evidence_power: float = 0.5,
        contact_contrast_gain: float = 2.0,
        **_: object,
    ):
        super().__init__()
        repair_variant = str(repair_variant).lower().replace("_", "").replace("-", "")
        if repair_variant in {"unified", "hrmdr"}:
            repair_variant = "unifiedhrmdr"
        if repair_variant not in {"unifiedhrmdr", "none"}:
            raise ValueError(f"Unsupported repair_variant: {repair_variant}")
        if repair_variant == "unifiedhrmdr" and not use_capc:
            raise ValueError("Unified HR-MDR requires use_capc=True.")
        if qc_evidence_mode not in {"legacy", "continuous", "relative", "calibrated"}:
            raise ValueError("Unsupported qc_evidence_mode")
        if not 0.0 < float(contact_evidence_power) <= 1.0:
            raise ValueError("contact_evidence_power must be in (0, 1].")
        if float(contact_contrast_gain) <= 0.0:
            raise ValueError("contact_contrast_gain must be positive.")
        decoder_variant = str(decoder_variant or "gsfpn").lower().replace("_", "").replace("-", "")
        if decoder_variant not in {"gsfpn", "legacy", "uperfpn", "upernet"}:
            raise ValueError(f"Unsupported decoder_variant: {decoder_variant}")
        decoder_variant = "gsfpn"
        if dims is None:
            dims = tuple(int(embed_dim * (2 ** i)) for i in range(len(depths)))
        self.model_config = {
            'capc_residual_scale_init': float(capc_residual_scale_init),
            'capc_prior_scale_init': float(capc_prior_scale_init),
            "input_channels": int(input_channels),
            "num_classes": int(num_classes),
            "patch_size": int(patch_size),
            "embed_dim": int(embed_dim),
            "depths": tuple(int(v) for v in depths),
            "dims": tuple(int(v) for v in dims),
            "d_state": int(d_state),
            "d_conv": int(d_conv),
            "expand": int(expand),
            "drop_rate": float(drop_rate),
            "attn_drop_rate": float(attn_drop_rate),
            "drop_path_rate": float(drop_path_rate),
            "decoder_channels": int(decoder_channels),
            "ppm_channels": int(ppm_channels),
            "ppm_pool_scales": tuple(int(v) for v in ppm_pool_scales),
            "decoder_variant": decoder_variant,
            "use_checkpoint": bool(use_checkpoint),
            "use_separation_head": bool(use_separation_head),
            "use_gland_ssm_block": bool(use_gland_ssm_block),
            "gland_ssm_block_stages": tuple(int(v) for v in gland_ssm_block_stages),
            "gland_ssm_scales": tuple(int(v) for v in gland_ssm_scales),
            "sgv_compact_ratio": float(sgv_compact_ratio),
            "sgv_scan_guidance": bool(sgv_scan_guidance),
            "sgv_adapter_placement": str(sgv_adapter_placement),
            "use_bssf_lite": bool(use_bssf_lite),
            "repair_variant": repair_variant,
            "residual_scale_init": float(residual_scale_init),
            "use_capc": bool(use_capc),
            "hr_min_gate": float(hr_min_gate),
            "hr_what_gradient_scale": float(hr_what_gradient_scale),
            "hr_morphology_dilations": tuple(int(value) for value in hr_morphology_dilations),
            "hr_image_feature_channels": int(hr_image_feature_channels),
            "hr_cue_feature_channels": int(hr_cue_feature_channels),
            "capc_cue_channels": int(capc_cue_channels),
            "capc_contact_floor": float(capc_contact_floor),
            "capc_use_discrepancy": bool(capc_use_discrepancy),
            "capc_use_contact_cue": bool(capc_use_contact_cue),
            "capc_use_sgv_prior": bool(capc_use_sgv_prior),
            "hr_use_cue_guidance": bool(hr_use_cue_guidance),
            "qc_evidence_mode": str(qc_evidence_mode),
            "contact_evidence_power": float(contact_evidence_power),
            "contact_contrast_gain": float(contact_contrast_gain),
            "encoder": "VMamba-style SS2D/VSS",
            "adapter": "SGV: structure-guided pretrained VSS adapter",
            "skip_decoder": "structure-calibrated CAPC-GSFPN: SGV prior calibrates CAPC boundary evidence",
            "scale_fusion": "original GSFPN scale-selective fusion",
            "repair_head": "Unified HR-MDR" if repair_variant == "unifiedhrmdr" else "none",
        }
        self.use_separation_head = bool(use_separation_head)
        self.capc_use_sgv_prior = bool(capc_use_sgv_prior)
        self.repair_variant = repair_variant
        self.decoder_variant = decoder_variant
        self.encoder = VSSEncoder(
            in_chans=int(input_channels),
            patch_size=int(patch_size),
            embed_dim=int(embed_dim),
            depths=tuple(int(v) for v in depths),
            dims=tuple(int(v) for v in dims),
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            drop_rate=float(drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            drop_path_rate=float(drop_path_rate),
            patch_norm=True,
            use_checkpoint=bool(use_checkpoint),
            use_gland_ssm_block=bool(use_gland_ssm_block),
            gland_ssm_block_stages=tuple(int(v) for v in gland_ssm_block_stages),
            gland_ssm_scales=tuple(int(v) for v in gland_ssm_scales),
            sgv_compact_ratio=float(sgv_compact_ratio),
            sgv_scan_guidance=bool(sgv_scan_guidance),
            sgv_adapter_placement=str(sgv_adapter_placement),
            residual_scale_init=float(residual_scale_init),
        )
        self.decode_head = GlandSelectiveFPNHead(
            in_channels=tuple(int(v) for v in dims),
            channels=int(decoder_channels),
            ppm_channels=int(ppm_channels),
            pool_scales=tuple(int(v) for v in ppm_pool_scales),
            use_bssf_lite=bool(use_bssf_lite),
            use_capc=bool(use_capc),
            capc_cue_channels=int(capc_cue_channels),
            capc_contact_floor=float(capc_contact_floor),
            capc_use_discrepancy=bool(capc_use_discrepancy),
            capc_use_contact_cue=bool(capc_use_contact_cue),
            qc_evidence_mode=qc_evidence_mode,
            contact_evidence_power=contact_evidence_power,
            contact_contrast_gain=contact_contrast_gain,
            structure_prior_channels=int(self.encoder.structure_prior_channels),
            prior_scale_init=float(capc_prior_scale_init),
            residual_scale_init=float(residual_scale_init),
            capc_residual_scale_init=float(capc_residual_scale_init),
        )
        self.main_head = SegHead(int(decoder_channels), int(num_classes))
        self.boundary_head = SegHead(int(decoder_channels), 1)
        self.hr_mdr = UnifiedHRMDR(
            image_channels=int(input_channels),
            decoder_channels=int(decoder_channels),
            context_channels=int(dims[-1]),
            dilations=tuple(int(value) for value in hr_morphology_dilations),
            min_gate=float(hr_min_gate),
            what_gradient_scale=float(hr_what_gradient_scale),
            image_feature_channels=int(hr_image_feature_channels),
            cue_feature_channels=int(hr_cue_feature_channels),
            num_cue_scales=len(dims) - 1,
            use_cue_guidance=bool(hr_use_cue_guidance),
            use_contact_cue=bool(capc_use_contact_cue),
            qc_evidence_mode=qc_evidence_mode,
            contact_evidence_power=contact_evidence_power,
            contact_contrast_gain=contact_contrast_gain,
        ) if self.repair_variant == "unifiedhrmdr" else None

    def load_vmamba_pretrained(self, ckpt_path: str | Path, *, strict_shape: bool = True) -> dict[str, int]:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            # The original VMamba checkpoint includes YACS configuration metadata.
            # Older PyTorch versions use the legacy loader for this trusted file.
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        else:
            from yacs.config import CfgNode

            with safe_globals([CfgNode]):
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)))
        else:
            state = ckpt
        own = self.state_dict()
        loaded = {}

        def add_wrapped_ssm_candidate(items: list[str], name: str) -> None:
            items.append(name)
            parts = name.split(".")
            for i, part in enumerate(parts[:-2]):
                if part == "blocks" and i + 1 < len(parts):
                    items.append(".".join(parts[:i + 2] + ["ssm"] + parts[i + 2:]))
                    return

        for raw_key, value in state.items():
            if not torch.is_tensor(value):
                continue
            key = raw_key.replace("module.", "")
            for prefix in ("module.", "model.", "net.", "vmunet.", "backbone.", "encoder."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            candidates = []
            add_wrapped_ssm_candidate(candidates, key)
            if key.startswith("patch_embed."):
                add_wrapped_ssm_candidate(candidates, "encoder." + key)
            if key.startswith("layers."):
                mapped = key.replace("layers.", "encoder.stages.", 1)
                add_wrapped_ssm_candidate(candidates, mapped)
            if key.startswith("stages."):
                add_wrapped_ssm_candidate(candidates, "encoder." + key)
            for cand in candidates:
                if cand in own and (not strict_shape or own[cand].shape == value.shape):
                    if own[cand].shape == value.shape:
                        loaded[cand] = value
                        break
        missing_before = len(own) - len(loaded)
        own.update(loaded)
        self.load_state_dict(own, strict=False)
        def is_custom_tensor(key: str) -> bool:
            if (
                "dfs_blocks" in key
                or "skip_selector" in key
                or "decode_head" in key
                or "hr_mdr" in key
            ):
                return True
            if ".blocks." not in key or ".ssm." in key:
                return False
            return any(
                name in key
                for name in (
                    "input_proj",
                    "branch_input_proj",
                    "branches",
                    "scale_gate",
                    "structure_gate",
                    "pre_condition",
                    "output_proj",
                    "pre_scale",
                    "post_scale",
                )
            )

        custom_tensors = sum(1 for key in own if is_custom_tensor(key))
        return {
            "loaded": len(loaded),
            "total_model_tensors": len(own),
            "unmatched_model_tensors": missing_before,
            "new_custom_tensors_initialized": custom_tensors,
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False, aux_mode: str | None = None):
        input_hw = x.shape[-2:]
        feats, structure_priors = self.encoder(x, return_structure_priors=True)
        # Path intervention: retain SGV features and CAPC capacity, but omit
        # only the explicit SGV structure-prior input to CAPC.
        decoded, diagnostics = self.decode_head(
            feats,
            structure_priors=structure_priors if self.capc_use_sgv_prior else None,
        )
        capc_boundary_logits = diagnostics.pop("capc_boundary_logits", [])
        capc_contact_logits = diagnostics.pop("capc_contact_logits", [])
        capc_gate_maps = diagnostics.pop("capc_gate_maps", [])
        capc_contact_evidence_maps = diagnostics.pop("capc_contact_evidence_maps", [])
        visual_mode = str(aux_mode or "").lower() in {"visual", "viz", "visualization"}
        hr_diagnostics: dict[str, torch.Tensor] = {}
        refined = decoded
        if self.hr_mdr is not None:
            if not capc_boundary_logits or not capc_contact_logits:
                raise RuntimeError("Unified HR-MDR requires CAPC boundary and contact logits.")
            refined, hr_diagnostics = self.hr_mdr(
                image=x,
                decoder_feature=decoded,
                deep_context=feats[-1],
                capc_boundary_logits=capc_boundary_logits,
                capc_contact_logits=capc_contact_logits,
                return_visual_maps=visual_mode,
            )

        def upsample(logits: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
            if logits.shape[-2:] == input_hw:
                return logits
            return F.interpolate(logits, size=input_hw, mode=mode, align_corners=False if mode == "bilinear" else None)

        main_logits = upsample(self.main_head(refined))
        boundary_logits = upsample(self.boundary_head(refined))
        aux = {"boundary_logits": boundary_logits}
        if return_aux:
            # The instance mask used by the contract variant is trained from
            # this same foreground/boundary probability, with no new head.
            aux["instance_logits"] = joint_instance_logits(main_logits, boundary_logits)
            aux["instance_probability"] = torch.sigmoid(aux["instance_logits"])
        aux.update(hr_diagnostics)
        if visual_mode:
            aux["structure_priors"] = [prior.detach() for prior in structure_priors]
            aux["capc_gate_maps"] = capc_gate_maps
            aux["capc_contact_evidence_maps"] = capc_contact_evidence_maps
        if capc_boundary_logits:
            aux["capc_boundary_logits"] = capc_boundary_logits
            aux["boundary_distance_logits"] = upsample(capc_boundary_logits[0])
        if capc_contact_logits:
            aux["capc_contact_logits"] = capc_contact_logits
            aux["contact_logits"] = upsample(capc_contact_logits[0])
        if capc_contact_logits and self.use_separation_head:
            contact_logits = aux["contact_logits"]
            aux["separation_core_logits"] = contact_logits
            aux["separation_field_logits"] = contact_logits
        aux.update(diagnostics)
        if return_aux:
            return main_logits, aux
        return main_logits


    def get_runtime_info(self) -> dict[str, object]:
        return {
            "name": "SGMNet",
            "encoder": "VMamba-style SS2D/VSS",
            "adapter": "SGV",
            "skip_decoder": self.model_config["skip_decoder"],
            "selective_scan_available": is_mamba_available(),
            **self.model_config,
        }


VMUNet = DSMNetRes
DSMNetFBAux = DSMNetRes
