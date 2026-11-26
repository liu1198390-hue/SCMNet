# coding=utf-8
"""Losses for foreground/boundary prediction and CAPC gate supervision."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_outputs(outputs):
    if isinstance(outputs, tuple) and len(outputs) == 2:
        return outputs[0], outputs[1]
    if isinstance(outputs, dict):
        return outputs["foreground_logits"], outputs
    return outputs, {}


def _resize_target(target: torch.Tensor, logits: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if target.shape[-2:] == logits.shape[-2:]:
        return target
    return F.interpolate(target, size=logits.shape[-2:], mode=mode, align_corners=False if mode == "bilinear" else None)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = _resize_target(target, logits).float().flatten(1)
    # A 464x464 fp16 probability map can exceed 65504 when reduced. Keep
    # Dice's sigmoid and all pixel sums in float32 under AMP.
    probability = torch.sigmoid(logits.float()).flatten(1)
    overlap = (probability * target).sum(dim=1)
    score = (2.0 * overlap + eps) / (probability.sum(dim=1) + target.sum(dim=1) + eps)
    return 1.0 - score.mean()


def focal_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
    pos_weight: float | None = None,
) -> torch.Tensor:
    target = _resize_target(target, logits)
    weight = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    focal = (1.0 - p_t).clamp(0.0, 1.0).pow(float(gamma))
    if alpha is not None:
        focal = focal * (float(alpha) * target + (1.0 - float(alpha)) * (1.0 - target))
    return (bce * focal).mean()


def boundary_difference_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    focus: torch.Tensor | None,
) -> torch.Tensor:
    target = _resize_target(target, logits)
    if focus is None:
        focus = torch.ones_like(target)
    else:
        focus = _resize_target(focus, logits, mode="nearest")
    focus = torch.maximum(focus.to(target.dtype), (target > 0).to(target.dtype))
    return (torch.abs(torch.sigmoid(logits) - target) * focus).sum() / (focus.sum() + 1e-6)


def proximity_regression_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    focus: torch.Tensor | None,
    background_weight: float,
) -> torch.Tensor:
    target = _resize_target(target, logits)
    if focus is None:
        focus = (target > 0).to(target.dtype)
    else:
        focus = _resize_target(focus, logits, mode="nearest").to(target.dtype)
    focus = torch.maximum(focus, (target > 0).to(target.dtype))
    background_weight = min(max(float(background_weight), 0.0), 1.0)
    weight = background_weight + (1.0 - background_weight) * focus
    error = torch.sigmoid(logits) - target
    l1 = (error.abs() * weight).sum() / (weight.sum() + 1e-6)
    mse = (error.square() * weight).sum() / (weight.sum() + 1e-6)
    return 0.6 * l1 + 0.4 * mse


class ForegroundBoundaryAuxLoss(nn.Module):
    """Final two-head loss plus multi-scale supervision for CAPC gates."""

    def __init__(
        self,
        boundary_weight: float = 0.15,
        capc_boundary_weight: float = 0.08,
        capc_contact_weight: float = 0.15,
        foreground_pos_weight: float | None = None,
        boundary_pos_weight: float | None = None,
        contact_pos_weight: float | None = 8.0,
        contact_empty_weight: float = 1.0,
        foreground_continuity_weight: float = 0.0,
        foreground_gap_weight: float = 0.0,
        instance_contract_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = None,
        boundary_diff_weight: float = 0.2,
        distance_background_weight: float = 0.1,
        use_soft_boundary: bool = True,
    ):
        super().__init__()
        self.boundary_weight = float(boundary_weight)
        self.capc_boundary_weight = float(capc_boundary_weight)
        self.capc_contact_weight = float(capc_contact_weight)
        self.foreground_pos_weight = foreground_pos_weight
        self.boundary_pos_weight = boundary_pos_weight
        self.contact_pos_weight = contact_pos_weight
        self.contact_empty_weight = float(contact_empty_weight)
        self.foreground_continuity_weight = float(foreground_continuity_weight)
        self.foreground_gap_weight = float(foreground_gap_weight)
        self.instance_contract_weight = float(instance_contract_weight)
        if not 0.0 <= self.contact_empty_weight <= 1.0:
            raise ValueError("contact_empty_weight must be in [0, 1]")
        if self.foreground_continuity_weight < 0.0:
            raise ValueError("foreground_continuity_weight must be >= 0")
        if self.foreground_gap_weight < 0.0:
            raise ValueError("foreground_gap_weight must be >= 0")
        if self.instance_contract_weight < 0.0:
            raise ValueError("instance_contract_weight must be >= 0")
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = focal_alpha
        self.boundary_diff_weight = float(boundary_diff_weight)
        self.distance_background_weight = float(distance_background_weight)
        self.use_soft_boundary = bool(use_soft_boundary)

    def _mean_capc_boundary_loss(
        self,
        logits: Sequence[torch.Tensor],
        target: torch.Tensor,
        focus: torch.Tensor,
    ) -> torch.Tensor:
        if not logits:
            return target.new_zeros(())
        return torch.stack(
            [
                proximity_regression_loss(
                    item, target, focus, background_weight=self.distance_background_weight
                )
                for item in logits
            ]
        ).mean()

    def _mean_capc_contact_loss(
        self,
        logits: Sequence[torch.Tensor],
        target: torch.Tensor,
        focus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not logits:
            return target.new_zeros(())
        losses = []
        for item in logits:
            resized_target = _resize_target(target, item)
            resized_focus = (
                torch.ones_like(resized_target)
                if focus is None
                else _resize_target(focus, item, mode="nearest").to(resized_target.dtype)
            )
            resized_focus = torch.maximum(resized_focus, (resized_target > 0).to(resized_target.dtype))
            if not bool((resized_focus > 0).any()):
                resized_focus = torch.ones_like(resized_target)
            has_contact = (resized_target.flatten(1) > 0).any(dim=1)
            sample_weight = torch.where(
                has_contact,
                torch.ones_like(has_contact, dtype=item.dtype),
                torch.full_like(has_contact, self.contact_empty_weight, dtype=item.dtype),
            )
            pos_weight = None if self.contact_pos_weight is None else torch.as_tensor(
                self.contact_pos_weight, dtype=item.dtype, device=item.device
            )
            bce = F.binary_cross_entropy_with_logits(
                item, resized_target, pos_weight=pos_weight, reduction="none"
            )
            probability = torch.sigmoid(item.float())
            p_t = probability * resized_target + (1.0 - probability) * (1.0 - resized_target)
            focal_map = (1.0 - p_t).clamp(0.0, 1.0).pow(float(self.focal_gamma))
            if self.focal_alpha is not None:
                focal_map = focal_map * (
                    float(self.focal_alpha) * resized_target
                    + (1.0 - float(self.focal_alpha)) * (1.0 - resized_target)
                )
            focal = (
                bce * focal_map * resized_focus * sample_weight[:, None, None, None]
            ).sum() / (resized_focus.sum() + 1e-6)
            if bool((resized_target > 0).any()):
                probability_flat = (probability * resized_focus).flatten(1)
                target_flat = (resized_target * resized_focus).flatten(1)
                overlap = (probability_flat * target_flat).sum(dim=1)
                dice_score = (2.0 * overlap + 1e-6) / (
                    probability_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-6
                )
                dice = ((1.0 - dice_score) * sample_weight).mean()
            else:
                dice = item.new_zeros(())
            losses.append(0.5 * focal + 0.5 * dice)
        return torch.stack(losses).mean()

    def forward(self, outputs, batch: Dict[str, torch.Tensor]):
        foreground_logits, aux = split_outputs(outputs)
        if not isinstance(aux, dict) or not torch.is_tensor(aux.get("boundary_logits")):
            raise KeyError("SGMNet auxiliary output must contain boundary_logits.")

        foreground_target = batch["foreground"]
        instance_probability = aux.get("instance_probability")
        instance_logits = aux.get("instance_logits")
        if self.instance_contract_weight > 0 and not torch.is_tensor(instance_probability):
            raise KeyError("instance_probability required for instance contract loss")
        if self.instance_contract_weight > 0 and not torch.is_tensor(instance_logits):
            foreground = foreground_logits.float()
            boundary = aux["boundary_logits"].float()
            instance_logits = -torch.logaddexp(-foreground, boundary+F.softplus(-foreground))
        hard_boundary = batch["boundary"]
        contact_core = batch.get("separation", hard_boundary.new_zeros(hard_boundary.shape))
        boundary_target = batch.get("boundary_soft", hard_boundary) if self.use_soft_boundary else hard_boundary
        boundary_focus = batch.get("boundary_focus", hard_boundary)

        foreground_bce = F.binary_cross_entropy_with_logits(
            foreground_logits,
            _resize_target(foreground_target, foreground_logits),
            pos_weight=None if self.foreground_pos_weight is None else torch.as_tensor(
                self.foreground_pos_weight, dtype=foreground_logits.dtype, device=foreground_logits.device
            ),
        )
        foreground_dice = dice_loss_from_logits(foreground_logits, foreground_target)
        continuity_loss = foreground_logits.new_zeros(())
        if self.foreground_continuity_weight > 0:
            if not torch.is_tensor(batch.get("narrow_continuity")):
                raise KeyError("narrow_continuity target required for foreground continuity loss")
            continuity_mask = _resize_target(
                batch["narrow_continuity"], foreground_logits, mode="nearest"
            ).float()
            if bool((continuity_mask > 0).any()):
                if self.instance_contract_weight > 0:
                    # Protect thin same-instance necks in the probability that
                    # will actually be thresholded for instance recovery.
                    continuity_error = F.softplus(-instance_logits.float())
                else:
                    continuity_error = F.softplus(-foreground_logits.float())
                continuity_loss = (continuity_error * continuity_mask).sum() / continuity_mask.sum().clamp_min(1.0)
        gap_loss = foreground_logits.new_zeros(())
        gap_pixel_fraction = foreground_logits.new_zeros(())
        if self.foreground_gap_weight > 0:
            if not torch.is_tensor(batch.get("separation")):
                raise KeyError("separation target required for foreground gap loss")
            resized_foreground = _resize_target(foreground_target, foreground_logits, mode="nearest")
            gap_mask = (
                (_resize_target(contact_core, foreground_logits, mode="nearest") > 0.5)
                & (resized_foreground < 0.5)
            ).float()
            gap_loss = (
                F.softplus(foreground_logits.float()) * gap_mask
            ).sum() / gap_mask.sum().clamp_min(1.0)
            gap_pixel_fraction = gap_mask.mean()
        foreground_loss = (
            foreground_bce + foreground_dice
            + self.foreground_continuity_weight * continuity_loss
            + self.foreground_gap_weight * gap_loss
        )

        boundary_logits = aux["boundary_logits"]
        boundary_focal = focal_bce_logits(
            boundary_logits,
            boundary_target,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            pos_weight=self.boundary_pos_weight,
        )
        boundary_dice = dice_loss_from_logits(boundary_logits, boundary_target)
        boundary_hard_dice = dice_loss_from_logits(boundary_logits, hard_boundary)
        boundary_difference = boundary_difference_loss(boundary_logits, boundary_target, boundary_focus)
        boundary_loss = (
            0.5 * boundary_focal
            + 0.25 * boundary_dice
            + 0.25 * boundary_hard_dice
            + self.boundary_diff_weight * boundary_difference
        )

        capc_boundary_logits = aux.get("capc_boundary_logits", [])
        boundary_proximity = batch.get("boundary_distance", boundary_target).clamp(0.0, 1.0)
        capc_boundary_loss = self._mean_capc_boundary_loss(
            capc_boundary_logits, boundary_proximity, boundary_focus
        )

        capc_contact_logits = aux.get("capc_contact_logits", [])
        contact_proximity = batch.get("separation_distance", contact_core).clamp(0.0, 1.0)
        contact_target = torch.maximum(contact_core, 0.75 * contact_proximity)
        contact_focus = batch.get("separation_focus", None)
        capc_contact_loss = self._mean_capc_contact_loss(capc_contact_logits, contact_target, contact_focus)

        instance_contract_loss = foreground_logits.new_zeros(())
        instance_contract_bce = foreground_logits.new_zeros(())
        instance_contract_dice = foreground_logits.new_zeros(())
        if self.instance_contract_weight > 0:
            # The evaluation unit is the annotated instance mask. The train
            # batch carries geometrically augmented IDs; validation IDs are
            # resized here to the logit lattice with nearest-neighbor sampling.
            if not torch.is_tensor(batch.get("object_instance")):
                raise KeyError("object_instance required for instance contract loss")
            inside_target = _resize_target(
                (batch["object_instance"] > 0).float(), foreground_logits, mode="nearest"
            ).float()
            instance_contract_bce = F.binary_cross_entropy_with_logits(instance_logits, inside_target)
            instance_contract_dice = dice_loss_from_logits(instance_logits, inside_target)
            instance_contract_loss = instance_contract_bce + instance_contract_dice

        total = (
            foreground_loss
            + self.boundary_weight * boundary_loss
            + self.capc_boundary_weight * capc_boundary_loss
            + self.capc_contact_weight * capc_contact_loss
            + self.instance_contract_weight * instance_contract_loss
        )
        logs = {
            "loss": float(total.detach().cpu()),
            "loss_fg": float(foreground_loss.detach().cpu()),
            "fg_bce": float(foreground_bce.detach().cpu()),
            "fg_dice_loss": float(foreground_dice.detach().cpu()),
            "loss_continuity": float(continuity_loss.detach().cpu()),
            "loss_boundary": float(boundary_loss.detach().cpu()),
            "boundary_focal": float(boundary_focal.detach().cpu()),
            "boundary_dice_loss": float(boundary_dice.detach().cpu()),
            "boundary_hard_dice_loss": float(boundary_hard_dice.detach().cpu()),
            "boundary_diff": float(boundary_difference.detach().cpu()),
            "loss_capc_boundary": float(capc_boundary_loss.detach().cpu()),
            "loss_capc_contact": float(capc_contact_loss.detach().cpu()),
            "contact_core_pixel_fraction": float((contact_core > 0).float().mean().detach().cpu()),
            "contact_positive_sample_fraction": float(
                (contact_core.flatten(1) > 0).any(dim=1).float().mean().detach().cpu()
            ),
        }
        if self.foreground_gap_weight > 0:
            logs["loss_gap"] = float(gap_loss.detach().cpu())
            logs["gap_pixel_fraction"] = float(gap_pixel_fraction.detach().cpu())
        if self.instance_contract_weight > 0:
            logs["loss_instance_contract"] = float(instance_contract_loss.detach().cpu())
            logs["instance_contract_bce"] = float(instance_contract_bce.detach().cpu())
            logs["instance_contract_dice"] = float(instance_contract_dice.detach().cpu())
        if torch.is_tensor(batch.get("narrow_continuity")):
            logs["continuity_pixel_fraction"] = float(
                (batch["narrow_continuity"] > 0).float().mean().detach().cpu()
            )
        diagnostic_names = (
            "capc_gate_mean",
            "capc_correction_abs_mean",
            "capc_discrepancy_mean",
            "capc_prior_boundary_abs_mean",
            "capc_prior_scale",
            "hr_gate_mean",
            "hr_gate_emphasis_mean",
            "hr_gate_emphasis_active_ratio",
            "hr_gate_emphasis_p95",
            "hr_boundary_evidence_mean",
            "hr_contact_evidence_mean",
            "hr_signed_cue_abs_mean",
            "hr_delta_abs_mean",
            "hr_gated_delta_abs_mean",
            "hr_repair_ratio",
            "hr_contact_only_ratio",
            "hr_contact_only_emphasis_mean",
            "hr_contact_only_delta_abs_mean",
            "hr_contact_only_gated_delta_abs_mean",
            "hr_contact_only_repair_ratio",
            "hr_contact_extra_permission_mean",
            "hr_contact_extra_gate_mean",
        )
        for name in diagnostic_names:
            value = aux.get(name)
            if torch.is_tensor(value) and value.numel() == 1:
                logs[name] = float(value.detach().cpu())
        return total, logs


__all__ = ["ForegroundBoundaryAuxLoss"]
