"""Foreground, affinity, contact, identity and continuity losses for SCMNet."""

from __future__ import annotations

import random

import torch
from torch import nn
from torch.nn import functional as F

from models import SCMNet


def _edge_slices(height, width, dy, dx):
    ya, yb = max(0, -dy), min(height, height - dy)
    xa, xb = max(0, -dx), min(width, width - dx)
    return (slice(ya, yb), slice(xa, xb)), (slice(ya + dy, yb + dy), slice(xa + dx, xb + dx))


def relation_targets(instances, offsets, return_cross=False):
    """Return same-ID edges and valid pairs. Long positive edges need clear line of sight."""
    ids = instances[:, 0].long()
    batch, height, width = ids.shape
    target = ids.new_zeros((batch, len(offsets), height, width), dtype=torch.float32)
    valid = torch.zeros_like(target, dtype=torch.bool)
    cross = torch.zeros_like(valid)
    for channel, (dy, dx) in enumerate(offsets):
        source, destination = _edge_slices(height, width, dy, dx)
        first = ids[:, source[0], source[1]]
        last = ids[:, destination[0], destination[1]]
        pair_valid = (first > 0) | (last > 0)
        same = (first > 0) & (first == last)
        if max(abs(dx), abs(dy)) > 1:
            steps = max(abs(dx), abs(dy))
            for step in range(1, steps):
                middle = ids[
                    :,
                    source[0].start + step * dy // steps : source[0].stop + step * dy // steps,
                    source[1].start + step * dx // steps : source[1].stop + step * dx // steps,
                ]
                same = same & (middle == first)
        target[:, channel, source[0], source[1]] = same.float()
        valid[:, channel, source[0], source[1]] = pair_valid
        cross[:, channel, source[0], source[1]] = (first > 0) & (last > 0) & (first != last)
    return (target, valid, cross) if return_cross else (target, valid)


def balanced_edge_bce(logits, target, valid, cross):
    positive = valid & (target > 0.5)
    ordinary_negative = valid & (target <= 0.5) & ~cross
    terms, weights = [], []
    if positive.any():
        terms.append(F.binary_cross_entropy_with_logits(logits[positive].float(), target[positive]))
        weights.append(0.4)
    if cross.any():
        terms.append(F.binary_cross_entropy_with_logits(logits[cross].float(), target[cross]))
        weights.append(0.4)
    if ordinary_negative.any():
        per_negative = F.binary_cross_entropy_with_logits(
            logits[ordinary_negative].float(), target[ordinary_negative], reduction="none"
        )
        # Hard negatives include false bridges and tissue artifacts.
        count = max(1, int(per_negative.numel() * 0.25))
        terms.append(per_negative.topk(count).values.mean())
        weights.append(0.2)
    return (
        sum(term * weight for term, weight in zip(terms, weights)) / sum(weights)
        if terms
        else logits.sum() * 0.0
    )


class RegionRelationLoss(nn.Module):
    def __init__(self, relation_weight=0.5, stage_weight=0.2, hard_region_weight=0.2):
        super().__init__()
        self.relation_weight = float(relation_weight)
        self.stage_weight = float(stage_weight)
        self.hard_region_weight = float(hard_region_weight)

    def forward(self, region_logits, aux, instances):
        target = (instances > 0).float()
        bce = F.binary_cross_entropy_with_logits(region_logits.float(), target, reduction="none")
        region_bce = bce.mean()
        hard = bce.flatten(1).topk(max(1, bce.shape[-1] * bce.shape[-2] // 5), dim=1).values.mean()
        probability = region_logits.float().sigmoid()
        dice = (
            1.0
            - (
                (2 * (probability * target).sum(dim=(2, 3)) + 1.0)
                / (probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1.0)
            ).mean()
        )
        edge_target, valid, cross = relation_targets(
            instances, SCMNet.relation_offsets, return_cross=True
        )
        edge = balanced_edge_bce(aux["relation_logits"], edge_target, valid, cross)
        stage_terms = []
        for logits in aux["encoder_relation_logits"]:
            small_ids = F.interpolate(
                instances.float(), size=logits.shape[-2:], mode="nearest"
            ).long()
            small_target, small_valid, small_cross = relation_targets(
                small_ids, ((0, -1), (-1, 0)), return_cross=True
            )
            stage_terms.append(balanced_edge_bce(logits, small_target, small_valid, small_cross))
        stage = sum(stage_terms) / len(stage_terms) if stage_terms else edge.new_zeros(())
        loss = (
            region_bce
            + dice
            + self.hard_region_weight * hard
            + self.relation_weight * edge
            + self.stage_weight * stage
        )
        logs = {
            "loss": loss.detach(),
            "region_bce": region_bce.detach(),
            "region_dice": dice.detach(),
            "hard_region": hard.detach(),
            "relation": edge.detach(),
            "stage_relation": stage.detach(),
        }
        return loss, logs


@torch.no_grad()
def contact_gap_targets(instances: torch.Tensor, radius: int = 6) -> torch.Tensor:
    """GT background within ``radius`` pixels of at least two gland IDs."""
    ids = instances[:, 0].long()
    if radius < 1:
        raise ValueError("contact radius must be positive")
    # A window contains two distinct glands exactly when its largest positive
    # label is greater than its smallest positive label. This needs only two
    # pooling passes, regardless of the number of glands in a crop.
    values = ids.float().unsqueeze(1)
    kernel = 2 * radius + 1
    max_id = F.max_pool2d(values, kernel, stride=1, padding=radius)
    sentinel = float(2**24)
    nonzero_values = torch.where(values > 0, values, sentinel)
    min_positive = -F.max_pool2d(-nonzero_values, kernel, stride=1, padding=radius)
    return (ids.unsqueeze(1) == 0) & (max_id > min_positive) & (min_positive < sentinel)


def hard_pair_loss(
    embedding: torch.Tensor, instances: torch.Tensor, hard_limit: int = 2048
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull gland pixels together; repel nearby pixels from different glands."""
    ids = instances[:, 0].long()
    _, _, height, width = embedding.shape
    fixed = [(0, d) for d in (5, 6, 8, 10)]
    fixed += [(d, 0) for d in (5, 6, 8, 10)]
    fixed += [(d, d) for d in (5, 6, 8, 10)]
    fixed += [(d, -d) for d in (5, 6, 8, 10)]
    random_offsets = []
    for _ in range(8):
        dy, dx = random.randint(-12, 12), random.randint(-12, 12)
        if 2 <= max(abs(dy), abs(dx)) <= 12:
            random_offsets.append((dy, dx))
    positive_terms, negative_terms = [], []
    for dy, dx in fixed + random_offsets:
        if abs(dy) >= height or abs(dx) >= width:
            continue
        first_slice, last_slice = _edge_slices(height, width, dy, dx)
        first_id = ids[:, first_slice[0], first_slice[1]]
        last_id = ids[:, last_slice[0], last_slice[1]]
        same = (first_id > 0) & (first_id == last_id)
        different = (first_id > 0) & (last_id > 0) & (first_id != last_id)
        if not (same.any() or different.any()):
            continue
        first_z = embedding[:, :, first_slice[0], first_slice[1]]
        last_z = embedding[:, :, last_slice[0], last_slice[1]]
        similarity = (first_z.float() * last_z.float()).sum(dim=1)
        if same.any():
            error = F.softplus((0.75 - similarity[same]) * 8.0) / 8.0
            positive_terms.append(error.topk(min(hard_limit, error.numel())).values.mean())
        if different.any():
            error = F.softplus((similarity[different] - 0.25) * 8.0) / 8.0
            negative_terms.append(error.topk(min(hard_limit, error.numel())).values.mean())
    zero = embedding.sum() * 0.0
    positive = torch.stack(positive_terms).mean() if positive_terms else zero
    negative = torch.stack(negative_terms).mean() if negative_terms else zero
    return positive, negative


def same_instance_path_loss(
    region_logits: torch.Tensor, instances: torch.Tensor, hard_limit: int = 2048
) -> torch.Tensor:
    """Penalize the weakest foreground pixel on short GT-inside gland paths.

    A path is supervised only when every traversed pixel has the same positive
    instance ID. It cannot teach the foreground branch to bridge two glands or
    to fill a lumen that the ground truth marks as background.
    """
    ids = instances[:, 0].long()
    logits = region_logits[:, 0].float()
    _, height, width = ids.shape
    terms = []
    for distance in (2, 4, 8):
        for unit_y, unit_x in ((0, 1), (1, 0), (1, 1), (1, -1)):
            dy, dx = distance * unit_y, distance * unit_x
            if abs(dy) >= height or abs(dx) >= width:
                continue
            source, _ = _edge_slices(height, width, dy, dx)
            first_id = ids[:, source[0], source[1]]
            valid = first_id > 0
            bottleneck = logits[:, source[0], source[1]]
            for step in range(1, distance + 1):
                y = slice(source[0].start + step * unit_y, source[0].stop + step * unit_y)
                x = slice(source[1].start + step * unit_x, source[1].stop + step * unit_x)
                valid = valid & (ids[:, y, x] == first_id)
                bottleneck = torch.minimum(bottleneck, logits[:, y, x])
            if valid.any():
                errors = F.softplus(-bottleneck[valid])
                count = min(hard_limit, max(1, (errors.numel() + 19) // 20))
                terms.append(errors.topk(count).values.mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


class SCMNetLoss(nn.Module):
    def __init__(
        self,
        relation_weight=0.5,
        stage_weight=0.2,
        hard_region_weight=0.2,
        contact_weight=0.35,
        pair_weight=0.25,
        gap_weight=0.15,
        contact_radius=6,
        continuity_weight=0.15,
    ):
        super().__init__()
        self.base = RegionRelationLoss(relation_weight, stage_weight, hard_region_weight)
        self.contact_weight = float(contact_weight)
        self.pair_weight = float(pair_weight)
        self.gap_weight = float(gap_weight)
        self.contact_radius = int(contact_radius)
        self.continuity_weight = float(continuity_weight)

    def forward(self, region_logits, aux, instances):
        base, logs = self.base(region_logits, aux, instances)
        target = contact_gap_targets(instances, radius=self.contact_radius).float()
        contact_logits = aux["contact_logits"].float()
        positive = target.sum()
        negative = target.numel() - positive
        pos_weight = (negative / positive.clamp_min(1)).clamp(max=30.0)
        contact_bce = F.binary_cross_entropy_with_logits(
            contact_logits, target, pos_weight=pos_weight
        )
        contact_probability = contact_logits.sigmoid()
        if positive.item() > 0:
            contact_dice = 1.0 - (
                (2.0 * (contact_probability * target).sum() + 1.0)
                / (contact_probability.sum() + positive + 1.0)
            )
            contact_loss = contact_bce + 0.5 * contact_dice
            gap_region = F.softplus(region_logits.float()[target.bool()]).mean()
        else:
            contact_loss = contact_bce
            gap_region = region_logits.sum() * 0.0
        pair_positive, pair_negative = hard_pair_loss(aux["instance_embedding"], instances)
        path_continuity = same_instance_path_loss(region_logits, instances)
        total = (
            base
            + self.contact_weight * contact_loss
            + self.pair_weight * (pair_positive + 1.5 * pair_negative)
            + self.gap_weight * gap_region
            + self.continuity_weight * path_continuity
        )
        logs.update(
            {
                "loss": total.detach(),
                "contact": contact_loss.detach(),
                "pair_positive": pair_positive.detach(),
                "pair_negative": pair_negative.detach(),
                "gap_region": gap_region.detach(),
                "path_continuity": path_continuity.detach(),
                "contact_fraction": target.mean().detach(),
            }
        )
        return total, logs
