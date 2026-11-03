"""Evaluation helpers shared by training and whole-image inference."""

import csv

import numpy as np
from PIL import Image

from models import SCMNet


def read_instances(path):
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.int32)
        arr = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    return arr.astype(np.int32)


def _pixel_metrics(pred, gt):
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return {
        "pixel_F1": 2 * tp / max(2 * tp + fp + fn, 1),
        "pixel_IoU": tp / max(tp + fp + fn, 1),
        "pixel_precision": tp / max(tp + fp, 1),
        "pixel_recall": tp / max(tp + fn, 1),
        "fp_pixels": fp,
        "fn_pixels": fn,
    }


def _instance_error_counts(pred, gt):
    """Count GT glands touched by >1 predicted gland, and vice versa."""
    both = (pred > 0) & (gt > 0)
    if not both.any():
        return {"merged_gt": 0, "split_gt": 0, "missed_gt": int(np.unique(gt[gt > 0]).size)}
    pairs, overlaps = np.unique(
        np.stack((gt[both], pred[both]), axis=1), axis=0, return_counts=True
    )
    all_ids, gt_sizes = np.unique(gt[gt > 0], return_counts=True)
    relevant = overlaps >= np.maximum(10, 0.05 * gt_sizes[np.searchsorted(all_ids, pairs[:, 0])])
    pairs = pairs[relevant]
    if not len(pairs):
        return {"merged_gt": 0, "split_gt": 0, "missed_gt": int(np.unique(gt[gt > 0]).size)}
    gt_ids, gt_counts = np.unique(pairs[:, 0], return_counts=True)
    _, pred_counts = np.unique(pairs[:, 1], return_counts=True)
    all_gt = np.unique(gt[gt > 0])
    return {
        "merged_gt": int(pred_counts[pred_counts > 1].sum()),
        "split_gt": int((gt_counts > 1).sum()),
        "missed_gt": int(all_gt.size - gt_ids.size),
    }


def _relation_metrics(relation, gt, threshold):
    positive_hits = positive_total = negative_hits = negative_total = 0
    height, width = gt.shape
    for channel, (dy, dx) in enumerate(SCMNet.relation_offsets):
        ya, yb = max(0, -dy), min(height, height - dy)
        xa, xb = max(0, -dx), min(width, width - dx)
        first = gt[ya:yb, xa:xb]
        last = gt[ya + dy : yb + dy, xa + dx : xb + dx]
        positive = (first > 0) & (first == last)
        if channel >= 4:
            steps = max(abs(dx), abs(dy))
            for step in range(1, steps):
                middle = gt[
                    ya + step * dy // steps : yb + step * dy // steps,
                    xa + step * dx // steps : xb + step * dx // steps,
                ]
                positive &= middle == first
        negative = ((first > 0) | (last > 0)) & ~positive
        decision = relation[channel, ya:yb, xa:xb] >= (
            max(threshold, 0.8) if channel >= 4 else threshold
        )
        positive_hits += int(np.count_nonzero(decision & positive))
        positive_total += int(np.count_nonzero(positive))
        negative_hits += int(np.count_nonzero(decision & negative))
        negative_total += int(np.count_nonzero(negative))
    return {
        "relation_recall": positive_hits / max(positive_total, 1),
        "relation_false_bridge_rate": negative_hits / max(negative_total, 1),
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
