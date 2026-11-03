"""Standalone data, reproducibility and object-metric utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_images(directory: Path | str):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def find_mask(
    mask_dir: Path | str, stem: str, role: str = "mask", image_path: Optional[Path | str] = None
) -> Path:
    mask_dir = Path(mask_dir)
    checked = []

    for ext in IMG_EXTS:
        p = mask_dir / f"{stem}{ext}"
        checked.append(p)
        if p.exists():
            return p

    for suffix in ["_mask", "_label", "_gt", "_anno", "_annotation", "_anno_label"]:
        for ext in IMG_EXTS:
            p = mask_dir / f"{stem}{suffix}{ext}"
            checked.append(p)
            if p.exists():
                return p

    image_part = f"\n  image_path: {Path(image_path).resolve()}" if image_path is not None else ""
    preview = "\n  - ".join(str(p) for p in checked[:24])
    raise FileNotFoundError(
        f"Cannot find {role} mask for stem={stem}\n"
        f"  mask_dir: {mask_dir.resolve()}"
        f"{image_part}\n"
        f"  checked_candidates(first {min(len(checked), 24)}):\n  - {preview}"
    )


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def image_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    """SCMNet input normalization: RGB [0,255] -> [-1,1] CHW tensor."""
    arr = img_rgb.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()


def sliding_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    pos = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if pos[-1] != last:
        pos.append(last)
    return pos


def reflect_pad_image_np(img: np.ndarray, tile_size: int) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    pad_h = max(0, int(tile_size) - h)
    pad_w = max(0, int(tile_size) - w)
    if pad_h == 0 and pad_w == 0:
        return img, h, w
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"), h, w


def center_weight_np(tile_size: int, eps: float = 0.05) -> np.ndarray:
    one = np.hanning(int(tile_size)).astype(np.float32)
    w = np.outer(one, one).astype(np.float32)
    w = w / max(float(w.max()), 1e-8)
    return np.maximum(w, float(eps)).astype(np.float32)


def _as_numpy_bool(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(bool)


def _instance_label_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def _glass_hausdorff(mask_s, mask_g) -> float:
    from scipy.spatial import cKDTree

    s = _as_numpy_bool(mask_s)
    g = _as_numpy_bool(mask_g)
    if not s.any() and not g.any():
        return 0.0
    if not s.any() or not g.any():
        h, w = g.shape[:2]
        return float(np.sqrt(h * h + w * w))
    row_ind, col_ind = np.indices(s.shape)
    x = np.column_stack((row_ind[g], col_ind[g]))
    y = np.column_stack((row_ind[s], col_ind[s]))
    tree_y = cKDTree(y)
    dist1, _ = tree_y.query(x)
    tree_x = cKDTree(x)
    dist2, _ = tree_x.query(y)
    return float(max(np.max(dist1), np.max(dist2)))


def glass_object_f1score(pred_inst, gt_inst):
    """Exact GlaS/MILDNet/DCAN ObjectF1 matching rule."""
    S = _instance_label_np(pred_inst)
    G = _instance_label_np(gt_inst)
    listS = np.unique(S)
    listS = listS[listS != 0]
    listG = np.unique(G)
    listG = listG[listG != 0]
    numS, numG = len(listS), len(listG)
    if numS == 0 and numG == 0:
        return 1.0, 0, 0, 0
    if numS == 0 or numG == 0:
        return 0.0, 0, int(numS), int(numG)

    temp = np.zeros((numS, 3), dtype=int)
    temp[:, 0] = listS.astype(int)
    for i in range(numS):
        inter_gt = G[S == temp[i, 0]]
        inter_gt = inter_gt[inter_gt != 0]
        if inter_gt.size > 0:
            ids, counts = np.unique(inter_gt, return_counts=True)
            temp[i, 1] = int(ids[int(np.argmax(counts))])
    for i in range(numS):
        if temp[i, 1] != 0:
            seg_obj = S == temp[i, 0]
            gt_obj = G == temp[i, 1]
            area_overlap = np.logical_and(seg_obj, gt_obj).sum()
            area_gt = gt_obj.sum()
            if area_overlap / max(float(area_gt), 1.0) > 0.5:
                temp[i, 2] = 1
    tp = int(np.sum(temp[:, 2] == 1))
    fp = int(np.sum(temp[:, 2] == 0))
    fn = int(numG - tp)
    precision = tp / max(tp + fp, 1e-10)
    recall = tp / max(tp + fn, 1e-10)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-10)
    return float(f1), tp, fp, fn


def glass_object_dice(pred_inst, gt_inst) -> float:
    def _dice(a, b):
        inter = np.logical_and(a, b).sum()
        return 2.0 * inter / max(float(a.sum() + b.sum()), 1e-10)

    S = _instance_label_np(pred_inst)
    G = _instance_label_np(gt_inst)
    listS = np.unique(S)
    listS = listS[listS != 0]
    listG = np.unique(G)
    listG = listG[listG != 0]
    numS, numG = len(listS), len(listG)
    if numS == 0 and numG == 0:
        return 1.0
    if numS == 0 or numG == 0:
        return 0.0

    total_area_s = float((S > 0).sum())
    total_area_g = float((G > 0).sum())
    temp1 = 0.0
    for sid in listS:
        si = S == sid
        inter = G[si]
        inter = inter[inter != 0]
        if inter.size != 0:
            gid = int(np.bincount(inter.astype(int)).argmax())
            gi = G == gid
        else:
            gi = np.zeros_like(G, dtype=bool)
        omega = float(si.sum()) / max(total_area_s, 1e-10)
        temp1 += omega * _dice(gi, si)
    temp2 = 0.0
    for gid in listG:
        gi = G == gid
        inter = S[gi]
        inter = inter[inter != 0]
        if inter.size != 0:
            sid = int(np.bincount(inter.astype(int)).argmax())
            si = S == sid
        else:
            si = np.zeros_like(S, dtype=bool)
        omega = float(gi.sum()) / max(total_area_g, 1e-10)
        temp2 += omega * _dice(gi, si)
    return float((temp1 + temp2) / 2.0)


def glass_object_hausdorff(pred_inst, gt_inst) -> float:
    S = _instance_label_np(pred_inst)
    G = _instance_label_np(gt_inst)
    listS = np.unique(S)
    listS = listS[listS != 0]
    listG = np.unique(G)
    listG = listG[listG != 0]
    if len(listS) == 0 and len(listG) == 0:
        return 0.0
    if len(listS) == 0 or len(listG) == 0:
        h, w = G.shape[:2]
        return float(np.sqrt(h * h + w * w))
    total_area_s = float((S > 0).sum())
    total_area_g = float((G > 0).sum())
    temp1 = 0.0
    for sid in listS:
        si = S == sid
        inter = G[si]
        inter = inter[inter != 0]
        if inter.size > 0:
            gid = int(np.bincount(inter.astype(int)).argmax())
            gi = G == gid
        else:
            dists = [_glass_hausdorff(G == gid, si) for gid in listG]
            gi = G == listG[int(np.argmin(dists))]
        omega = float(si.sum()) / max(total_area_s, 1e-10)
        temp1 += omega * _glass_hausdorff(gi, si)
    temp2 = 0.0
    for gid in listG:
        gi = G == gid
        inter = S[gi]
        inter = inter[inter != 0]
        if inter.size > 0:
            sid = int(np.bincount(inter.astype(int)).argmax())
            si = S == sid
        else:
            dists = [_glass_hausdorff(gi, S == sid) for sid in listS]
            si = S == listS[int(np.argmin(dists))]
        omega = float(gi.sum()) / max(total_area_g, 1e-10)
        temp2 += omega * _glass_hausdorff(gi, si)
    return float((temp1 + temp2) / 2.0)


def glass_object_metrics(pred_inst, gt_inst) -> Dict[str, float]:
    f1, tp, fp, fn = glass_object_f1score(pred_inst, gt_inst)
    dice = glass_object_dice(pred_inst, gt_inst)
    haus = glass_object_hausdorff(pred_inst, gt_inst)
    gt_shape = _instance_label_np(gt_inst).shape[:2]
    image_diagonal = max(float(np.hypot(*gt_shape)), 1.0)
    haus_quality = 1.0 - min(float(haus) / image_diagonal, 1.0)
    score = 0.30 * float(f1) + 0.40 * float(dice) + 0.30 * haus_quality
    precision = tp / max(tp + fp, 1e-10)
    recall = tp / max(tp + fn, 1e-10)
    return {
        "recall": float(recall),
        "precision": float(precision),
        "F1": float(f1),
        "Dice": float(dice),
        "IoU": float(dice / max(2.0 - dice, 1e-10)),
        "Hausdorff": float(haus),
        "HausdorffQuality": float(haus_quality),
        "Score": float(score),
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
    }


def mean_glass_object_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "F1": 0.0,
            "Dice": 0.0,
            "IoU": 0.0,
            "Hausdorff": 0.0,
            "HausdorffQuality": 0.0,
            "Score": 0.0,
        }
    keys = ["recall", "precision", "F1", "Dice", "IoU", "Hausdorff", "HausdorffQuality", "Score"]
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


class Tee:
    """Write progress to the terminal and a UTF-8 log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()
