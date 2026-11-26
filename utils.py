# coding=utf-8
"""Common utilities for foreground-main + boundary-aux GlaS experiments."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_tuple_int(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


def list_images(directory: Path | str):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def find_mask(mask_dir: Path | str, stem: str, role: str = "mask", image_path: Optional[Path | str] = None) -> Path:
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


def first_existing_dir(root: Path | str, split: str, names: Sequence[str]):
    root = Path(root)
    for name in names:
        d = root / name / split
        if d.exists():
            return d, name
    return None, None


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
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


def read_split_file(path: Path | str) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path | str, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dsmnet_image_to_tensor_np(img_rgb: np.ndarray) -> torch.Tensor:
    """DSMNet input normalization: RGB [0,255] -> [-1,1] CHW tensor."""
    arr = img_rgb.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()


def dsmnet_read_rgb(path: Path | str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


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


def _dsmnet_extract_probs(outputs, target_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(outputs, tuple):
        main_logits, aux = outputs
    elif isinstance(outputs, dict):
        main_logits, aux = outputs.get("main_logits", outputs.get("logits")), outputs.get("aux", {})
    else:
        main_logits, aux = outputs, {}
    if main_logits is None:
        raise RuntimeError("Cannot parse DSMNet main foreground logits from model output.")
    if main_logits.shape[-2:] != target_hw:
        main_logits = F.interpolate(main_logits, size=target_hw, mode="bilinear", align_corners=False)
    fg = torch.sigmoid(main_logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    def aux_prob(name: str) -> np.ndarray:
        if isinstance(aux, dict) and torch.is_tensor(aux.get(name, None)):
            t = aux[name]
            if t.shape[-2:] != target_hw:
                t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
            return torch.sigmoid(t)[0, 0].detach().cpu().numpy().astype(np.float32)
        return np.zeros(target_hw, dtype=np.float32)

    separation_core = aux_prob("separation_core_logits")
    separation_field = aux_prob("separation_field_logits")
    separation_blocker = np.maximum(separation_core, 0.5 * separation_field).astype(np.float32, copy=False)
    return fg, aux_prob("boundary_logits"), separation_blocker, aux_prob("boundary_distance_logits")


@torch.no_grad()
def dsmnet_predict_sliding_np(
    model: torch.nn.Module,
    img_rgb: np.ndarray,
    device: torch.device,
    tile_size: int = 464,
    stride: int = 232,
    fusion: str = "uniform",
    center_weight_eps: float = 0.05,
    return_boundary_distance: bool = False,
    return_instance_probability: bool = False,
) -> tuple[np.ndarray, ...]:
    """Whole-image DSMNet inference using DCAN-style sliding windows.

    Defaults match DCAN's whole-image protocol: tile=464, stride=232 and
    uniform overlap averaging. When requested, the joint instance probability
    is fused per tile, preserving the probability supervised during training.
    """
    tile_size = int(tile_size)
    stride = int(stride)
    img_pad, ori_h, ori_w = reflect_pad_image_np(img_rgb, tile_size)
    h, w = img_pad.shape[:2]
    ys = sliding_positions(h, tile_size, stride)
    xs = sliding_positions(w, tile_size, stride)

    if fusion == "uniform":
        weight = np.ones((tile_size, tile_size), dtype=np.float32)
    elif fusion == "center_weighted":
        weight = center_weight_np(tile_size, eps=center_weight_eps)
    else:
        raise ValueError(f"Unsupported fusion={fusion!r}")

    fg_sum = np.zeros((h, w), dtype=np.float32)
    bd_sum = np.zeros((h, w), dtype=np.float32)
    bd_distance_sum = np.zeros((h, w), dtype=np.float32)
    sep_sum = np.zeros((h, w), dtype=np.float32)
    instance_sum = np.zeros((h, w), dtype=np.float32) if return_instance_probability else None
    weight_sum = np.zeros((h, w), dtype=np.float32)

    was_training = model.training
    model.eval()
    for y0 in ys:
        for x0 in xs:
            patch = img_pad[y0:y0 + tile_size, x0:x0 + tile_size]
            x = dsmnet_image_to_tensor_np(patch).unsqueeze(0).to(device, non_blocking=True)
            outputs = model(x, return_aux=True, aux_mode="loss")
            fg, bd, sep, bd_distance = _dsmnet_extract_probs(outputs, (tile_size, tile_size))
            if return_instance_probability:
                # E[fg*(1-bd)] differs from E[fg]*(1-E[bd]) in overlaps.
                # The former matches the per-tile training objective.
                aux = outputs[1] if isinstance(outputs, tuple) else outputs.get("aux", {}) if isinstance(outputs, dict) else {}
                joint = aux.get("instance_probability") if isinstance(aux, dict) else None
                if torch.is_tensor(joint):
                    if joint.shape[-2:] != (tile_size, tile_size):
                        joint = F.interpolate(joint.float(), size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                    joint = joint[0, 0].detach().float().cpu().numpy()
                else:
                    joint = fg*(1.0-bd)
                instance_sum[y0:y0+tile_size, x0:x0+tile_size] += joint*weight
            fg_sum[y0:y0 + tile_size, x0:x0 + tile_size] += fg * weight
            bd_sum[y0:y0 + tile_size, x0:x0 + tile_size] += bd * weight
            bd_distance_sum[y0:y0 + tile_size, x0:x0 + tile_size] += bd_distance * weight
            sep_sum[y0:y0 + tile_size, x0:x0 + tile_size] += sep * weight
            weight_sum[y0:y0 + tile_size, x0:x0 + tile_size] += weight
    if was_training:
        model.train()

    denom = np.maximum(weight_sum, 1e-8)
    probabilities = (
        (fg_sum / denom)[:ori_h, :ori_w],
        (bd_sum / denom)[:ori_h, :ori_w],
        (sep_sum / denom)[:ori_h, :ori_w],
    )
    if return_boundary_distance:
        probabilities += ((bd_distance_sum / denom)[:ori_h, :ori_w],)
    if return_instance_probability:
        probabilities += ((instance_sum / denom)[:ori_h, :ori_w],)
    return probabilities


@torch.no_grad()
def binary_confusion_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = torch.nn.functional.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits)
    pred = prob >= float(threshold)
    gt = target >= 0.5
    tp = torch.logical_and(pred, gt).sum().float()
    tn = torch.logical_and(~pred, ~gt).sum().float()
    fp = torch.logical_and(pred, ~gt).sum().float()
    fn = torch.logical_and(~pred, gt).sum().float()
    return {"tp": float(tp.item()), "tn": float(tn.item()), "fp": float(fp.item()), "fn": float(fn.item())}


def metrics_from_confusion(c: Dict[str, float]) -> Dict[str, float]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + 1e-10),
        "iou": tp / (tp + fp + fn + 1e-10),
        "acc": (tp + tn) / (tp + tn + fp + fn + 1e-10),
        "recall": tp / (tp + fn + 1e-10),
        "precision": tp / (tp + fp + 1e-10),
        "specificity": tn / (tn + fp + 1e-10),
    }


def new_confusion() -> Dict[str, float]:
    return {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}


def add_confusion(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for k in ["tp", "tn", "fp", "fn"]:
        dst[k] += float(src[k])


def mean_dict(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(dicts)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


@torch.no_grad()
def binary_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Return binary segmentation metrics from logits using the shared confusion implementation."""
    return metrics_from_confusion(binary_confusion_from_logits(logits, target, threshold=threshold))


def _as_numpy_bool(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(bool)


def symmetric_hausdorff_np(mask_a, mask_b) -> float:
    """Symmetric Hausdorff distance for binary masks. Returns image diagonal if one side is empty."""
    from scipy.spatial.distance import directed_hausdorff

    a = np.argwhere(_as_numpy_bool(mask_a))
    b = np.argwhere(_as_numpy_bool(mask_b))
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        h, w = _as_numpy_bool(mask_a).shape[:2]
        return float(np.sqrt(h * h + w * w))
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


def object_level_single_and_accum_np(pred_binary, gt_binary):
    """
    Object-level metrics used for validation-time model selection.

    This mirrors the test script's GlaS-style object metrics: object F1 is based
    on connected-component matching, while Dice/IoU/Hausdorff are area-weighted
    object-level scores.
    """
    from skimage import measure, morphology

    pred_binary = _as_numpy_bool(pred_binary)
    gt_binary = _as_numpy_bool(gt_binary)

    pred_labeled = measure.label(pred_binary > 0, connectivity=2)
    gt_labeled = measure.label(gt_binary > 0, connectivity=2)
    gt_labeled = morphology.remove_small_objects(gt_labeled > 0, min_size=3)
    gt_labeled = measure.label(gt_labeled, connectivity=2)

    ns = int(pred_labeled.max())
    ng = int(gt_labeled.max())

    if ng == 0 and ns == 0:
        single = {"recall": 1.0, "precision": 1.0, "F1": 1.0, "Dice": 1.0, "IoU": 1.0, "Hausdorff": 0.0}
        return single, np.zeros(11, dtype=float)

    if ng == 0 or ns == 0:
        h, w = gt_binary.shape[:2]
        penalty = float(np.sqrt(h * h + w * w))
        single = {
            "recall": 0.0 if ng > 0 else 1.0,
            "precision": 0.0 if ns > 0 else 1.0,
            "F1": 0.0,
            "Dice": 0.0,
            "IoU": 0.0,
            "Hausdorff": penalty,
        }
        gt_area = float((gt_labeled > 0).sum())
        pred_area = float((pred_labeled > 0).sum())
        acc = np.array([0, ns, ng, 0, 0, 0, 0, penalty * gt_area, penalty * pred_area, gt_area, pred_area], dtype=float)
        return single, acc

    tp = 0.0
    fp = 0.0
    for j in range(1, ns + 1):
        pred_j = pred_labeled == j
        overlap_parts = pred_j * gt_labeled
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]
        if obj_no.size == 0:
            fp += 1
            continue
        obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
        gt_obj = obj_no[int(np.argmax(obj_areas))]
        gt_obj_area = np.sum(gt_labeled == gt_obj)
        overlap_area = np.max(obj_areas)
        if float(overlap_area) / (gt_obj_area + 1e-10) >= 0.5:
            tp += 1
        else:
            fp += 1

    fn = float(ng) - tp
    recall = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)

    pred_area_total = float((pred_labeled > 0).sum())
    gt_area_total = float((gt_labeled > 0).sum())
    dice_g = iou_g = haus_g = 0.0
    dice_s = iou_s = haus_s = 0.0

    for i in range(1, ng + 1):
        gt_i = gt_labeled == i
        overlap_parts = gt_i * pred_labeled
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]
        gamma_i = float(gt_i.sum()) / (gt_area_total + 1e-10)
        if obj_no.size == 0:
            dice_i = 0.0
            iou_i = 0.0
            haus_i = min(symmetric_hausdorff_np(pred_labeled == j, gt_i) for j in range(1, ns + 1))
        else:
            obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
            seg_obj = obj_no[int(np.argmax(obj_areas))]
            pred_i = pred_labeled == seg_obj
            overlap_area = float(np.max(obj_areas))
            dice_i = 2 * overlap_area / (pred_i.sum() + gt_i.sum() + 1e-10)
            iou_i = overlap_area / (pred_i.sum() + gt_i.sum() - overlap_area + 1e-10)
            haus_i = symmetric_hausdorff_np(pred_i, gt_i)
        dice_g += gamma_i * dice_i
        iou_g += gamma_i * iou_i
        haus_g += gamma_i * haus_i

    for j in range(1, ns + 1):
        pred_j = pred_labeled == j
        overlap_parts = pred_j * gt_labeled
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]
        sigma_j = float(pred_j.sum()) / (pred_area_total + 1e-10)
        if obj_no.size == 0:
            dice_j = 0.0
            iou_j = 0.0
            haus_j = min(symmetric_hausdorff_np(pred_j, gt_labeled == i) for i in range(1, ng + 1))
        else:
            gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
            gt_obj = obj_no[int(np.argmax(gt_areas))]
            gt_j = gt_labeled == gt_obj
            overlap_area = float(np.max(gt_areas))
            dice_j = 2 * overlap_area / (pred_j.sum() + gt_j.sum() + 1e-10)
            iou_j = overlap_area / (pred_j.sum() + gt_j.sum() - overlap_area + 1e-10)
            haus_j = symmetric_hausdorff_np(pred_j, gt_j)
        dice_s += sigma_j * dice_j
        iou_s += sigma_j * iou_j
        haus_s += sigma_j * haus_j

    single = {
        "recall": recall,
        "precision": precision,
        "F1": f1,
        "Dice": (dice_g + dice_s) / 2,
        "IoU": (iou_g + iou_s) / 2,
        "Hausdorff": (haus_g + haus_s) / 2,
    }
    acc = np.array(
        [
            tp, fp, fn,
            dice_g * gt_area_total, dice_s * pred_area_total,
            iou_g * gt_area_total, iou_s * pred_area_total,
            haus_g * gt_area_total, haus_s * pred_area_total,
            gt_area_total, pred_area_total,
        ],
        dtype=float,
    )
    return single, acc


def object_metrics_from_accum_np(acc: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, dice_g, dice_s, iou_g, iou_s, haus_g, haus_s, gt_area, pred_area = acc
    return {
        "recall": tp / (tp + fn + 1e-10),
        "precision": tp / (tp + fp + 1e-10),
        "F1": 2 * tp / (2 * tp + fp + fn + 1e-10),
        "Dice": (dice_g / (gt_area + 1e-10) + dice_s / (pred_area + 1e-10)) / 2,
        "IoU": (iou_g / (gt_area + 1e-10) + iou_s / (pred_area + 1e-10)) / 2,
        "Hausdorff": (haus_g / (gt_area + 1e-10) + haus_s / (pred_area + 1e-10)) / 2,
    }


def object_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Batch-averaged object metrics from foreground logits and object/interior target masks."""
    if logits.shape[-2:] != target.shape[-2:]:
        logits = torch.nn.functional.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits).detach().cpu()
    target_cpu = target.detach().cpu()
    acc = np.zeros(11, dtype=float)
    for b in range(prob.shape[0]):
        pred_b = (prob[b, 0].numpy() >= float(threshold))
        gt_b = (target_cpu[b, 0].numpy() >= 0.5)
        _, one_acc = object_level_single_and_accum_np(pred_b, gt_b)
        acc += one_acc
    return object_metrics_from_accum_np(acc)


# -----------------------------------------------------------------------------
# Exact MILDNet/DCAN/GlaS-style object metrics for main-table reporting.
# These operate on INSTANCE LABEL MAPS, not binary accumulators.
# Aggregation for a split should be per-image mean, matching the uploaded
# MILDNet/DCAN testing scripts.
# -----------------------------------------------------------------------------

def connected_components_from_binary_np(mask, connectivity: int = 2) -> np.ndarray:
    from skimage import measure
    arr = _as_numpy_bool(mask)
    return measure.label(arr, connectivity=int(connectivity)).astype(np.int32)


def mild_dcan_postprocess_instance_np(
    prob_or_binary,
    threshold: float = 0.5,
    opening_radius: int = 5,
    min_size: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """MILDNet/DCAN-style foreground postprocess: threshold -> opening -> CC."""
    from skimage import measure, morphology
    arr = np.asarray(prob_or_binary)
    binary = arr.astype(bool) if arr.dtype == bool else (arr >= float(threshold))
    if int(opening_radius) > 0:
        binary = morphology.opening(binary, footprint=morphology.disk(int(opening_radius)))
    if int(min_size) > 0:
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=int(min_size))
    inst = measure.label(binary.astype(bool), connectivity=2).astype(np.int32)
    return binary.astype(bool), inst


def dcan_postprocess_instance_np(
    obj_prob: np.ndarray,
    contour_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    contour_threshold: float = 0.5,
    opening_radius: int = 5,
    min_size: int = 100,
    fill_holes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """DCAN whole-image postprocess: object & not contour -> opening -> holes/min-size -> CC."""
    import cv2

    obj = np.asarray(obj_prob, dtype=np.float32)
    if contour_prob is None:
        contour = np.zeros_like(obj, dtype=np.float32)
    else:
        contour = np.asarray(contour_prob, dtype=np.float32)
        if contour.shape != obj.shape:
            raise ValueError(f"contour_prob shape {contour.shape} != obj_prob shape {obj.shape}")

    binary = (obj >= float(threshold)) & (contour < float(contour_threshold))
    if int(opening_radius) > 0:
        r = int(opening_radius)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)

    if fill_holes:
        h, w = binary.shape
        padded = np.pad(binary.astype(np.uint8), ((1, 1), (1, 1)), mode="constant", constant_values=0)
        flood = padded.copy()
        mask = np.zeros((h + 4, w + 4), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 2)
        holes = flood[1:-1, 1:-1] == 0
        binary = binary | holes

    if int(min_size) > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        kept = np.zeros_like(binary, dtype=np.uint8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= int(min_size):
                kept[lab == i] = 1
        binary = kept.astype(bool)

    _, inst = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return binary.astype(bool), inst.astype(np.int32)


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
    listS = np.unique(S); listS = listS[listS != 0]
    listG = np.unique(G); listG = listG[listG != 0]
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
    listS = np.unique(S); listS = listS[listS != 0]
    listG = np.unique(G); listG = listG[listG != 0]
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
    listS = np.unique(S); listS = listS[listS != 0]
    listG = np.unique(G); listG = listG[listG != 0]
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
