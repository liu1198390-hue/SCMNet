# coding=utf-8
"""Training/validation engine for DSMNet-Res.

The engine is intentionally clean: no legacy branch diagnostics.  Validation
keeps the same foreground + soft-blocker object protocol so results remain
comparable with the existing DSMNet experiments.
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage import feature, measure, morphology, segmentation

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from utils import (
    add_confusion,
    binary_confusion_from_logits,
    binary_metrics_from_logits,
    dsmnet_predict_sliding_np,
    dsmnet_read_rgb,
    glass_object_metrics,
    mean_glass_object_metrics,
    metrics_from_confusion,
    new_confusion,
)


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _get_arg(args, name: str, default):
    return getattr(args, name, default) if args is not None else default


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes:02d}m{secs:02d}s"


def _format_metric(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _print_progress(phase: str, epoch, n: int, total, start: float, metrics: Dict[str, object]) -> None:
    step_text = f"{n}/{total}" if total is not None else str(n)
    metric_text = ", ".join(f"{k}={_format_metric(v)}" for k, v in metrics.items())
    elapsed = _format_seconds(time.time() - start)
    print(f"{phase} {epoch} [{step_text}] elapsed={elapsed} | {metric_text}", flush=True)


def _make_iterator(loader, phase: str, epoch, args, log_interval: int):
    total = len(loader) if hasattr(loader, "__len__") else None
    mode = str(_get_arg(args, "progress_mode", "simple")).lower()
    if mode == "auto":
        mode = "tqdm" if tqdm is not None else "simple"
    if mode == "tqdm" and tqdm is not None:
        pbar = tqdm(
            loader,
            desc=f"{phase} {epoch}",
            ncols=140,
            miniters=max(1, int(log_interval)),
            mininterval=float(_get_arg(args, "progress_mininterval", 10.0)),
            leave=bool(_get_arg(args, "tqdm_leave", False)),
        )
        return pbar, pbar, mode, total
    return loader, None, mode, total


def _should_print(mode: str, n: int, total, log_interval: int) -> bool:
    if mode in {"none", "off", "false", "0"}:
        return False
    if total is not None and n >= total:
        return True
    return n > 0 and n % max(1, int(log_interval)) == 0


def _main_logits(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    if isinstance(outputs, dict):
        return outputs.get("main_logits", outputs.get("logits"))
    return outputs


def _aux(outputs):
    if isinstance(outputs, tuple) and len(outputs) >= 2 and isinstance(outputs[1], dict):
        return outputs[1]
    if isinstance(outputs, dict):
        return outputs.get("aux", {})
    return {}


def _mean(meters: Dict[str, float], key: str, n: int, default: float = 0.0) -> float:
    return float(meters.get(key, default)) / max(1, int(n))


def _update_meters(meters: Dict[str, float], values: Dict[str, float]) -> None:
    for k, v in values.items():
        try:
            meters[k] = meters.get(k, 0.0) + float(v)
        except Exception:
            continue


def _object_score(obj_metrics: Dict[str, float]) -> float:
    return float(
        obj_metrics.get(
            "Score",
            0.30 * float(obj_metrics["F1"])
            + 0.40 * float(obj_metrics["Dice"])
            + 0.30 * float(obj_metrics["HausdorffQuality"]),
        )
    )


def soft_blocker_postprocess(
    fg_prob: np.ndarray,
    bd_prob: np.ndarray,
    sep_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    sure_threshold: float = 0.65,
    blocker_threshold: float = 0.60,
    keep_threshold: float = 0.75,
    blocker_mode: str = "boundary",
    opening_radius: int = 5,
    min_area: int = 100,
    fill_holes: bool = True,
    watershed: bool = False,
    watershed_min_distance: int = 12,
    instance_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fg = np.asarray(fg_prob, dtype=np.float32)
    bd = np.asarray(bd_prob, dtype=np.float32)
    sep = np.asarray(sep_prob, dtype=np.float32)
    if blocker_mode == "contract":
        blocker = bd
    elif blocker_mode == "boundary":
        blocker = bd
    elif blocker_mode == "separation":
        blocker = sep
    elif blocker_mode == "max":
        blocker = np.maximum(bd, sep)
    else:
        blocker = 0.5 * (bd + sep)

    if blocker_mode == "contract":
        # Match the differentiable probability supervised in the model loss.
        joint = fg*(1.0-bd) if instance_probability is None else np.asarray(instance_probability, dtype=np.float32)
        if joint.shape != fg.shape:
            raise ValueError("Joint instance probability and foreground sizes differ")
        binary = joint >= float(threshold)
    else:
        fg_mask = fg >= float(threshold)
        cut = blocker >= float(blocker_threshold)
        keep = fg >= float(keep_threshold)
        binary = fg_mask & (~cut | keep)

    if int(opening_radius) > 0:
        binary = morphology.opening(binary.astype(bool), footprint=morphology.disk(int(opening_radius)))
    if fill_holes:
        binary = ndi.binary_fill_holes(binary).astype(bool)
    if int(min_area) > 0:
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=int(min_area))
    if not watershed:
        return binary.astype(bool), measure.label(binary.astype(bool), connectivity=2).astype(np.int32)

    distance = ndi.distance_transform_edt(binary)
    sure_fg = (fg >= float(sure_threshold)) & binary
    labels_for_peaks = sure_fg.astype(np.uint8) if np.any(sure_fg) else binary.astype(np.uint8)
    coords = feature.peak_local_max(
        distance,
        labels=labels_for_peaks,
        min_distance=max(1, int(watershed_min_distance)),
        exclude_border=False,
    )
    marker_mask = np.zeros_like(binary, dtype=bool)
    if coords.size > 0:
        marker_mask[coords[:, 0], coords[:, 1]] = True
    markers = measure.label(marker_mask, connectivity=2)
    if markers.max() == 0:
        return binary.astype(bool), measure.label(binary.astype(bool), connectivity=2).astype(np.int32)
    elevation = -distance + 0.25 * blocker.astype(np.float32)
    inst = segmentation.watershed(elevation, markers=markers, mask=binary.astype(bool)).astype(np.int32)
    return (inst > 0), inst


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    args=None,
    scaler=None,
    amp=None,
    grad_clip=None,
    log_interval=None,
    scheduler=None,
    global_step: int = 0,
):
    if amp is None:
        amp = bool(_get_arg(args, "amp", False))
    if grad_clip is None:
        grad_clip = float(_get_arg(args, "grad_clip", 0.0))
    if log_interval is None:
        log_interval = int(_get_arg(args, "log_interval", 50))
    accumulation_steps = int(_get_arg(args, "gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    model.train()
    meters: Dict[str, float] = {"amp_overflow_updates": 0.0}
    n_batches = 0
    valid_batches = 0
    consecutive_amp_overflows = 0
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator, pbar, mode, total_batches = _make_iterator(loader, "Train", epoch, args, log_interval)
    optimizer.zero_grad(set_to_none=True)
    for batch in iterator:
        batch = move_batch_to_device(batch, device)

        with torch.amp.autocast("cuda", enabled=bool(amp and device.type == "cuda")):
            outputs = model(batch["image"], return_aux=True, aux_mode="loss")
            loss, logs = criterion(outputs, batch)

        if not torch.isfinite(loss):
            if bool(_get_arg(args, "fail_on_nonfinite", False)) or accumulation_steps > 1:
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, batch={n_batches + 1}/{total_batches}. "
                    "Inspect the preceding finite batch and model activations before restarting."
                )
            meters["skipped_nonfinite"] = meters.get("skipped_nonfinite", 0.0) + 1.0
            n_batches += 1
            global_step += 1
            continue

        batch_index = n_batches
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total_batches - group_start)
        normalized_loss = loss / group_size
        update_due = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == total_batches
        did_step = False
        if scaler is not None and amp and device.type == "cuda":
            scaler.scale(normalized_loss).backward()
            if update_due:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_step = scaler.get_scale() >= scale_before
                if did_step:
                    consecutive_amp_overflows = 0
                else:
                    meters["amp_overflow_updates"] += 1.0
                    consecutive_amp_overflows += 1
                    if bool(_get_arg(args, "fail_on_nonfinite", False)) and consecutive_amp_overflows >= 16:
                        raise FloatingPointError(
                            f"AMP skipped {consecutive_amp_overflows} consecutive optimizer updates "
                            f"at epoch={epoch}, batch={n_batches + 1}/{total_batches}."
                        )
        else:
            normalized_loss.backward()
            if update_due:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                did_step = True
        if update_due:
            if scheduler is not None and did_step:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            fg_metrics = binary_metrics_from_logits(_main_logits(outputs).detach(), batch["foreground"], threshold=0.5)

        global_step += 1
        n_batches += 1
        valid_batches += 1
        _update_meters(meters, {**logs, **{f"fg_{k}": v for k, v in fg_metrics.items()}})

        if _should_print(mode, n_batches, total_batches, log_interval):
            train_metrics = {
                "loss": _mean(meters, "loss", valid_batches),
                "fg_dice": _mean(meters, "fg_dice", valid_batches),
                "bd_loss": _mean(meters, "loss_boundary", valid_batches),
                "capc_contact": _mean(meters, "loss_capc_contact", valid_batches),
            }
            if mode == "tqdm" and pbar is not None:
                pbar.set_postfix({k: _format_metric(v) for k, v in train_metrics.items()})
            else:
                _print_progress("Train", epoch, n_batches, total_batches, start, train_metrics)

    avg = {
        k: (v if k in {"skipped_nonfinite", "amp_overflow_updates"} else v / max(1, valid_batches))
        for k, v in meters.items()
    }
    avg.setdefault("loss", float("nan"))
    avg.setdefault("fg_dice", 0.0)
    avg["valid_batches"] = float(valid_batches)
    avg["skipped_nonfinite"] = float(meters.get("skipped_nonfinite", 0.0))
    avg["amp_overflow_updates"] = float(meters["amp_overflow_updates"])
    if device.type == "cuda":
        avg["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        avg["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    avg["time"] = time.time() - start
    return avg, global_step


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch=0, args=None, amp=None, threshold=None):
    if amp is None:
        amp = bool(_get_arg(args, "amp", False))
    if threshold is None:
        threshold = float(_get_arg(args, "threshold", 0.5))
    boundary_threshold = float(_get_arg(args, "boundary_metric_threshold", 0.5))
    separation_threshold = float(_get_arg(args, "separation_metric_threshold", 0.3))
    compute_object = bool(_get_arg(args, "compute_val_object_metrics", False))
    val_full_mode = str(_get_arg(args, "val_full_mode", "sliding")).lower()
    tile_size = int(_get_arg(args, "val_full_tile_size", 464))
    stride = int(_get_arg(args, "val_full_stride", 232))
    fusion = str(_get_arg(args, "val_full_fusion", "uniform"))
    center_eps = float(_get_arg(args, "val_full_center_weight_eps", 0.05))
    contour_threshold = float(_get_arg(args, "val_object_contour_threshold", 0.5))
    blocker_threshold = _get_arg(args, "val_dsmnet_blocker_threshold", None)
    blocker_threshold = contour_threshold if blocker_threshold is None else float(blocker_threshold)
    keep_threshold = float(_get_arg(args, "val_dsmnet_keep_threshold", 0.75))
    sure_threshold = float(_get_arg(args, "val_dsmnet_sure_threshold", 0.65))
    blocker_mode = str(_get_arg(args, "val_dsmnet_blocker_mode", "boundary"))
    watershed = bool(_get_arg(args, "val_dsmnet_watershed", False))
    watershed_min_distance = int(_get_arg(args, "val_dsmnet_watershed_min_distance", 12))
    opening_radius = int(_get_arg(args, "val_object_opening_radius", 5))
    min_size = int(_get_arg(args, "val_object_min_size", 100))
    log_interval = int(_get_arg(args, "val_log_interval", _get_arg(args, "log_interval", 50)))

    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss_meters: Dict[str, float] = {}
    n_batches = 0
    start = time.time()
    fg_conf = new_confusion()
    inside_conf = new_confusion()
    bd_conf = new_confusion()
    sep_conf = new_confusion()
    has_bd = has_sep = False
    obj_rows = []
    obj_available = False

    iterator, pbar, mode, total_batches = _make_iterator(loader, "Val", epoch, args, log_interval)
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=bool(amp and device.type == "cuda")):
            outputs = model(batch["image"], return_aux=True, aux_mode="loss")
            loss, logs = criterion(outputs, batch)
        n_batches += 1
        _update_meters(loss_meters, logs)
        main_logits = _main_logits(outputs)
        aux = _aux(outputs)

        if val_full_mode == "sliding":
            image_paths = batch.get("image_path", None)
            if image_paths is None:
                raise RuntimeError("Sliding validation requires batch['image_path'].")
            if isinstance(image_paths, (str, bytes)):
                image_paths = [image_paths]
            gt_instances = batch.get("object_instance", batch["inside"]).detach().cpu().numpy()
            for b, image_path in enumerate(image_paths):
                img_rgb = dsmnet_read_rgb(image_path)
                probability_maps = dsmnet_predict_sliding_np(
                    model,
                    img_rgb,
                    device,
                    tile_size=tile_size,
                    stride=stride,
                    fusion=fusion,
                    center_weight_eps=center_eps,
                    return_instance_probability=(blocker_mode == "contract"),
                )
                fg_prob, bd_prob, sep_prob = probability_maps[:3]
                instance_probability = probability_maps[3] if blocker_mode == "contract" else None
                pred_bin, pred_inst = soft_blocker_postprocess(
                    fg_prob,
                    bd_prob,
                    sep_prob,
                    threshold=float(threshold),
                    sure_threshold=sure_threshold,
                    blocker_threshold=blocker_threshold,
                    keep_threshold=keep_threshold,
                    blocker_mode=blocker_mode,
                    opening_radius=opening_radius,
                    min_area=min_size,
                    fill_holes=True,
                    watershed=watershed,
                    watershed_min_distance=watershed_min_distance,
                    instance_probability=instance_probability,
                )
                gt_inst = gt_instances[b, 0].astype(np.int32)
                gt_bin = gt_inst > 0
                add_confusion(
                    fg_conf,
                    {
                        "tp": float(np.logical_and(pred_bin, gt_bin).sum()),
                        "tn": float(np.logical_and(~pred_bin, ~gt_bin).sum()),
                        "fp": float(np.logical_and(pred_bin, ~gt_bin).sum()),
                        "fn": float(np.logical_and(~pred_bin, gt_bin).sum()),
                    },
                )
                if compute_object:
                    obj_rows.append(glass_object_metrics(pred_inst, gt_inst))
                    obj_available = True
        else:
            add_confusion(fg_conf, binary_confusion_from_logits(main_logits, batch["foreground"], threshold=threshold))

        add_confusion(inside_conf, binary_confusion_from_logits(main_logits, batch["inside"], threshold=threshold))

        if torch.is_tensor(aux.get("boundary_logits", None)):
            has_bd = True
            add_confusion(bd_conf, binary_confusion_from_logits(aux["boundary_logits"], batch["boundary"], threshold=boundary_threshold))
        contact_logits = aux.get("contact_logits", aux.get("separation_core_logits", None))
        if torch.is_tensor(contact_logits) and torch.is_tensor(batch.get("separation", None)):
            has_sep = True
            contact_core = batch["separation"]
            contact_distance = batch.get("separation_distance", contact_core).clamp(0.0, 1.0)
            contact_target = torch.maximum(contact_core, 0.75 * contact_distance)
            contact_target = (contact_target >= separation_threshold).to(dtype=contact_target.dtype)
            add_confusion(
                sep_conf,
                binary_confusion_from_logits(contact_logits, contact_target, threshold=separation_threshold),
            )

        fg_now = metrics_from_confusion(fg_conf)
        bd_now = metrics_from_confusion(bd_conf) if has_bd else {"dice": 0.0}
        obj_text = "off"
        if compute_object and obj_available:
            obj_now = mean_glass_object_metrics(obj_rows)
            obj_text = f"{_object_score(obj_now):.4f}"
        if _should_print(mode, n_batches, total_batches, log_interval):
            val_metrics = {
                "loss": _mean(loss_meters, "loss", n_batches),
                "fg_dice": fg_now.get("dice", 0.0),
                "bd_dice": bd_now.get("dice", 0.0),
                "obj": obj_text,
            }
            if mode == "tqdm" and pbar is not None:
                pbar.set_postfix({k: _format_metric(v) for k, v in val_metrics.items()})
            else:
                _print_progress("Val", epoch, n_batches, total_batches, start, val_metrics)

    avg = {k: v / max(1, n_batches) for k, v in loss_meters.items()}
    if device.type == "cuda":
        avg["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        avg["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    avg.update({f"fg_{k}": v for k, v in metrics_from_confusion(fg_conf).items()})
    avg.update({f"inside_{k}": v for k, v in metrics_from_confusion(inside_conf).items()})
    if has_bd:
        avg.update({f"bd_{k}": v for k, v in metrics_from_confusion(bd_conf).items()})
    if has_sep:
        avg.update({f"sep_{k}": v for k, v in metrics_from_confusion(sep_conf).items()})

    if compute_object and obj_available:
        obj_metrics = mean_glass_object_metrics(obj_rows)
        avg.update({f"obj_{k}": v for k, v in obj_metrics.items()})
        avg["obj_score"] = _object_score(obj_metrics)
    elif compute_object:
        avg["obj_metric_error"] = 1.0
    avg["time"] = time.time() - start
    return avg
