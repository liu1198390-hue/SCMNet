# coding=utf-8
"""Evaluate DSMNet-Res checkpoints with foreground + soft-blocker objects."""
from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import csv
import datetime as _dt
import io
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from configs.model import build_model_config
from engine import soft_blocker_postprocess
from models.scmnet import ContactAwarePyramidCorrection, DSMNetRes, UnifiedHRMDR
from utils import (
    add_confusion,
    center_weight_np,
    dsmnet_image_to_tensor_np,
    dsmnet_predict_sliding_np,
    find_mask,
    first_existing_dir,
    glass_object_metrics,
    list_images,
    mean_glass_object_metrics,
    metrics_from_confusion,
    new_confusion,
    parse_tuple_int,
    reflect_pad_image_np,
    sliding_positions,
)


class TeeStream(io.TextIOBase):
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def add_bool_argument(parser: argparse.ArgumentParser, option: str, default: bool) -> None:
    """Add --flag/--no-flag while remaining compatible with Python 3.7+."""
    destination = option.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(option, dest=destination, action="store_true")
    group.add_argument(f"--no-{option[2:]}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: bool(default)})


def install_test_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(old_stdout, log_file)
    sys.stderr = TeeStream(old_stderr, log_file)

    def close_log():
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    atexit.register(close_log)
    print(f"\n===== TEST { _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S') } =====")
    print(" ".join([Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]]))
    print(f"=> test log append: {log_path}")


def read_instance_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.uint32)
        arr = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    return arr.astype(np.int32)


def resolve_dirs(root, split, foreground_dir_name, inside_dir_name, boundary_dir_name, object_dir_name):
    root = Path(root)
    img_dir = root / "images" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    fg_dir, fg_source = first_existing_dir(root, split, [foreground_dir_name, "labels_foreground", "labels_binary", "labels_class"])
    inside_dir, inside_source = first_existing_dir(root, split, [inside_dir_name, "labels_inside", "labels_class"])
    bd_dir, bd_source = first_existing_dir(root, split, [boundary_dir_name, "labels_boundary", "labels_class"])
    obj_dir, obj_source = first_existing_dir(root, split, [object_dir_name, "labels_instance_cc", "labels_instance", "labels_instances"])
    if fg_dir is None:
        raise FileNotFoundError("Foreground label dir not found.")
    if bd_dir is None:
        raise FileNotFoundError("Boundary label dir not found.")
    if obj_dir is None:
        raise FileNotFoundError("Object instance label dir not found.")
    return img_dir, fg_dir, inside_dir, bd_dir, obj_dir, fg_source, inside_source, bd_source, obj_source


def normalize_image(img: Image.Image, input_size: int):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    ten = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
    ten = F.interpolate(ten, size=(input_size, input_size), mode="bilinear", align_corners=False)
    return ten.squeeze(0)


def resize_prob(prob: np.ndarray, size_wh: Tuple[int, int]):
    return np.asarray(
        Image.fromarray(np.clip(prob * 255, 0, 255).astype(np.uint8)).resize(size_wh, Image.BILINEAR)
    ).astype(np.float32) / 255.0


def load_json(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(args):
    run_cfg = load_json(args.run_config)
    if run_cfg is not None:
        cfg = dict(run_cfg.get("model_config", {}))
        if not cfg:
            raise ValueError("Run configuration must contain a nonempty model_config.")
        run_args = run_cfg.get("args", {})
        args.input_size = int(run_args.get("input_size", run_cfg.get("input_size", args.input_size)))
    else:
        cfg = build_model_config(
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depths=parse_tuple_int(args.depths),
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            drop_path_rate=args.drop_path_rate,
            decoder_channels=args.decoder_channels,
            ppm_channels=args.ppm_channels,
            ppm_pool_scales=parse_tuple_int(args.ppm_pool_scales),
            decoder_variant=args.decoder_variant,
            use_checkpoint=False,
            use_separation_head=not args.no_separation_head,
            use_gland_ssm_block=args.use_gland_ssm_block,
            gland_ssm_block_stages=parse_tuple_int(args.gland_ssm_block_stages),
            gland_ssm_scales=parse_tuple_int(args.gland_ssm_scales),
            sgv_compact_ratio=args.sgv_compact_ratio,
            sgv_scan_guidance=args.sgv_scan_guidance,
            sgv_adapter_placement=args.sgv_adapter_placement,
            use_bssf_lite=args.use_bssf_lite,
            repair_variant=args.repair_variant,
            residual_scale_init=args.residual_scale_init,
            capc_residual_scale_init=args.capc_residual_scale_init,
            capc_prior_scale_init=args.capc_prior_scale_init,
            use_capc=args.use_capc,
            hr_min_gate=args.hr_min_gate,
            hr_what_gradient_scale=args.hr_what_gradient_scale,
            hr_morphology_dilations=parse_tuple_int(args.hr_morphology_dilations),
            hr_image_feature_channels=args.hr_image_feature_channels,
            hr_cue_feature_channels=args.hr_cue_feature_channels,
            capc_cue_channels=args.capc_cue_channels,
            capc_contact_floor=args.capc_contact_floor,
            capc_use_discrepancy=args.capc_use_discrepancy,
            capc_use_contact_cue=args.capc_use_contact_cue,
            capc_use_sgv_prior=args.capc_use_sgv_prior,
            hr_use_cue_guidance=args.hr_use_cue_guidance,
            qc_evidence_mode=args.qc_evidence_mode,
            contact_evidence_power=args.contact_evidence_power,
            contact_contrast_gain=args.contact_contrast_gain,
        )
    cfg["use_checkpoint"] = False
    return DSMNetRes(**cfg), cfg


def load_checkpoint(model, ckpt_path, strict=False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        saved_config = ckpt.get("model_config")
        if strict and isinstance(saved_config, dict):
            # These settings change the forward paths without changing tensor shapes.
            cue_fields = (
                "sgv_scan_guidance",
                "capc_use_discrepancy",
                "capc_use_sgv_prior",
                "capc_use_contact_cue",
                "hr_use_cue_guidance",
                "qc_evidence_mode",
                "contact_evidence_power",
                "contact_contrast_gain",
                "capc_contact_floor",
                "hr_min_gate",
            )
            mismatches = [
                f"{key}: checkpoint={saved_config[key]!r}, evaluation={model.model_config[key]!r}"
                for key in cue_fields
                if (
                    key in saved_config and key in model.model_config
                    and saved_config[key] != model.model_config[key]
                )
            ]
            if mismatches:
                raise ValueError(
                    "Checkpoint cue configuration mismatch. Use its matching run_config.json. "
                    + ", ".join(mismatches)
                )
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt
    model_state = model.state_dict()
    loaded = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        kk = k.replace("module.", "")
        candidates = [kk, kk[4:] if kk.startswith("net.") else "net." + kk]
        for cand in candidates:
            if cand in model_state and model_state[cand].shape == v.shape:
                loaded[cand] = v
                break
    if len(loaded) == 0:
        raise RuntimeError("No checkpoint tensors matched model.")
    if strict and len(loaded) != len(model_state):
        missing = sorted(set(model_state) - set(loaded))
        raise RuntimeError(f"Checkpoint/model mismatch: {len(missing)} tensors missing; first keys: {missing[:10]}")
    model.load_state_dict(loaded, strict=strict)
    print(f"=> loaded checkpoint tensors: {len(loaded)}/{len(model_state)}")


@contextmanager
def contact_disabled(model):
    """A same-checkpoint inference intervention; restore every cue flag."""
    modules = [
        module for module in model.modules()
        if isinstance(module, (ContactAwarePyramidCorrection, UnifiedHRMDR))
    ]
    if not modules:
        raise RuntimeError("The model has no CAPC/HR-MDR contact path to disable.")
    states = [module.use_contact_cue for module in modules]
    try:
        for module in modules:
            module.use_contact_cue = False
        yield
    finally:
        for module, enabled in zip(modules, states):
            module.use_contact_cue = enabled


@torch.no_grad()
def predict(model, img: Image.Image, args):
    ow, oh = img.size
    if args.mode == "sliding":
        img_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        return dsmnet_predict_sliding_np(
            model,
            img_rgb,
            args.device,
            tile_size=args.tile_size,
            stride=args.stride,
            fusion=args.fusion,
            center_weight_eps=args.center_weight_eps,
            return_boundary_distance=True,
            return_instance_probability=(args.dsmnet_blocker_mode == "contract"),
        )

    x = normalize_image(img, args.input_size).unsqueeze(0).to(args.device, non_blocking=True)
    outputs = model(x, return_aux=True, aux_mode="loss")
    if isinstance(outputs, tuple):
        main_logits, aux = outputs
    else:
        main_logits, aux = outputs, {}
    fg_prob = torch.sigmoid(main_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    instance_prob = None
    if args.dsmnet_blocker_mode == "contract" and isinstance(aux, dict):
        if torch.is_tensor(aux.get("instance_probability")):
            instance_prob = aux["instance_probability"][0, 0].detach().float().cpu().numpy()
    bd_prob = None
    bd_distance_prob = None
    sep_prob = None
    if isinstance(aux, dict) and torch.is_tensor(aux.get("boundary_logits", None)):
        bd_prob = torch.sigmoid(aux["boundary_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
    if isinstance(aux, dict) and torch.is_tensor(aux.get("boundary_distance_logits", None)):
        bd_distance_prob = torch.sigmoid(aux["boundary_distance_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
    if isinstance(aux, dict) and torch.is_tensor(aux.get("separation_core_logits", None)):
        sep_core = torch.sigmoid(aux["separation_core_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
        if torch.is_tensor(aux.get("separation_field_logits", None)):
            sep_field = torch.sigmoid(aux["separation_field_logits"])[0, 0].detach().cpu().numpy().astype(np.float32)
            sep_prob = np.maximum(sep_core, 0.5 * sep_field).astype(np.float32, copy=False)
        else:
            sep_prob = sep_core
    if fg_prob.shape != (oh, ow):
        fg_prob = resize_prob(fg_prob, (ow, oh))
    if bd_prob is None:
        bd_prob = np.zeros_like(fg_prob, dtype=np.float32)
    elif bd_prob.shape != (oh, ow):
        bd_prob = resize_prob(bd_prob, (ow, oh))
    if bd_distance_prob is None:
        bd_distance_prob = np.zeros_like(fg_prob, dtype=np.float32)
    elif bd_distance_prob.shape != (oh, ow):
        bd_distance_prob = resize_prob(bd_distance_prob, (ow, oh))
    if sep_prob is None:
        sep_prob = np.zeros_like(fg_prob, dtype=np.float32)
    elif sep_prob.shape != (oh, ow):
        sep_prob = resize_prob(sep_prob, (ow, oh))
    result = (fg_prob, bd_prob, sep_prob, bd_distance_prob)
    if args.dsmnet_blocker_mode == "contract":
        if instance_prob is None:
            raise RuntimeError("Contract inference requires the trained joint instance probability")
        if instance_prob.shape != (oh, ow):
            instance_prob = np.asarray(
                Image.fromarray(instance_prob.astype(np.float32)).resize((ow, oh), Image.BILINEAR),
                dtype=np.float32,
            )
        result += (instance_prob,)
    return result


@torch.no_grad()
def predict_feature_maps(model, img: Image.Image, args):
    """Extract internal responses with the same tile geometry as the prediction."""
    image = np.asarray(img.convert("RGB"), dtype=np.uint8)
    oh, ow = image.shape[:2]
    if args.mode == "sliding":
        tile = int(args.tile_size)
        padded, _, _ = reflect_pad_image_np(image, tile)
        ys = sliding_positions(padded.shape[0], tile, int(args.stride))
        xs = sliding_positions(padded.shape[1], tile, int(args.stride))
        weight = (np.ones((tile, tile), dtype=np.float32) if args.fusion == "uniform"
                  else center_weight_np(tile, eps=args.center_weight_eps))
    else:
        tile = int(args.input_size)
        padded = np.asarray(img.resize((tile, tile), Image.BILINEAR), dtype=np.uint8)
        ys = xs = [0]
        weight = np.ones((tile, tile), dtype=np.float32)

    sums = {}
    weights = np.zeros(padded.shape[:2], dtype=np.float32)
    for y0 in ys:
        for x0 in xs:
            patch = padded[y0:y0 + tile, x0:x0 + tile]
            x = dsmnet_image_to_tensor_np(patch).unsqueeze(0).to(args.device, non_blocking=True)
            outputs = model(x, return_aux=True, aux_mode="visual")
            aux = outputs[1] if isinstance(outputs, tuple) else {}
            if not isinstance(aux, dict):
                raise RuntimeError("Visualization forward did not return auxiliary maps.")
            structure = aux.get("structure_priors") or []
            boundary = aux.get("capc_boundary_logits") or []
            contact = aux.get("capc_contact_logits") or []
            capc_gates = aux.get("capc_gate_maps") or []
            capc_contacts = aux.get("capc_contact_evidence_maps") or []
            gate = aux.get("hr_gate_map")
            maps = {}
            if structure and torch.is_tensor(structure[0]) and structure[0].shape[1] >= 2:
                maps.update(S_h=structure[0][:, 0:1], S_v=structure[0][:, 1:2])
            for scale, cue in enumerate(boundary):
                if torch.is_tensor(cue):
                    maps["Q_i" if scale == 0 else f"Q_{scale}"] = torch.sigmoid(cue)
            for scale, cue in enumerate(contact):
                if torch.is_tensor(cue):
                    maps["C_i" if scale == 0 else f"C_{scale}"] = torch.sigmoid(cue)
            for scale, cue in enumerate(capc_gates):
                if torch.is_tensor(cue):
                    maps["G_CAPC" if scale == 0 else f"G_CAPC_{scale}"] = cue
            for scale, cue in enumerate(capc_contacts):
                if torch.is_tensor(cue):
                    maps["C_effective" if scale == 0 else f"C_effective_{scale}"] = cue
            if torch.is_tensor(gate):
                maps["G"] = gate
            for source, name in (
                ("hr_boundary_evidence_map", "Q_where"),
                ("hr_contact_evidence_map", "C_where"),
                ("hr_contact_extra_gate_map", "C_extra_gate"),
                ("hr_gated_delta_abs_map", "repair_abs"),
            ):
                cue = aux.get(source)
                if torch.is_tensor(cue):
                    maps[name] = cue
            maps["P_foreground"] = torch.sigmoid(outputs[0])
            if torch.is_tensor(aux.get("instance_probability")):
                maps["P_instance"] = aux["instance_probability"]
            if torch.is_tensor(aux.get("boundary_logits")):
                maps["P_boundary"] = torch.sigmoid(aux["boundary_logits"])
            for name, tensor in maps.items():
                if tensor.shape[-2:] != (tile, tile):
                    tensor = F.interpolate(tensor, size=(tile, tile), mode="bilinear", align_corners=False)
                value = tensor[0, 0].detach().float().cpu().numpy().astype(np.float32)
                if name not in sums:
                    sums[name] = np.zeros(padded.shape[:2], dtype=np.float32)
                sums[name][y0:y0 + tile, x0:x0 + tile] += value * weight
            weights[y0:y0 + tile, x0:x0 + tile] += weight
    if not sums:
        return {}
    result = {name: value / np.maximum(weights, 1e-8) for name, value in sums.items()}
    if args.mode == "sliding":
        return {name: value[:oh, :ow] for name, value in result.items()}
    return {name: np.asarray(Image.fromarray(value, mode="F").resize((ow, oh), Image.BILINEAR),
                             dtype=np.float32) for name, value in result.items()}


def signed_map_to_uint8(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(value, [1.0, 99.0])
    if high <= low + 1e-6:
        low, high = float(value.min()), float(value.max())
    if high <= low + 1e-6:
        return np.full(value.shape, 128, dtype=np.uint8)
    return np.clip((value - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def save_feature_visuals(feature_maps, out_dir: Path, name: str):
    out_dir = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "raw_maps.npz", **feature_maps)
    for map_name in ("S_h", "S_v", "Q_i", "C_i", "C_effective", "Q_where", "C_where",
                     "C_extra_gate", "G_CAPC", "G", "P_foreground", "P_boundary", "P_instance",
                     "Q_1", "Q_2", "C_1", "C_2", "C_effective_1", "C_effective_2"):
        if map_name not in feature_maps:
            continue
        value = feature_maps[map_name]
        if map_name not in {"S_h", "S_v"}:
            image = np.clip(value * 255.0, 0, 255).astype(np.uint8)
        else:
            image = signed_map_to_uint8(value)
        save_uint8(out_dir / f"{map_name}.png", image)
    metadata = {
        key: {"mean": float(np.mean(value)), "p99": float(np.percentile(value, 99)),
              "max": float(np.max(value))}
        for key, value in feature_maps.items()
    }
    (out_dir / "map_statistics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"=> feature visuals saved: {out_dir}")


def _instance_seams(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    seams = np.zeros(labels.shape, dtype=bool)
    adjacent = (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] > 0) & (labels[:, :-1] > 0)
    seams[:, 1:] |= adjacent
    adjacent = (labels[1:] != labels[:-1]) & (labels[1:] > 0) & (labels[:-1] > 0)
    seams[1:] |= adjacent
    return seams


def _instance_black_white(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    image = np.repeat(((labels > 0).astype(np.uint8) * 255)[..., None], 3, axis=2)
    image[_instance_seams(labels)] = 0
    return image


def _cue_color(value: np.ndarray, upper: float, channel: int) -> np.ndarray:
    scaled = np.clip(np.asarray(value, dtype=np.float32) / upper, 0.0, 1.0)
    image = np.zeros((*scaled.shape, 3), dtype=np.uint8)
    image[..., channel] = np.rint(255.0 * scaled).astype(np.uint8)
    return image


def save_qc_review_panel(image_rgb, gt_inst, pred_inst, feature_maps, out_dir: Path, name: str, evidence_mode="legacy"):
    """Diagnostic panels from real predictions; every cue uses a fixed stated scale."""
    required = ("Q_i", "C_i", "C_where", "C_extra_gate")
    if any(key not in feature_maps for key in required):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    calibrated = evidence_mode == "calibrated"
    c_upper, permission_upper = (1.0, 1.0) if calibrated else (0.1, 0.2)
    probability_key = "P_instance" if calibrated and "P_instance" in feature_maps else "P_foreground"
    panels = [
        np.asarray(image_rgb, dtype=np.uint8),
        _instance_black_white(gt_inst),
        _instance_black_white(pred_inst),
        _cue_color(feature_maps[probability_key], 1.0, 2),
        _cue_color(feature_maps["Q_i"], 1.0, 2),
        _cue_color(feature_maps["C_i"], c_upper, 0),
        _cue_color(feature_maps["C_where"], permission_upper, 0),
        _cue_color(feature_maps["C_extra_gate"], permission_upper, 0),
    ]
    titles = ("H&E", "GT", "Prediction", "P instance [0,1]" if probability_key == "P_instance" else "P foreground [0,1]",
              "Q raw [0,1]", f"C raw [0,{c_upper:g}]", f"C in WHERE [0,{permission_upper:g}]",
              f"C extra gate [0,{permission_upper:g}]")
    top = make_labeled_strip(panels[:4], titles[:4])
    bottom = make_labeled_strip(panels[4:], titles[4:])
    save_uint8(out_dir / f"{name}_qc_review.png", np.concatenate([top, bottom], axis=0))
    save_uint8(out_dir / f"{name}_Q_blue.png", panels[4])
    save_uint8(out_dir / f"{name}_C_raw_red_{'0to1' if calibrated else '0to0p1'}.png", panels[5])
    save_uint8(out_dir / f"{name}_C_where_red.png", panels[6])


def save_contact_counterfactual_panel(
    image_rgb, gt_inst, pred_with, pred_without, feature_maps,
    metrics_with, metrics_without, out_dir: Path, name: str,
    evidence_mode="legacy",
):
    """Show true Q/C responses beside a same-checkpoint C-off intervention."""
    required = ("Q_i", "C_i", "C_where", "C_extra_gate")
    if any(key not in feature_maps for key in required):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    c_upper, permission_upper = (1.0, 1.0) if evidence_mode == "calibrated" else (0.1, 0.2)
    panels = [
        np.asarray(image_rgb, dtype=np.uint8),
        _instance_black_white(gt_inst),
        _instance_black_white(pred_without),
        _instance_black_white(pred_with),
        _cue_color(feature_maps["Q_i"], 1.0, 2),
        _cue_color(feature_maps["C_i"], c_upper, 0),
        _cue_color(feature_maps["C_where"], permission_upper, 0),
        _cue_color(feature_maps["C_extra_gate"], permission_upper, 0),
    ]
    titles = (
        "H&E", "GT",
        f"C off (F1 {metrics_without['F1']:.3f})",
        f"C on (F1 {metrics_with['F1']:.3f})",
        "Q raw [0,1]", f"C raw [0,{c_upper:g}]",
        f"C in WHERE [0,{permission_upper:g}]", f"C extra gate [0,{permission_upper:g}]",
    )
    top = make_labeled_strip(panels[:4], titles[:4])
    bottom = make_labeled_strip(panels[4:], titles[4:])
    save_uint8(out_dir / f"{name}_contact_effect_review.png", np.concatenate([top, bottom], axis=0))


def pixel_confusion(pred_binary, gt_binary):
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)
    return {
        "tp": float(np.logical_and(pred, gt).sum()),
        "tn": float(np.logical_and(~pred, ~gt).sum()),
        "fp": float(np.logical_and(pred, ~gt).sum()),
        "fn": float(np.logical_and(~pred, gt).sum()),
    }


def pixel_metrics_from_confusion(c):
    m = metrics_from_confusion(c)
    return {
        "pixel_acc": m["acc"],
        "pixel_iou": m["iou"],
        "pixel_recall": m["recall"],
        "pixel_precision": m["precision"],
        "pixel_F1": m["dice"],
        "pixel_specificity": m["specificity"],
    }


def save_uint8(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_uint16(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint16)).save(path)


def save_visuals(gt_inst, pred_bin, pred_inst, fg_prob, bd_prob, sep_prob, bd_distance_prob, out_dir, name, instance_probability=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_uint8(out_dir / f"{name}_prob_foreground.png", np.clip(fg_prob * 255, 0, 255))
    save_uint8(out_dir / f"{name}_prob_boundary.png", np.clip(bd_prob * 255, 0, 255))
    save_uint8(out_dir / f"{name}_prob_boundary_distance.png", np.clip(bd_distance_prob * 255, 0, 255))
    save_uint8(out_dir / f"{name}_prob_separation.png", np.clip(sep_prob * 255, 0, 255))
    save_uint8(out_dir / f"{name}_pred_binary.png", pred_bin.astype(np.uint8) * 255)
    save_uint16(out_dir / f"{name}_pred_instance.png", pred_inst.astype(np.uint16))
    save_uint16(out_dir / f"{name}_gt_instance.png", gt_inst.astype(np.uint16))
    if instance_probability is not None:
        save_uint8(out_dir / f"{name}_prob_instance.png", np.clip(instance_probability*255, 0, 255))


def render_instances(instance_map: np.ndarray) -> np.ndarray:
    labels = instance_map.astype(np.int32, copy=False)
    rendered = np.zeros((*labels.shape, 3), dtype=np.uint8)
    rendered[labels > 0] = np.array([255, 255, 255], dtype=np.uint8)

    boundaries = np.zeros(labels.shape, dtype=bool)
    vertical_diff = (labels[1:, :] != labels[:-1, :]) & ((labels[1:, :] != 0) | (labels[:-1, :] != 0))
    horizontal_diff = (labels[:, 1:] != labels[:, :-1]) & ((labels[:, 1:] != 0) | (labels[:, :-1] != 0))
    boundaries[1:, :] |= vertical_diff
    boundaries[:-1, :] |= vertical_diff
    boundaries[:, 1:] |= horizontal_diff
    boundaries[:, :-1] |= horizontal_diff
    rendered[boundaries] = np.array([128, 128, 128], dtype=np.uint8)
    return rendered


def probability_to_rgb(probability: np.ndarray) -> np.ndarray:
    gray = np.clip(probability * 255, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def make_labeled_strip(panels, titles, title_height: int = 32) -> np.ndarray:
    if len(panels) != len(titles):
        raise ValueError("The number of panels and titles must match.")
    target_height, target_width = panels[0].shape[:2]
    font_size = max(18, min(36, round(target_width/40)))
    label_font = None
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(candidate).is_file():
            label_font = ImageFont.truetype(candidate, font_size)
            break
    if label_font is None:
        label_font = ImageFont.load_default()
    title_height = max(title_height, font_size+16)
    labeled_panels = []
    for panel, title in zip(panels, titles):
        image = Image.fromarray(panel.astype(np.uint8), mode="RGB")
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.BILINEAR)
        canvas = Image.new("RGB", (target_width, target_height + title_height), "white")
        draw = ImageDraw.Draw(canvas)
        text_box = draw.textbbox((0, 0), title, font=label_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        draw.text(
            ((target_width - text_width) // 2, (title_height - text_height) // 2 - text_box[1]),
            title,
            fill="black",
            font=label_font,
        )
        canvas.paste(image, (0, title_height))
        labeled_panels.append(np.asarray(canvas))
    return np.concatenate(labeled_panels, axis=1)


def save_paper_visuals(image_rgb, gt_inst, pred_inst, fg_prob, bd_prob, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    input_rgb = np.asarray(image_rgb, dtype=np.uint8)
    gt_color = render_instances(gt_inst)
    pred_color = render_instances(pred_inst)
    foreground_rgb = probability_to_rgb(fg_prob)
    boundary_rgb = probability_to_rgb(bd_prob)

    save_uint8(out_dir / f"{name}_input.png", input_rgb)
    save_uint8(out_dir / f"{name}_gt_instance_color.png", gt_color)
    save_uint8(out_dir / f"{name}_pred_instance_color.png", pred_color)
    save_uint8(
        out_dir / f"{name}_comparison_triptych.png",
        make_labeled_strip(
            [input_rgb, gt_color, pred_color],
            ["Input", "GT Instances", "Prediction"],
        ),
    )
    save_uint8(
        out_dir / f"{name}_method_outputs.png",
        make_labeled_strip(
            [input_rgb, foreground_rgb, boundary_rgb, pred_color],
            ["Input", "Foreground", "Boundary", "Final Instance"],
        ),
    )


def write_metrics(path, title, metrics):
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Evaluate SCMNet with boundary-guided instance recovery.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run_config", default=None)
    p.add_argument("--data_path", required=True)
    p.add_argument("--split", default="testA")
    p.add_argument("--foreground_dir_name", default="labels_foreground")
    p.add_argument("--inside_dir_name", default="labels_inside")
    p.add_argument("--boundary_dir_name", default="labels_boundary")
    p.add_argument("--object_dir_name", "--label_subdir", dest="object_dir_name", default="labels_instance_cc")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--mode", default="sliding", choices=["sliding", "resize"])
    p.add_argument("--tile_size", type=int, default=464)
    p.add_argument("--stride", type=int, default=232)
    p.add_argument("--fusion", default="uniform", choices=["uniform", "center_weighted"])
    p.add_argument("--center_weight_eps", type=float, default=0.05)
    p.add_argument("--input_size", type=int, default=464)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--opening_radius", type=int, default=5)
    p.add_argument("--min_area", type=int, default=100)
    p.add_argument("--dsmnet_sure_threshold", type=float, default=0.65)
    p.add_argument("--dsmnet_blocker_threshold", type=float, default=0.60)
    p.add_argument("--dsmnet_keep_threshold", type=float, default=0.75)
    p.add_argument("--dsmnet_blocker_mode", default="boundary", choices=["max", "mean", "boundary", "separation", "contract"])
    p.add_argument("--dsmnet_watershed", action="store_true")
    p.add_argument("--dsmnet_watershed_min_distance", type=int, default=12)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--save_flag", action="store_true")
    p.add_argument("--save_feature_visuals", action="store_true")
    p.add_argument("--save_feature_visuals_all", action="store_true", help="Save internal maps for every test image.")
    p.add_argument("--no_paper_vis", action="store_true", help="Skip legacy triptych and strip exports.")
    p.add_argument("--strict_checkpoint", action="store_true",
                   help="Require matching model tensors and check saved cue settings when available.")
    p.add_argument("--feature_visual_name", default="", help="Image stem to visualize; empty selects the first test image.")
    p.add_argument("--feature_visual_names", default="", help="Comma-separated image stems for Q/C review panels.")
    p.add_argument(
        "--contact_counterfactual_names", default="",
        help="Comma-separated stems to re-evaluate with C disabled using the same checkpoint.",
    )
    p.add_argument("--no_train_log", action="store_true")
    p.add_argument("--postprocess", action="store_true")
    p.add_argument("--no_fill_holes", action="store_true")
    p.add_argument("--boundary_threshold", type=float, default=0.5)
    p.add_argument("--separation_threshold", type=float, default=0.5)
    p.add_argument("--separation_radius", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=96)
    p.add_argument("--depths", default="2,2,9,2")
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=3)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--drop_path_rate", type=float, default=0.2)
    p.add_argument("--decoder_channels", type=int, default=128)
    p.add_argument("--ppm_channels", type=int, default=128)
    p.add_argument("--ppm_pool_scales", default="1,2,3,6")
    p.add_argument("--decoder_variant", default="gsfpn", choices=["gsfpn"])
    p.add_argument("--no_separation_head", action="store_true")
    add_bool_argument(p, "--use_gland_ssm_block", default=True)
    p.add_argument("--gland_ssm_block_stages", default="0,1,2,3")
    p.add_argument("--gland_ssm_scales", default="3,5,7")
    p.add_argument("--sgv_compact_ratio", type=float, default=1.0)
    add_bool_argument(p, "--sgv_scan_guidance", default=False)
    p.add_argument("--sgv_adapter_placement", default="all", choices=["all", "last"])
    add_bool_argument(p, "--use_bssf_lite", default=False)
    p.add_argument("--repair_variant", default="unified_hrmdr", choices=["unified_hrmdr", "none"])
    p.add_argument("--residual_scale_init", type=float, default=5e-3)
    p.add_argument("--capc_residual_scale_init", type=float, default=1e-2)
    p.add_argument("--capc_prior_scale_init", type=float, default=0.10)
    add_bool_argument(p, "--use_capc", default=True)
    p.add_argument("--hr_min_gate", type=float, default=0.05)
    p.add_argument("--hr_what_gradient_scale", type=float, default=0.50)
    p.add_argument("--hr_morphology_dilations", default="1,3,5")
    p.add_argument("--hr_image_feature_channels", type=int, default=32)
    p.add_argument("--hr_cue_feature_channels", type=int, default=16)
    p.add_argument("--capc_cue_channels", type=int, default=16)
    p.add_argument("--capc_contact_floor", type=float, default=0.25)
    add_bool_argument(p, "--capc_use_discrepancy", default=True)
    add_bool_argument(p, "--capc_use_contact_cue", default=True)
    add_bool_argument(p, "--capc_use_sgv_prior", default=True)
    add_bool_argument(p, "--hr_use_cue_guidance", default=True)
    p.add_argument("--qc_evidence_mode", default="legacy", choices=["legacy", "continuous", "relative", "calibrated"])
    p.add_argument("--contact_evidence_power", type=float, default=0.5)
    p.add_argument("--contact_contrast_gain", type=float, default=2.0)
    return p


def main():
    args = build_arg_parser().parse_args()
    if not args.no_train_log:
        install_test_log(Path(__file__).resolve().parent / "train_log.txt")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "This model uses a CUDA-only selective-scan kernel; check GPU visibility and CUDA initialization."
        )
    args.device = torch.device(args.device)

    img_dir, fg_dir, inside_dir, bd_dir, obj_dir, fg_source, inside_source, bd_source, obj_source = resolve_dirs(
        args.data_path,
        args.split,
        args.foreground_dir_name,
        args.inside_dir_name,
        args.boundary_dir_name,
        args.object_dir_name,
    )
    model, model_cfg = build_model(args)
    print(f"=> images: {img_dir}")
    print(f"=> foreground labels: {fg_dir} ({fg_source})")
    print(f"=> boundary labels: {bd_dir} ({bd_source})")
    print(f"=> object labels: {obj_dir} ({obj_source})")
    print("=> project: SGMNet")
    print(f"=> architecture: SGV + CAPC-GSFPN + repair={model_cfg.get('repair_variant', args.repair_variant)}")
    print(
        f"=> inference: mode={args.mode}, tile={args.tile_size}, stride={args.stride}, "
        f"fusion={args.fusion}, resize_input_size={args.input_size}"
    )
    print(
        f"=> postprocess: fg>={args.threshold}, blocker_mode={args.dsmnet_blocker_mode}, "
        f"blocker>={args.dsmnet_blocker_threshold}, keep_fg>={args.dsmnet_keep_threshold}, "
        f"opening_radius={args.opening_radius}, min_area={args.min_area}, watershed={args.dsmnet_watershed}"
    )

    model = model.to(args.device)
    load_checkpoint(model, args.ckpt, strict=args.strict_checkpoint)
    model.eval()

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = list_images(img_dir)
    print(f"=> images: {len(imgs)} | split={args.split}")

    pix_conf = new_confusion()
    object_rows = []
    per_image_rows = []
    counterfactual_rows = []
    feature_visual_saved = False
    feature_visual_target = str(args.feature_visual_name or "").strip()
    feature_visual_names = {part.strip() for part in args.feature_visual_names.split(",") if part.strip()}
    counterfactual_names = {
        part.strip() for part in args.contact_counterfactual_names.split(",") if part.strip()
    }
    start = time.time()
    for idx, img_path in enumerate(imgs, 1):
        t0 = time.time()
        stem = img_path.stem
        img = Image.open(img_path).convert("RGB")
        gt_inst = read_instance_label(find_mask(obj_dir, stem, role="object_instance", image_path=img_path))
        gt_bin = gt_inst > 0

        probability_maps = predict(model, img, args)
        fg_prob, bd_prob, sep_prob, bd_distance_prob = probability_maps[:4]
        instance_probability = probability_maps[4] if len(probability_maps) == 5 else None
        feature_maps = None
        if args.save_feature_visuals_all or stem in feature_visual_names or stem in counterfactual_names or (args.save_feature_visuals and not feature_visual_saved and (
            (feature_visual_target and stem == feature_visual_target)
            or (not feature_visual_target and idx == 1)
        )):
            feature_maps = predict_feature_maps(model, img, args)
            save_feature_visuals(feature_maps, out_dir / "feature_visuals", stem)
            if args.save_feature_visuals and not feature_visual_names:
                feature_visual_saved = True
        pred_bin, pred_inst = soft_blocker_postprocess(
            fg_prob,
            bd_prob,
            sep_prob,
            threshold=args.threshold,
            sure_threshold=args.dsmnet_sure_threshold,
            blocker_threshold=args.dsmnet_blocker_threshold,
            keep_threshold=args.dsmnet_keep_threshold,
            blocker_mode=args.dsmnet_blocker_mode,
            opening_radius=args.opening_radius,
            min_area=args.min_area,
            fill_holes=not args.no_fill_holes,
            watershed=args.dsmnet_watershed,
            watershed_min_distance=args.dsmnet_watershed_min_distance,
            instance_probability=instance_probability,
        )
        if stem in counterfactual_names:
            with contact_disabled(model):
                without_maps = predict(model, img, args)
                fg_without, bd_without, sep_without, _ = without_maps[:4]
                instance_without_probability = without_maps[4] if len(without_maps) == 5 else None
            _, inst_without = soft_blocker_postprocess(
                fg_without,
                bd_without,
                sep_without,
                threshold=args.threshold,
                sure_threshold=args.dsmnet_sure_threshold,
                blocker_threshold=args.dsmnet_blocker_threshold,
                keep_threshold=args.dsmnet_keep_threshold,
                blocker_mode=args.dsmnet_blocker_mode,
                opening_radius=args.opening_radius,
                min_area=args.min_area,
                fill_holes=not args.no_fill_holes,
                watershed=args.dsmnet_watershed,
                watershed_min_distance=args.dsmnet_watershed_min_distance,
                instance_probability=instance_without_probability,
            )
            metrics_with = glass_object_metrics(pred_inst, gt_inst)
            metrics_without = glass_object_metrics(inst_without, gt_inst)
            counterfactual_rows.append({
                "image": stem,
                "object_F1_with_C": metrics_with["F1"],
                "object_F1_without_C": metrics_without["F1"],
                "object_F1_C_effect": metrics_with["F1"] - metrics_without["F1"],
                "object_Dice_with_C": metrics_with["Dice"],
                "object_Dice_without_C": metrics_without["Dice"],
                "object_Dice_C_effect": metrics_with["Dice"] - metrics_without["Dice"],
                "num_objects_with_C": int(pred_inst.max()),
                "num_objects_without_C": int(inst_without.max()),
                "foreground_mean_abs_difference": float(np.abs(fg_prob - fg_without).mean()),
                "boundary_mean_abs_difference": float(np.abs(bd_prob - bd_without).mean()),
                "instance_binary_changed_pixels": int(np.count_nonzero((pred_inst > 0) != (inst_without > 0))),
                "instance_seam_changed_pixels": int(np.count_nonzero(
                    _instance_seams(pred_inst) ^ _instance_seams(inst_without)
                )),
            })
            counterfactual_dir = out_dir / "contact_counterfactual"
            counterfactual_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                counterfactual_dir / f"{stem}.npz",
                foreground_with_C=fg_prob,
                foreground_without_C=fg_without,
                boundary_with_C=bd_prob,
                boundary_without_C=bd_without,
                instance_with_C=pred_inst,
                instance_without_C=inst_without,
                gt_instance=gt_inst,
                image_rgb=np.asarray(img, dtype=np.uint8),
                **({"instance_probability_with_C": instance_probability,
                    "instance_probability_without_C": instance_without_probability}
                   if instance_probability is not None else {}),
            )
            if feature_maps is not None:
                save_contact_counterfactual_panel(
                    np.asarray(img, dtype=np.uint8), gt_inst, pred_inst, inst_without,
                    feature_maps, metrics_with, metrics_without,
                    counterfactual_dir, stem,
                    evidence_mode=model_cfg.get("qc_evidence_mode", "legacy"),
                )
        if feature_maps is not None:
            save_qc_review_panel(
                np.asarray(img, dtype=np.uint8), gt_inst, pred_inst,
                feature_maps, out_dir / "qc_review", stem,
                evidence_mode=model_cfg.get("qc_evidence_mode", "legacy"),
            )
        c = pixel_confusion(pred_bin, gt_bin)
        add_confusion(pix_conf, c)
        pm = pixel_metrics_from_confusion(c)
        om = glass_object_metrics(pred_inst, gt_inst)
        object_rows.append(om)
        row = {
            "image": stem,
            **pm,
            "object_recall": om["recall"],
            "object_precision": om["precision"],
            "object_F1": om["F1"],
            "object_Dice": om["Dice"],
            "object_IoU": om["IoU"],
            "object_Hausdorff": om["Hausdorff"],
            "object_HausdorffQuality": om["HausdorffQuality"],
            "object_Score": om["Score"],
            "TP": om["TP"],
            "FP": om["FP"],
            "FN": om["FN"],
            "num_pred_objects": int(pred_inst.max()),
            "num_gt_objects": int(gt_inst.max()),
        }
        per_image_rows.append(row)
        if args.save_flag:
            save_visuals(
                gt_inst,
                pred_bin,
                pred_inst,
                fg_prob,
                bd_prob,
                sep_prob,
                bd_distance_prob,
                out_dir / "vis",
                stem,
                instance_probability=instance_probability,
            )
            if not args.no_paper_vis:
                save_paper_visuals(
                np.asarray(img, dtype=np.uint8),
                gt_inst,
                pred_inst,
                fg_prob,
                bd_prob,
                out_dir / "paper_vis",
                stem,
                )
        print(
            f"[{idx}/{len(imgs)}] {stem}: PixelF1={pm['pixel_F1']:.4f}, PixelIoU={pm['pixel_iou']:.4f} | "
            f"ObjF1={om['F1']:.4f}, ObjDice={om['Dice']:.4f}, H={om['Hausdorff']:.4f}, "
            f"Score={om['Score']:.4f} | {time.time()-t0:.1f}s"
        )

    pix_avg = pixel_metrics_from_confusion(pix_conf)
    obj_avg = mean_glass_object_metrics(object_rows)
    print("\n========== Pixel-level foreground support ==========")
    for k, v in pix_avg.items():
        print(f"{k}: {v:.4f}")
    print("\n========== Object-level metrics ==========")
    for k, v in obj_avg.items():
        print(f"{k}: {v:.4f}")

    protocol = "dsmnet_res1"
    write_metrics(out_dir / f"{args.split}_{protocol}_foreground_pixel_result.txt", "Pixel-level foreground support metrics", pix_avg)
    write_metrics(out_dir / f"{args.split}_{protocol}_object_result.txt", "Object-level metrics", obj_avg)
    with open(out_dir / f"{args.split}_{protocol}_per_image.csv", "w", newline="", encoding="utf-8") as f:
        if per_image_rows:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)
    if counterfactual_rows:
        with open(out_dir / "contact_counterfactual.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(counterfactual_rows[0].keys()))
            writer.writeheader()
            writer.writerows(counterfactual_rows)
    flat_metrics = {
        **{f"foreground_{k}": v for k, v in pix_avg.items()},
        **{f"object_{k}": v for k, v in obj_avg.items()},
    }
    with open(out_dir / f"{args.split}_{protocol}_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "protocol", *flat_metrics.keys()])
        writer.writeheader()
        writer.writerow({"split": args.split, "protocol": protocol, **flat_metrics})
    with open(out_dir / f"{args.split}_{protocol}_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "project": "SGMNet",
                "architecture": "SGV + CAPC-GSFPN + Unified HR-MDR",
                "eval_protocol": protocol,
                "object_target": str(obj_dir),
                "prediction_postprocess": {
                    "inference_mode": args.mode,
                    "tile_size": int(args.tile_size),
                    "stride": int(args.stride),
                    "fusion": args.fusion,
                    "center_weight_eps": args.center_weight_eps,
                    "joint_probability_fusion": "per_tile" if args.dsmnet_blocker_mode == "contract" else None,
                    "threshold": args.threshold,
                    "sure_threshold": args.dsmnet_sure_threshold,
                    "blocker_threshold": args.dsmnet_blocker_threshold,
                    "keep_threshold": args.dsmnet_keep_threshold,
                    "blocker_mode": args.dsmnet_blocker_mode,
                    "opening_radius": args.opening_radius,
                    "min_area": args.min_area,
                    "fill_holes": not args.no_fill_holes,
                    "watershed": args.dsmnet_watershed,
                },
                "pixel_foreground_support": pix_avg,
                "object": obj_avg,
                "flat_metrics": flat_metrics,
                "model_config": model_cfg,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n=> results saved to: {out_dir}")
    if args.save_feature_visuals and not args.save_feature_visuals_all and not feature_visual_saved and not feature_visual_names:
        print(f"[warning] requested feature image was not found: {feature_visual_target}")
    print(f"=> total time: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
