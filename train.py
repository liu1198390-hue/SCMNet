# coding=utf-8
"""Train SGMNet with foreground/object validation protocol."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from configs.model import build_model_config
from datasets.gland import GlandForegroundBoundaryAuxDataset
from engine import train_one_epoch, validate_one_epoch
from losses.segmentation import ForegroundBoundaryAuxLoss
from models.scmnet import DSMNetRes, is_mamba_available
from utils import make_generator, parse_tuple_int, read_split_file, seed_worker, set_seed, write_json


BEST_CKPT_METRICS = ("obj_F1", "obj_Dice", "obj_Hausdorff", "obj_score")
ALLOWED_CHECKPOINT_NAMES = {
    "best_obj_F1.pth",
    "best_obj_Dice.pth",
    "best_obj_Hausdorff.pth",
    "best_obj_score.pth",
    "latest.pth",
}
OBJECT_BEST_METRICS = {"obj_F1", "obj_Dice", "obj_Hausdorff", "obj_score"}
MAXIMIZE_METRICS = {"fg_dice", "obj_F1", "obj_Dice", "obj_IoU", "obj_score"}
MINIMIZE_METRICS = {"loss", "obj_Hausdorff"}


def add_bool_argument(parser: argparse.ArgumentParser, option: str, default: bool) -> None:
    """Add --flag/--no-flag while remaining compatible with Python 3.7+."""
    destination = option.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(option, dest=destination, action="store_true")
    group.add_argument(f"--no-{option[2:]}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: bool(default)})


def resolve_vmamba_pretrained_path(path_text: str) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "VM-UNet-main" / "pre_trained_weights" / "vmamba_small_e238_ema.pth",
        repo_root / "Swin-UMamba-main" / "data" / "pretrained" / "vmamba" / "vmamba_tiny_e292.pth",
    ]
    if text.lower() in {"auto", "default"}:
        for path in candidates:
            if path.exists():
                return str(path)
        searched = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(f"VMamba pretrained path auto-search failed. Searched:\n{searched}")

    path = Path(text).expanduser()
    if not path.exists():
        searched = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            f"VMamba pretrained path not found: {path}\n"
            f"Use --vmamba_pretrained_path auto or one of the known history paths:\n{searched}"
        )
    return str(path)


def make_run_dir(args):
    run_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"sgmnet_{args.exp_name}_seed{args.seed}"
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def normalize_repair_variant(repair_variant: str) -> str:
    variant = str(repair_variant or "unified_hrmdr").lower().replace("_", "").replace("-", "")
    if variant in {"unified", "hrmdr"}:
        variant = "unifiedhrmdr"
    if variant not in {"unifiedhrmdr", "none"}:
        raise ValueError(f"Unsupported repair_variant: {repair_variant}")
    return variant


def _adamw_parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "residual_scale" in name
            or bool(getattr(parameter, "_no_weight_decay", False))
        )
        (no_decay_parameters if no_decay else decay_parameters).append(parameter)
    return [
        {"params": decay_parameters, "weight_decay": float(weight_decay)},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]


def cleanup_checkpoint_dir(ckpt_dir: Path) -> None:
    for path in ckpt_dir.glob("*.pth"):
        if path.name not in ALLOWED_CHECKPOINT_NAMES:
            path.unlink()


def _metric_value(logs: Dict[str, float], metric_name: str) -> float:
    if metric_name == "loss":
        return float(logs["loss"])
    if metric_name == "fg_dice":
        return float(logs["fg_dice"])
    if metric_name.startswith("obj_"):
        if metric_name not in logs:
            raise KeyError(f"Validation log does not contain {metric_name}.")
        return float(logs[metric_name])
    raise KeyError(f"Unknown metric: {metric_name}")


def _is_better(metric_name: str, value: float, best_value: float) -> bool:
    return value < best_value if metric_name in MINIMIZE_METRICS else value > best_value


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    args,
    model_cfg: Dict,
    selected_metric_name: str,
    selected_metric_value: float,
    best_trackers: Dict[str, float],
    best_epochs: Dict[str, int],
    global_step: int = 0,
    scaler=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "args": vars(args),
            "selected_metric_name": selected_metric_name,
            "selected_metric_value": float(selected_metric_value),
            "best_metric_name": selected_metric_name,
            "best_metric": float(selected_metric_value),
            "best_trackers": dict(best_trackers),
            "best_epochs": dict(best_epochs),
            "global_step": int(global_step),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "model_config": model_cfg,
        "project": "SGMNet",
        "architecture": "SGV + CAPC-GSFPN + Unified HR-MDR",
        "target_protocol": "foreground = inside OR boundary; object metrics = labels_instance_cc",
        },
        path,
    )


def _format_float(value: float) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def write_best_summary(run_dir: Path, best_trackers: Dict[str, float], best_epochs: Dict[str, int], selected_metric: str, selected_best: float) -> None:
    summary = {
        "selected_metric": selected_metric,
        "selected_best": float(selected_best),
        "selected_best_epoch": int(best_epochs.get(selected_metric, 0)),
        "best": {
            name: {
                "value": float(value),
                "epoch": int(best_epochs.get(name, 0)),
                "checkpoint": f"checkpoints/best_{name}.pth",
            }
            for name, value in best_trackers.items()
        },
    }
    write_json(run_dir / "best_summary.json", summary)
    lines = [
        f"selected_metric: {selected_metric}",
        f"selected_best: {_format_float(selected_best)}",
        f"selected_best_epoch: {best_epochs.get(selected_metric, 0)}",
        "",
        "metric              value       epoch  checkpoint",
        "------------------  ----------  -----  -------------------------------",
    ]
    for name in BEST_CKPT_METRICS:
        lines.append(
            f"{name:<18}  {_format_float(best_trackers.get(name, float('nan'))):>10}  "
            f"{best_epochs.get(name, 0):>5}  checkpoints/best_{name}.pth"
        )
    (run_dir / "best_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train SCMNet with SGV, CAPC and MDR.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--train_data_path", default="")
    parser.add_argument("--val_data_path", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--foreground_dir_name", default="labels_foreground")
    parser.add_argument("--inside_dir_name", default="labels_inside")
    parser.add_argument("--boundary_dir_name", default="labels_boundary")
    parser.add_argument("--object_dir_name", default="labels_instance_cc")
    parser.add_argument("--input_size", type=int, default=464)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--vflip_prob", type=float, default=0.5)
    parser.add_argument("--rotate90_prob", type=float, default=0.5)
    parser.add_argument("--color_jitter_prob", type=float, default=0.6)
    parser.add_argument("--color_jitter_brightness", type=float, default=0.20)
    parser.add_argument("--color_jitter_contrast", type=float, default=0.20)
    parser.add_argument("--color_jitter_saturation", type=float, default=0.15)
    parser.add_argument("--color_jitter_hue", type=float, default=0.03)
    parser.add_argument("--random_scale_crop_prob", type=float, default=0.6)
    parser.add_argument("--random_scale_min", type=float, default=0.75)
    parser.add_argument("--random_scale_max", type=float, default=1.25)
    parser.add_argument("--elastic_prob", type=float, default=0.35)
    parser.add_argument("--elastic_alpha", type=float, default=30.0)
    parser.add_argument("--elastic_sigma", type=float, default=5.0)
    parser.add_argument("--soft_boundary_radius1", type=int, default=1)
    parser.add_argument("--soft_boundary_radius2", type=int, default=2)
    parser.add_argument("--soft_boundary_weight1", type=float, default=0.6)
    parser.add_argument("--soft_boundary_weight2", type=float, default=0.3)
    parser.add_argument("--boundary_focus_radius", type=int, default=2)
    parser.add_argument("--separation_radius", type=int, default=4)
    parser.add_argument("--contact_target_mode", default="legacy", choices=["legacy", "instance_gap"])
    parser.add_argument("--contact_search_radius", type=int, default=5)
    parser.add_argument("--contact_empty_weight", type=float, default=1.0)
    parser.add_argument("--foreground_continuity_weight", type=float, default=0.0)
    parser.add_argument("--foreground_gap_weight", type=float, default=0.0)
    parser.add_argument("--instance_contract_weight", type=float, default=0.0)
    parser.add_argument("--continuity_max_radius", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--fail_on_nonfinite", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--val_log_interval", type=int, default=50)
    parser.add_argument("--progress_mode", default="simple", choices=["auto", "tqdm", "simple", "none"])
    parser.add_argument("--progress_mininterval", type=float, default=10.0)
    parser.add_argument("--tqdm_leave", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary_metric_threshold", type=float, default=0.5)
    parser.add_argument("--separation_metric_threshold", type=float, default=0.3)
    parser.add_argument("--non_deterministic", action="store_true")
    parser.add_argument("--val_full_mode", default="sliding", choices=["sliding", "resize"])
    parser.add_argument("--val_full_tile_size", type=int, default=464)
    parser.add_argument("--val_full_stride", type=int, default=232)
    parser.add_argument("--val_full_fusion", default="uniform", choices=["uniform", "center_weighted"])
    parser.add_argument("--val_full_center_weight_eps", type=float, default=0.05)

    parser.add_argument("--boundary_weight", type=float, default=0.15)
    parser.add_argument("--capc_boundary_weight", type=float, default=0.08)
    parser.add_argument("--capc_contact_weight", type=float, default=0.15)
    parser.add_argument("--foreground_pos_weight", type=float, default=None)
    parser.add_argument("--boundary_pos_weight", type=float, default=None)
    parser.add_argument("--contact_pos_weight", type=float, default=8.0)
    parser.add_argument("--no_soft_boundary", action="store_true")
    parser.add_argument("--boundary_diff_weight", type=float, default=0.2)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--distance_background_weight", type=float, default=0.1)

    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--depths", default="2,2,9,2")
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=3)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--attn_drop_rate", type=float, default=0.0)
    parser.add_argument("--drop_path_rate", type=float, default=0.2)
    parser.add_argument("--decoder_channels", type=int, default=128)
    parser.add_argument("--ppm_channels", type=int, default=128)
    parser.add_argument("--ppm_pool_scales", default="1,2,3,6")
    parser.add_argument("--decoder_variant", default="gsfpn", choices=["gsfpn"])
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--no_separation_head", action="store_true")
    add_bool_argument(parser, "--use_gland_ssm_block", default=True)
    parser.add_argument("--gland_ssm_block_stages", default="0,1,2,3")
    parser.add_argument("--gland_ssm_scales", default="3,5,7")
    parser.add_argument("--sgv_compact_ratio", type=float, default=1.0, help="SGV internal branch width ratio at the selected adapter positions; 1.0 keeps the original adapter.")
    add_bool_argument(parser, "--sgv_scan_guidance", default=False)
    parser.add_argument("--sgv_adapter_placement", default="all", choices=["all", "last"])
    add_bool_argument(parser, "--use_bssf_lite", default=False)
    parser.add_argument("--repair_variant", default="unified_hrmdr", choices=["unified_hrmdr", "none"])
    parser.add_argument("--residual_scale_init", type=float, default=5e-3)
    parser.add_argument("--capc_residual_scale_init", type=float, default=1e-2)
    parser.add_argument("--capc_prior_scale_init", type=float, default=0.10)
    add_bool_argument(parser, "--use_capc", default=True)
    parser.add_argument("--hr_min_gate", type=float, default=0.05)
    parser.add_argument("--hr_what_gradient_scale", type=float, default=0.50)
    parser.add_argument("--hr_morphology_dilations", default="1,3,5")
    parser.add_argument("--hr_image_feature_channels", type=int, default=32)
    parser.add_argument("--hr_cue_feature_channels", type=int, default=16)
    parser.add_argument("--capc_cue_channels", type=int, default=16)
    parser.add_argument("--capc_contact_floor", type=float, default=0.25)
    add_bool_argument(parser, "--capc_use_discrepancy", default=True)
    add_bool_argument(parser, "--capc_use_contact_cue", default=True)
    add_bool_argument(parser, "--capc_use_sgv_prior", default=True)
    add_bool_argument(parser, "--hr_use_cue_guidance", default=True)
    parser.add_argument("--qc_evidence_mode", default="legacy", choices=["legacy", "continuous", "relative", "calibrated"])
    parser.add_argument("--contact_evidence_power", type=float, default=0.5)
    parser.add_argument("--contact_contrast_gain", type=float, default=2.0)
    parser.add_argument("--vmamba_pretrained_path", default="auto")

    parser.add_argument("--compute_val_object_metrics", action="store_true", default=True)
    parser.add_argument("--no_compute_val_object_metrics", dest="compute_val_object_metrics", action="store_false")
    parser.add_argument("--best_metric", default="obj_score", choices=["fg_dice", "loss", "obj_F1", "obj_Dice", "obj_IoU", "obj_Hausdorff", "obj_score"])
    parser.add_argument("--val_object_opening_radius", type=int, default=5)
    parser.add_argument("--val_object_contour_threshold", type=float, default=0.5)
    parser.add_argument("--val_object_min_size", type=int, default=100)
    parser.add_argument("--val_dsmnet_blocker_mode", default="boundary", choices=["max", "mean", "boundary", "separation", "contract"])
    parser.add_argument("--val_dsmnet_blocker_threshold", type=float, default=0.60)
    parser.add_argument("--val_dsmnet_keep_threshold", type=float, default=0.75)
    parser.add_argument("--val_dsmnet_sure_threshold", type=float, default=0.65)
    parser.add_argument("--val_dsmnet_watershed", action="store_true")
    parser.add_argument("--val_dsmnet_watershed_min_distance", type=int, default=12)

    parser.add_argument("--exp_name", default="full")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_ckpt", default="")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_start", type=int, default=80)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    return parser


def build_model(args):
    args.repair_variant = normalize_repair_variant(args.repair_variant)
    cfg = build_model_config(
        input_channels=3,
        num_classes=1,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depths=parse_tuple_int(args.depths),
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        decoder_channels=args.decoder_channels,
        ppm_channels=args.ppm_channels,
        ppm_pool_scales=parse_tuple_int(args.ppm_pool_scales),
        decoder_variant=args.decoder_variant,
        use_checkpoint=not args.no_checkpoint,
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
    model = DSMNetRes(**cfg)
    pretrained_path = resolve_vmamba_pretrained_path(args.vmamba_pretrained_path)
    if pretrained_path:
        info = model.load_vmamba_pretrained(pretrained_path)
        print(f"=> loaded VMamba encoder pretrain: {pretrained_path} | {info}")
        cfg["vmamba_pretrained_path"] = str(pretrained_path)
        cfg["vmamba_pretrained_loaded"] = info
    return model, cfg


def append_history(path: Path, epoch: int, train_logs, val_logs, lr: float):
    row = {"epoch": epoch, "lr": lr}
    row.update({f"train_{k}": v for k, v in train_logs.items()})
    row.update({f"val_{k}": v for k, v in val_logs.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as source:
            fieldnames = next(csv.reader(source))
        unexpected = set(row) - set(fieldnames)
        if unexpected:
            raise ValueError(f"History schema changed; new columns: {sorted(unexpected)}")
    else:
        fieldnames = list(row)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = build_arg_parser().parse_args()
    if args.foreground_continuity_weight > 0 and args.continuity_max_radius < 1:
        raise ValueError("continuity_max_radius must be >= 1 when continuity loss is enabled")
    if args.foreground_gap_weight > 0 and args.contact_target_mode != "instance_gap":
        raise ValueError("foreground_gap_weight requires contact_target_mode=instance_gap")
    if args.best_metric in OBJECT_BEST_METRICS:
        args.compute_val_object_metrics = True
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    deterministic = not args.non_deterministic
    set_seed(args.seed, deterministic=deterministic)
    if not torch.cuda.is_available() or not is_mamba_available():
        raise RuntimeError("Training requires CUDA and a compatible selective-scan extension.")
    device = torch.device("cuda")

    run_dir = make_run_dir(args)
    ckpt_dir = run_dir / "checkpoints"
    cleanup_checkpoint_dir(ckpt_dir)

    train_data_path = Path(args.train_data_path) if args.train_data_path else Path(args.data_path)
    val_data_path = Path(args.val_data_path) if args.val_data_path else Path(args.data_path)

    train_set = GlandForegroundBoundaryAuxDataset(
        train_data_path,
        split=args.train_split,
        input_size=args.input_size,
        train=True,
        augment=not args.no_augment,
        foreground_dir_name=args.foreground_dir_name,
        inside_dir_name=args.inside_dir_name,
        boundary_dir_name=args.boundary_dir_name,
        object_dir_name=args.object_dir_name,
        strict_consistency=False,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        rotate90_prob=args.rotate90_prob,
        color_jitter_prob=args.color_jitter_prob,
        color_jitter_brightness=args.color_jitter_brightness,
        color_jitter_contrast=args.color_jitter_contrast,
        color_jitter_saturation=args.color_jitter_saturation,
        color_jitter_hue=args.color_jitter_hue,
        random_scale_crop_prob=args.random_scale_crop_prob,
        random_scale_min=args.random_scale_min,
        random_scale_max=args.random_scale_max,
        elastic_prob=args.elastic_prob,
        elastic_alpha=args.elastic_alpha,
        elastic_sigma=args.elastic_sigma,
        soft_boundary_radius1=args.soft_boundary_radius1,
        soft_boundary_radius2=args.soft_boundary_radius2,
        soft_boundary_weight1=args.soft_boundary_weight1,
        soft_boundary_weight2=args.soft_boundary_weight2,
        boundary_focus_radius=args.boundary_focus_radius,
        separation_radius=args.separation_radius,
        contact_target_mode=args.contact_target_mode,
        contact_search_radius=args.contact_search_radius,
        continuity_max_radius=args.continuity_max_radius if args.foreground_continuity_weight > 0 else 0,
    )
    val_set = GlandForegroundBoundaryAuxDataset(
        val_data_path,
        split=args.val_split,
        input_size=args.input_size,
        train=False,
        augment=False,
        foreground_dir_name=args.foreground_dir_name,
        inside_dir_name=args.inside_dir_name,
        boundary_dir_name=args.boundary_dir_name,
        object_dir_name=args.object_dir_name,
        strict_consistency=False,
        soft_boundary_radius1=args.soft_boundary_radius1,
        soft_boundary_radius2=args.soft_boundary_radius2,
        soft_boundary_weight1=args.soft_boundary_weight1,
        soft_boundary_weight2=args.soft_boundary_weight2,
        boundary_focus_radius=args.boundary_focus_radius,
        separation_radius=args.separation_radius,
        contact_target_mode=args.contact_target_mode,
        contact_search_radius=args.contact_search_radius,
        continuity_max_radius=args.continuity_max_radius if args.foreground_continuity_weight > 0 else 0,
    )

    loader_generator = make_generator(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / args.gradient_accumulation_steps)
    total_steps = max(1, optimizer_steps_per_epoch * int(args.epochs))

    model, model_cfg = build_model(args)
    model = model.to(device)
    criterion = ForegroundBoundaryAuxLoss(
        boundary_weight=args.boundary_weight,
        capc_boundary_weight=args.capc_boundary_weight,
        capc_contact_weight=args.capc_contact_weight,
        foreground_pos_weight=args.foreground_pos_weight,
        boundary_pos_weight=args.boundary_pos_weight,
        contact_pos_weight=args.contact_pos_weight,
        contact_empty_weight=args.contact_empty_weight,
        foreground_continuity_weight=args.foreground_continuity_weight,
        foreground_gap_weight=args.foreground_gap_weight,
        instance_contract_weight=args.instance_contract_weight,
        use_soft_boundary=not args.no_soft_boundary,
        boundary_diff_weight=args.boundary_diff_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        distance_background_weight=args.distance_background_weight,
    )
    optimizer_groups = _adamw_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    run_config = {
        "args": vars(args),
        "model_config": model_cfg,
        "runtime": {
            "device": str(device),
            "amp": bool(args.amp),
            "weight_decay_policy": "matrix/kernel weights only; bias, norm, residual scales and VMamba-marked parameters excluded",
            "deterministic": deterministic,
            "selective_scan_available": is_mamba_available(),
            "steps_per_epoch": int(steps_per_epoch),
            "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
            "effective_batch_size": int(args.batch_size * args.gradient_accumulation_steps),
            "total_steps": int(total_steps),
        },
        "data_path": str(val_data_path),
        "train_data_path": str(train_data_path),
        "val_data_path": str(val_data_path),
        "split_files": {
            "train": read_split_file(train_data_path / "splits" / f"{args.train_split}.txt"),
            "val": read_split_file(val_data_path / "splits" / f"{args.val_split}.txt"),
        },
        "project": "SGMNet",
        "architecture": "SGV + CAPC-GSFPN + HR-MDR",
        "method_modules": [
            "SGV structure-guided pretrained VSS adapter",
            "CAPC contact-aware correction inside GSFPN top-down fusion",
            f"{model_cfg.get('repair_head', args.repair_variant)} terminal morphology repair",
        ],
        "target_protocol": (
            "foreground and boundary plus object-instance-supervised probability = sigmoid(fg)*(1-sigmoid(boundary)); "
            "object metrics = labels_instance_cc"
            if args.instance_contract_weight > 0 else
            "binary foreground = inside OR boundary; object metrics = labels_instance_cc"
        ),
        "model_selection": {
            "checkpoints": ["obj_F1", "obj_Dice", "obj_Hausdorff", "obj_score", "latest"],
            "hausdorff_quality": "1 - min(object_hausdorff / image_diagonal, 1)",
            "score": "0.30 * object_f1 + 0.40 * object_dice + 0.30 * hausdorff_quality",
            "selection_split": "validation only",
        },
        "whole_image_validation": {
            "mode": args.val_full_mode,
            "tile_size": int(args.val_full_tile_size),
            "stride": int(args.val_full_stride),
            "fusion": args.val_full_fusion,
            "postprocess": {
                "threshold": float(args.threshold),
                "blocker_mode": args.val_dsmnet_blocker_mode,
                "blocker_threshold": float(args.val_dsmnet_blocker_threshold),
                "keep_threshold": float(args.val_dsmnet_keep_threshold),
                "sure_threshold": float(args.val_dsmnet_sure_threshold),
                "opening_radius": int(args.val_object_opening_radius),
                "min_size": int(args.val_object_min_size),
            },
        },
    }
    write_json(ckpt_dir / "run_config.json", run_config)

    print(f"=> run_dir: {run_dir}")
    print(f"=> train_data_path: {train_data_path}")
    print(f"=> val_data_path: {val_data_path}")
    print(f"=> train samples: {len(train_set)} | val images: {len(val_set)}")
    print(f"=> device: {device} | amp={args.amp} | deterministic={deterministic}")
    print(f"=> architecture: SGMNet | SGV | CAPC-GSFPN | repair={args.repair_variant}")
    print(
        "=> custom modules: "
        f"SGV={args.use_gland_ssm_block}@{args.gland_ssm_block_stages}/scales={args.gland_ssm_scales}, "
        f"CAPC={args.use_capc}/cue={args.capc_cue_channels}/floor={args.capc_contact_floor:g}/"
        f"residual={args.capc_residual_scale_init:g}/prior={args.capc_prior_scale_init:g}/"
        f"sgv_prior={args.capc_use_sgv_prior}, "
        f"GSFPN selection={args.use_bssf_lite}, repair={args.repair_variant}, "
        f"HR-MDR=min_gate={args.hr_min_gate:g}/what_grad={args.hr_what_gradient_scale:g}/"
        f"dilations={args.hr_morphology_dilations}/image={args.hr_image_feature_channels}/cue={args.hr_cue_feature_channels}, "
        f"residual_scale_init={args.residual_scale_init:g}"
    )
    print(f"=> selective_scan_available={is_mamba_available()}")
    print(
        f"=> validation: {args.val_full_mode}, tile={args.val_full_tile_size}, stride={args.val_full_stride}, "
        f"fusion={args.val_full_fusion}, threshold={args.threshold}, blocker={args.val_dsmnet_blocker_mode}@{args.val_dsmnet_blocker_threshold}"
    )
    print(f"=> selected best metric: {args.best_metric}")
    print(
        f"=> supervision: CAPC boundary={args.capc_boundary_weight:g}, "
        f"CAPC contact={args.capc_contact_weight:g}"
    )

    best_trackers = {
        name: float("inf") if name in MINIMIZE_METRICS else -float("inf")
        for name in BEST_CKPT_METRICS
    }
    best_epochs = {name: 0 for name in BEST_CKPT_METRICS}
    selected_best = float("inf") if args.best_metric in MINIMIZE_METRICS else -float("inf")
    bad_epochs = 0
    history_path = run_dir / "history.csv"
    start_epoch = 1
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume_ckpt) if args.resume_ckpt else ckpt_dir / "latest.pth"
        if not resume_path.exists():
            raise FileNotFoundError(f"Cannot resume: checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"Invalid resume checkpoint: {resume_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        saved_trackers = checkpoint.get("best_trackers")
        if isinstance(saved_trackers, dict):
            for name in BEST_CKPT_METRICS:
                if name in saved_trackers:
                    best_trackers[name] = float(saved_trackers[name])
        saved_epochs = checkpoint.get("best_epochs")
        if isinstance(saved_epochs, dict):
            for name in BEST_CKPT_METRICS:
                if name in saved_epochs:
                    best_epochs[name] = int(saved_epochs[name])
        selected_best = float(best_trackers.get(args.best_metric, selected_best))
        completed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        global_step = int(checkpoint.get("global_step", completed_epoch * steps_per_epoch))
        print(
            f"=> resumed from {resume_path} | completed_epoch={completed_epoch}, "
            f"next_epoch={start_epoch}, global_step={global_step}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_logs, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            args,
            scaler,
            scheduler=scheduler,
            global_step=global_step,
        )
        val_logs = validate_one_epoch(model, val_loader, criterion, device, epoch, args)
        if args.fail_on_nonfinite and not math.isfinite(float(val_logs.get("loss", float("nan")))):
            raise FloatingPointError(f"Non-finite validation loss at epoch={epoch}.")
        if args.compute_val_object_metrics:
            val_logs["obj_score"] = float(val_logs.get("obj_Score", 0.0))

        lr = optimizer.param_groups[0]["lr"]
        append_history(history_path, epoch, train_logs, val_logs, lr)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_logs['loss']:.4f}, train_fg_dice={train_logs.get('fg_dice', 0.0):.4f}, "
            f"val_loss={val_logs.get('loss', 0.0):.4f}, val_fg_dice={val_logs.get('fg_dice', 0.0):.4f}, "
            f"val_bd_dice={val_logs.get('bd_dice', 0.0):.4f}, val_contact_dice={val_logs.get('sep_dice', 0.0):.4f}, "
            f"val_obj_F1={val_logs.get('obj_F1', 0.0):.4f}, val_obj_Dice={val_logs.get('obj_Dice', 0.0):.4f}, "
            f"val_obj_H={val_logs.get('obj_Hausdorff', 0.0):.4f}, val_obj_score={val_logs.get('obj_score', 0.0):.4f}, "
            f"amp_overflow={int(train_logs.get('amp_overflow_updates', 0))}, "
            f"train_cuda_peak={train_logs.get('cuda_peak_allocated_gb', 0.0):.2f}GiB, "
            f"val_cuda_peak={val_logs.get('cuda_peak_allocated_gb', 0.0):.2f}GiB, lr={lr:.6g}"
        )

        metrics = {"train": train_logs, "val": val_logs}
        for metric_name in BEST_CKPT_METRICS:
            if metric_name not in val_logs:
                continue
            value = _metric_value(val_logs, metric_name)
            if _is_better(metric_name, value, best_trackers[metric_name]):
                best_trackers[metric_name] = value
                best_epochs[metric_name] = int(epoch)
                save_checkpoint(
                    ckpt_dir / f"best_{metric_name}.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    metrics,
                    args,
                    model_cfg,
                    metric_name,
                    value,
                    best_trackers,
                    best_epochs,
                    global_step,
                    scaler=scaler,
                )
                write_best_summary(run_dir, best_trackers, best_epochs, args.best_metric, selected_best)
                print(f"=> new best {metric_name}: {value:.6f} at epoch {epoch}")

        selected_value = _metric_value(val_logs, args.best_metric)
        if _is_better(args.best_metric, selected_value, selected_best):
            selected_best = selected_value
            bad_epochs = 0
            print(f"=> new best selected metric ({args.best_metric}): {selected_value:.6f}")
        else:
            bad_epochs += 1

        save_checkpoint(
            ckpt_dir / "latest.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            metrics,
            args,
            model_cfg,
            args.best_metric,
            selected_value,
            best_trackers,
            best_epochs,
            global_step,
            scaler=scaler,
        )
        write_best_summary(run_dir, best_trackers, best_epochs, args.best_metric, selected_best)
        if args.early_stop and epoch >= args.early_stop_start and bad_epochs >= args.early_stop_patience:
            print(f"=> early stop at epoch {epoch}, best selected {args.best_metric}={selected_best:.6f}")
            break

    print(
        "Done. "
        f"best_obj_F1={best_trackers['obj_F1']:.4f}@epoch{best_epochs['obj_F1']} | "
        f"best_obj_Dice={best_trackers['obj_Dice']:.4f}@epoch{best_epochs['obj_Dice']} | "
        f"best_obj_Hausdorff={best_trackers['obj_Hausdorff']:.4f}@epoch{best_epochs['obj_Hausdorff']} | "
        f"best_obj_score={best_trackers['obj_score']:.4f}@epoch{best_epochs['obj_score']} | "
        f"best_selected_{args.best_metric}={selected_best:.4f}"
    )
    cleanup_checkpoint_dir(ckpt_dir)
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
