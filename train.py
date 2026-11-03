"""Train SCMNet; select checkpoints with whole-image validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader, Subset

from datasets import GlandDataset
from losses import SCMNetLoss
from models import SCMNet
from models.scan import is_mamba_available
from test import evaluate_split
from utils import Tee, count_parameters, make_generator, seed_worker, set_seed


def _save(path, epoch, model, optimizer, scaler, best, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best": best,
        "model_config": model.model_config,
        "model_version": model.model_version,
        "seed": args.seed,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def _append_history(path, row):
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(args):
    if not torch.cuda.is_available() or not is_mamba_available():
        raise RuntimeError("SCMNet requires CUDA and the VMamba selective scan extension")
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda")
    print(
        f"[ENV] torch={torch.__version__} CUDA={torch.version.cuda} "
        f"GPU={torch.cuda.get_device_name(0)} numpy={np.__version__} Pillow={pillow_version}",
        flush=True,
    )
    dataset = GlandDataset(args.train_data_path, input_size=args.input_size, augment=True)
    if args.train_limit:
        dataset = Subset(dataset, list(range(min(args.train_limit, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    model = SCMNet(use_checkpoint=args.use_checkpoint).to(device)
    criterion = SCMNetLoss(
        args.relation_weight,
        args.stage_weight,
        args.hard_region_weight,
        args.contact_weight,
        args.pair_weight,
        args.gap_weight,
        args.contact_radius,
        args.continuity_weight,
    )
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim < 2
            or name.endswith("bias")
            or getattr(parameter, "_no_weight_decay", False)
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start_epoch = 1
    best = {"object_F1": -math.inf, "object_Dice": -math.inf, "object_Hausdorff": math.inf}
    if args.resume:
        checkpoint = torch.load(str(args.resume), map_location="cpu", weights_only=True)
        if checkpoint.get("model_version") not in {model.model_version, "scmnet_relation_v2"}:
            raise RuntimeError("Resume checkpoint model version mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        best.update(checkpoint.get("best", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[RESUME] epoch={start_epoch}", flush=True)
    else:
        if not args.pretrained:
            raise ValueError("--pretrained VMamba Tiny checkpoint is required")
        information = model.load_vmamba_pretrained(args.pretrained)
        if information["loaded_tensors"] < 100:
            raise RuntimeError(f"VMamba pretrained mapping loaded too few tensors: {information}")
        print("[PRETRAINED] " + json.dumps(information), flush=True)
    print(
        f"[MODEL] trainable_parameters={count_parameters(model):,} train_patches={len(dataset)}",
        flush=True,
    )
    config = vars(args).copy()
    config.update(
        {
            "model_config": model.model_config,
            "trainable_parameters": count_parameters(model),
            "supervision": "region, same-instance edges, contact gaps, identity pairs and within-instance foreground paths from augmented instance IDs",
            "selection": "full-image val object F1",
            "postprocess": "contact-barrier and identity-gated relation components; no opening or hole filling",
            "augmentation": {
                "horizontal_flip_p": 0.5,
                "vertical_flip_p": 0.5,
                "rotation_90_degrees": [0, 90, 180, 270],
                "random_scale_p": 0.6,
                "random_scale_range": [0.75, 1.25],
                "brightness_p": 0.6,
                "brightness_range": [0.8, 1.2],
                "contrast_range": [0.8, 1.2],
                "color_range": [0.85, 1.15],
            },
        }
    )
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in config.items()
    }
    config_path = args.output_dir / "run_config.json"
    if args.resume and config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        locked = (
            "epochs",
            "lr",
            "min_lr",
            "weight_decay",
            "grad_clip",
            "amp",
            "deterministic",
            "val_split",
            "train_data_path",
            "val_data_path",
            "seed",
            "batch_size",
            "input_size",
            "stride",
            "fusion",
            "relation_weight",
            "stage_weight",
            "hard_region_weight",
            "contact_weight",
            "pair_weight",
            "gap_weight",
            "continuity_weight",
            "contact_radius",
            "candidate_threshold",
            "seed_threshold",
            "edge_threshold",
            "contact_threshold",
            "similarity_threshold",
            "min_area",
        )
        changed = [key for key in locked if previous.get(key) != config.get(key)]
        if changed:
            raise RuntimeError(f"Resume configuration changed: {changed}; use a new --output_dir")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    history = args.output_dir / "history.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
            1 + math.cos(math.pi * (epoch - 1) / args.epochs)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        sums = {
            name: 0.0
            for name in (
                "loss",
                "region_bce",
                "region_dice",
                "hard_region",
                "relation",
                "stage_relation",
                "contact",
                "pair_positive",
                "pair_negative",
                "gap_region",
                "path_continuity",
                "contact_fraction",
            )
        }
        for step, batch in enumerate(loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            instances = batch["object_instance"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                region, aux = model(images, return_aux=True)
                loss, parts = criterion(region, aux, instances)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for name, value in parts.items():
                sums[name] += float(value)
            if step % args.log_interval == 0 or step == len(loader):
                print(
                    f"[TRAIN] epoch={epoch}/{args.epochs} step={step}/{len(loader)} "
                    f"loss={sums['loss'] / step:.4f} region={sums['region_bce'] / step:.4f} "
                    f"relation={sums['relation'] / step:.4f} "
                    f"contact={sums['contact'] / step:.4f} "
                    f"pair_neg={sums['pair_negative'] / step:.4f} "
                    f"gap={sums['gap_region'] / step:.4f} "
                    f"path={sums['path_continuity'] / step:.4f} lr={lr:.6g}",
                    flush=True,
                )
        train_seconds = round(time.time() - t0, 2)
        model.eval()
        with torch.no_grad():
            validation = evaluate_split(
                model,
                args.val_data_path,
                args.val_split,
                device,
                output_dir=None,
                save_visuals=False,
                tile_size=args.input_size,
                stride=args.stride,
                fusion=args.fusion,
                candidate_threshold=args.candidate_threshold,
                seed_threshold=args.seed_threshold,
                edge_threshold=args.edge_threshold,
                contact_threshold=args.contact_threshold,
                similarity_threshold=args.similarity_threshold,
                min_area=args.min_area,
                limit=args.val_limit,
            )
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_seconds": train_seconds,
            "epoch_seconds": round(time.time() - t0, 2),
            **{f"train_{key}": value / len(loader) for key, value in sums.items()},
            **{
                f"val_{key}": value
                for key, value in validation.items()
                if isinstance(value, (int, float))
            },
        }
        _append_history(history, row)
        print(
            f"[EPOCH] {epoch}/{args.epochs} val_F1={validation['object_F1']:.5f} "
            f"val_Dice={validation['object_Dice']:.5f} val_H={validation['object_Hausdorff']:.3f} "
            f"val_merge_case={validation['merge_case']:.4f} "
            f"val_split_case={validation['split_case']:.4f} "
            f"train_seconds={train_seconds} epoch_seconds={row['epoch_seconds']}",
            flush=True,
        )
        ckpt_dir = args.output_dir / "checkpoints"
        for metric, filename, higher in (
            ("object_F1", "best_obj_F1.pth", True),
            ("object_Dice", "best_obj_Dice.pth", True),
            ("object_Hausdorff", "best_obj_Hausdorff.pth", False),
        ):
            value = float(validation[metric])
            if (value > best[metric]) if higher else (value < best[metric]):
                best[metric] = value
                best[metric + "_epoch"] = epoch
                _save(ckpt_dir / filename, epoch, model, optimizer, scaler, best, args)
        _save(ckpt_dir / "latest.pth", epoch, model, optimizer, scaler, best, args)
        (args.output_dir / "best_summary.txt").write_text(
            "selected_metric: object_F1\nselected_best: {:.6f}\nselected_best_epoch: {}\n\n".format(
                best["object_F1"], best.get("object_F1_epoch", 0)
            )
            + "\n".join(f"{key}: {value}" for key, value in best.items())
            + "\n",
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data_path", type=Path, required=True)
    parser.add_argument("--val_data_path", type=Path, required=True)
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=464)
    parser.add_argument("--stride", type=int, default=232)
    parser.add_argument(
        "--fusion", choices=("uniform", "center_weighted"), default="center_weighted"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--relation_weight", type=float, default=0.5)
    parser.add_argument("--stage_weight", type=float, default=0.2)
    parser.add_argument("--hard_region_weight", type=float, default=0.2)
    parser.add_argument("--contact_weight", type=float, default=0.35)
    parser.add_argument("--pair_weight", type=float, default=0.25)
    parser.add_argument("--gap_weight", type=float, default=0.15)
    parser.add_argument("--continuity_weight", type=float, default=0.15)
    parser.add_argument("--contact_radius", type=int, default=6)
    parser.add_argument("--candidate_threshold", type=float, default=0.35)
    parser.add_argument("--seed_threshold", type=float, default=0.65)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--contact_threshold", type=float, default=0.65)
    parser.add_argument("--similarity_threshold", type=float, default=0.45)
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train.log").open("a", encoding="utf-8") as log:
        stdout, stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = Tee(stdout, log)
            sys.stderr = Tee(stderr, log)
            print(
                "[ARGS] " + json.dumps({key: str(value) for key, value in vars(args).items()}),
                flush=True,
            )
            train(args)
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
