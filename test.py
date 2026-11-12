"""Sliding-window SCMNet evaluation and instance-mask export."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.nn import functional as F

from models import SCMNet
from postprocess import OFFSETS, decode_instances
from utils import (
    Tee,
    _instance_error_counts,
    _pixel_metrics,
    _write_csv,
    center_weight_np,
    find_mask,
    glass_object_metrics,
    image_to_tensor,
    list_images,
    mean_glass_object_metrics,
    read_instances,
    reflect_pad_image_np,
    sliding_positions,
)


def load_model(path, device):
    """Strictly load current checkpoints or checkpoints from the original run."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") not in {
        SCMNet.model_version,
        "scmnet_relation_v2",
    }:
        raise ValueError("Unsupported checkpoint: expected SCMNet model metadata")
    config = dict(checkpoint["model_config"])
    config.pop("model_version", None)
    model = SCMNet(**config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint


@torch.no_grad()
def predict_image(
    model,
    image,
    device,
    tile_size=464,
    stride=232,
    fusion="center_weighted",
    return_features=False,
):
    if tile_size < 32 or not 1 <= stride <= tile_size:
        raise ValueError("Require tile_size >= 32 and 1 <= stride <= tile_size")
    if fusion not in {"uniform", "center_weighted"}:
        raise ValueError("fusion must be uniform or center_weighted")
    padded, original_height, original_width = reflect_pad_image_np(image, tile_size)
    height, width = padded.shape[:2]
    weight = (
        np.ones((tile_size, tile_size), dtype=np.float32)
        if fusion == "uniform"
        else center_weight_np(tile_size)
    )
    embedding_dim = int(model.model_config.get("embedding_dim", 8))
    base_channels = 10 + embedding_dim
    accumulator = np.zeros(
        (base_channels + 7 if return_features else base_channels, height, width),
        dtype=np.float32,
    )
    denominator = np.zeros((height, width), dtype=np.float32)
    for y in sliding_positions(height, tile_size, stride):
        for x in sliding_positions(width, tile_size, stride):
            tile = image_to_tensor(padded[y : y + tile_size, x : x + tile_size])
            region, aux = model(tile.unsqueeze(0).to(device), return_aux=True)
            output = torch.cat(
                (
                    region.sigmoid(),
                    aux["relation_logits"].sigmoid(),
                    aux["contact_logits"].sigmoid(),
                    aux["instance_embedding"],
                ),
                dim=1,
            )
            if return_features:
                extra = []
                for stage in aux["encoder_relation_logits"]:
                    extra.append(
                        F.interpolate(
                            stage.sigmoid(),
                            size=(tile_size, tile_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                for alpha in aux["fusion_alpha"]:
                    extra.append(
                        F.interpolate(
                            alpha,
                            size=(tile_size, tile_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                output = torch.cat((output, *extra), dim=1)
            values = output[0].float().cpu().numpy()
            accumulator[:, y : y + tile_size, x : x + tile_size] += values * weight
            denominator[y : y + tile_size, x : x + tile_size] += weight
    values = accumulator[:, :original_height, :original_width] / np.maximum(
        denominator[None, :original_height, :original_width], 1e-8
    )
    if not np.isfinite(values).all():
        raise RuntimeError("Model produced non-finite maps")
    embedding = values[10:base_channels]
    embedding = embedding / np.maximum(
        np.linalg.norm(embedding, axis=0, keepdims=True), 1e-6
    )
    features = {}
    if return_features:
        for index, name in enumerate(
            (
                "stage0_left",
                "stage0_top",
                "stage1_left",
                "stage1_top",
                "fusion_alpha0",
                "fusion_alpha1",
                "fusion_alpha2",
            ),
            start=base_channels,
        ):
            features[name] = values[index]
    return values[0], values[1:9], values[9], embedding, features


def decode_boundary_instances(
    region,
    relation,
    foreground_threshold,
    boundary_threshold,
    keep_threshold,
    opening_radius,
    min_area,
):
    """Block low-affinity foreground edges, then clean and label the mask."""
    foreground = region >= foreground_threshold
    boundary_score = np.zeros_like(region)
    height, width = region.shape
    # Only one-pixel affinities localize a boundary to the edge endpoints.
    for channel, (dy, dx) in enumerate(OFFSETS[:4]):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        source = np.s_[y0:y1, x0:x1]
        target = np.s_[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        score = np.where(
            foreground[source] & foreground[target], 1 - relation[channel][source], 0
        )
        np.maximum(boundary_score[source], score, out=boundary_score[source])
        np.maximum(boundary_score[target], score, out=boundary_score[target])

    binary = foreground & (
        (boundary_score < boundary_threshold) | (region >= keep_threshold)
    )
    if opening_radius:
        coords = np.arange(-opening_radius, opening_radius + 1)
        disk = coords[:, None] ** 2 + coords[None, :] ** 2 <= opening_radius**2
        eroded = ndi.binary_erosion(binary, structure=disk, border_value=1)
        binary = ndi.binary_dilation(eroded, structure=disk, border_value=0)
    binary = ndi.binary_fill_holes(binary)
    if min_area > 1:
        components, _ = ndi.label(binary)
        sizes = np.bincount(components.ravel())
        binary &= sizes[components] >= min_area
    labels, _ = ndi.label(binary, structure=np.ones((3, 3), dtype=bool))
    return binary, labels, boundary_score


def save_predictions(directory, stem, image, gt, labels, region, boundary_score=None):
    """Save integer instance IDs, foreground probability and a visual triptych."""
    for name in ("instances", "probability", "preview"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels.astype(np.int32)).save(
        directory / "instances" / f"{stem}.tiff"
    )
    Image.fromarray(region.astype(np.float32)).save(
        directory / "probability" / f"{stem}.tiff"
    )
    if boundary_score is not None:
        scores = directory / "boundary_score"
        scores.mkdir(exist_ok=True)
        Image.fromarray(boundary_score.astype(np.float32)).save(scores / f"{stem}.tiff")
    panels = [image]
    for mask in (gt, labels):
        binary = np.repeat(((mask > 0) * 255).astype(np.uint8)[..., None], 3, axis=2)
        # Retain visible separation between touching instance IDs.
        border = np.zeros(mask.shape, dtype=bool)
        border[:, 1:] |= (
            (mask[:, 1:] != mask[:, :-1]) & (mask[:, 1:] > 0) & (mask[:, :-1] > 0)
        )
        border[1:] |= (mask[1:] != mask[:-1]) & (mask[1:] > 0) & (mask[:-1] > 0)
        binary[border] = 0
        panels.append(binary)
    Image.fromarray(np.concatenate(panels, axis=1)).save(
        directory / "preview" / f"{stem}.png"
    )


def evaluate_split(
    model,
    data_root,
    split,
    device,
    output_dir=None,
    save_visuals=False,
    tile_size=464,
    stride=232,
    fusion="center_weighted",
    candidate_threshold=0.35,
    seed_threshold=0.65,
    edge_threshold=0.5,
    contact_threshold=0.65,
    similarity_threshold=0.45,
    min_area=100,
    limit=0,
    postprocess="graph",
    foreground_threshold=0.5,
    boundary_threshold=0.60,
    keep_threshold=0.75,
    opening_radius=5,
):
    """Report image-averaged object metrics for the selected instance decoder."""
    if postprocess not in {"graph", "boundary"}:
        raise ValueError("postprocess must be graph or boundary")
    if postprocess == "boundary" and (
        not 0 <= foreground_threshold <= keep_threshold <= 1
        or not 0 < boundary_threshold <= 1
        or opening_radius < 0
        or min_area < 0
    ):
        raise ValueError(
            "Invalid boundary-mode thresholds, opening radius or minimum area"
        )
    device = torch.device(device)
    data_root = Path(data_root)
    image_dir = data_root / "images" / split
    label_dir = data_root / "labels_instance_cc" / split
    images = list_images(image_dir)
    if not images or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Missing images or instance labels: {data_root} / {split}"
        )
    if limit:
        images = images[:limit]
    model.eval()
    rows, object_metrics = [], []
    start = time.perf_counter()
    for number, image_path in enumerate(images, 1):
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gt = read_instances(
            find_mask(label_dir, image_path.stem, "instance", image_path)
        )
        if gt.shape != image.shape[:2]:
            raise ValueError(f"Image and label dimensions differ: {image_path}")
        region, relation, contact, embedding, _ = predict_image(
            model, image, device, tile_size, stride, fusion
        )
        boundary_score = None
        if postprocess == "boundary":
            binary, labels, boundary_score = decode_boundary_instances(
                region,
                relation,
                foreground_threshold,
                boundary_threshold,
                keep_threshold,
                opening_radius,
                min_area,
            )
        else:
            binary, labels = decode_instances(
                region,
                relation,
                contact,
                embedding,
                candidate_threshold,
                seed_threshold,
                edge_threshold,
                contact_threshold,
                similarity_threshold,
                min_area,
            )
        obj = glass_object_metrics(labels, gt)
        errors = _instance_error_counts(labels, gt)
        row = {
            "image": image_path.stem,
            **{
                f"object_{key}": obj[key]
                for key in ("F1", "Dice", "IoU", "Hausdorff", "TP", "FP", "FN")
            },
            **_pixel_metrics(binary, gt > 0),
            **errors,
            "merge_case": int(errors["merged_gt"] > 0),
            "split_case": int(errors["split_gt"] > 0),
        }
        rows.append(row)
        object_metrics.append(obj)
        if output_dir is not None and save_visuals:
            save_predictions(
                Path(output_dir),
                image_path.stem,
                image,
                gt,
                labels,
                region,
                boundary_score,
            )
        print(
            f"[{split} {number}/{len(images)}] {image_path.stem} "
            f"F1={obj['F1']:.4f} Dice={obj['Dice']:.4f}",
            flush=True,
        )
    mean_obj = mean_glass_object_metrics(object_metrics)
    summary = {
        "split": split,
        "count": len(rows),
        **{
            f"object_{key}": mean_obj[key] for key in ("F1", "Dice", "IoU", "Hausdorff")
        },
        **{
            key: float(np.mean([r[key] for r in rows]))
            for key in (
                "pixel_F1",
                "pixel_IoU",
                "pixel_precision",
                "pixel_recall",
                "merge_case",
                "split_case",
            )
        },
        "elapsed_s": round(time.perf_counter() - start, 2),
    }
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _write_csv(directory / "per_image.csv", rows)
        _write_csv(directory / "metrics_summary.csv", [summary])
        (directory / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf8"
        )
    print("[SUMMARY] " + json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--save_predictions", "--save_flag", action="store_true")
    parser.add_argument("--tile_size", type=int, default=464)
    parser.add_argument("--stride", type=int, default=232)
    parser.add_argument(
        "--fusion", choices=("uniform", "center_weighted"), default="center_weighted"
    )
    parser.add_argument(
        "--postprocess",
        choices=("graph", "boundary"),
        default="graph",
        help="Instance decoder (default: graph). Boundary mode uses one-pixel affinities.",
    )
    parser.add_argument("--foreground_threshold", type=float, default=0.5)
    parser.add_argument("--boundary_threshold", type=float, default=0.60)
    parser.add_argument("--keep_threshold", type=float, default=0.75)
    parser.add_argument("--opening_radius", type=int, default=5)
    parser.add_argument("--candidate_threshold", type=float, default=0.35)
    parser.add_argument("--seed_threshold", type=float, default=0.65)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--contact_threshold", type=float, default=0.65)
    parser.add_argument("--similarity_threshold", type=float, default=0.45)
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; use --device cpu only for small-input checks"
        )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.ckpt, device)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        model_config=model.model_config, checkpoint_epoch=checkpoint.get("epoch")
    )
    if args.postprocess == "boundary":
        config["boundary_source"] = (
            "maximum one-minus-affinity over adjacent foreground pairs"
        )
    (args.save_dir / "test_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf8"
    )
    stdout, stderr = sys.stdout, sys.stderr
    with (args.save_dir / "test.log").open("w", encoding="utf8") as log:
        try:
            sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
            evaluate_split(
                model,
                args.data_path,
                args.split,
                device,
                output_dir=args.save_dir,
                save_visuals=args.save_predictions,
                tile_size=args.tile_size,
                stride=args.stride,
                fusion=args.fusion,
                candidate_threshold=args.candidate_threshold,
                seed_threshold=args.seed_threshold,
                edge_threshold=args.edge_threshold,
                contact_threshold=args.contact_threshold,
                similarity_threshold=args.similarity_threshold,
                min_area=args.min_area,
                limit=args.limit,
                postprocess=args.postprocess,
                foreground_threshold=args.foreground_threshold,
                boundary_threshold=args.boundary_threshold,
                keep_threshold=args.keep_threshold,
                opening_radius=args.opening_radius,
            )
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
