# coding=utf-8
"""Generate coverage-first offline patches for gland segmentation.

This script is a safer alternative to pure random 500x500 patch sampling.
It always keeps a sliding-window grid that covers each training image, then
adds foreground-aware random patches only if a minimum total patch budget is
requested.

Expected input layout:
  data_root/images/{split}/*.png
  data_root/labels_instance_cc/{split}/*.png

Compatible output layout:
  out_root/images/{split}
  out_root/labels_instance/{split}
  out_root/labels_instance_cc/{split}
  out_root/labels_foreground/{split}
  out_root/labels_inside/{split}
  out_root/labels_boundary/{split}
  out_root/qc_overlay/{split}
  out_root/manifest_{split}.csv
  out_root/split_config_{split}.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def numeric_key(p: Path):
    m = re.search(r"(\d+)", p.stem)
    return (int(m.group(1)) if m else 10**12, p.name)


def list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    files: List[Path] = []
    for ext in IMG_EXTS:
        files.extend(d.glob(f"*{ext}"))
    return sorted([p for p in files if p.is_file()], key=numeric_key)


def strip_label_suffix(stem: str) -> str:
    suffixes = [
        "_anno_label",
        "_annotation_label",
        "_anno",
        "_annotation",
        "_mask",
        "_label",
        "_gt",
        "_seg",
        "_segmentation",
        "_instance",
        "_instances",
        "-anno-label",
        "-annotation-label",
        "-anno",
        "-annotation",
        "-mask",
        "-label",
        "-gt",
    ]
    out = stem
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


def find_label(img_path: Path, label_dir: Path) -> Optional[Path]:
    stem = img_path.stem
    suffixes = ["", "_anno", "_annotation", "_mask", "_label", "_gt", "_instance", "_instances"]
    for ext in IMG_EXTS:
        for suffix in suffixes:
            p = label_dir / f"{stem}{suffix}{ext}"
            if p.exists():
                return p
    if label_dir.exists():
        hits = [
            p
            for p in label_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS and strip_label_suffix(p.stem) == stem
        ]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise RuntimeError(f"Ambiguous labels for {img_path.name}: {[x.name for x in hits]}")
    return None


def resolve_dirs(data_root: Path, split: str, label_subdir: str):
    img_candidates = [
        data_root / "images" / split,
        data_root / split / "images",
        data_root / split / "Images",
        data_root / "Images" / split,
    ]
    label_candidates = [
        data_root / label_subdir / split,
        data_root / split / label_subdir,
        data_root / "labels_instance_cc" / split,
        data_root / "labels_instance" / split,
        data_root / "labels_instances" / split,
        data_root / "Annotation" / split,
        data_root / split / "Annotation",
    ]
    img_dir = next((p for p in img_candidates if p.exists()), None)
    label_dir = next((p for p in label_candidates if p.exists()), None)
    if img_dir is None:
        raise FileNotFoundError(f"Cannot find images for split={split}; tried {img_candidates}")
    if label_dir is None:
        raise FileNotFoundError(f"Cannot find labels for split={split}; tried {label_candidates}")
    return img_dir, label_dir


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def read_label(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.uint32)
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
            arr = rgb[..., 0]
        else:
            arr = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    if arr.ndim != 2:
        raise ValueError(f"Label must be 2D or RGB-like: {path}, shape={arr.shape}")
    return arr.astype(np.int32, copy=False)


def load_pairs(data_root: Path, split: str, label_subdir: str):
    img_dir, label_dir = resolve_dirs(data_root, split, label_subdir)
    pairs = []
    for img in list_images(img_dir):
        lab = find_label(img, label_dir)
        if lab is None:
            raise FileNotFoundError(f"Cannot find label for {img.name} in {label_dir}")
        pairs.append((img, lab))
    return pairs


def pad_to_min(img: np.ndarray, inst: np.ndarray, size: int):
    h, w = inst.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if ph == 0 and pw == 0:
        return img, inst
    pad_img = ((0, ph), (0, pw), (0, 0))
    pad_inst = ((0, ph), (0, pw))
    pad_mode = "reflect" if h > 1 and w > 1 else "edge"
    img_pad = np.pad(img, pad_img, mode=pad_mode)
    inst_pad = np.pad(inst, pad_inst, mode="constant", constant_values=0)
    return img_pad, inst_pad


def axis_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, max(1, length - patch_size + 1), stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return sorted(set(int(x) for x in positions))


def grid_coords(h: int, w: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    ys = axis_positions(h, patch_size, stride)
    xs = axis_positions(w, patch_size, stride)
    return [(y, x) for y in ys for x in xs]


def find_instance_boundaries(inst: np.ndarray) -> np.ndarray:
    try:
        from skimage.segmentation import find_boundaries

        return find_boundaries(inst, mode="outer")
    except Exception:
        boundary = np.zeros(inst.shape, dtype=bool)
        diff = inst[:, 1:] != inst[:, :-1]
        boundary[:, 1:] |= diff
        boundary[:, :-1] |= diff
        diff = inst[1:, :] != inst[:-1, :]
        boundary[1:, :] |= diff
        boundary[:-1, :] |= diff
        return boundary


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=False)
    try:
        from skimage.morphology import binary_dilation, disk

        return binary_dilation(mask, disk(radius))
    except Exception:
        try:
            from scipy import ndimage as ndi
        except Exception as exc:
            raise RuntimeError("Boundary dilation requires scikit-image or scipy.") from exc
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        structure = (yy * yy + xx * xx) <= radius * radius
        return ndi.binary_dilation(mask, structure=structure)


def instance_targets(inst: np.ndarray, boundary_radius: int = 1):
    fg = (inst > 0).astype(np.uint8)
    if fg.max() == 0:
        boundary = np.zeros_like(fg, dtype=np.uint8)
    else:
        boundary = find_instance_boundaries(inst)
        if boundary_radius > 0:
            boundary = dilate_bool(boundary, boundary_radius)
        boundary = boundary.astype(np.uint8)
    inside = (fg.astype(bool) & ~boundary.astype(bool)).astype(np.uint8)
    return fg, inside, boundary


def make_overlay(img: np.ndarray, fg: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    out = img.copy().astype(np.float32)
    green = np.zeros_like(out)
    green[..., 1] = 255
    red = np.zeros_like(out)
    red[..., 0] = 255
    out = np.where(fg[..., None].astype(bool), out * 0.55 + green * 0.45, out)
    out = np.where(boundary[..., None].astype(bool), out * 0.35 + red * 0.65, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_png(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def clean_split(out_root: Path, split: str):
    for sub in [
        "images",
        "labels_instance",
        "labels_instance_cc",
        "labels_foreground",
        "labels_inside",
        "labels_boundary",
        "qc_overlay",
    ]:
        shutil.rmtree(out_root / sub / split, ignore_errors=True)
    for p in [
        out_root / f"manifest_{split}.csv",
        out_root / f"coverage_{split}.csv",
        out_root / f"split_config_{split}.json",
    ]:
        if p.exists():
            p.unlink()


def write_patch(
    out_root: Path,
    split: str,
    img_stem: str,
    patch_img: np.ndarray,
    patch_inst: np.ndarray,
    y0: int,
    x0: int,
    patch_idx: int,
    kind: str,
    boundary_radius: int,
    qc_remaining: int,
):
    fg, inside, boundary = instance_targets(patch_inst, boundary_radius=boundary_radius)
    name = f"{img_stem}_{kind}{patch_idx:04d}_y{y0:04d}_x{x0:04d}.png"
    save_png(out_root / "images" / split / name, patch_img)
    inst_out = patch_inst.astype(np.uint16, copy=False)
    save_png(out_root / "labels_instance" / split / name, inst_out)
    save_png(out_root / "labels_instance_cc" / split / name, inst_out)
    save_png(out_root / "labels_foreground" / split / name, (fg * 255).astype(np.uint8))
    save_png(out_root / "labels_inside" / split / name, (inside * 255).astype(np.uint8))
    save_png(out_root / "labels_boundary" / split / name, (boundary * 255).astype(np.uint8))
    if qc_remaining > 0:
        overlay = make_overlay(patch_img, fg, boundary)
        save_png(out_root / "qc_overlay" / split / name, overlay)
    return {
        "split": split,
        "patch": name,
        "source_image": "",
        "source_label": "",
        "kind": kind,
        "y": int(y0),
        "x": int(x0),
        "patch_size": int(patch_img.shape[0]),
        "foreground_ratio": float(fg.mean()),
        "inside_ratio": float(inside.mean()),
        "boundary_ratio": float(boundary.mean()),
        "num_instances": int(len(np.unique(patch_inst)) - (1 if 0 in np.unique(patch_inst) else 0)),
    }


def choose_weighted_index(weights: Sequence[float], rng: random.Random) -> int:
    total = float(sum(weights))
    if total <= 0:
        return rng.randrange(len(weights))
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += float(w)
        if r <= acc:
            return i
    return len(weights) - 1


def random_foreground_coord(
    inst: np.ndarray,
    patch_size: int,
    rng: random.Random,
    min_foreground_ratio: float,
    min_boundary_ratio: float,
    boundary_radius: int,
    max_trials: int,
) -> Tuple[int, int]:
    h, w = inst.shape[:2]
    max_y = max(0, h - patch_size)
    max_x = max(0, w - patch_size)
    fg = inst > 0
    fg_ys, fg_xs = np.where(fg)

    best: Tuple[float, int, int] = (-1.0, 0, 0)
    for _ in range(max(1, max_trials)):
        if len(fg_ys) > 0 and rng.random() < 0.85:
            j = rng.randrange(len(fg_ys))
            cy, cx = int(fg_ys[j]), int(fg_xs[j])
            y0 = min(max(0, cy - rng.randint(0, patch_size - 1)), max_y)
            x0 = min(max(0, cx - rng.randint(0, patch_size - 1)), max_x)
        else:
            y0 = rng.randint(0, max_y) if max_y > 0 else 0
            x0 = rng.randint(0, max_x) if max_x > 0 else 0
        patch = inst[y0 : y0 + patch_size, x0 : x0 + patch_size]
        fg_patch, _, bd_patch = instance_targets(patch, boundary_radius=boundary_radius)
        fg_ratio = float(fg_patch.mean())
        bd_ratio = float(bd_patch.mean())
        score = fg_ratio + 2.0 * bd_ratio
        if score > best[0]:
            best = (score, y0, x0)
        if fg_ratio >= min_foreground_ratio or bd_ratio >= min_boundary_ratio:
            return y0, x0
    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", "--src_root", dest="data_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--label_subdir", default="labels_instance_cc")
    parser.add_argument("--patch_size", type=int, default=500)
    parser.add_argument("--grid_stride", type=int, default=250)
    parser.add_argument(
        "--target_total_patches",
        type=int,
        default=0,
        help="Minimum split-level patch count. Grid patches are always kept; foreground-aware patches fill any shortfall.",
    )
    parser.add_argument(
        "--target_patches_per_image",
        type=int,
        default=0,
        help="Minimum patch count for each source image. Grid patches are always kept; supplements fill any per-image shortfall.",
    )
    parser.add_argument("--min_foreground_ratio", type=float, default=0.02)
    parser.add_argument("--min_boundary_ratio", type=float, default=0.002)
    parser.add_argument("--max_trials_per_patch", type=int, default=80)
    parser.add_argument("--boundary_radius", type=int, default=1)
    parser.add_argument("--qc_num", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.patch_size <= 0:
        raise ValueError("--patch_size must be positive")
    if args.grid_stride <= 0:
        raise ValueError("--grid_stride must be positive")

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clean_split(out_root, args.split)

    pairs = load_pairs(data_root, args.split, args.label_subdir)
    if not pairs:
        raise RuntimeError(f"No images found for split={args.split} under {data_root}")

    rng = random.Random(int(args.seed))
    rows: List[Dict[str, object]] = []
    image_cache = []
    used_coords: Dict[str, set[Tuple[int, int]]] = {}
    per_image_counts: Dict[str, int] = {}
    supplement_index: Dict[str, int] = {}
    image_summaries: Dict[str, Dict[str, object]] = {}
    qc_remaining = int(args.qc_num)
    grid_count = 0

    print("========== Coverage patch generation ==========")
    print(f"data_root={data_root}")
    print(f"out_root={out_root}")
    print(f"split={args.split}, patch_size={args.patch_size}, grid_stride={args.grid_stride}")

    for img_path, label_path in pairs:
        img = read_rgb(img_path)
        inst = read_label(label_path)
        if img.shape[:2] != inst.shape[:2]:
            raise ValueError(f"Shape mismatch {img_path.name}: image={img.shape[:2]}, label={inst.shape[:2]}")
        orig_h, orig_w = inst.shape[:2]
        img, inst = pad_to_min(img, inst, args.patch_size)
        h, w = inst.shape[:2]
        coords = grid_coords(h, w, args.patch_size, args.grid_stride)
        stem = img_path.stem
        used_coords[stem] = set(coords)
        per_image_counts[stem] = 0
        supplement_index[stem] = 0
        image_summaries[stem] = {
            "split": args.split,
            "source_image": img_path.name,
            "source_label": label_path.name,
            "orig_h": int(orig_h),
            "orig_w": int(orig_w),
            "padded_h": int(h),
            "padded_w": int(w),
            "grid_patches": int(len(coords)),
            "supplement_patches": 0,
            "total_patches": int(len(coords)),
            "foreground_ratio": float((inst > 0).mean()),
        }
        image_cache.append((img_path, label_path, img, inst, float((inst > 0).mean()) + 1e-6))

        for patch_idx, (y0, x0) in enumerate(coords):
            patch_img = img[y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]
            patch_inst = inst[y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]
            row = write_patch(
                out_root,
                args.split,
                img_path.stem,
                patch_img,
                patch_inst,
                y0,
                x0,
                patch_idx,
                "g",
                args.boundary_radius,
                qc_remaining,
            )
            row["source_image"] = img_path.name
            row["source_label"] = label_path.name
            rows.append(row)
            per_image_counts[stem] += 1
            grid_count += 1
            qc_remaining = max(0, qc_remaining - 1)

    def add_supplement(img_i: int):
        nonlocal qc_remaining
        img_path, label_path, img, inst, _ = image_cache[img_i]
        stem = img_path.stem
        used = used_coords[stem]
        y0, x0 = random_foreground_coord(
            inst,
            args.patch_size,
            rng,
            args.min_foreground_ratio,
            args.min_boundary_ratio,
            args.boundary_radius,
            args.max_trials_per_patch,
        )
        retry = 0
        while (y0, x0) in used and retry < 20:
            y0, x0 = random_foreground_coord(
                inst,
                args.patch_size,
                rng,
                args.min_foreground_ratio,
                args.min_boundary_ratio,
                args.boundary_radius,
                args.max_trials_per_patch,
            )
            retry += 1
        used.add((y0, x0))

        patch_img = img[y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]
        patch_inst = inst[y0 : y0 + args.patch_size, x0 : x0 + args.patch_size]
        patch_idx = supplement_index[stem]
        supplement_index[stem] += 1
        row = write_patch(
            out_root,
            args.split,
            stem,
            patch_img,
            patch_inst,
            y0,
            x0,
            patch_idx,
            "s",
            args.boundary_radius,
            qc_remaining,
        )
        row["source_image"] = img_path.name
        row["source_label"] = label_path.name
        rows.append(row)
        per_image_counts[stem] += 1
        image_summaries[stem]["supplement_patches"] = int(image_summaries[stem]["supplement_patches"]) + 1
        image_summaries[stem]["total_patches"] = int(per_image_counts[stem])
        qc_remaining = max(0, qc_remaining - 1)

    target_per_image = max(0, int(args.target_patches_per_image))
    if target_per_image > 0:
        per_image_needed = 0
        for img_i, (img_path, *_rest) in enumerate(image_cache):
            stem = img_path.stem
            need = max(0, target_per_image - per_image_counts[stem])
            per_image_needed += need
            for _ in range(need):
                add_supplement(img_i)
        if per_image_needed > 0:
            print(
                f"=> grid patches={grid_count}; added per-image foreground-aware supplements={per_image_needed} "
                f"to reach target_patches_per_image={target_per_image}"
            )
        else:
            print(
                f"=> all images already have >= target_patches_per_image={target_per_image}; "
                "keeping all grid patches to preserve coverage."
            )

    supplement_needed = max(0, int(args.target_total_patches) - len(rows))
    if supplement_needed > 0:
        weights = [x[4] for x in image_cache]
        print(f"=> grid patches={grid_count}; adding foreground-aware supplements={supplement_needed}")
        for _ in range(supplement_needed):
            img_i = choose_weighted_index(weights, rng)
            add_supplement(img_i)
    elif int(args.target_total_patches) > 0:
        print(
            f"=> grid patches={grid_count}, already >= target_total_patches={args.target_total_patches}; "
            "keeping all grid patches to preserve coverage."
        )

    manifest = out_root / f"manifest_{args.split}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "patch",
        "source_image",
        "source_label",
        "kind",
        "y",
        "x",
        "patch_size",
        "foreground_ratio",
        "inside_ratio",
        "boundary_ratio",
        "num_instances",
    ]
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    coverage = out_root / f"coverage_{args.split}.csv"
    coverage_fields = [
        "split",
        "source_image",
        "source_label",
        "orig_h",
        "orig_w",
        "padded_h",
        "padded_w",
        "grid_patches",
        "supplement_patches",
        "total_patches",
        "foreground_ratio",
    ]
    with open(coverage, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=coverage_fields)
        writer.writeheader()
        writer.writerows(image_summaries.values())

    config = {
        "script": "make_coverage_patches.py",
        "purpose": "coverage-first offline patches for gland segmentation",
        "data_root": str(data_root),
        "out_root": str(out_root),
        "split": args.split,
        "label_subdir": args.label_subdir,
        "patch_size": int(args.patch_size),
        "grid_stride": int(args.grid_stride),
        "target_total_patches": int(args.target_total_patches),
        "target_patches_per_image": int(args.target_patches_per_image),
        "min_foreground_ratio": float(args.min_foreground_ratio),
        "min_boundary_ratio": float(args.min_boundary_ratio),
        "boundary_radius": int(args.boundary_radius),
        "seed": int(args.seed),
        "num_source_images": len(pairs),
        "num_grid_patches": int(grid_count),
        "num_supplement_patches": int(len(rows) - grid_count),
        "num_patches": int(len(rows)),
        "coverage_policy": (
            "all grid patches are kept; target_patches_per_image and target_total_patches "
            "are minimum patch budgets, not caps"
        ),
    }
    (out_root / f"split_config_{args.split}.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"Done. split={args.split}, images={len(pairs)}, grid={grid_count}, "
        f"supplement={len(rows) - grid_count}, total={len(rows)}"
    )
    print(f"manifest={manifest}")
    print(f"coverage={coverage}")
    print(f"qc_overlay={out_root / 'qc_overlay' / args.split}")


if __name__ == "__main__":
    main()
