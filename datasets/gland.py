# coding=utf-8
"""Dataset for binary foreground segmentation with boundary auxiliary supervision.

This version keeps the original target protocol and directory logic, but adds
medical-segmentation augmentations that are safe for masks:

- ColorJitter: image only.
- Random scale crop/pad: image + foreground + inside + boundary + instance, same params.
- Elastic deformation: image + foreground + inside + boundary + instance, same field.
- 90-degree rotation uses transpose/rot90 style operations so non-square images
  are not cropped by PIL.rotate(..., expand=False).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from torchvision.transforms import ColorJitter
except Exception:  # pragma: no cover - fallback for minimal environments
    ColorJitter = None

try:
    from scipy.ndimage import (
        gaussian_filter,
        map_coordinates,
        binary_dilation as scipy_binary_dilation,
        distance_transform_edt,
        label as scipy_label,
    )
except Exception:  # pragma: no cover
    gaussian_filter = None
    map_coordinates = None
    scipy_binary_dilation = None
    distance_transform_edt = None
    scipy_label = None

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover - baseline runs do not require the continuity target
    skeletonize = None

from utils import find_mask, first_existing_dir, list_images


try:
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.Transpose.FLIP_TOP_BOTTOM
    ROTATE_90 = Image.Transpose.ROTATE_90
    ROTATE_180 = Image.Transpose.ROTATE_180
    ROTATE_270 = Image.Transpose.ROTATE_270
except AttributeError:  # Pillow < 9
    FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.FLIP_TOP_BOTTOM
    ROTATE_90 = Image.ROTATE_90
    ROTATE_180 = Image.ROTATE_180
    ROTATE_270 = Image.ROTATE_270


def _read_mask_array(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D or RGB-like: {path}, shape={arr.shape}")
    return arr


def _binary_from_mask(path: Path, kind: str) -> np.ndarray:
    """
    Supported kinds:
      foreground -> class mask values 1 or 2, or binary mask > 0
      inside     -> class mask value 1, or binary mask > 0
      boundary   -> class mask value 2, or binary mask > 0
    """
    arr = _read_mask_array(path)
    vals = set(np.unique(arr).tolist())
    if vals.issubset({0, 1, 2}):
        if kind == "foreground":
            return ((arr == 1) | (arr == 2)).astype(np.uint8)
        if kind == "inside":
            return (arr == 1).astype(np.uint8)
        if kind == "boundary":
            return (arr == 2).astype(np.uint8)
        raise ValueError(f"Unknown mask kind: {kind}")
    return (arr > 0).astype(np.uint8)



def _read_instance_array(path: Path) -> np.ndarray:
    """Read an instance-label map. RGB encodings are converted to unique integer IDs."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.uint32)
        arr = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    if arr.ndim != 2:
        raise ValueError(f"Instance mask must be 2D or RGB-like: {path}, shape={arr.shape}")
    return arr.astype(np.int32)


def _instance_to_pil(arr: np.ndarray) -> Image.Image:
    """Store instance IDs in a nearest-neighbor safe PIL image."""
    return Image.fromarray(arr.astype(np.int32, copy=False), mode="I")


def _pil_to_instance_np(mask: Image.Image) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.uint32)
        arr = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    return arr.astype(np.int32, copy=False)

def _rotate_90_no_crop(img: Image.Image, k: int) -> Image.Image:
    k = int(k) % 4
    if k == 1:
        return img.transpose(ROTATE_90)
    if k == 2:
        return img.transpose(ROTATE_180)
    if k == 3:
        return img.transpose(ROTATE_270)
    return img


def _resize_pil(img: Image.Image, size: Tuple[int, int], is_mask: bool) -> Image.Image:
    return img.resize(size, Image.NEAREST if is_mask else Image.BILINEAR)


def _crop_or_pad_to_size(
    img: Image.Image,
    target_size: Tuple[int, int],
    crop_left: int,
    crop_top: int,
    paste_x: int,
    paste_y: int,
    fill=0,
) -> Image.Image:
    """Apply precomputed crop/pad parameters so all masks stay aligned."""
    target_w, target_h = target_size
    src_w, src_h = img.size

    crop_right = crop_left + min(src_w, target_w)
    crop_bottom = crop_top + min(src_h, target_h)
    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    out = Image.new(img.mode, target_size, fill)
    out.paste(cropped, (paste_x, paste_y))
    return out


def _elastic_deform_image_and_masks(
    img: Image.Image,
    masks: Sequence[Image.Image],
    alpha: float,
    sigma: float,
) -> Tuple[Image.Image, list[Image.Image]]:
    """Apply one elastic field to RGB image and all masks."""
    if gaussian_filter is None or map_coordinates is None:
        return img, list(masks)

    img_arr = np.asarray(img.convert("RGB"))
    h, w = img_arr.shape[:2]

    dx = gaussian_filter((np.random.rand(h, w) * 2.0 - 1.0), sigma=float(sigma), mode="reflect") * float(alpha)
    dy = gaussian_filter((np.random.rand(h, w) * 2.0 - 1.0), sigma=float(sigma), mode="reflect") * float(alpha)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = (yy + dy, xx + dx)

    out_img = np.empty_like(img_arr)
    for c in range(img_arr.shape[2]):
        out_img[..., c] = map_coordinates(img_arr[..., c], coords, order=1, mode="reflect").reshape(h, w)

    out_masks = []
    for mask in masks:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        warped = map_coordinates(arr, coords, order=0, mode="constant", cval=0).reshape(h, w)
        out_masks.append(Image.fromarray((warped > 127).astype(np.uint8) * 255))

    return Image.fromarray(out_img.astype(np.uint8)), out_masks


def _elastic_deform_image_masks_and_instances(
    img: Image.Image,
    masks: Sequence[Image.Image],
    instances: Sequence[Image.Image],
    alpha: float,
    sigma: float,
) -> Tuple[Image.Image, list[Image.Image], list[Image.Image]]:
    """Apply one elastic field to RGB, binary masks, and integer instance IDs."""
    if gaussian_filter is None or map_coordinates is None:
        return img, list(masks), list(instances)

    img_arr = np.asarray(img.convert("RGB"))
    h, w = img_arr.shape[:2]

    dx = gaussian_filter((np.random.rand(h, w) * 2.0 - 1.0), sigma=float(sigma), mode="reflect") * float(alpha)
    dy = gaussian_filter((np.random.rand(h, w) * 2.0 - 1.0), sigma=float(sigma), mode="reflect") * float(alpha)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = (yy + dy, xx + dx)

    out_img = np.empty_like(img_arr)
    for c in range(img_arr.shape[2]):
        out_img[..., c] = map_coordinates(img_arr[..., c], coords, order=1, mode="reflect").reshape(h, w)

    out_masks = []
    for mask in masks:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        warped = map_coordinates(arr, coords, order=0, mode="constant", cval=0).reshape(h, w)
        out_masks.append(Image.fromarray((warped > 127).astype(np.uint8) * 255))

    out_instances = []
    for inst in instances:
        arr = _pil_to_instance_np(inst)
        warped = map_coordinates(arr, coords, order=0, mode="constant", cval=0).reshape(h, w)
        out_instances.append(_instance_to_pil(warped.astype(np.int32, copy=False)))

    return Image.fromarray(out_img.astype(np.uint8)), out_masks, out_instances


def _disk_structure(radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _binary_dilate_np(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = mask.astype(bool)
    radius = int(radius)
    if radius <= 0:
        return mask.copy()
    if scipy_binary_dilation is not None:
        return scipy_binary_dilation(mask, structure=_disk_structure(radius))
    # Small pure-numpy fallback. It is slower but avoids silently failing in
    # minimal environments without scipy.
    pad = radius
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(_disk_structure(radius))
    ys = ys - radius
    xs = xs - radius
    for dy, dx in zip(ys.tolist(), xs.tolist()):
        out |= padded[pad + dy:pad + dy + mask.shape[0], pad + dx:pad + dx + mask.shape[1]]
    return out


def make_soft_boundary_np(
    boundary: np.ndarray,
    radius1: int = 1,
    radius2: int = 2,
    weight1: float = 0.6,
    weight2: float = 0.3,
) -> np.ndarray:
    """Multi-band soft boundary target: 1.0 on original contour, lower weights on rings."""
    b0 = boundary.astype(bool)
    d1 = _binary_dilate_np(b0, radius1)
    d2 = _binary_dilate_np(b0, radius2)
    soft = np.zeros(boundary.shape, dtype=np.float32)
    soft[d2] = float(weight2)
    soft[d1] = float(weight1)
    soft[b0] = 1.0
    return soft


def make_boundary_focus_np(boundary: np.ndarray, radius: int = 2) -> np.ndarray:
    """Region where boundary errors are emphasized by BoundaryDiff loss."""
    return _binary_dilate_np(boundary.astype(bool), radius).astype(np.float32)


def make_truncated_proximity_field_np(
    core_mask: np.ndarray,
    radius: int,
    support_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Truncated distance-to-structure proximity field in [0, 1].

    Value 1.0 lies on the target structure. The response decays linearly with
    Euclidean distance and is truncated to 0 outside the specified radius.
    """
    core = core_mask.astype(bool)
    radius = max(1, int(radius))
    if not core.any():
        return np.zeros(core.shape, dtype=np.float32)

    if distance_transform_edt is not None:
        dist = distance_transform_edt(~core).astype(np.float32)
    else:
        # Fallback: coarse ring approximation via iterative dilations.
        dist = np.full(core.shape, float(radius) + 1.0, dtype=np.float32)
        current = core.copy()
        dist[current] = 0.0
        for r in range(1, radius + 1):
            current = _binary_dilate_np(current, 1)
            update = current & (dist > float(radius))
            dist[update] = float(r)

    field = 1.0 - np.clip(dist / float(radius), 0.0, 1.0)
    if support_mask is not None:
        field = field * support_mask.astype(np.float32)
    field[core] = 1.0
    return field.astype(np.float32)


def make_boundary_distance_target_np(
    boundary: np.ndarray,
    radius: int = 6,
) -> np.ndarray:
    boundary_bool = boundary.astype(bool)
    if not boundary_bool.any():
        return np.zeros_like(boundary, dtype=np.float32)
    support = _binary_dilate_np(boundary_bool, max(1, int(radius)))
    return make_truncated_proximity_field_np(boundary_bool, radius=radius, support_mask=support)


def make_violation_target_np(
    inside: np.ndarray,
    boundary: np.ndarray,
    boundary_distance: np.ndarray,
    separation_distance: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """
    Conservative structural-priority proxy kept for backward compatibility.

    The target highlights regions where local merge/split mistakes are most
    likely to hurt object-level metrics, without relying on unstable online
    topology estimation from current predictions.
    """
    inside_bool = inside.astype(bool)
    boundary_bool = boundary.astype(bool)
    if not inside_bool.any():
        return np.zeros_like(boundary_distance, dtype=np.float32)
    narrow_inside_band = _binary_dilate_np(boundary_bool, max(1, int(radius))) & inside_bool
    violation = np.maximum.reduce(
        [
            0.65 * boundary_distance.astype(np.float32),
            1.00 * separation_distance.astype(np.float32),
            0.45 * narrow_inside_band.astype(np.float32),
        ]
    )
    return np.clip(violation, 0.0, 1.0).astype(np.float32)


def make_separation_target_np(
    inside: np.ndarray,
    boundary: np.ndarray,
    radius: int = 4,
    min_neighbor_components: int = 2,
    instance_labels: Optional[np.ndarray] = None,
    target_mode: str = "legacy",
    contact_search_radius: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split-band targets for the separation branch.

    If instance labels are available, positives are generated only where a
    boundary/gap neighborhood sees at least two different gland IDs.  The final
    target is represented by:

      - core:   hard touching-gland split centerline, used for Dice/F1 metrics.
      - field:  truncated distance / ordinal split field in [0, 1].
      - level:  0=background, 1=near-split band, 2=core split pixels.
    """
    inside_bool = inside.astype(bool)
    boundary_bool = boundary.astype(bool)
    if target_mode not in {"legacy", "instance_gap"}:
        raise ValueError(f"Unsupported contact target mode: {target_mode}")
    if not inside_bool.any() or (target_mode == "legacy" and not boundary_bool.any()):
        zeros = np.zeros_like(boundary_bool, dtype=np.float32)
        return zeros, zeros, np.zeros_like(boundary_bool, dtype=np.uint8)

    if instance_labels is not None:
        labels = np.asarray(instance_labels).astype(np.int32, copy=False)
        if labels.shape != inside_bool.shape:
            raise ValueError(f"instance_labels shape {labels.shape} != inside/boundary shape {inside_bool.shape}")
        labels = labels.copy()
        labels[labels < 0] = 0
        if int(labels.max()) <= 1:
            labels = None
    else:
        labels = None

    if labels is None:
        if scipy_label is not None:
            labels, _ = scipy_label(inside_bool)
        else:
            labels = inside_bool.astype(np.int32)

    core = np.zeros_like(boundary_bool, dtype=bool)
    if target_mode == "instance_gap":
        # Two *different* annotated glands must reach the same gap pixel.  This
        # avoids treating the entire contour of one gland as a contact cue and
        # tolerates an imperfectly positioned hand-drawn boundary centerline.
        search_r = max(1, int(contact_search_radius))
        if int(labels.max()) > 1:
            overlap_count = np.zeros_like(boundary_bool, dtype=np.uint16)
            for obj_id in np.unique(labels):
                if obj_id > 0:
                    ys, xs = np.where(labels == obj_id)
                    if ys.size == 0:
                        continue
                    y0 = max(0, int(ys.min()) - search_r)
                    y1 = min(labels.shape[0], int(ys.max()) + search_r + 1)
                    x0 = max(0, int(xs.min()) - search_r)
                    x1 = min(labels.shape[1], int(xs.max()) + search_r + 1)
                    roi = labels[y0:y1, x0:x1] == obj_id
                    overlap_count[y0:y1, x0:x1] += _binary_dilate_np(
                        roi, search_r
                    ).astype(np.uint16)
            core = (
                (overlap_count >= int(min_neighbor_components))
                & ~inside_bool
            )
    else:
        ys, xs = np.where(boundary_bool)
        # Preserve the published protocol exactly in the default branch.
        search_r = max(1, min(3, int(radius)))
        h, w = boundary_bool.shape
        for y, x in zip(ys.tolist(), xs.tolist()):
            y0, y1 = max(0, y - search_r), min(h, y + search_r + 1)
            x0, x1 = max(0, x - search_r), min(w, x + search_r + 1)
            ids = np.unique(labels[y0:y1, x0:x1])
            ids = ids[ids > 0]
            if len(ids) >= int(min_neighbor_components):
                core[y, x] = True

    # Keep the original fallback only for the original target protocol.  An
    # empty instance-gap target means this crop has no verified near-contact.
    if target_mode == "legacy" and not core.any() and int(labels.max()) > 1:
        overlap_count = np.zeros_like(boundary_bool, dtype=np.uint16)
        ids = [int(x) for x in np.unique(labels) if int(x) > 0]
        for obj_id in ids:
            m = labels == obj_id
            if m.any():
                overlap_count += _binary_dilate_np(m, search_r).astype(np.uint16)
        core = (overlap_count >= int(min_neighbor_components)) & _binary_dilate_np(boundary_bool, max(1, search_r))

    if not core.any():
        zeros = np.zeros_like(boundary_bool, dtype=np.float32)
        return zeros, zeros, np.zeros_like(boundary_bool, dtype=np.uint8)

    inner_r = max(1, int(radius))
    outer_r = max(inner_r + 1, int(radius) * 2)
    boundary_band = _binary_dilate_np(
        core if target_mode == "instance_gap" else boundary_bool, outer_r
    )
    foreground_band = _binary_dilate_np(inside_bool, outer_r)
    support = boundary_band & foreground_band
    core_band = _binary_dilate_np(core, inner_r) & support
    field = make_truncated_proximity_field_np(core, radius=outer_r, support_mask=support)
    field[core_band] = np.maximum(field[core_band], 0.6)
    field[core] = 1.0
    if gaussian_filter is not None:
        sigma = max(1.0, float(radius) * 0.75)
        field = gaussian_filter(field, sigma=sigma)
        maxv = float(field.max())
        if maxv > 0:
            field = field / maxv
        field = np.clip(field, 0.0, 1.0)
        field[core_band] = np.maximum(field[core_band], 0.6)
        field[core] = 1.0

    level = np.zeros_like(boundary_bool, dtype=np.uint8)
    level[(field > 0.05) & support] = 1
    level[core] = 2
    core_float = core.astype(np.float32)
    return core_float, field.astype(np.float32), level


def make_narrow_continuity_target_np(
    inside: np.ndarray,
    instance_labels: np.ndarray,
    max_radius: int = 4,
) -> np.ndarray:
    """Mark thin medial strips *inside* annotated instances for FG supervision.

    A centerline pixel is selected only if its distance to that instance's
    exterior is at most max_radius.  One-pixel dilation gives a stable training
    target without crossing into another gland or an annotated gap.
    """
    if skeletonize is None or distance_transform_edt is None:
        raise RuntimeError("Narrow-continuity supervision requires scipy and scikit-image")
    labels = np.asarray(instance_labels)
    inside_bool = np.asarray(inside).astype(bool)
    if labels.shape != inside_bool.shape:
        raise ValueError(f"Instance shape {labels.shape} != inside shape {inside_bool.shape}")
    result = np.zeros_like(inside_bool, dtype=bool)
    for obj_id in np.unique(labels):
        if obj_id <= 0:
            continue
        gland = (labels == obj_id) & inside_bool
        if not gland.any():
            continue
        ys, xs = np.where(gland)
        y0, y1 = max(0, int(ys.min()) - 1), min(gland.shape[0], int(ys.max()) + 2)
        x0, x1 = max(0, int(xs.min()) - 1), min(gland.shape[1], int(xs.max()) + 2)
        crop = gland[y0:y1, x0:x1]
        medial = skeletonize(crop)
        narrow = medial & (distance_transform_edt(crop) <= max(1, int(max_radius)))
        result[y0:y1, x0:x1] |= _binary_dilate_np(narrow, 1) & crop
    return result.astype(np.float32)


class GlandForegroundBoundaryAuxDataset(Dataset):
    """
    GlaS foreground-main + boundary-aux dataset.

    Preferred layout:
        root/
        ├── images/{split}
        ├── labels_foreground/{split}   # red + green
        ├── labels_inside/{split}       # red only, optional for object-level evaluation
        ├── labels_boundary/{split}     # green only
        └── labels_instance_cc/{split}  # instance labels for MILDNet/DCAN-style validation

    Fallback:
        labels_class/{split}, where 0=background, 1=inside, 2=boundary.

    Returns a dict:
        image:      float tensor [3,H,W]
        foreground: float tensor [1,H,W]  # main binary target, inside OR boundary
        boundary:        float tensor [1,H,W]  # hard auxiliary contour target
        boundary_soft:   float tensor [1,H,W]  # multi-band soft contour target
        boundary_focus:  float tensor [1,H,W]  # dilated focus region for BoundaryDiff
        separation:      float tensor [1,H,W]  # soft adhesion-boundary band target
        inside:          float tensor [1,H,W]  # diagnostic target
        object_instance: int tensor [1,H,W]    # labels_instance_cc if available; validation object target
        name:            image stem
    """

    def __init__(
        self,
        root,
        split: str = "train",
        input_size: int = 512,
        train: bool = True,
        augment: bool = True,
        foreground_dir_name: str = "labels_foreground",
        inside_dir_name: str = "labels_inside",
        boundary_dir_name: str = "labels_boundary",
        object_dir_name: str = "labels_instance_cc",
        strict_consistency: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        color_jitter_prob: float = 0.6,
        color_jitter_brightness: float = 0.20,
        color_jitter_contrast: float = 0.20,
        color_jitter_saturation: float = 0.15,
        color_jitter_hue: float = 0.03,
        random_scale_crop_prob: float = 0.6,
        random_scale_min: float = 0.75,
        random_scale_max: float = 1.25,
        elastic_prob: float = 0.35,
        elastic_alpha: float = 30.0,
        elastic_sigma: float = 5.0,
        soft_boundary_radius1: int = 1,
        soft_boundary_radius2: int = 2,
        soft_boundary_weight1: float = 0.6,
        soft_boundary_weight2: float = 0.3,
        boundary_focus_radius: int = 2,
        separation_radius: int = 4,
        contact_target_mode: str = "legacy",
        contact_search_radius: int = 5,
        continuity_max_radius: int = 0,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = str(split)
        self.input_size = int(input_size)
        self.train = bool(train)
        self.augment = bool(augment and train)
        self.strict_consistency = bool(strict_consistency)

        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotate90_prob = float(rotate90_prob)
        self.color_jitter_prob = float(color_jitter_prob)
        self.random_scale_crop_prob = float(random_scale_crop_prob)
        self.random_scale_min = float(random_scale_min)
        self.random_scale_max = float(random_scale_max)
        self.elastic_prob = float(elastic_prob)
        self.elastic_alpha = float(elastic_alpha)
        self.elastic_sigma = float(elastic_sigma)
        self.soft_boundary_radius1 = int(soft_boundary_radius1)
        self.soft_boundary_radius2 = int(soft_boundary_radius2)
        self.soft_boundary_weight1 = float(soft_boundary_weight1)
        self.soft_boundary_weight2 = float(soft_boundary_weight2)
        self.boundary_focus_radius = int(boundary_focus_radius)
        self.separation_radius = int(separation_radius)
        self.contact_target_mode = str(contact_target_mode)
        self.contact_search_radius = int(contact_search_radius)
        self.continuity_max_radius = int(continuity_max_radius)

        if self.contact_target_mode not in {"legacy", "instance_gap"}:
            raise ValueError(f"Unsupported contact_target_mode: {self.contact_target_mode}")
        if self.contact_search_radius < 1:
            raise ValueError("contact_search_radius must be >= 1")
        if self.continuity_max_radius < 0:
            raise ValueError("continuity_max_radius must be >= 0")

        if self.random_scale_min <= 0 or self.random_scale_max <= 0:
            raise ValueError("random_scale_min/random_scale_max must be positive.")
        if self.random_scale_min > self.random_scale_max:
            raise ValueError("random_scale_min must be <= random_scale_max.")

        self.color_jitter = None
        if ColorJitter is not None:
            self.color_jitter = ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                hue=color_jitter_hue,
            )

        self.img_dir = self.root / "images" / self.split
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.foreground_dir, self.foreground_source = first_existing_dir(
            self.root, self.split, [foreground_dir_name, "labels_foreground", "labels_binary", "labels_class"]
        )
        self.inside_dir, self.inside_source = first_existing_dir(
            self.root, self.split, [inside_dir_name, "labels_inside", "labels_class"]
        )
        self.boundary_dir, self.boundary_source = first_existing_dir(
            self.root, self.split, [boundary_dir_name, "labels_boundary", "labels_class"]
        )
        self.object_dir, self.object_source = first_existing_dir(
            self.root, self.split, [object_dir_name, "labels_instance_cc", "labels_instance", "labels_instances"]
        )

        if self.foreground_dir is None:
            raise FileNotFoundError(
                f"Foreground label dir not found under {self.root}. Tried {foreground_dir_name}, labels_foreground, labels_binary, labels_class."
            )
        if self.boundary_dir is None:
            raise FileNotFoundError(
                f"Boundary label dir not found under {self.root}. Tried {boundary_dir_name}, labels_boundary, labels_class."
            )
        if self.inside_dir is None:
            print(f"[warning] inside labels not found for split={self.split}; inside will be derived as foreground minus boundary.")

        self.images = list_images(self.img_dir)
        if not self.images:
            raise RuntimeError(f"No images found in {self.img_dir}")

        self.items = []
        for img_path in self.images:
            fg_path = find_mask(self.foreground_dir, img_path.stem, role="foreground", image_path=img_path)
            bd_path = find_mask(self.boundary_dir, img_path.stem, role="boundary", image_path=img_path)
            inside_path = (
                find_mask(self.inside_dir, img_path.stem, role="inside", image_path=img_path)
                if self.inside_dir is not None
                else None
            )
            object_path = None
            if self.object_dir is not None:
                try:
                    object_path = find_mask(self.object_dir, img_path.stem, role="object_instance", image_path=img_path)
                except FileNotFoundError:
                    object_path = None
            self.items.append((img_path, fg_path, inside_path, bd_path, object_path))

        if self.contact_target_mode == "instance_gap" or self.continuity_max_radius > 0:
            missing = [item[0].name for item in self.items if item[4] is None]
            if missing:
                raise FileNotFoundError(
                    "Instance-aware training needs labels_instance_cc for every image; "
                    f"missing {len(missing)} examples, first: {missing[:3]}"
                )

        print(
            f"[GlandForegroundBoundaryAuxDataset] split={self.split}, n={len(self.items)}, images={self.img_dir}"
        )
        print(
            f"[GlandForegroundBoundaryAuxDataset] foreground={self.foreground_dir} ({self.foreground_source}), "
            f"inside={self.inside_dir} ({self.inside_source}), boundary={self.boundary_dir} ({self.boundary_source}), "
            f"object_instance={self.object_dir} ({self.object_source})"
        )

    def __len__(self):
        return len(self.items)

    def _random_scale_crop_pair(
        self,
        img: Image.Image,
        foreground: Image.Image,
        inside: Image.Image,
        boundary: Image.Image,
        object_instance: Optional[Image.Image] = None,
    ):
        w, h = img.size
        scale = random.uniform(self.random_scale_min, self.random_scale_max)
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))

        img = _resize_pil(img, new_size, is_mask=False)
        foreground = _resize_pil(foreground, new_size, is_mask=True)
        inside = _resize_pil(inside, new_size, is_mask=True)
        boundary = _resize_pil(boundary, new_size, is_mask=True)
        if object_instance is not None:
            object_instance = _resize_pil(object_instance, new_size, is_mask=True)

        target_size = (w, h)
        src_w, src_h = new_size
        crop_left = random.randint(0, src_w - w) if src_w > w else 0
        crop_top = random.randint(0, src_h - h) if src_h > h else 0
        cropped_w = min(src_w, w)
        cropped_h = min(src_h, h)
        paste_x = random.randint(0, w - cropped_w) if cropped_w < w else 0
        paste_y = random.randint(0, h - cropped_h) if cropped_h < h else 0

        img = _crop_or_pad_to_size(img, target_size, crop_left, crop_top, paste_x, paste_y, fill=0)
        foreground = _crop_or_pad_to_size(foreground, target_size, crop_left, crop_top, paste_x, paste_y, fill=0)
        inside = _crop_or_pad_to_size(inside, target_size, crop_left, crop_top, paste_x, paste_y, fill=0)
        boundary = _crop_or_pad_to_size(boundary, target_size, crop_left, crop_top, paste_x, paste_y, fill=0)
        if object_instance is not None:
            object_instance = _crop_or_pad_to_size(object_instance, target_size, crop_left, crop_top, paste_x, paste_y, fill=0)
        return img, foreground, inside, boundary, object_instance

    def _augment_pair(
        self,
        img: Image.Image,
        foreground: Image.Image,
        inside: Image.Image,
        boundary: Image.Image,
        object_instance: Optional[Image.Image] = None,
    ):
        if random.random() < self.hflip_prob:
            img = img.transpose(FLIP_LEFT_RIGHT)
            foreground = foreground.transpose(FLIP_LEFT_RIGHT)
            inside = inside.transpose(FLIP_LEFT_RIGHT)
            boundary = boundary.transpose(FLIP_LEFT_RIGHT)
            if object_instance is not None:
                object_instance = object_instance.transpose(FLIP_LEFT_RIGHT)

        if random.random() < self.vflip_prob:
            img = img.transpose(FLIP_TOP_BOTTOM)
            foreground = foreground.transpose(FLIP_TOP_BOTTOM)
            inside = inside.transpose(FLIP_TOP_BOTTOM)
            boundary = boundary.transpose(FLIP_TOP_BOTTOM)
            if object_instance is not None:
                object_instance = object_instance.transpose(FLIP_TOP_BOTTOM)

        if random.random() < self.rotate90_prob:
            k = random.choice([1, 2, 3])
            img = _rotate_90_no_crop(img, k)
            foreground = _rotate_90_no_crop(foreground, k)
            inside = _rotate_90_no_crop(inside, k)
            boundary = _rotate_90_no_crop(boundary, k)
            if object_instance is not None:
                object_instance = _rotate_90_no_crop(object_instance, k)

        if random.random() < self.random_scale_crop_prob:
            img, foreground, inside, boundary, object_instance = self._random_scale_crop_pair(
                img, foreground, inside, boundary, object_instance
            )

        if random.random() < self.elastic_prob:
            if object_instance is None:
                img, masks = _elastic_deform_image_and_masks(
                    img,
                    [foreground, inside, boundary],
                    alpha=self.elastic_alpha,
                    sigma=self.elastic_sigma,
                )
                foreground, inside, boundary = masks
            else:
                img, masks, instances = _elastic_deform_image_masks_and_instances(
                    img,
                    [foreground, inside, boundary],
                    [object_instance],
                    alpha=self.elastic_alpha,
                    sigma=self.elastic_sigma,
                )
                foreground, inside, boundary = masks
                object_instance = instances[0]

        # Photometric augmentation must never touch masks.
        if self.color_jitter is not None and random.random() < self.color_jitter_prob:
            img = self.color_jitter(img)

        return img, foreground, inside, boundary, object_instance

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path, fg_path, inside_path, bd_path, object_path = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        fg_arr = _binary_from_mask(fg_path, "foreground")
        bd_arr = _binary_from_mask(bd_path, "boundary")
        if inside_path is not None:
            inside_arr = _binary_from_mask(inside_path, "inside")
        else:
            inside_arr = np.logical_and(fg_arr > 0, bd_arr == 0).astype(np.uint8)
        object_arr = _read_instance_array(object_path) if object_path is not None else None

        if self.strict_consistency:
            expected = np.logical_or(inside_arr > 0, bd_arr > 0)
            if not np.array_equal(fg_arr > 0, expected):
                diff = int(np.logical_xor(fg_arr > 0, expected).sum())
                raise ValueError(f"foreground != inside OR boundary for {img_path.name}, diff={diff}")

        foreground = Image.fromarray(fg_arr.astype(np.uint8) * 255)
        inside = Image.fromarray(inside_arr.astype(np.uint8) * 255)
        boundary = Image.fromarray(bd_arr.astype(np.uint8) * 255)
        # Instance IDs are transformed with nearest-neighbor geometry so the
        # separation branch can learn true touching-gland contact regions.
        object_instance = _instance_to_pil(object_arr) if object_arr is not None else None
        metric_object_instance = object_instance.copy() if (object_instance is not None and not self.train) else None

        if self.augment:
            img, foreground, inside, boundary, object_instance = self._augment_pair(
                img, foreground, inside, boundary, object_instance
            )

        size = (self.input_size, self.input_size)
        img = img.resize(size, Image.BILINEAR)
        foreground = foreground.resize(size, Image.NEAREST)
        inside = inside.resize(size, Image.NEAREST)
        boundary = boundary.resize(size, Image.NEAREST)
        target_object_instance = object_instance.resize(size, Image.NEAREST) if object_instance is not None else None

        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5
        image_t = torch.from_numpy(img_np).permute(2, 0, 1).float()

        def pil_to_binary_np(mask: Image.Image) -> np.ndarray:
            arr = np.asarray(mask)
            if arr.ndim == 3:
                arr = arr[..., 0]
            return (arr > 127).astype(np.float32)

        fg_np = pil_to_binary_np(foreground)
        inside_np = pil_to_binary_np(inside)
        boundary_np = pil_to_binary_np(boundary)
        if target_object_instance is not None:
            target_object_instance_np = _pil_to_instance_np(target_object_instance)
        else:
            try:
                from scipy.ndimage import label as _ndi_label
                target_object_instance_np, _ = _ndi_label(inside_np.astype(bool))
            except Exception:
                target_object_instance_np = inside_np.astype(np.int32)

        if metric_object_instance is not None:
            metric_object_instance_np = _pil_to_instance_np(metric_object_instance)
        else:
            metric_object_instance_np = target_object_instance_np

        boundary_soft_np = make_soft_boundary_np(
            boundary_np,
            radius1=self.soft_boundary_radius1,
            radius2=self.soft_boundary_radius2,
            weight1=self.soft_boundary_weight1,
            weight2=self.soft_boundary_weight2,
        )
        boundary_focus_np = make_boundary_focus_np(boundary_np, radius=self.boundary_focus_radius)
        boundary_distance_np = make_boundary_distance_target_np(
            boundary_np,
            radius=max(self.boundary_focus_radius + 2, self.separation_radius + 1),
        )
        separation_core_np, separation_field_np, separation_level_np = make_separation_target_np(
            inside_np,
            boundary_np,
            radius=self.separation_radius,
            instance_labels=target_object_instance_np,
            target_mode=self.contact_target_mode,
            contact_search_radius=self.contact_search_radius,
        )
        separation_focus_np = np.maximum(boundary_focus_np, (separation_field_np > 0).astype(np.float32))
        continuity_np = (
            make_narrow_continuity_target_np(
                inside_np, target_object_instance_np, max_radius=self.continuity_max_radius
            )
            if self.continuity_max_radius > 0
            else np.zeros_like(fg_np, dtype=np.float32)
        )

        def to_tensor_np(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0).float()

        return {
            "image": image_t,
            "foreground": to_tensor_np(fg_np),
            "inside": to_tensor_np(inside_np),
            "object_instance": torch.from_numpy(metric_object_instance_np.astype(np.int32)).unsqueeze(0),
            "boundary": to_tensor_np(boundary_np),
            "boundary_soft": to_tensor_np(boundary_soft_np),
            "boundary_distance": to_tensor_np(boundary_distance_np),
            "boundary_focus": to_tensor_np(boundary_focus_np),
            "separation": to_tensor_np(separation_core_np),
            "separation_distance": to_tensor_np(separation_field_np),
            "separation_level": torch.from_numpy(separation_level_np.astype(np.int64, copy=False)).unsqueeze(0),
            "separation_focus": to_tensor_np(separation_focus_np),
            "narrow_continuity": to_tensor_np(continuity_np),
            "name": img_path.stem,
            "image_path": str(img_path),
        }
