"""Training patches with instance-preserving geometric and color augmentation.

Only instance IDs are authoritative. Foreground and edge targets are derived
after joint geometric augmentation by the loss, avoiding mask disagreement.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from utils import find_mask, list_images


class GlandDataset(Dataset):
    def __init__(self, root, split="train", input_size=464, augment=True):
        self.root = Path(root)
        self.size = int(input_size)
        self.augment = bool(augment)
        image_dir = self.root / "images" / split
        label_dir = self.root / "labels_instance_cc" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Need images and labels_instance_cc for {split} under {self.root}"
            )
        self.items = [
            (image, find_mask(label_dir, image.stem, "instance", image))
            for image in list_images(image_dir)
        ]
        if not self.items:
            raise RuntimeError(f"No training images found in {image_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, label_path = self.items[index]
        image = Image.open(image_path).convert("RGB")
        label_array = np.asarray(Image.open(label_path))
        if label_array.ndim == 3:
            rgb = label_array[..., :3].astype(np.int32)
            label_array = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
        label = Image.fromarray(label_array.astype(np.int32), mode="I")
        if self.augment:
            for transpose in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
                if random.random() < 0.5:
                    image, label = image.transpose(transpose), label.transpose(transpose)
            rotation = random.randrange(4)
            if rotation:
                op = (
                    None,
                    Image.Transpose.ROTATE_90,
                    Image.Transpose.ROTATE_180,
                    Image.Transpose.ROTATE_270,
                )[rotation]
                image, label = image.transpose(op), label.transpose(op)
            if random.random() < 0.6:
                scale = random.uniform(0.75, 1.25)
                width = max(16, round(image.width * scale))
                height = max(16, round(image.height * scale))
                image = image.resize((width, height), Image.Resampling.BILINEAR)
                label = label.resize((width, height), Image.Resampling.NEAREST)
            if random.random() < 0.6:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        if image.width < self.size or image.height < self.size:
            pad_w, pad_h = max(0, self.size - image.width), max(0, self.size - image.height)
            from PIL import ImageOps

            image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=255)
            label = ImageOps.expand(label, (0, 0, pad_w, pad_h), fill=0)
        if self.augment:
            left = random.randrange(image.width - self.size + 1)
            top = random.randrange(image.height - self.size + 1)
        else:
            left = (image.width - self.size) // 2
            top = (image.height - self.size) // 2
        box = (left, top, left + self.size, top + self.size)
        image, label = image.crop(box), label.crop(box)
        rgb = np.asarray(image, dtype=np.float32)
        inst = np.asarray(label, dtype=np.int32)
        return {
            "image": torch.from_numpy(((rgb / 255.0 - 0.5) / 0.5).transpose(2, 0, 1).copy()),
            "object_instance": torch.from_numpy(inst.copy()).unsqueeze(0),
            "name": image_path.stem,
        }
