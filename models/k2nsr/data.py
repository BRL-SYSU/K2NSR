"""Image loading and aligned HR/LR crops for K2NSR.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def image_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No supported images found in {root}")
    return paths


def load_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class TrainingImages(Dataset):
    """Crop aligned pairs or HR-only samples; never stretch training images."""

    def __init__(self, hr_directory: str | Path, *, lr_directory: str | Path | None = None,
                 resolution: int = 512, scale: int = 4) -> None:
        self.hr_root = Path(hr_directory)
        self.hr_paths = image_paths(self.hr_root)
        self.resolution = resolution
        self.scale = scale
        if resolution <= 0 or scale <= 0 or resolution % scale:
            raise ValueError("Crop resolution must be positive and divisible by the scale.")
        self.lr_paths = None
        if lr_directory is not None:
            lr_root = Path(lr_directory)
            lookup = {}
            for path in image_paths(lr_root):
                key = path.relative_to(lr_root).with_suffix("")
                if key in lookup:
                    raise ValueError(f"Ambiguous LR filename stem: {key}")
                lookup[key] = path
            keys = [path.relative_to(self.hr_root).with_suffix("") for path in self.hr_paths]
            if any(key not in lookup for key in keys):
                raise ValueError("Every HR image must have a matching relative LR filename stem.")
            self.lr_paths = [lookup[key] for key in keys]

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        hr = load_image(self.hr_paths[index])
        lr = load_image(self.lr_paths[index]) if self.lr_paths else None
        if lr is not None and tuple(hr.shape[-2:]) != tuple(size * self.scale for size in lr.shape[-2:]):
            raise ValueError(f"Unaligned pair dimensions: {self.hr_paths[index]}")
        height, width = hr.shape[-2:]
        if min(height, width) < self.resolution:
            raise ValueError(f"HR image is smaller than the requested crop: {self.hr_paths[index]}")
        if lr is not None:
            crop = self.resolution // self.scale
            top = int(torch.randint(lr.shape[-2] - crop + 1, (1,)))
            left = int(torch.randint(lr.shape[-1] - crop + 1, (1,)))
            lr = lr[:, top:top + crop, left:left + crop]
            top, left = top * self.scale, left * self.scale
        else:
            top = int(torch.randint(height - self.resolution + 1, (1,)))
            left = int(torch.randint(width - self.resolution + 1, (1,)))
        hr = hr[:, top:top + self.resolution, left:left + self.resolution]
        if torch.rand(()) < 0.5:
            hr = hr.flip(-1)
            if lr is not None:
                lr = lr.flip(-1)
        result = {"hr": hr}
        if lr is not None:
            result["lr"] = lr
        return result
