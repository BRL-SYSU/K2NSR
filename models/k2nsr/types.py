"""Validated tensor contracts shared by training and inference.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .config import validate_scales


@dataclass(frozen=True)
class ReconstructionOutput:
    patch_sizes: tuple[int, ...]
    features: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        validate_scales(self.patch_sizes)
        if len(self.features) != len(self.patch_sizes):
            raise ValueError("One feature map is required for each scale.")
        first = self.features[0]
        if first.ndim != 4 or not first.is_floating_point():
            raise ValueError("Features must be floating-point BCHW tensors.")
        for size, feature in zip(self.patch_sizes, self.features):
            if feature.shape != (*first.shape[:2], size, size):
                raise ValueError(f"Invalid feature shape at scale {size}: {tuple(feature.shape)}")
            if feature.device != first.device or feature.dtype != first.dtype:
                raise ValueError("All feature maps must share a device and dtype.")


@dataclass(frozen=True)
class ReconstructionTargets(ReconstructionOutput):
    code_indices: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.code_indices) != len(self.patch_sizes):
            raise ValueError("One code-index tensor is required for each target scale.")
        for size, indices in zip(self.patch_sizes, self.code_indices):
            if indices.shape != (self.features[0].shape[0], size * size) or indices.dtype != torch.long:
                raise ValueError("Code indices must be long tensors with shape [B, pn*pn].")
            if indices.device != self.features[0].device:
                raise ValueError("Targets and code indices must share a device.")


@dataclass(frozen=True)
class PrefixOutput:
    patch_sizes: tuple[int, ...]
    embeddings: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if self.patch_sizes:
            ReconstructionOutput(self.patch_sizes, self.embeddings)
        elif self.embeddings:
            raise ValueError("An empty prefix must contain no embeddings.")

    @property
    def start_scale_index(self) -> int:
        return len(self.patch_sizes)


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
