"""Parallel coarse-scale reconstruction with independent scale branches.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ReconstructorConfig, ScaleConfig
from .types import ReconstructionOutput


class ScaleBranch(nn.Module):
    """Project shared latent features to one scale without scale recurrence."""

    def __init__(self, scale: ScaleConfig, config: ReconstructorConfig) -> None:
        super().__init__()
        self.patch_size = scale.patch_size
        self.feature_transform = nn.Sequential(
            nn.Conv2d(config.channels, scale.width // 2, 3, padding=1),
            nn.LayerNorm([scale.width // 2, config.latent_size, config.latent_size]),
            nn.GELU(),
            nn.Conv2d(scale.width // 2, scale.width, 3, padding=1),
            nn.LayerNorm([scale.width, config.latent_size, config.latent_size]),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=scale.width, nhead=scale.heads,
            dim_feedforward=int(scale.width * config.mlp_ratio),
            dropout=config.dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, scale.layers, enable_nested_tensor=False)
        self.output_projection = nn.Sequential(
            nn.Linear(scale.width, scale.width // 2), nn.LayerNorm(scale.width // 2),
            nn.GELU(), nn.Linear(scale.width // 2, config.channels),
        )
        # TransformerEncoder clones one layer; initialize each clone independently.
        for block in self.transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        grid = F.adaptive_avg_pool2d(self.feature_transform(features), self.patch_size)
        tokens = grid.flatten(2).transpose(1, 2)
        tokens = self.output_projection(self.transformer(tokens))
        return tokens.transpose(1, 2).reshape(
            features.shape[0], -1, self.patch_size, self.patch_size,
        ).contiguous()


class ParallelReconstructor(nn.Module):
    """Return a BCHW feature map for every configured coarse scale."""

    def __init__(self, config: ReconstructorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReconstructorConfig()
        c = self.config.channels
        mid1, mid2 = self.config.feature_channels
        grid = self.config.latent_size
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, mid1, 3, padding=1), nn.LayerNorm([mid1, grid, grid]), nn.GELU(),
            nn.Conv2d(mid1, mid2, 3, padding=1), nn.LayerNorm([mid2, grid, grid]), nn.GELU(),
            nn.Conv2d(mid2, c, 3, padding=1), nn.LayerNorm([c, grid, grid]), nn.GELU(),
        )
        self.branches = nn.ModuleList(ScaleBranch(scale, self.config) for scale in self.config.scales)
        count = len(self.branches) if self.config.residual_per_scale else 1
        self.input_residual_scale = nn.Parameter(torch.full((count,), self.config.input_residual_init))
        self.feature_residual_scale = nn.Parameter(torch.full((count,), self.config.feature_residual_init))

    def forward(self, lr_latent: torch.Tensor) -> ReconstructionOutput:
        expected = (self.config.channels, self.config.latent_size, self.config.latent_size)
        if lr_latent.ndim != 4 or tuple(lr_latent.shape[1:]) != expected:
            raise ValueError(f"Expected latent shape [B, {expected}], got {tuple(lr_latent.shape)}.")
        features = self.feature_extractor(lr_latent)
        outputs = []
        for index, branch in enumerate(self.branches):
            gate = index if self.config.residual_per_scale else 0
            size = branch.patch_size
            outputs.append(
                branch(features)
                + self.input_residual_scale[gate] * F.adaptive_avg_pool2d(lr_latent, size)
                + self.feature_residual_scale[gate] * F.adaptive_avg_pool2d(features, size)
            )
        return ReconstructionOutput(self.config.patch_sizes, tuple(outputs))
