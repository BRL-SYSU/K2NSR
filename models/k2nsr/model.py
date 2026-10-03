"""K2NSR reconstruction model and frozen teacher supervision.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from .config import K2NSRConfig
from .parallel_reconstructor import ParallelReconstructor
from .types import ReconstructionOutput, ReconstructionTargets


class K2NSR(nn.Module):
    """Keep the teacher VAE frozen while adapting a separate LR encoder."""

    def __init__(self, vae: nn.Module, config: K2NSRConfig | None = None) -> None:
        super().__init__()
        self.config = config or K2NSRConfig()
        if tuple(vae.quantize.v_patch_nums) != self.config.patch_sizes:
            raise ValueError("VAE and K2NSR scale schedules differ.")
        if vae.Cvae != self.config.reconstructor.channels or vae.vocab_size != self.config.vocabulary_size:
            raise ValueError("VAE and reconstructor latent dimensions differ.")
        self.vae = vae.requires_grad_(False).eval()
        self.lr_encoder = deepcopy(vae.encoder)
        self.lr_quant_conv = deepcopy(vae.quant_conv)
        self.reconstructor = ParallelReconstructor(self.config.reconstructor)
        self._configure_encoder()

    def _configure_encoder(self) -> None:
        self.lr_encoder.requires_grad_(False)
        self.lr_quant_conv.requires_grad_(self.config.encoder.train_quant_conv)
        policy = self.config.encoder
        if policy.train_input_conv:
            self.lr_encoder.conv_in.requires_grad_(True)
        for level in policy.train_down_levels:
            if not 0 <= level < len(self.lr_encoder.down):
                raise ValueError(f"Invalid encoder downsampling level: {level}")
            self.lr_encoder.down[level].requires_grad_(True)
        if policy.train_mid:
            self.lr_encoder.mid.requires_grad_(True)
        if policy.train_norm_affine:
            for module in self.lr_encoder.modules():
                if isinstance(module, nn.GroupNorm):
                    module.requires_grad_(True)

    def train(self, mode: bool = True) -> K2NSR:
        super().train(mode)
        self.vae.eval()
        if not any(p.requires_grad for p in self.lr_encoder.parameters()):
            self.lr_encoder.eval()
        return self

    @property
    def codebook(self) -> torch.Tensor:
        return self.vae.quantize.embedding.weight.detach()

    def forward(self, lr: torch.Tensor) -> ReconstructionOutput:
        return self.reconstruct(lr)

    def reconstruct(self, lr: torch.Tensor) -> ReconstructionOutput:
        if lr.ndim != 4 or lr.shape[1] != 3 or min(lr.shape[-2:]) < self.vae.downsample:
            raise ValueError("LR images must be BCHW RGB tensors large enough for the VAE encoder.")
        latent = self.lr_quant_conv(self.lr_encoder(lr))
        size = self.config.reconstructor.latent_size
        if latent.shape[-2:] != (size, size):
            latent = F.interpolate(latent, size=(size, size), mode="bilinear", align_corners=False)
        return self.reconstructor(latent)

    @torch.no_grad()
    def encode_targets(self, hr: torch.Tensor) -> ReconstructionTargets:
        resolution = self.config.patch_sizes[-1] * self.vae.downsample
        if hr.ndim != 4 or hr.shape[1:] != (3, resolution, resolution):
            raise ValueError(f"HR images must have shape [B, 3, {resolution}, {resolution}].")
        self.vae.eval()
        # This repository's VAE returns both residual metadata and per-scale indices.
        _, all_indices = self.vae.img_to_idxBl(hr.float())
        sizes = self.config.reconstructor.patch_sizes
        indices = tuple(all_indices[:len(sizes)])
        features = tuple(
            F.embedding(index, self.codebook).transpose(1, 2).reshape(hr.shape[0], -1, size, size)
            for size, index in zip(sizes, indices)
        )
        return ReconstructionTargets(sizes, features, indices)

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)
