"""End-to-end fixed-resolution K2NSR super-resolution inference.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F

from models.k2nsr import GenerationConfig, K2NSR, PrefixOutput
from .prefill import build_prefix


class K2NSRPipeline(nn.Module):
    """Accept RGB images in [0, 1] and return fixed-resolution RGB images in [0, 1]."""

    def __init__(self, model: K2NSR, var: nn.Module, *, precision: str = "fp32") -> None:
        super().__init__()
        if tuple(var.patch_nums) != model.config.patch_sizes:
            raise ValueError("VAR and K2NSR scale schedules differ.")
        if var.vae_proxy[0] is not model.vae:
            raise ValueError("VAR and K2NSR must share the same frozen VAE/codebook.")
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Precision must be fp32, fp16, or bf16.")
        self.model = model.eval()
        self.var = var.requires_grad_(False).eval()
        self.precision = precision

    @torch.inference_mode()
    def generate(self, lr: torch.Tensor, generation_config: GenerationConfig | None = None) -> torch.Tensor:
        config = generation_config or GenerationConfig()
        if lr.ndim != 4 or lr.shape[1] != 3 or not lr.is_floating_point():
            raise ValueError("LR images must be floating-point BCHW RGB tensors.")
        if lr.shape[0] == 0 or not torch.isfinite(lr).all() or lr.min() < 0 or lr.max() > 1:
            raise ValueError("LR images must be nonempty, finite and in [0, 1].")
        if config.num_prefill_scales >= len(self.model.config.patch_sizes):
            raise ValueError("At least one VAR scale must remain for autoregressive generation.")
        device = next(self.model.parameters()).device
        if self.precision == "fp16" and device.type != "cuda":
            raise ValueError("FP16 inference requires CUDA; use fp32 on CPU.")
        self.eval()
        lr = lr.to(device=device, dtype=torch.float32)
        normalized = lr * 2 - 1
        resolution = self.model.config.patch_sizes[-1] * self.model.vae.downsample
        condition = F.interpolate(normalized, (resolution, resolution), mode="bicubic", align_corners=False).clamp(-1, 1)
        lr_size = resolution // 4
        reconstruction_input = F.interpolate(normalized, (lr_size, lr_size), mode="bicubic", align_corners=False).clamp(-1, 1)
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        context = torch.autocast(device.type, dtype=dtype) if self.precision != "fp32" else nullcontext()
        with context:
            if config.num_prefill_scales:
                predictions = self.model.reconstruct(reconstruction_input)
                prefix = build_prefix(predictions, self.model.codebook, config)
            else:
                prefix = PrefixOutput((), ())
            return self.var.generate_from_prefix(condition, prefix, config).float().clamp(0, 1)
