"""K2NSR configuration contracts.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


def validate_scales(scales: tuple[int, ...]) -> None:
    if not scales or any(type(size) is not int or size < 1 for size in scales):
        raise ValueError("Patch sizes must be positive integers.")
    if tuple(sorted(set(scales))) != scales:
        raise ValueError("Patch sizes must be strictly increasing.")


@dataclass(frozen=True)
class ScaleConfig:
    patch_size: int
    width: int
    layers: int
    heads: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (self.patch_size, self.width, self.layers, self.heads)):
            raise ValueError("Scale dimensions, layers and heads must be positive.")
        if self.width % self.heads or self.width % 2:
            raise ValueError("Scale width must be even and divisible by the head count.")


def p5_500m_scales() -> tuple[ScaleConfig, ...]:
    """Preserve the architecture of the original five-scale 500M experiment."""
    return tuple(ScaleConfig(*values) for values in (
        (1, 384, 4, 8), (2, 448, 6, 8), (3, 768, 7, 8),
        (4, 864, 8, 8), (6, 1728, 9, 16),
    ))


@dataclass(frozen=True)
class ReconstructorConfig:
    scales: tuple[ScaleConfig, ...] = field(default_factory=p5_500m_scales)
    channels: int = 32
    latent_size: int = 32
    feature_channels: tuple[int, int] = (128, 256)
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    input_residual_init: float = 0.1
    feature_residual_init: float = 0.1
    residual_per_scale: bool = True

    @property
    def patch_sizes(self) -> tuple[int, ...]:
        return tuple(scale.patch_size for scale in self.scales)

    def __post_init__(self) -> None:
        validate_scales(self.patch_sizes)
        if self.channels < 1 or self.latent_size < self.patch_sizes[-1]:
            raise ValueError("Invalid channel count or latent size.")
        if len(self.feature_channels) != 2 or min(self.feature_channels) < 1:
            raise ValueError("Exactly two positive feature channel counts are required.")
        if not 0 <= self.dropout < 1 or not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0:
            raise ValueError("Invalid dropout or MLP ratio.")


@dataclass(frozen=True)
class EncoderConfig:
    train_input_conv: bool = True
    train_down_levels: tuple[int, ...] = (0,)
    train_norm_affine: bool = True
    train_mid: bool = False
    train_quant_conv: bool = False


@dataclass(frozen=True)
class K2NSRConfig:
    reconstructor: ReconstructorConfig = field(default_factory=ReconstructorConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    patch_sizes: tuple[int, ...] = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    vocabulary_size: int = 4096
    vae_channels: int = 160
    shared_quant_residuals: int = 4

    def __post_init__(self) -> None:
        validate_scales(self.patch_sizes)
        predicted = self.reconstructor.patch_sizes
        if self.patch_sizes[:len(predicted)] != predicted:
            raise ValueError("Reconstructor scales must be a prefix of the VAE/VAR schedule.")
        if self.patch_sizes[-1] != self.reconstructor.latent_size:
            raise ValueError("The final VAE scale must equal the reconstructor latent size.")
        if min(self.vocabulary_size, self.vae_channels, self.shared_quant_residuals) < 1:
            raise ValueError("VAE dimensions must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> K2NSRConfig:
        values = dict(values)
        reconstructor = dict(values.pop("reconstructor", {}))
        if "scales" in reconstructor:
            reconstructor["scales"] = tuple(ScaleConfig(**scale) for scale in reconstructor["scales"])
        if "feature_channels" in reconstructor:
            reconstructor["feature_channels"] = tuple(reconstructor["feature_channels"])
        encoder = dict(values.pop("encoder", {}))
        if "train_down_levels" in encoder:
            encoder["train_down_levels"] = tuple(encoder["train_down_levels"])
        if "patch_sizes" in values:
            values["patch_sizes"] = tuple(values["patch_sizes"])
        return cls(reconstructor=ReconstructorConfig(**reconstructor),
                   encoder=EncoderConfig(**encoder), **values)


@dataclass(frozen=True)
class LossConfig:
    reconstruction_weight: float = 1.0
    codebook_weight: float = 0.001
    accumulation_weight: float = 0.2
    huber_delta: float | None = 1.0
    logit_scale: float = 20.0
    normalize_codebook: bool = False

    def __post_init__(self) -> None:
        weights = (self.reconstruction_weight, self.codebook_weight, self.accumulation_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError("Loss weights must be finite, nonnegative, and not all zero.")
        if self.huber_delta is not None and (not math.isfinite(self.huber_delta) or self.huber_delta <= 0):
            raise ValueError("Huber delta must be positive, or None for L1.")
        if not math.isfinite(self.logit_scale) or self.logit_scale <= 0:
            raise ValueError("Logit scale must be finite and positive.")


@dataclass(frozen=True)
class GenerationConfig:
    num_prefill_scales: int = 3
    quantize_prefix: bool = True
    cfg: float = 9.0
    top_k: int = 1
    top_p: float = 0.75
    seed: int | None = 42
    refine: bool = True

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in (self.num_prefill_scales, self.top_k)):
            raise ValueError("Prefill scale count and top-k must be nonnegative.")
        if not 0 <= self.top_p <= 1 or not math.isfinite(self.cfg) or self.cfg < 0:
            raise ValueError("Invalid sampling probability or guidance strength.")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2 ** 63):
            raise ValueError("Seed must be a nonnegative 63-bit integer or None.")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 80
    workers: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    precision: str = "fp32"
    degradation: str = "realesrgan"
    seed: int = 42
    log_every: int = 50
    save_every_epochs: int = 1

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.log_every, self.save_every_epochs) < 1 or self.workers < 0:
            raise ValueError("Training counts must be positive and worker count nonnegative.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("Invalid optimizer parameters.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Invalid training precision.")
        if self.degradation not in {"realesrgan", "bicubic"}:
            raise ValueError("Degradation must be realesrgan or bicubic.")


def load_experiment(path: str | None) -> dict[str, Any]:
    """Read a strict JSON configuration without machine-specific path defaults."""
    import json
    from pathlib import Path
    if path is None:
        return {}
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) - {"model", "loss", "training", "generation", "metadata"}:
        raise ValueError("Unsupported experiment configuration fields.")
    return values
