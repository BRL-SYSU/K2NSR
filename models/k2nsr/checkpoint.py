"""Versioned K2NSR checkpoints and explicit legacy P5 imports.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import torch

from .config import K2NSRConfig
from .model import K2NSR

FORMAT = "k2nsr"
VERSION = 1


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    # Checkpoints must come from a trusted source because upstream files contain Python metadata.
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("Expected a checkpoint dictionary.")
    return value


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_k2nsr(vae_path: str | Path, config: K2NSRConfig | None = None,
                *, model_path: str | Path | None = None, legacy_p5: bool = False,
                device: str | torch.device = "cpu") -> K2NSR:
    from models.vqvae import VQVAE
    checkpoint = read_checkpoint(model_path) if model_path else None
    if checkpoint is not None and not legacy_p5:
        if checkpoint.get("format") != FORMAT or checkpoint.get("version") != VERSION:
            raise ValueError("Unsupported K2NSR checkpoint; use explicit legacy P5 import when appropriate.")
        saved_config = K2NSRConfig.from_dict(checkpoint["config"])
        if config is not None and config != saved_config:
            raise ValueError("Supplied architecture does not match the checkpoint configuration.")
        config = saved_config
        if checkpoint.get("vae_sha256") != file_digest(vae_path):
            raise ValueError("The base VAE checkpoint differs from the one used during training.")
    config = config or K2NSRConfig()
    vae_checkpoint = read_checkpoint(vae_path)
    trainer = vae_checkpoint.get("trainer", {})
    if "vae_local" not in trainer:
        raise ValueError("Expected the upstream VQVAE checkpoint key trainer.vae_local.")
    saved_sizes = trainer.get("config", {}).get("patch_nums")
    if saved_sizes is not None and tuple(saved_sizes) != config.patch_sizes:
        raise ValueError("The VAE checkpoint scale schedule differs from the model configuration.")
    vae = VQVAE(vocab_size=config.vocabulary_size, z_channels=config.reconstructor.channels,
                ch=config.vae_channels, share_quant_resi=config.shared_quant_residuals,
                v_patch_nums=config.patch_sizes, test_mode=True)
    vae.load_state_dict(trainer["vae_local"], strict=True)
    model = K2NSR(vae, config)
    if checkpoint is not None:
        if legacy_p5:
            import_legacy_p5(model, checkpoint)
        else:
            model.reconstructor.load_state_dict(checkpoint["reconstructor"], strict=True)
            model.lr_encoder.load_state_dict(checkpoint["lr_encoder"], strict=True)
            model.lr_quant_conv.load_state_dict(checkpoint["lr_quant_conv"], strict=True)
    return model.to(device)


def import_legacy_p5(model: K2NSR, checkpoint: dict[str, Any]) -> None:
    """Map the old residual transformation state into the new scale-branch layout."""
    if tuple(checkpoint.get("patch_nums", ())) != model.config.reconstructor.patch_sizes:
        raise ValueError("Legacy and configured reconstruction scales differ.")
    if "transformation" not in checkpoint:
        raise ValueError("The legacy checkpoint has no transformation state.")
    mapped = {}
    branches = {"feature_transforms": "feature_transform", "transformers": "transformer",
                "output_projs": "output_projection"}
    for key, value in checkpoint["transformation"].items():
        if key.startswith("vae_student."):
            continue
        if key == "map_transformer.alpha_input":
            mapped["input_residual_scale"] = value.reshape(-1)
        elif key == "map_transformer.alpha_feat":
            mapped["feature_residual_scale"] = value.reshape(-1)
        elif key.startswith("map_transformer.base.feature_extractor."):
            mapped[key.removeprefix("map_transformer.base.")] = value
        elif key.startswith("map_transformer.base."):
            parts = key.removeprefix("map_transformer.base.").split(".", 2)
            if len(parts) != 3 or parts[0] not in branches:
                raise ValueError(f"Unsupported legacy state key: {key}")
            mapped[f"branches.{parts[1]}.{branches[parts[0]]}.{parts[2]}"] = value
        else:
            raise ValueError(f"Unsupported legacy state key: {key}")
    model.reconstructor.load_state_dict(mapped, strict=True)
    if checkpoint.get("vae_student_encoder") is not None:
        model.lr_encoder.load_state_dict(checkpoint["vae_student_encoder"], strict=True)
    if checkpoint.get("vae_student_quant_conv") is not None:
        model.lr_quant_conv.load_state_dict(checkpoint["vae_student_quant_conv"], strict=True)


def save_k2nsr_checkpoint(path: str | Path, model: K2NSR, *, vae_sha256: str,
                         training_state: dict[str, Any] | None = None) -> None:
    payload = {"format": FORMAT, "version": VERSION, "author": "HongyiFang",
               "date": "2025-12-11", "config": model.config.to_dict(), "vae_sha256": vae_sha256,
               "reconstructor": model.reconstructor.state_dict(),
               "lr_encoder": model.lr_encoder.state_dict(),
               "lr_quant_conv": model.lr_quant_conv.state_dict(),
               "training": training_state or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_var_backend(model: K2NSR, path: str | Path, *, depth: int = 24):
    from models.var import VAR_RoPE
    device = next(model.parameters()).device
    var = VAR_RoPE(vae_local=model.vae, num_classes=2, depth=depth,
                   controlnet_depth=depth, embed_dim=depth * 64, num_heads=depth,
                   patch_nums=model.config.patch_sizes, attn_l2_norm=True,
                   flash_if_available=False, fused_if_available=False).to(device)
    checkpoint = read_checkpoint(path)
    try:
        state = checkpoint["trainer"]["var_wo_ddp"]
    except KeyError as error:
        raise ValueError("Expected upstream VAR weights at trainer.var_wo_ddp.") from error
    var.load_state_dict(state, strict=True)
    return var.requires_grad_(False).eval()
