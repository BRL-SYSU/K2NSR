"""Scale-balanced reconstruction, codebook, and accumulation losses.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import LossConfig
from .types import LossOutput, ReconstructionOutput, ReconstructionTargets


def codebook_logits(features: torch.Tensor, codebook: torch.Tensor, *,
                    normalize: bool = False, scale: float = 1.0) -> torch.Tensor:
    """Return [B, pn*pn, vocabulary] scores using squared distance or cosine similarity."""
    if features.ndim != 4 or codebook.ndim != 2 or features.shape[1] != codebook.shape[1]:
        raise ValueError("Feature channels must match the codebook embedding dimension.")
    tokens = features.float().flatten(2).transpose(1, 2)
    codes = codebook.detach().float()
    if normalize:
        logits = F.normalize(tokens, dim=-1) @ F.normalize(codes, dim=-1).t()
    else:
        # The omitted negative token norm is constant across vocabulary entries.
        logits = 2 * (tokens @ codes.t()) - codes.square().sum(dim=-1)
    return logits * scale


class K2NSRLoss(nn.Module):
    """Match the old P5 objective without coupling it to the model or trainer."""

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def _distance(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.huber_delta is None:
            return F.l1_loss(prediction, target)
        return F.smooth_l1_loss(prediction, target, beta=self.config.huber_delta)

    def forward(self, predictions: ReconstructionOutput, targets: ReconstructionTargets,
                codebook: torch.Tensor) -> LossOutput:
        if predictions.patch_sizes != targets.patch_sizes:
            raise ValueError("Prediction and target scale schedules must match.")
        if any(p.shape != t.shape for p, t in zip(predictions.features, targets.features)):
            raise ValueError("Prediction and target feature shapes must match.")
        reconstruction = []
        classification = []
        # Keep distance and vocabulary operations in FP32 under mixed precision.
        with torch.autocast(device_type=predictions.features[0].device.type, enabled=False):
            for prediction, target, indices in zip(predictions.features, targets.features, targets.code_indices):
                reconstruction.append(self._distance(prediction.float(), target.detach().float()))
                if self.config.codebook_weight:
                    logits = codebook_logits(prediction, codebook,
                        normalize=self.config.normalize_codebook, scale=self.config.logit_scale)
                    classification.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), indices.reshape(-1)))
            residual = torch.stack(reconstruction).mean()
            classification_loss = torch.stack(classification).mean() if classification else residual.new_zeros(())
            accumulation = residual.new_zeros(())
            if self.config.accumulation_weight:
                size = predictions.patch_sizes[-1]
                accumulated_prediction = sum(F.interpolate(p.float(), (size, size), mode="bilinear", align_corners=False)
                                             for p in predictions.features)
                accumulated_target = sum(F.interpolate(t.detach().float(), (size, size), mode="bilinear", align_corners=False)
                                         for t in targets.features)
                accumulation = self._distance(accumulated_prediction, accumulated_target)
            total = (self.config.reconstruction_weight * residual
                     + self.config.codebook_weight * classification_loss
                     + self.config.accumulation_weight * accumulation)
        return LossOutput(total, {"reconstruction": residual, "codebook": classification_loss,
                                  "accumulation": accumulation})
