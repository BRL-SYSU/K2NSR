"""Convert coarse reconstruction outputs into validated VAR prefixes.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from models.k2nsr.config import GenerationConfig
from models.k2nsr.losses import codebook_logits
from models.k2nsr.types import PrefixOutput, ReconstructionOutput


@torch.no_grad()
def build_prefix(predictions: ReconstructionOutput, codebook: torch.Tensor,
                 config: GenerationConfig) -> PrefixOutput:
    count = config.num_prefill_scales
    if count > len(predictions.patch_sizes):
        raise ValueError("Requested prefix is longer than the reconstructor schedule.")
    embeddings = []
    for size, feature in zip(predictions.patch_sizes[:count], predictions.features[:count]):
        if config.quantize_prefix:
            with torch.autocast(device_type=feature.device.type, enabled=False):
                indices = codebook_logits(feature, codebook).argmax(dim=-1)
                feature = F.embedding(indices, codebook).transpose(1, 2)
                feature = feature.reshape(indices.shape[0], -1, size, size)
            feature = feature.to(dtype=predictions.features[0].dtype)
        embeddings.append(feature)
    return PrefixOutput(predictions.patch_sizes[:count], tuple(embeddings))
