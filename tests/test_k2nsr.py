"""Numerical and state-lifecycle checks with tiny CPU models.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.k2nsr import (EncoderConfig, GenerationConfig, K2NSR, K2NSRConfig, K2NSRLoss,
                          LossConfig, ParallelReconstructor, PrefixOutput, ReconstructorConfig,
                          ReconstructionOutput, ReconstructionTargets, ScaleConfig)
from models.k2nsr.checkpoint import import_legacy_p5, read_checkpoint, save_k2nsr_checkpoint
from models.k2nsr.losses import codebook_logits
from inference.prefill import build_prefix
from models.var import VAR_RoPE


def tiny_config() -> K2NSRConfig:
    return K2NSRConfig(
        reconstructor=ReconstructorConfig(scales=(ScaleConfig(1, 8, 2, 2), ScaleConfig(2, 8, 2, 2)),
                                          channels=4, latent_size=4, feature_channels=(4, 8), dropout=0),
        patch_sizes=(1, 2, 4), vocabulary_size=8, vae_channels=4,
    )


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 4, 1)
        self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(4, 4, 1), nn.GroupNorm(2, 4))])
        self.mid = nn.Conv2d(4, 4, 1)

    def forward(self, image):
        return self.mid(self.down[0](self.conv_in(image)))


class TinyQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.v_patch_nums = (1, 2, 4)

    def get_next_autoregressive_input(self, index, count, latent, embedding):
        latent = latent + F.interpolate(embedding, size=(4, 4), mode="bicubic")
        size = self.v_patch_nums[min(index + 1, count - 1)]
        return latent, F.adaptive_avg_pool2d(latent, size)


class TinyVAE(nn.Module):
    Cvae = 4
    vocab_size = 8
    downsample = 1

    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.quant_conv = nn.Conv2d(4, 4, 1)
        self.quantize = TinyQuantizer()

    def img_to_idxBl(self, image):
        latent = self.quant_conv(self.encoder(image))
        indices = [codebook_logits(F.adaptive_avg_pool2d(latent, size), self.quantize.embedding.weight).argmax(-1)
                   for size in self.quantize.v_patch_nums]
        return [], indices

    def fhat_to_img(self, latent):
        return latent[:, :3].tanh()


class Cache(nn.Module):
    def __init__(self):
        super().__init__()
        self.caching = False

    def kv_caching(self, enabled):
        self.caching = enabled


class RecordingBlock(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.attn = Cache()
        self.lengths = []
        self.fail = fail

    def forward(self, x, cond_BD, freqs_cis, attn_bias):
        if self.fail:
            raise RuntimeError("Injected block failure")
        self.lengths.append(x.shape[1])
        return x + cond_BD.unsqueeze(1) * 0.1


class TinyCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 4, 1)

    def forward(self, image):
        return self.projection(F.adaptive_avg_pool2d(image, 4)), None


class TinyRefiner(nn.Module):
    def sample(self, z, temperature, cfg):
        return torch.randn(z.shape[0], 4, device=z.device) * 0.01


class TinyVAR(VAR_RoPE):
    """Exercise the real generation loop without allocating a pretrained backbone."""

    def __init__(self, fail=False):
        nn.Module.__init__(self)
        self.patch_nums = (1, 2, 4)
        self.Cvae = self.C = 4
        self.num_classes = 2
        self.num_stages_minus_1 = 2
        self.context_token = self.first_l = 17
        self.label_B_flag = True
        self.con_embedding = TinyCondition()
        self.class_emb = nn.Embedding(2, 4)
        self.pos_start = nn.Parameter(torch.zeros(1, 17, 4))
        self.lvl_embed = nn.Embedding(3, 4)
        self.register_buffer("lvl_1L", torch.tensor([[0] * 17 + [1] * 4 + [2] * 16]))
        self.freqs_cis = torch.ones(37, 2)
        self.shared_ada_lin = nn.Identity()
        self.word_embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([RecordingBlock(fail)])
        self.head = nn.Linear(4, 8)
        self.decoder_norm = nn.LayerNorm(4)
        self.diffloss = TinyRefiner()
        vae = TinyVAE()
        self.vae_proxy = (vae,)
        self.vae_quant_proxy = (vae.quantize,)

    def get_logits(self, hidden, conditioning):
        return self.head(hidden)


class K2NSRChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_architecture_roundtrip_and_parameter_budget(self):
        config = K2NSRConfig()
        self.assertEqual(config, K2NSRConfig.from_dict(config.to_dict()))
        with torch.device("meta"):
            model = ParallelReconstructor()
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertTrue(450_000_000 < count < 550_000_000, count)

    def test_shapes_gradients_and_independent_layer_initialization(self):
        model = ParallelReconstructor(tiny_config().reconstructor)
        latent = torch.randn(2, 4, 4, 4, requires_grad=True)
        output = model(latent)
        self.assertEqual([tuple(t.shape) for t in output.features], [(2, 4, 1, 1), (2, 4, 2, 2)])
        sum(t.square().mean() for t in output.features).backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        layers = model.branches[0].transformer.layers
        self.assertFalse(torch.equal(layers[0].linear1.weight, layers[1].linear1.weight))

    def test_loss_matches_old_objective_and_detaches_teacher(self):
        predictions = tuple(torch.randn(2, 4, size, size, requires_grad=True) for size in (1, 2))
        indices = (torch.randint(8, (2, 1)), torch.randint(8, (2, 4)))
        codebook = torch.randn(8, 4, requires_grad=True)
        target_features = tuple(F.embedding(index, codebook).transpose(1, 2).reshape(2, 4, size, size)
                                for size, index in zip((1, 2), indices))
        prediction = ReconstructionOutput((1, 2), predictions)
        target = ReconstructionTargets((1, 2), target_features, indices)
        config = LossConfig()
        result = K2NSRLoss(config)(prediction, target, codebook)
        residual = torch.stack([F.smooth_l1_loss(p, t.detach()) for p, t in zip(predictions, target_features)]).mean()
        classification = torch.stack([F.cross_entropy(codebook_logits(p, codebook, scale=20).reshape(-1, 8), idx.flatten())
                                      for p, idx in zip(predictions, indices)]).mean()
        cumulative_prediction = sum(F.interpolate(p, (2, 2), mode="bilinear", align_corners=False) for p in predictions)
        cumulative_target = sum(F.interpolate(t.detach(), (2, 2), mode="bilinear", align_corners=False) for t in target_features)
        expected = residual + 0.001 * classification + 0.2 * F.smooth_l1_loss(cumulative_prediction, cumulative_target)
        torch.testing.assert_close(result.total, expected)
        result.total.backward()
        self.assertIsNone(codebook.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictions))

    def test_prefix_nearest_codebook_and_bounds(self):
        feature = torch.randn(2, 4, 2, 2)
        prediction = ReconstructionOutput((1, 2), (feature[:, :, :1, :1], feature))
        codebook = torch.randn(8, 4)
        prefix = build_prefix(prediction, codebook, GenerationConfig(num_prefill_scales=2))
        tokens = feature.flatten(2).transpose(1, 2)
        indices = (tokens.unsqueeze(-2) - codebook).square().sum(-1).argmin(-1)
        expected = F.embedding(indices, codebook).transpose(1, 2).reshape_as(feature)
        torch.testing.assert_close(prefix.embeddings[1], expected)
        with self.assertRaises(ValueError):
            build_prefix(prediction, codebook, GenerationConfig(num_prefill_scales=3))
        self.assertEqual(build_prefix(prediction, codebook, GenerationConfig(num_prefill_scales=0)).start_scale_index, 0)

    def test_teacher_freeze_and_target_contract(self):
        model = K2NSR(TinyVAE(), tiny_config()).train()
        self.assertFalse(model.vae.training)
        self.assertFalse(any(p.requires_grad for p in model.vae.parameters()))
        self.assertTrue(model.lr_encoder.conv_in.weight.requires_grad)
        self.assertFalse(model.lr_encoder.mid.weight.requires_grad)
        targets = model.encode_targets(torch.randn(2, 3, 4, 4))
        self.assertEqual(targets.patch_sizes, (1, 2))
        self.assertFalse(any(t.requires_grad for t in targets.features))
        model(torch.randn(2, 3, 2, 2)).features[-1].sum().backward()
        self.assertIsNotNone(model.lr_encoder.conv_in.weight.grad)

    def test_explicit_legacy_conversion_and_native_save(self):
        source = K2NSR(TinyVAE(), tiny_config())
        legacy = {}
        reverse = {"feature_transform": "feature_transforms", "transformer": "transformers",
                   "output_projection": "output_projs"}
        for key, value in source.reconstructor.state_dict().items():
            if key == "input_residual_scale":
                old = "map_transformer.alpha_input"
            elif key == "feature_residual_scale":
                old = "map_transformer.alpha_feat"
            elif key.startswith("branches."):
                _, index, branch, rest = key.split(".", 3)
                old = f"map_transformer.base.{reverse[branch]}.{index}.{rest}"
            else:
                old = "map_transformer.base." + key
            legacy[old] = value
        destination = K2NSR(TinyVAE(), tiny_config())
        import_legacy_p5(destination, {"patch_nums": (1, 2), "transformation": legacy})
        for key, value in source.reconstructor.state_dict().items():
            torch.testing.assert_close(destination.reconstructor.state_dict()[key], value)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_k2nsr_checkpoint(path, destination, vae_sha256="test")
            saved = read_checkpoint(path)
            self.assertEqual((saved["format"], saved["version"], saved["author"]), ("k2nsr", 1, "HongyiFang"))
            self.assertNotIn("vae", saved)

    def test_var_prefill_refinement_and_rng_isolation(self):
        var = TinyVAR().eval()
        condition = torch.randn(2, 3, 4, 4)
        prefix = PrefixOutput((1, 2), (torch.zeros(2, 4, 1, 1), torch.zeros(2, 4, 2, 2)))
        config = GenerationConfig(num_prefill_scales=2)
        state = torch.get_rng_state().clone()
        output = var.generate_from_prefix(condition, prefix, config)
        self.assertEqual(tuple(output.shape), (2, 3, 4, 4))
        self.assertEqual(var.blocks[0].lengths, [17, 4, 16])
        self.assertFalse(var.blocks[0].attn.caching)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(output, var.generate_from_prefix(condition, prefix, config))

    def test_baseline_wrapper_uses_same_generation_and_cache_cleans_on_error(self):
        var = TinyVAR().eval()
        condition = torch.randn(1, 3, 4, 4)
        config = GenerationConfig(num_prefill_scales=0, cfg=2, seed=42)
        reference = var.generate_from_prefix(condition, PrefixOutput((), ()), config)
        actual = var.autoregressive_infer_cfg(1, None, condition, None, torch.zeros(1, dtype=torch.long),
                                             g_seed=42, cfg=2, top_k=1, top_p=0.75)
        torch.testing.assert_close(reference, actual)
        failing = TinyVAR(fail=True)
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            failing.generate_from_prefix(condition, PrefixOutput((), ()), config)
        self.assertFalse(failing.blocks[0].attn.caching)


if __name__ == "__main__":
    unittest.main()
