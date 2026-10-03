"""Generate fixed-resolution SR images using K2NSR coarse-scale prefill.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
import dist

from models.k2nsr import GenerationConfig, K2NSRConfig
from models.k2nsr.config import load_experiment
from models.k2nsr.checkpoint import build_k2nsr, build_var_backend
from models.k2nsr.data import image_paths, load_image
from inference import K2NSRPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional experiment JSON; checkpoint architecture is used by default")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--var-checkpoint", required=True)
    parser.add_argument("--legacy-p5", action="store_true", help="Explicitly import the old P5-500M checkpoint")
    parser.add_argument("--num-prefill-scales", type=int)
    parser.add_argument("--continuous-prefix", action="store_true")
    parser.add_argument("--var-depth", type=int, default=24)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    experiment = load_experiment(args.config)
    config = K2NSRConfig.from_dict(experiment["model"]) if "model" in experiment else None
    generation = GenerationConfig(**experiment.get("generation", {}))
    if args.num_prefill_scales is not None:
        generation = replace(generation, num_prefill_scales=args.num_prefill_scales)
    if args.continuous_prefix:
        generation = replace(generation, quantize_prefix=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        dist.set_gpu_id(device.index or 0)
    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("Choose an output directory outside the input tree.")
    paths = image_paths(input_root)
    outputs = [output_root / path.relative_to(input_root).with_suffix(".png") for path in paths]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Input filename stems collide when converted to PNG.")
    model = build_k2nsr(args.vae_checkpoint, config, model_path=args.checkpoint,
                       legacy_p5=args.legacy_p5, device=device)
    var = build_var_backend(model, args.var_checkpoint, depth=args.var_depth)
    pipeline = K2NSRPipeline(model, var, precision=args.precision)
    for source, destination in zip(paths, outputs):
        result = pipeline.generate(load_image(source).unsqueeze(0), generation)[0]
        array = result.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(array)).save(destination)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
