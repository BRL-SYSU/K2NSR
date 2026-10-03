"""Train a parallel reconstructor with paired images or online degradation.

Author: HongyiFang
Date: 2025-12-11
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import os
import random
import sys

# Support both `python -m scripts.train_k2nsr` and direct script execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from models.k2nsr import K2NSRConfig, K2NSRLoss, LossConfig
from models.k2nsr.config import TrainingConfig, load_experiment
from models.k2nsr.checkpoint import build_k2nsr, file_digest, read_checkpoint, save_k2nsr_checkpoint
from models.k2nsr.data import TrainingImages


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


@torch.no_grad()
def prepare_batch(batch: dict, device: torch.device, degradation) -> tuple[torch.Tensor, torch.Tensor]:
    if "lr" in batch:
        hr, lr = batch["hr"].to(device), batch["lr"].to(device)
    elif degradation is None:
        hr = batch["hr"].to(device)
        lr = F.interpolate(hr, scale_factor=0.25, mode="bicubic", align_corners=False).clamp(0, 1)
    else:
        # The upstream adapter accepts one HWC numpy image and returns an aligned pair.
        pairs = [degradation.degrade_process(image.permute(1, 2, 0).numpy()) for image in batch["hr"]]
        hr = torch.cat([pair[0] for pair in pairs])
        lr = torch.cat([pair[1] for pair in pairs])
    return hr * 2 - 1, lr * 2 - 1


def train(args: argparse.Namespace) -> None:
    experiment = load_experiment(args.config)
    config = K2NSRConfig.from_dict(experiment.get("model", {}))
    settings = TrainingConfig(**experiment.get("training", {}))
    loss_config = LossConfig(**experiment.get("loss", {}))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if settings.precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training requires CUDA.")
    if world_size > 1:
        distributed.init_process_group(backend="nccl" if device.type == "cuda" and os.name != "nt" else "gloo")
    rank = distributed.get_rank() if distributed.is_initialized() else 0
    try:
        random.seed(settings.seed + rank)
        np.random.seed(settings.seed + rank)
        torch.manual_seed(settings.seed + rank)
        model = build_k2nsr(args.vae_checkpoint, config, model_path=args.resume, device=device)
        model.train()
        criterion = K2NSRLoss(loss_config)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=settings.learning_rate,
                                      weight_decay=settings.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and settings.precision == "fp16")
        resolution = config.patch_sizes[-1] * model.vae.downsample
        dataset = TrainingImages(args.hr_dir, lr_directory=args.lr_dir, resolution=resolution)
        sampler = DistributedSampler(dataset, seed=settings.seed) if world_size > 1 else None
        loader = DataLoader(dataset, batch_size=settings.batch_size, sampler=sampler,
                            shuffle=sampler is None, num_workers=settings.workers,
                            pin_memory=device.type == "cuda", drop_last=True, worker_init_fn=seed_worker)
        if len(loader) == 0:
            raise ValueError("The dataset is smaller than one full per-rank training batch.")
        degradation = None
        if args.lr_dir is None and settings.degradation == "realesrgan":
            from dataloader.realesrgan import RealESRGAN_degradation
            options = args.degradation_config or str(Path(__file__).resolve().parents[1] / "dataloader/params_realesrgan.yml")
            degradation = RealESRGAN_degradation(opt_path=options, device=device)
        start_epoch = step = 0
        if args.resume:
            state = read_checkpoint(args.resume)["training"]
            if state.get("world_size") != world_size or state.get("loss_config") != asdict(loss_config):
                raise ValueError("Resume requires the same world size and loss configuration.")
            previous = state.get("configuration", {})
            for key, value in asdict(settings).items():
                if key not in {"epochs", "log_every", "save_every_epochs"} and previous.get(key) != value:
                    raise ValueError(f"Resume training setting differs: {key}")
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_epoch, step = state["next_epoch"], state["global_step"]
            restore_rng(state["rng_states"][rank])
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                          broadcast_buffers=False) if world_size > 1 else model
        vae_hash = file_digest(args.vae_checkpoint)
        dtype = torch.float16 if settings.precision == "fp16" else torch.bfloat16
        if rank == 0:
            count = sum(p.numel() for p in model.reconstructor.parameters())
            print(f"Parallel Reconstructor: {count / 1e6:.2f}M parameters; scales={config.reconstructor.patch_sizes}")
        for epoch in range(start_epoch, settings.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                hr, lr = prepare_batch(batch, device, degradation)
                targets = model.encode_targets(hr)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=dtype, enabled=settings.precision != "fp32"):
                    predictions = wrapped(lr)
                    losses = criterion(predictions, targets, model.codebook)
                scaler.scale(losses.total).backward()
                scaler.unscale_(optimizer)
                if settings.grad_clip:
                    torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), settings.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                if step % settings.log_every == 0:
                    metrics = torch.stack([losses.total.detach(), *[term.detach() for term in losses.terms.values()]])
                    if world_size > 1:
                        distributed.all_reduce(metrics)
                        metrics /= world_size
                    if rank == 0:
                        print(f"epoch={epoch + 1} step={step} loss={metrics[0].item():.6f} "
                              f"terms={dict(zip(losses.terms, metrics[1:].tolist()))}")
            if (epoch + 1) % settings.save_every_epochs == 0 or epoch + 1 == settings.epochs:
                states = [None] * world_size
                if world_size > 1:
                    distributed.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                if rank == 0:
                    output = Path(args.output_dir) / f"k2nsr_epoch{epoch + 1:04d}.pth"
                    save_k2nsr_checkpoint(output, model, vae_sha256=vae_hash,
                        training_state={"next_epoch": epoch + 1, "global_step": step,
                                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                                        "rng_states": states, "world_size": world_size,
                                        "configuration": asdict(settings), "loss_config": asdict(loss_config)})
                    print(f"Saved {output}")
                if world_size > 1:
                    distributed.barrier()
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Experiment JSON configuration")
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--lr-dir", help="Optional aligned LR directory; otherwise generate degradation online")
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", help="Versioned K2NSR epoch checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--degradation-config", help="RealESRGAN degradation YAML")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
