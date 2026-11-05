"""Training loop shared by the GAN and diffusion models."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .common import describe_device, pick_device
from .config import TrainConfig, config_from_dict
from .data import (
    PatchSampler,
    batch_to_tensors,
    colorize_labels,
    find_naip_paths,
    load_scenes,
    resolve_use_labels,
    save_grid,
    tensor_to_false_color,
    tensor_to_rgb,
)


def make_trainer(cfg: TrainConfig, use_labels: bool, device: torch.device):
    if cfg.model == "gan":
        from .gan import GANTrainer

        return GANTrainer(cfg, use_labels, device)
    if cfg.model == "diffusion":
        from .diffusion import DiffusionTrainer

        return DiffusionTrainer(cfg, use_labels, device)
    raise ValueError(f"Unknown model {cfg.model!r}, expected 'gan' or 'diffusion'")


def save_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_trainer_from_checkpoint(path: str | Path, device: torch.device):
    """For inference: returns (trainer with EMA weights loaded, config, use_labels)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(checkpoint["config"])
    trainer = make_trainer(cfg, checkpoint["use_labels"], device)
    if "trainer" in checkpoint:
        trainer.load_state_dict(checkpoint["trainer"])
    else:
        trainer.load_ema_state_dict(checkpoint["ema"])
    return trainer, cfg, checkpoint["use_labels"]


def plot_losses(csv_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return
    columns = [c for c in rows[0] if c not in ("step", "seconds")]
    steps = np.array([int(r["step"]) for r in rows])
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 2.2 * len(columns)), sharex=True)
    for ax, column in zip(np.atleast_1d(axes), columns):
        values = np.array([float(r[column]) for r in rows])
        ax.plot(steps, values, alpha=0.3, color="C0")
        window = max(len(values) // 50, 1)
        if window > 1:
            smooth = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1 :], smooth, color="C0")
        ax.set_ylabel(column)
    np.atleast_1d(axes)[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(png_path, dpi=80)
    plt.close(fig)


def save_samples(trainer, fixed_images, fixed_labels, path: Path, device) -> None:
    real, labels = batch_to_tensors(fixed_images, fixed_labels, device)
    n, _, h, w = real.shape
    fakes = [trainer.generate(labels, n, h, w, seed=seed) for seed in (0, 1)]
    rows = [] if fixed_labels is None else [colorize_labels(fixed_labels)]
    rows += [tensor_to_rgb(real), tensor_to_rgb(fakes[0]), tensor_to_rgb(fakes[1])]
    rows += [tensor_to_false_color(real), tensor_to_false_color(fakes[0])]
    save_grid(rows, path)


def train(cfg: TrainConfig, resume: str | None = None) -> Path:
    device = pick_device(cfg.device)
    print(f"Device: {describe_device(device)}")
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    checkpoint = None
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        use_labels = checkpoint["use_labels"]
        run_dir = Path(resume).resolve().parent.parent
    else:
        use_labels = resolve_use_labels(cfg.labels, find_naip_paths(cfg.naip_dir))
        run_name = cfg.run_name or f"{cfg.model}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path(cfg.runs_dir) / run_name
    print(f"Model: {cfg.model}, conditioned on labels: {use_labels}, run directory: {run_dir}")

    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump({**dataclasses.asdict(cfg), "use_labels": use_labels}, f, indent=2)

    scenes = load_scenes(cfg.naip_dir, use_labels)
    # A separate sampler (fixed seed) for the patches shown in sample grids, so that
    # they stay the same across resumed runs
    fixed_images, fixed_labels = PatchSampler(
        scenes, cfg.patch_size, max_nodata_fraction=cfg.max_nodata_fraction, seed=1234
    ).sample(cfg.n_fixed_samples)
    step = 0 if checkpoint is None else checkpoint["step"]
    sampler = PatchSampler(
        scenes,
        cfg.patch_size,
        augment=cfg.augment,
        max_nodata_fraction=cfg.max_nodata_fraction,
        seed=cfg.seed + step,
    )

    trainer = make_trainer(cfg, use_labels, device)
    if checkpoint is not None:
        trainer.load_state_dict(checkpoint["trainer"])
        print(f"Resumed from {resume} at step {step}")
        del checkpoint

    def checkpoint_payload(**extra):
        return {
            "config": dataclasses.asdict(cfg),
            "use_labels": use_labels,
            "step": step,
            **extra,
        }

    csv_path = run_dir / "losses.csv"
    csv_file = open(csv_path, "a", newline="")
    writer = None
    running: dict[str, list[float]] = {}
    start = time.time()

    progress = tqdm(range(step, cfg.steps), initial=step, total=cfg.steps, dynamic_ncols=True)
    for step in progress:
        images, labels = sampler.sample(cfg.batch_size)
        losses = trainer.train_step(*batch_to_tensors(images, labels, device))
        for key, value in losses.items():
            running.setdefault(key, []).append(value)
        done = step + 1

        if done % cfg.log_every == 0:
            means = {key: float(np.mean(values)) for key, values in running.items()}
            running = {}
            progress.set_postfix({k: f"{v:.3f}" for k, v in means.items()})
            if writer is None:
                fields = ["step", "seconds", *means]
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                if csv_file.tell() == 0:
                    writer.writeheader()
            writer.writerow({"step": done, "seconds": round(time.time() - start), **means})
            csv_file.flush()

        if done % cfg.sample_every == 0 or done == cfg.steps:
            save_samples(
                trainer, fixed_images, fixed_labels, run_dir / "samples" / f"{done:07d}.png", device
            )
            plot_losses(csv_path, run_dir / "losses.png")

        if done % cfg.checkpoint_every == 0 or done == cfg.steps:
            step = done
            checkpoints = run_dir / "checkpoints"
            save_checkpoint(checkpoints / "last.pt", checkpoint_payload(trainer=trainer.state_dict()))
            save_checkpoint(
                checkpoints / f"ema_{done:07d}.pt", checkpoint_payload(ema=trainer.ema_state_dict())
            )

    csv_file.close()
    print(f"Done. Samples and checkpoints are in {run_dir}")
    return run_dir
