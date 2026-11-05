"""Generating images from a trained checkpoint, and scoring them with KID."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .common import describe_device, pick_device
from .data import (
    PatchSampler,
    batch_to_tensors,
    colorize_labels,
    load_scene,
    load_scenes,
    save_grid,
    tensor_to_false_color,
    tensor_to_rgb,
)
from .train import load_trainer_from_checkpoint


def _generate(trainer, labels, n, h, w, seed, sample_steps, guidance_scale):
    kwargs = {}
    if trainer.cfg.model != "diffusion":
        return trainer.generate(labels, n, h, w, seed=seed)
    if sample_steps is not None:
        kwargs["steps"] = sample_steps
    if guidance_scale is not None:
        kwargs["guidance_scale"] = guidance_scale
    return trainer.generate(labels, n, h, w, seed=seed, **kwargs)


def sample_patches(
    checkpoint: str,
    out: str,
    n: int = 8,
    size: int | None = None,
    seed: int = 0,
    naip_dir: str | None = None,
    sample_steps: int | None = None,
    guidance_scale: float | None = None,
    device: str = "auto",
) -> Path:
    """A grid of generated patches, next to real patches (and their label maps)."""
    torch_device = pick_device(device)
    print(f"Device: {describe_device(torch_device)}")
    trainer, cfg, use_labels = load_trainer_from_checkpoint(checkpoint, torch_device)
    size = size or cfg.patch_size
    scenes = load_scenes(naip_dir or cfg.naip_dir, use_labels)
    images, label_maps = PatchSampler(scenes, size, seed=seed).sample(n)
    real, labels = batch_to_tensors(images, label_maps, torch_device)
    fakes = [
        _generate(trainer, labels, n, size, size, seed + i, sample_steps, guidance_scale)
        for i in range(2)
    ]
    rows = [] if label_maps is None else [colorize_labels(label_maps)]
    rows += [tensor_to_rgb(real)] + [tensor_to_rgb(f) for f in fakes]
    rows += [tensor_to_false_color(real), tensor_to_false_color(fakes[0])]
    out_path = Path(out) / f"patches_seed{seed}.png"
    save_grid(rows, out_path)
    print(f"Saved {out_path}")
    return out_path


def sample_large(
    checkpoint: str,
    out: str,
    scene: str,
    row: int = 0,
    col: int = 0,
    size: int = 1024,
    n: int = 1,
    seed: int = 0,
    sample_steps: int | None = None,
    guidance_scale: float | None = None,
    device: str = "auto",
) -> Path:
    """Generate a large image from the label map of a window of a real NAIP scene.

    Both models are (mostly) convolutional, so the output can be much larger than
    the patches the model was trained on.
    """
    torch_device = pick_device(device)
    print(f"Device: {describe_device(torch_device)}")
    trainer, _, use_labels = load_trainer_from_checkpoint(checkpoint, torch_device)
    data = load_scene(Path(scene), use_labels)
    _, height, width = data.image.shape
    size = min(size, height - row, width - col) // 16 * 16
    image = data.image[None, :, row : row + size, col : col + size]
    label_map = (
        None if data.labels is None else data.labels[None, row : row + size, col : col + size]
    )
    real, labels = batch_to_tensors(image, label_map, torch_device)

    out_dir = Path(out) / f"{Path(scene).stem}_r{row}_c{col}_s{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_rgb(real)[0]).save(out_dir / "real.png")
    if label_map is not None:
        Image.fromarray(colorize_labels(label_map)[0]).save(out_dir / "labels.png")
    for i in range(n):
        fake = _generate(trainer, labels, 1, size, size, seed + i, sample_steps, guidance_scale)
        Image.fromarray(tensor_to_rgb(fake)[0]).save(out_dir / f"generated_{seed + i}.png")
        Image.fromarray(tensor_to_false_color(fake)[0]).save(
            out_dir / f"generated_{seed + i}_nir_r_g.png"
        )
    print(f"Saved images to {out_dir}")
    return out_dir


def kid(
    checkpoint: str,
    n: int = 1000,
    batch_size: int = 16,
    seed: int = 0,
    naip_dir: str | None = None,
    sample_steps: int | None = None,
    guidance_scale: float | None = None,
    device: str = "auto",
) -> tuple[float, float]:
    """Kernel Inception Distance between real and generated RGB patches (lower is better).

    KID is unbiased (unlike FID), so it is meaningful with ~1000 images. Use the
    same n and seed when comparing checkpoints. Generated images use the label maps
    of the real patches they are compared with, when the model uses labels.
    """
    try:
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError as e:
        raise SystemExit("KID needs the metrics extra: uv sync --extra metrics") from e

    torch_device = pick_device(device)
    trainer, cfg, use_labels = load_trainer_from_checkpoint(checkpoint, torch_device)
    scenes = load_scenes(naip_dir or cfg.naip_dir, use_labels)
    sampler = PatchSampler(scenes, cfg.patch_size, seed=seed)
    metric = KernelInceptionDistance(subset_size=min(n, 1000), normalize=False).to(torch_device)

    def to_uint8_rgb(x):
        return torch.from_numpy(tensor_to_rgb(x)).permute(0, 3, 1, 2).to(torch_device)

    for start in tqdm(range(0, n, batch_size), desc="KID"):
        b = min(batch_size, n - start)
        images, label_maps = sampler.sample(b)
        real, labels = batch_to_tensors(images, label_maps, torch_device)
        size = cfg.patch_size
        fake = _generate(trainer, labels, b, size, size, seed + start, sample_steps, guidance_scale)
        metric.update(to_uint8_rgb(real), real=True)
        metric.update(to_uint8_rgb(fake), real=False)
    mean, std = (float(v) for v in metric.compute())
    print(f"KID: {mean:.5f} ± {std:.5f} (n={n}, checkpoint={checkpoint})")
    return mean, std
