"""Small helpers shared by the GAN and diffusion models."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


def pick_device(setting: str = "auto") -> torch.device:
    if setting != "auto":
        return torch.device(setting)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    props = torch.cuda.get_device_properties(device)
    return (
        f"cuda ({props.name}, {props.total_memory / 2**30:.1f} GiB, "
        f"compute capability {props.major}.{props.minor})"
    )


def seeded_randn(shape: tuple[int, ...], seed: int | None, device: torch.device) -> torch.Tensor:
    """Gaussian noise that is reproducible across devices (drawn on the CPU)."""
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device)


@contextmanager
def seeded(seed: int | None, device: torch.device):
    """Seed torch's global RNG inside the block, restoring its previous state afterwards."""
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        yield


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float, step: int) -> None:
    # Warm up: early in training, the EMA should follow the model closely
    decay = min(decay, (1 + step) / (10 + step))
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.lerp_(p, 1 - decay)
    for b_ema, b in zip(ema.buffers(), model.buffers()):
        b_ema.copy_(b)


def count_params(module: nn.Module) -> str:
    return f"{sum(p.numel() for p in module.parameters()) / 1e6:.1f}M"


def group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise AssertionError
