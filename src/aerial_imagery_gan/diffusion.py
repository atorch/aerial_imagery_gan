"""A small pixel-space diffusion model, trained with rectified flow / flow matching.

* Training: pick t in (0, 1) (logit-normal, as in Stable Diffusion 3), mix
  x_t = (1 - t) * image + t * noise, and regress the velocity (noise - image) with
  a U-Net. This is a plain MSE loss: no adversarial game, so training is stable,
  and the loss goes down as samples improve.
* Sampling: integrate from pure noise (t=1) to an image (t=0) with Euler steps.
* Labels (when used) are concatenated to the U-Net input. They are dropped for a
  fraction of training examples, which enables classifier-free guidance at
  sampling time: a guidance scale above 1 makes samples follow the labels more closely.
* The U-Net is convolutional (plus self-attention at the lowest resolution), so it
  can also sample images larger than the training patches.
"""

from __future__ import annotations

import copy
import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .common import count_params, ema_update, group_norm, seeded_randn
from .config import N_BANDS, N_LABEL_CHANNELS, TrainConfig


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    args = 1000 * t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm0 = group_norm(in_channels)
        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_channels)
        self.norm1 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv0(F.silu(self.norm0(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm1(h) * (1 + scale) + shift
        h = self.conv1(self.dropout(F.silu(h)))
        return self.skip(x) + h


class Attention(nn.Module):
    def __init__(self, channels: int, head_dim: int = 64):
        super().__init__()
        self.heads = max(channels // head_dim, 1)
        self.max_query_chunk = 4096
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(n, 3, self.heads, c // self.heads, h * w)
        q, k, v = (t.transpose(-1, -2) for t in qkv.unbind(1))  # (n, heads, hw, d)
        if q.shape[2] <= self.max_query_chunk:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # Large images (sampling beyond the training size): bound the attention
            # matrix's memory in case no memory-efficient kernel is available
            out = torch.cat(
                [
                    F.scaled_dot_product_attention(q_chunk, k, v)
                    for q_chunk in q.split(self.max_query_chunk, dim=2)
                ],
                dim=2,
            )
        return x + self.proj(out.transpose(-1, -2).reshape(n, c, h, w))


class UNet(nn.Module):
    def __init__(
        self,
        label_channels: int,
        channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 2, 4),
        n_res_blocks: int = 2,
        attention: bool = True,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.label_channels = label_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.downsample_factor = 2 ** (len(channel_mults) - 1)
        emb_dim = 4 * channels
        self.time_embed = nn.Sequential(
            nn.Linear(channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.input_conv = nn.Conv2d(N_BANDS + label_channels, channels, 3, padding=1)

        n_levels = len(channel_mults)
        self.down = nn.ModuleList()
        skip_channels = [channels]
        ch = channels
        for level, mult in enumerate(channel_mults):
            for _ in range(n_res_blocks):
                use_attention = attention and level == n_levels - 1
                self.down.append(self._block(ch, channels * mult, emb_dim, dropout, use_attention))
                ch = channels * mult
                skip_channels.append(ch)
            if level != n_levels - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skip_channels.append(ch)

        self.mid = nn.ModuleList(
            [
                ResBlock(ch, ch, emb_dim, dropout),
                Attention(ch) if attention else nn.Identity(),
                ResBlock(ch, ch, emb_dim, dropout),
            ]
        )

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            for _ in range(n_res_blocks + 1):
                use_attention = attention and level == n_levels - 1
                self.up.append(
                    self._block(
                        ch + skip_channels.pop(), channels * mult, emb_dim, dropout, use_attention
                    )
                )
                ch = channels * mult
            if level != 0:
                self.up.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch, ch, 3, padding=1)
                    )
                )

        self.output = nn.Sequential(group_norm(ch), nn.SiLU(), nn.Conv2d(ch, N_BANDS, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.base_channels = channels

    @staticmethod
    def _block(cin, cout, emb_dim, dropout, use_attention):
        block = nn.ModuleList([ResBlock(cin, cout, emb_dim, dropout)])
        if use_attention:
            block.append(Attention(cout))
        return block

    @staticmethod
    def _apply_block(block, x, emb):
        x = block[0](x, emb)
        for layer in block[1:]:
            x = layer(x)
        return x

    def _run_block(self, block, x, emb):
        if not isinstance(block, nn.ModuleList):
            return block(x)  # downsample / upsample
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            # Recompute activations in the backward pass instead of storing them:
            # roughly halves activation memory for ~30% more compute per step
            return checkpoint(self._apply_block, block, x, emb, use_reentrant=False)
        return self._apply_block(block, x, emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None):
        if self.label_channels:
            if labels is None:
                labels = x.new_zeros(x.shape[0], self.label_channels, *x.shape[2:])
            x = torch.cat([x, labels], dim=1)
        emb = self.time_embed(timestep_embedding(t, self.base_channels))

        h = self.input_conv(x)
        skips = [h]
        for block in self.down:
            h = self._run_block(block, h, emb)
            skips.append(h)

        h = self.mid[0](h, emb)
        h = self.mid[1](h)
        h = self.mid[2](h, emb)

        for block in self.up:
            if isinstance(block, nn.ModuleList):
                h = torch.cat([h, skips.pop()], dim=1)
            h = self._run_block(block, h, emb)
        return self.output(h)


class DiffusionTrainer:
    def __init__(self, cfg: TrainConfig, use_labels: bool, device: torch.device):
        self.cfg = cfg
        self.use_labels = use_labels
        self.device = device
        mults = tuple(int(m) for m in cfg.unet_channel_mults.split(","))
        label_channels = N_LABEL_CHANNELS if use_labels else 0
        self.model = UNet(
            label_channels,
            cfg.unet_channels,
            mults,
            attention=cfg.unet_attention,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )
        self.model.to(device)
        self.model_ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr_diffusion, weight_decay=0)
        self.step = 0
        print(f"U-Net: {count_params(self.model)} params")

    def train_step(self, real: torch.Tensor, labels: torch.Tensor | None) -> dict[str, float]:
        self.model.train()
        n = real.shape[0]
        t = torch.sigmoid(torch.randn(n, device=self.device))  # logit-normal
        noise = torch.randn_like(real)
        t4 = t[:, None, None, None]
        x_t = (1 - t4) * real + t4 * noise
        target = noise - real

        if labels is not None and self.cfg.label_dropout > 0:
            keep = torch.rand(n, device=self.device) >= self.cfg.label_dropout
            labels = labels * keep[:, None, None, None]

        loss = F.mse_loss(self.model(x_t, t, labels), target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        ema_update(self.model_ema, self.model, self.cfg.ema_decay, self.step)
        self.step += 1
        return {"loss": loss.item(), "grad_norm": grad_norm.item()}

    def _velocity(self, model, x, t, labels, guidance_scale):
        t = t.expand(x.shape[0])
        if labels is None or guidance_scale == 1.0:
            return model(x, t, labels)
        v_cond, v_uncond = model(
            torch.cat([x, x]), torch.cat([t, t]), torch.cat([labels, torch.zeros_like(labels)])
        ).chunk(2)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor | None,
        n: int,
        height: int,
        width: int,
        seed: int | None = None,
        use_ema: bool = True,
        steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        f = self.model.downsample_factor
        if height % f or width % f:
            raise ValueError(f"Diffusion output height and width must be multiples of {f}")
        model = self.model_ema if use_ema else self.model
        was_training = model.training
        model.eval()
        steps = steps or self.cfg.sample_steps
        guidance_scale = self.cfg.guidance_scale if guidance_scale is None else guidance_scale

        x = seeded_randn((n, N_BANDS, height, width), seed, self.device)
        times = torch.linspace(1.0, 0.0, steps + 1, device=self.device)
        for t, t_next in pairwise(times):
            x = x + (t_next - t) * self._velocity(model, x, t, labels, guidance_scale)
        model.train(was_training)
        return x.clamp(-1, 1)

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.model_ema.load_state_dict(state["model_ema"])
        self.opt.load_state_dict(state["opt"])
        self.step = state["step"]

    def ema_state_dict(self) -> dict:
        return self.model_ema.state_dict()

    def load_ema_state_dict(self, state: dict) -> None:
        self.model_ema.load_state_dict(state)
