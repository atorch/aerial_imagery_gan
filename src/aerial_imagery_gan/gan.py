"""A label-conditional GAN in the style of SPADE / GauGAN (Park et al. 2019).

What changed relative to the 2020 version of this repo, and why:

* Generator: instead of a U-Net mapping full-resolution noise to an image, the
  generator starts from low-resolution *spatial* noise (1/16 of the output size)
  and upsamples through residual blocks. Label maps modulate every block via
  SPADE (spatially-adaptive normalization), so labels are not "washed out" by
  normalization layers, and per-pixel noise injection (as in StyleGAN) gives the
  fine texture of tree canopies and fields some randomness. The generator is
  fully convolutional, so it can produce images much larger than the training
  patches from a large label map.
* Discriminator: multi-scale PatchGAN with spectral normalization, which
  classifies overlapping patches instead of the whole image, and sees the label
  map along with the image (when labels are used).
* Loss: hinge loss plus discriminator feature matching (paired when labels are
  available, batch-mean matching as in Salimans et al. 2016 otherwise).
* Two time-scale update rule (a faster discriminator), Adam with beta1=0, and an
  exponential moving average (EMA) of the generator weights for sampling, which
  usually gives visibly cleaner images than the raw generator.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from .common import count_params, ema_update, seeded, seeded_randn
from .config import N_BANDS, N_LABEL_CHANNELS, TrainConfig


class NoiseInjection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class SPADE(nn.Module):
    """BatchNorm without learned affine params, then a scale and shift predicted per pixel
    from the label map."""

    def __init__(self, channels: int, label_channels: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels, affine=False)
        self.shared = nn.Sequential(nn.Conv2d(label_channels, hidden, 3, padding=1), nn.ReLU())
        self.gamma = nn.Conv2d(hidden, channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = F.interpolate(labels, size=x.shape[2:], mode="nearest")
        h = self.shared(labels)
        return self.norm(x) * (1 + self.gamma(h)) + self.beta(h)


class SPADEResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, label_channels: int):
        super().__init__()
        mid = min(in_channels, out_channels)
        self.norm0 = SPADE(in_channels, label_channels)
        self.conv0 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.noise0 = NoiseInjection(mid)
        self.norm1 = SPADE(mid, label_channels)
        self.conv1 = nn.Conv2d(mid, out_channels, 3, padding=1)
        self.noise1 = NoiseInjection(out_channels)
        self.learned_shortcut = in_channels != out_channels
        if self.learned_shortcut:
            self.norm_s = SPADE(in_channels, label_channels)
            self.conv_s = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shortcut = self.conv_s(self.norm_s(x, labels)) if self.learned_shortcut else x
        h = self.noise0(self.conv0(F.leaky_relu(self.norm0(x, labels), 0.2)))
        h = self.noise1(self.conv1(F.leaky_relu(self.norm1(h, labels), 0.2)))
        return shortcut + h


class Generator(nn.Module):
    n_upsamples = 4
    downsample_factor = 2**n_upsamples  # output size / latent size

    def __init__(self, label_channels: int, channels: int = 256, z_channels: int = 64):
        super().__init__()
        self.label_channels = label_channels
        self.z_channels = z_channels
        # Without labels, SPADE gets a constant map, which reduces it to plain BatchNorm
        spade_channels = max(label_channels, 1)
        # e.g. 256, 256, 256, 128, 64, 32 (the last width is at full resolution)
        widths = [channels, channels] + [max(channels // 2**i, 16) for i in range(self.n_upsamples)]
        self.input_conv = nn.Conv2d(z_channels, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList(
            SPADEResBlock(widths[i], widths[i + 1], spade_channels)
            for i in range(self.n_upsamples + 1)
        )
        self.output_conv = nn.Conv2d(widths[-1], N_BANDS, 3, padding=1)

    def latent_shape(self, n: int, height: int, width: int) -> tuple[int, ...]:
        f = self.downsample_factor
        if height % f or width % f:
            raise ValueError(f"GAN output height and width must be multiples of {f}")
        return (n, self.z_channels, height // f, width // f)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        n, _, h, w = z.shape
        if labels is None:
            f = self.downsample_factor
            labels = torch.ones(n, 1, h * f, w * f, device=z.device, dtype=z.dtype)
        x = self.input_conv(z)
        for i, block in enumerate(self.blocks):
            if i > 0:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = block(x, labels)
        return torch.tanh(self.output_conv(F.leaky_relu(x, 0.2)))


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN (pix2pix) with spectral norm. Returns intermediate features too."""

    def __init__(self, in_channels: int, channels: int = 64):
        super().__init__()
        c = channels
        spec = [(in_channels, c, 2), (c, 2 * c, 2), (2 * c, 4 * c, 2), (4 * c, 8 * c, 1)]
        self.layers = nn.ModuleList(
            nn.Sequential(
                spectral_norm(nn.Conv2d(cin, cout, 4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2),
            )
            for cin, cout, stride in spec
        )
        self.output = spectral_norm(nn.Conv2d(8 * c, 1, 4, padding=1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        features.append(self.output(x))
        return features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels: int, channels: int = 64, n_scales: int = 2):
        super().__init__()
        self.discriminators = nn.ModuleList(
            PatchDiscriminator(in_channels, channels) for _ in range(n_scales)
        )

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        outputs = []
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                x = F.avg_pool2d(x, 3, stride=2, padding=1, count_include_pad=False)
            outputs.append(discriminator(x))
        return outputs


def discriminator_hinge_loss(real_outputs, fake_outputs) -> torch.Tensor:
    losses = [
        F.relu(1 - real[-1]).mean() + F.relu(1 + fake[-1]).mean()
        for real, fake in zip(real_outputs, fake_outputs)
    ]
    return sum(losses) / len(losses)


def generator_hinge_loss(fake_outputs) -> torch.Tensor:
    return sum(-fake[-1].mean() for fake in fake_outputs) / len(fake_outputs)


def feature_matching_loss(real_outputs, fake_outputs, paired: bool) -> torch.Tensor:
    losses = []
    for real, fake in zip(real_outputs, fake_outputs):
        for real_feature, fake_feature in zip(real[:-1], fake[:-1]):
            real_feature = real_feature.detach()
            if not paired:
                real_feature, fake_feature = real_feature.mean(0), fake_feature.mean(0)
            losses.append(F.l1_loss(fake_feature, real_feature))
    return sum(losses) / len(losses)


class GANTrainer:
    def __init__(self, cfg: TrainConfig, use_labels: bool, device: torch.device):
        self.cfg = cfg
        self.use_labels = use_labels
        self.device = device
        label_channels = N_LABEL_CHANNELS if use_labels else 0

        self.generator = Generator(label_channels, cfg.gan_channels, cfg.gan_z_channels).to(device)
        self.discriminator = MultiScaleDiscriminator(
            N_BANDS + label_channels, cfg.disc_channels, cfg.disc_scales
        ).to(device)
        self.generator_ema = copy.deepcopy(self.generator).eval().requires_grad_(False)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=cfg.lr_generator, betas=(0.0, 0.999)
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=cfg.lr_discriminator, betas=(0.0, 0.999)
        )
        self.step = 0
        print(
            f"Generator: {count_params(self.generator)} params, "
            f"discriminator: {count_params(self.discriminator)} params"
        )

    def _discriminate(self, images: torch.Tensor, labels: torch.Tensor | None):
        x = images if labels is None else torch.cat([images, labels], dim=1)
        return self.discriminator(x)

    def _random_latent(self, images: torch.Tensor) -> torch.Tensor:
        n, _, h, w = images.shape
        return torch.randn(self.generator.latent_shape(n, h, w), device=self.device)

    def train_step(self, real: torch.Tensor, labels: torch.Tensor | None) -> dict[str, float]:
        self.generator.train()

        # Discriminator step
        self.discriminator.requires_grad_(True)
        with torch.no_grad():
            fake = self.generator(self._random_latent(real), labels)
        real_outputs = self._discriminate(real, labels)
        fake_outputs = self._discriminate(fake, labels)
        loss_d = discriminator_hinge_loss(real_outputs, fake_outputs)
        self.opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        self.opt_d.step()

        # Generator step
        self.discriminator.requires_grad_(False)
        fake = self.generator(self._random_latent(real), labels)
        fake_outputs = self._discriminate(fake, labels)
        with torch.no_grad():
            real_outputs = self._discriminate(real, labels)
        loss_adv = generator_hinge_loss(fake_outputs)
        loss_fm = feature_matching_loss(real_outputs, fake_outputs, paired=labels is not None)
        loss_g = loss_adv + self.cfg.feature_matching_weight * loss_fm
        self.opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        self.opt_g.step()

        ema_update(self.generator_ema, self.generator, self.cfg.ema_decay, self.step)
        self.step += 1

        return {
            "loss_d": loss_d.item(),
            "loss_g_adv": loss_adv.item(),
            "loss_g_fm": loss_fm.item(),
            # Mean discriminator score (last scale) on real and fake patches
            "d_real": real_outputs[-1][-1].mean().item(),
            "d_fake": fake_outputs[-1][-1].mean().item(),
        }

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor | None,
        n: int,
        height: int,
        width: int,
        seed: int | None = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        generator = self.generator_ema if use_ema else self.generator
        was_training = generator.training
        generator.eval()
        z = seeded_randn(generator.latent_shape(n, height, width), seed, self.device)
        with seeded(seed, self.device):  # noise injection layers draw noise too
            images = generator(z, labels)
        generator.train(was_training)
        return images

    def state_dict(self) -> dict:
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "generator_ema": self.generator_ema.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "step": self.step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.generator_ema.load_state_dict(state["generator_ema"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.step = state["step"]

    def ema_state_dict(self) -> dict:
        return self.generator_ema.state_dict()

    def load_ema_state_dict(self, state: dict) -> None:
        self.generator_ema.load_state_dict(state)
