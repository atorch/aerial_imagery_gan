"""Configuration: label classes and training hyperparameters."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class LabelClass:
    name: str
    # Directory (sibling of naip/) holding a single-band raster with the same
    # filename and the same width/height as each NAIP scene.
    directory: str
    codes: tuple[int, ...]
    # RGB color used when visualizing label maps
    color: tuple[int, int, int]


# Label classes, in PRIORITY order: when a pixel matches several classes (a road
# running through a CDL "developed" pixel, say), the first match wins. Pixels that
# match nothing get the implicit class 0, "other".
LABEL_CLASSES: tuple[LabelClass, ...] = (
    LabelClass("buildings", "building_annotations", (1,), (220, 30, 30)),
    LabelClass("roads", "road_annotations", (1,), (250, 250, 250)),
    LabelClass("cdl_water", "cdl_annotations", (83, 111), (40, 90, 220)),
    LabelClass("cdl_developed", "cdl_annotations", (82, 121, 122, 123, 124), (150, 150, 150)),
    LabelClass("cdl_forest", "cdl_annotations", (63, 141, 142, 143), (20, 100, 30)),
    LabelClass("cdl_pasture", "cdl_annotations", (176,), (150, 210, 90)),
    LabelClass("cdl_corn_soy", "cdl_annotations", (1, 5, 26), (230, 200, 60)),
)
OTHER_COLOR = (60, 40, 30)

# Number of label channels the models see when conditioning on labels
# (one-hot, including the implicit "other" class 0).
N_LABEL_CHANNELS = len(LABEL_CLASSES) + 1

N_BANDS = 4  # NAIP: R, G, B, NIR


@dataclass
class TrainConfig:
    """Every field is also a command line flag (underscores become dashes)."""

    model: str = "gan"  # "gan" or "diffusion"
    run_name: str = ""  # defaults to "<model>_<timestamp>"
    runs_dir: str = "runs"
    naip_dir: str = "naip"
    # "auto": condition on labels if label rasters exist for every scene
    labels: str = "auto"  # "auto", "on" or "off"

    patch_size: int = 128
    batch_size: int = 8
    steps: int = 200_000
    # Random flips and 90 degree rotations, applied to images and labels together.
    # Off by default: shadows in NAIP have a consistent direction, and rotations
    # would teach the model that any shadow direction is fine.
    augment: bool = False
    # Reject patches where more than this fraction of pixels is nodata (all bands 0)
    max_nodata_fraction: float = 0.01

    ema_decay: float = 0.999
    log_every: int = 50
    sample_every: int = 1000
    checkpoint_every: int = 5000
    n_fixed_samples: int = 8
    seed: int = 0
    device: str = "auto"

    # --- GAN ---
    gan_channels: int = 256  # channels at the lowest resolution of the generator
    gan_z_channels: int = 64  # channels of the spatial latent noise (at 1/16 resolution)
    disc_channels: int = 64
    disc_scales: int = 2  # number of PatchGAN discriminators, at successively halved resolution
    lr_generator: float = 1e-4
    lr_discriminator: float = 4e-4
    feature_matching_weight: float = 10.0

    # --- diffusion (rectified flow / flow matching) ---
    unet_channels: int = 64
    unet_channel_mults: str = "1,2,2,4"
    unet_attention: bool = True  # self-attention at the lowest resolution
    # Trade compute for memory (needed for the defaults to fit in 4 GB of GPU memory)
    gradient_checkpointing: bool = True
    lr_diffusion: float = 2e-4
    label_dropout: float = 0.1  # for classifier-free guidance
    sample_steps: int = 50
    guidance_scale: float = 2.0


def config_from_dict(d: dict) -> TrainConfig:
    names = {f.name for f in dataclasses.fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in d.items() if k in names})
