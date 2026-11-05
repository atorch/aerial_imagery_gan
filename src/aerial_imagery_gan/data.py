"""Reading NAIP scenes (and optional label rasters), sampling patches, saving images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image

from .config import LABEL_CLASSES, N_BANDS, N_LABEL_CHANNELS, OTHER_COLOR


@dataclass
class Scene:
    path: Path
    image: np.ndarray  # (4, H, W) uint8, bands R G B NIR
    labels: np.ndarray | None  # (H, W) uint8 class index (0 = "other"), or None


def find_naip_paths(naip_dir: str | Path) -> list[Path]:
    naip_dir = Path(naip_dir)
    paths = sorted(p for p in naip_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    if not paths:
        raise FileNotFoundError(f"No .tif files found in {naip_dir.resolve()}")
    return paths


def label_raster_paths(naip_path: Path) -> dict[str, Path]:
    """Label rasters live in sibling directories of naip/, with the same filename."""
    directories = sorted({c.directory for c in LABEL_CLASSES})
    return {d: naip_path.parent.parent / d / naip_path.name for d in directories}


def labels_available(naip_paths: list[Path]) -> tuple[bool, list[Path]]:
    missing = [
        p for naip in naip_paths for p in label_raster_paths(naip).values() if not p.exists()
    ]
    return not missing, missing


def resolve_use_labels(labels_setting: str, naip_paths: list[Path]) -> bool:
    if labels_setting == "off":
        return False
    available, missing = labels_available(naip_paths)
    if labels_setting == "on" and not available:
        listed = "\n  ".join(str(p) for p in missing[:10])
        raise FileNotFoundError(f"--labels on, but label rasters are missing, e.g.:\n  {listed}")
    if labels_setting == "auto" and not available:
        print(f"Label rasters missing ({len(missing)} files), training an unconditional model")
    return available


def read_naip(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count < N_BANDS:
            raise ValueError(f"{path} has {src.count} bands, expected {N_BANDS} (R G B NIR)")
        x = src.read(indexes=list(range(1, N_BANDS + 1)))
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def read_class_map(naip_path: Path, shape: tuple[int, int]) -> np.ndarray:
    rasters = {}
    for directory, path in label_raster_paths(naip_path).items():
        with rasterio.open(path) as src:
            rasters[directory] = src.read(1)
        if rasters[directory].shape != shape:
            raise ValueError(
                f"{path} has shape {rasters[directory].shape}, but the NAIP scene has {shape}"
            )

    class_map = np.zeros(shape, dtype=np.uint8)
    assigned = np.zeros(shape, dtype=bool)
    for index, label_class in enumerate(LABEL_CLASSES, start=1):
        match = np.isin(rasters[label_class.directory], label_class.codes) & ~assigned
        class_map[match] = index
        assigned |= match
    return class_map


def load_scene(path: Path, use_labels: bool) -> Scene:
    print(f"Reading {path}")
    image = read_naip(path)
    labels = read_class_map(path, image.shape[1:]) if use_labels else None
    return Scene(path=path, image=image, labels=labels)


def load_scenes(naip_dir: str | Path, use_labels: bool) -> list[Scene]:
    scenes = [load_scene(p, use_labels) for p in find_naip_paths(naip_dir)]
    if use_labels:
        # Every 4th row and column is plenty for a frequency table
        counts = sum(
            np.bincount(s.labels[::4, ::4].ravel(), minlength=N_LABEL_CHANNELS) for s in scenes
        )
        names = ["other"] + [c.name for c in LABEL_CLASSES]
        total = counts.sum()
        print("Label class frequencies:")
        for name, count in zip(names, counts):
            print(f"  {name:>14}: {100 * count / total:5.1f}%")
    return scenes


class PatchSampler:
    """Draws random square patches (and matching label patches) from in-memory scenes."""

    def __init__(
        self,
        scenes: list[Scene],
        patch_size: int,
        augment: bool = False,
        max_nodata_fraction: float = 0.01,
        seed: int = 0,
    ):
        for scene in scenes:
            if min(scene.image.shape[1:]) < patch_size:
                raise ValueError(f"{scene.path} is smaller than patch_size={patch_size}")
        self.scenes = scenes
        self.patch_size = patch_size
        self.augment = augment
        self.max_nodata_fraction = max_nodata_fraction
        self.rng = np.random.default_rng(seed)
        areas = np.array([s.image.shape[1] * s.image.shape[2] for s in scenes], dtype=float)
        self.scene_weights = areas / areas.sum()

    def _one(self) -> tuple[np.ndarray, np.ndarray | None]:
        p = self.patch_size
        for _ in range(100):
            scene = self.scenes[self.rng.choice(len(self.scenes), p=self.scene_weights)]
            _, height, width = scene.image.shape
            row = self.rng.integers(0, height - p + 1)
            col = self.rng.integers(0, width - p + 1)
            image = scene.image[:, row : row + p, col : col + p]
            if (image == 0).all(axis=0).mean() <= self.max_nodata_fraction:
                break
        labels = None if scene.labels is None else scene.labels[row : row + p, col : col + p]

        if self.augment:
            k = self.rng.integers(4)
            flip = self.rng.integers(2)
            image = np.rot90(image, k, axes=(1, 2))
            labels = None if labels is None else np.rot90(labels, k)
            if flip:
                image = image[:, :, ::-1]
                labels = None if labels is None else labels[:, ::-1]
        return image, labels

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray | None]:
        pairs = [self._one() for _ in range(n)]
        images = np.stack([image for image, _ in pairs])
        labels = None if pairs[0][1] is None else np.stack([labels for _, labels in pairs])
        return images, labels


# ---- tensors ---------------------------------------------------------------


def images_to_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 (N, 4, H, W) -> float32 in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    return x.float() / 127.5 - 1.0


def labels_to_onehot(labels: np.ndarray | None, device: torch.device) -> torch.Tensor | None:
    """uint8 class map (N, H, W) -> float32 one-hot (N, N_LABEL_CHANNELS, H, W)."""
    if labels is None:
        return None
    x = torch.from_numpy(np.ascontiguousarray(labels)).to(device).long()
    return F.one_hot(x, N_LABEL_CHANNELS).permute(0, 3, 1, 2).float()


def batch_to_tensors(
    images: np.ndarray, labels: np.ndarray | None, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return images_to_tensor(images, device), labels_to_onehot(labels, device)


# ---- visualization -----------------------------------------------------------


def tensor_to_rgb(x: torch.Tensor) -> np.ndarray:
    """float (N, 4, H, W) in [-1, 1] -> uint8 (N, H, W, 3), RGB bands only."""
    x = ((x[:, :3].detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round()
    return x.byte().permute(0, 2, 3, 1).numpy()


def tensor_to_false_color(x: torch.Tensor) -> np.ndarray:
    """NIR, R, G as RGB: vegetation shows up red. Useful for checking the NIR band."""
    return tensor_to_rgb(x[:, [3, 0, 1]])


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """uint8 class map (N, H, W) -> uint8 (N, H, W, 3)."""
    palette = np.array([OTHER_COLOR] + [c.color for c in LABEL_CLASSES], dtype=np.uint8)
    return palette[labels]


def save_grid(rows: list[np.ndarray], path: str | Path, pad: int = 2) -> None:
    """Save rows of images (each an (N, H, W, 3) uint8 array) as one PNG."""
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    height, width = rows[0].shape[1:3]
    canvas = np.full(
        (n_rows * (height + pad) + pad, n_cols * (width + pad) + pad, 3), 255, dtype=np.uint8
    )
    for i, row in enumerate(rows):
        for j, image in enumerate(row):
            top, left = pad + i * (height + pad), pad + j * (width + pad)
            canvas[top : top + height, left : left + width] = image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
