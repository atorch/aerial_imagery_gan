"""Fast CPU smoke tests on small synthetic GeoTIFFs (no real NAIP data needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from aerial_imagery_gan import data
from aerial_imagery_gan.cli import main
from aerial_imagery_gan.config import LABEL_CLASSES, N_LABEL_CHANNELS, TrainConfig
from aerial_imagery_gan.train import make_trainer

SIZE = 160


def write_raster(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = array if array.ndim == 3 else array[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
        crs="EPSG:26915",
        transform=from_origin(500_000, 4_600_000, 1.0, 1.0),
    ) as dst:
        dst.write(array)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for name in ["m_0000000_ne_15_1_20170101.tif", "m_0000001_sw_15_1_20170101.tif"]:
        image = rng.integers(0, 256, size=(4, SIZE, SIZE), dtype=np.uint8)
        image[:, :20, :] = 0  # a nodata strip along the top
        write_raster(tmp_path / "naip" / name, image)
        cdl = np.full((SIZE, SIZE), 141, dtype=np.uint8)  # forest
        cdl[:, SIZE // 2 :] = 176  # pasture
        write_raster(tmp_path / "cdl_annotations" / name, cdl)
        roads = np.zeros((SIZE, SIZE), dtype=np.uint8)
        roads[70:75, :] = 1
        write_raster(tmp_path / "road_annotations" / name, roads)
        buildings = np.zeros((SIZE, SIZE), dtype=np.uint8)
        buildings[100:110, 100:110] = 1
        write_raster(tmp_path / "building_annotations" / name, buildings)
    return tmp_path


def class_index(name: str) -> int:
    return 1 + [c.name for c in LABEL_CLASSES].index(name)


def test_class_map_priority(dataset: Path):
    path = data.find_naip_paths(dataset / "naip")[0]
    scene = data.load_scene(path, use_labels=True)
    assert scene.image.shape == (4, SIZE, SIZE)
    assert scene.labels[72, 10] == class_index("roads")  # road wins over forest
    assert scene.labels[105, 105] == class_index("buildings")  # building wins over pasture
    assert scene.labels[150, 10] == class_index("cdl_forest")
    assert scene.labels[150, 150] == class_index("cdl_pasture")


def test_labels_auto_detection(dataset: Path):
    paths = data.find_naip_paths(dataset / "naip")
    assert data.resolve_use_labels("auto", paths)
    assert not data.resolve_use_labels("off", paths)
    (dataset / "road_annotations" / paths[0].name).unlink()
    assert not data.resolve_use_labels("auto", paths)
    with pytest.raises(FileNotFoundError):
        data.resolve_use_labels("on", paths)


def test_patch_sampler_skips_nodata(dataset: Path):
    scenes = data.load_scenes(dataset / "naip", use_labels=True)
    sampler = data.PatchSampler(scenes, 32, augment=True, seed=0)
    images, labels = sampler.sample(16)
    assert images.shape == (16, 4, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (16, 32, 32)
    assert ((images == 0).all(axis=1).mean(axis=(1, 2)) <= 0.01).all()
    onehot = data.labels_to_onehot(labels, "cpu")
    assert onehot.shape == (16, N_LABEL_CHANNELS, 32, 32)


TINY = {
    "batch_size": 2,
    "gan_channels": 32,
    "gan_z_channels": 8,
    "disc_channels": 8,
    "unet_channels": 16,
    "unet_channel_mults": "1,2,2",
    "sample_steps": 3,
    "device": "cpu",
}


@pytest.mark.parametrize("model", ["gan", "diffusion"])
@pytest.mark.parametrize("use_labels", [True, False])
def test_train_step_and_generate(dataset: Path, model: str, use_labels: bool):
    import torch

    cfg = TrainConfig(model=model, patch_size=64, **TINY)
    trainer = make_trainer(cfg, use_labels, torch.device("cpu"))
    scenes = data.load_scenes(dataset / "naip", use_labels)
    images, labels = data.PatchSampler(scenes, 64).sample(2)
    real, onehot = data.batch_to_tensors(images, labels, torch.device("cpu"))
    for _ in range(2):
        losses = trainer.train_step(real, onehot)
    assert all(np.isfinite(v) for v in losses.values())

    # Fully convolutional: generate at a larger size than the training patches
    big_labels = (
        data.labels_to_onehot(scenes[0].labels[None, :128, :128], "cpu") if use_labels else None
    )
    a = trainer.generate(big_labels, 1, 128, 128, seed=3)
    b = trainer.generate(big_labels, 1, 128, 128, seed=3)
    assert a.shape == (1, 4, 128, 128)
    assert a.min() >= -1 and a.max() <= 1
    assert torch.equal(a, b), "same seed should give the same image"


@pytest.mark.parametrize("model", ["gan", "diffusion"])
def test_cli_train_resume_sample(dataset: Path, model: str, monkeypatch):
    monkeypatch.chdir(dataset)
    tiny_flags = [f"--{k.replace('_', '-')}={v}" for k, v in TINY.items() if k != "device"] + [
        "--device=cpu"
    ]
    common = ["--patch-size=64", "--log-every=1", "--sample-every=2", "--checkpoint-every=2"]
    main(["train", f"--model={model}", "--run-name=test", "--steps=4", *common, *tiny_flags])

    run = dataset / "runs" / "test"
    assert (run / "samples" / "0000004.png").exists()
    assert (run / "losses.png").exists()
    assert (run / "checkpoints" / "ema_0000004.pt").exists()

    main(["train", "--resume", str(run / "checkpoints" / "last.pt"), "--steps=6"])
    assert (run / "checkpoints" / "ema_0000006.pt").exists()
    assert len((run / "losses.csv").read_text().strip().splitlines()) == 1 + 6

    main(
        [
            "sample",
            "--checkpoint",
            str(run / "checkpoints" / "ema_0000006.pt"),
            "--n=2",
            "--device=cpu",
        ]
    )
    assert (dataset / "generated" / "patches_seed0.png").exists()

    scene = data.find_naip_paths(dataset / "naip")[0]
    main(
        [
            "large",
            "--checkpoint",
            str(run / "checkpoints" / "last.pt"),
            "--scene",
            str(scene),
            "--size=128",
            "--device=cpu",
        ]
    )
    out = dataset / "generated" / f"{scene.stem}_r0_c0_s128"
    assert (out / "generated_0.png").exists() and (out / "labels.png").exists()
