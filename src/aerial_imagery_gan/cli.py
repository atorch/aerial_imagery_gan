"""Command line interface: `uv run aerial {train,sample,large,kid} --help`."""

from __future__ import annotations

import argparse
import dataclasses

import torch

from .config import TrainConfig, config_from_dict


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    """One flag per TrainConfig field. Defaults are suppressed so that we can tell which
    flags were passed explicitly (those override the config stored in a checkpoint)."""
    defaults = TrainConfig()
    for field in dataclasses.fields(TrainConfig):
        flag = "--" + field.name.replace("_", "-")
        default = getattr(defaults, field.name)
        if isinstance(default, bool):
            parser.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
                help=f"(default: {default})",
            )
        else:
            parser.add_argument(
                flag,
                type=type(default),
                default=argparse.SUPPRESS,
                help=f"(default: {default})",
                metavar=field.type.upper() if isinstance(field.type, str) else None,
            )


def _sampling_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="last.pt or ema_*.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--sample-steps", type=int, default=None, help="diffusion only: number of Euler steps"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=None, help="diffusion only: CFG scale"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aerial", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train a GAN or diffusion model")
    train_parser.add_argument("--resume", help="path to a last.pt checkpoint to continue from")
    _add_config_flags(train_parser)

    sample_parser = commands.add_parser("sample", help="grid of generated vs real patches")
    _sampling_flags(sample_parser)
    sample_parser.add_argument("--n", type=int, default=8)
    sample_parser.add_argument("--size", type=int, default=None, help="defaults to patch size")
    sample_parser.add_argument("--naip-dir", default=None)
    sample_parser.add_argument("--out", default="generated")

    large_parser = commands.add_parser(
        "large", help="generate a large image from the labels of a window of a real scene"
    )
    _sampling_flags(large_parser)
    large_parser.add_argument("--scene", required=True, help="path to a NAIP .tif")
    large_parser.add_argument("--row", type=int, default=0)
    large_parser.add_argument("--col", type=int, default=0)
    large_parser.add_argument("--size", type=int, default=1024)
    large_parser.add_argument("--n", type=int, default=1)
    large_parser.add_argument("--out", default="generated")

    kid_parser = commands.add_parser(
        "kid", help="Kernel Inception Distance (needs --extra metrics)"
    )
    _sampling_flags(kid_parser)
    kid_parser.add_argument("--n", type=int, default=1000)
    kid_parser.add_argument("--batch-size", type=int, default=16)
    kid_parser.add_argument("--naip-dir", default=None)

    args = vars(parser.parse_args(argv))
    command = args.pop("command")

    if command == "train":
        from .train import train

        resume = args.pop("resume")
        if resume:
            stored = torch.load(resume, map_location="cpu", weights_only=False)["config"]
            cfg = config_from_dict({**stored, **args})
        else:
            cfg = TrainConfig(**args)
        train(cfg, resume=resume)
    elif command == "sample":
        from .sample import sample_patches

        sample_patches(**args)
    elif command == "large":
        from .sample import sample_large

        sample_large(**args)
    elif command == "kid":
        from .sample import kid

        kid(**args)


if __name__ == "__main__":
    main()
