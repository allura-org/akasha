#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "Pillow",
# ]
# ///
"""Generate random noise images."""

import argparse
import os
import random
import sys
from pathlib import Path

from PIL import Image


def make_noise_image(
    width: int,
    height: int,
    seed: int | None = None,
) -> Image.Image:
    """Return a new RGB image filled with random bytes."""
    rng = random.Random(seed)
    size = width * height * 3
    data = rng.randbytes(size)
    return Image.frombytes("RGB", (width, height), data)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random noise images.",
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=10,
        help="Number of images to generate (default: 10).",
    )
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=512,
        help="Image width in pixels (default: 512).",
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=512,
        help="Image height in pixels (default: 512).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="noise",
        help="Filename prefix (default: noise).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        help="Output image format/extension, e.g. png, jpg, webp (default: png).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible noise.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.count <= 0:
        print("error: --count must be positive", file=sys.stderr)
        return 1
    if args.width <= 0 or args.height <= 0:
        print("error: width and height must be positive", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    ext = args.format.lstrip(".")
    digits = max(len(str(args.count)), 3)

    for i in range(args.count):
        img = make_noise_image(args.width, args.height, args.seed)
        filename = f"{args.prefix}_{i:0{digits}d}.{ext}"
        out_path = args.output / filename
        img.save(out_path)
        print(out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
