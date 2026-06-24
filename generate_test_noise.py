#!/usr/bin/env python3
"""Generate a random noise PNG in test_imgs/ for watcher testing."""
import argparse
import os
import random
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "--user", "Pillow"])
    from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Generate random noise images.")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="test_imgs")
    parser.add_argument("--prefix", type=str, default="noise")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        pixels = bytes(random.randrange(256) for _ in range(args.width * args.height * 3))
        img = Image.frombytes("RGB", (args.width, args.height), pixels)
        name = f"{args.prefix}_{i:04d}_{random.randrange(1_000_000):06d}.png"
        path = out_dir / name
        img.save(path)
        print(path)


if __name__ == "__main__":
    main()
