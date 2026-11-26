"""Write simple train-mask overlays for selected normalized frame names."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


def map_files(root: Path) -> dict[str, Path]:
    return {re.sub(r"_frame_", "_", path.stem): path
            for path in root.glob("*.png")}


def read(path: Path) -> np.ndarray:
    image = Image.open(path)
    data = np.asarray(image)
    if data.ndim == 3 and data.shape[-1] == 4:
        return data[..., 3] >= 128
    return np.asarray(image.convert("L")) >= 128


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="first mask RGBA images")
    parser.add_argument("second", type=Path, help="image-v2/fallback mask RGBA images")
    parser.add_argument("output", type=Path)
    parser.add_argument("frames", nargs="+")
    args = parser.parse_args()
    first, second = map_files(args.first), map_files(args.second)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.frames:
        a, b = read(first[name]), read(second[name])
        if a.shape != b.shape:
            raise ValueError(f"resolution mismatch for {name}: {a.shape} vs {b.shape}")
        overlay = np.zeros((*a.shape, 3), dtype=np.uint8)
        overlay[a] = (220, 50, 50)       # first input only
        overlay[b] = (50, 190, 60)       # ours only
        overlay[a & b] = (230, 210, 40)  # agreement
        Image.fromarray(overlay).save(args.output / f"{name}_mask_diff.png")


if __name__ == "__main__":
    main()
