"""Compare two training-only binary mask sets by normalized frame name."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def mask_map(root: Path) -> dict[str, Path]:
    result = {}
    for path in root.glob("*.png"):
        name = re.sub(r"_frame_", "_", path.stem)
        result[name] = path
    return result


def read_mask(path: Path) -> np.ndarray:
    image = Image.open(path)
    if image.mode == "RGBA":
        return np.asarray(image)[..., 3] >= 128
    return np.asarray(image.convert("L")) >= 128


def compare(first: Path, second: Path) -> dict:
    a, b = mask_map(first), mask_map(second)
    names = sorted(a.keys() & b.keys())
    rows = []
    for name in names:
        m1, m2 = read_mask(a[name]), read_mask(b[name])
        if m1.shape != m2.shape:
            raise ValueError(f"shape mismatch for {name}: {m1.shape} vs {m2.shape}")
        intersection = int(np.count_nonzero(m1 & m2))
        union = int(np.count_nonzero(m1 | m2))
        rows.append({"name": name,
                     "iou": float(intersection / union) if union else 1.0,
                     "first_area": int(m1.sum()), "second_area": int(m2.sum()),
                     "relative_area_delta": float((m2.sum() - m1.sum()) /
                                                   max(int(m1.sum()), 1))})
    rows.sort(key=lambda row: row["iou"])
    return {
        "first_dir": str(first), "second_dir": str(second),
        "first_files": len(a), "second_files": len(b), "shared_frames": len(names),
        "first_only": sorted(a.keys() - b.keys()),
        "second_only": sorted(b.keys() - a.keys()),
        "mean_iou": float(np.mean([row["iou"] for row in rows])) if rows else None,
        "median_iou": float(np.median([row["iou"] for row in rows])) if rows else None,
        "per_frame": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.first, args.second)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("first_files", "second_files", "shared_frames",
                       "first_only", "second_only", "mean_iou", "median_iou")},
                     indent=2))


if __name__ == "__main__":
    main()
