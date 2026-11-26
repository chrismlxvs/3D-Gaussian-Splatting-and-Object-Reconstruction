"""Evaluate a set of user-owned plush meshes on the frozen 12-view mask set.

Uses the same CPU first-hit ray
casting and stride as mesh_silhouette_eval.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_silhouette_eval import evaluate_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--corrected-frame52", type=Path, required=True)
    parser.add_argument("--mesh", action="append", required=True,
                        help="repeat as NAME=/absolute/path/to/mesh.ply")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    metadata = json.loads(args.cameras_json.read_text(encoding="utf-8"))
    intrinsics = tuple(float(metadata[k]) for k in ("fx", "fy", "cx", "cy"))

    meshes: dict[str, Path] = {}
    for spec in args.mesh:
        if "=" not in spec:
            parser.error("--mesh must be NAME=PATH")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not name or name in meshes:
            parser.error(f"empty or repeated mesh label: {name!r}")
        meshes[name] = path

    per_mesh = {}
    for name, path in meshes.items():
        rows = evaluate_mesh(path, metadata["views"], metadata["width"],
                             metadata["height"], intrinsics,
                             args.validation_dir, args.corrected_frame52,
                             args.stride, None)
        per_mesh[name] = {
            "path": str(path),
            "mean_iou": float(np.mean([row["iou"] for row in rows])),
            "per_view": rows,
        }
    report = {
        "validation_views": len(metadata["views"]),
        "stride": args.stride,
        "comparison_scope": "selected local iterations and outputs",
        "frame52_mask": str(args.corrected_frame52),
        "validation_used_for_geometry_or_cleanup": False,
        "meshes": per_mesh,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: row["mean_iou"] for name, row in per_mesh.items()}, indent=2))


if __name__ == "__main__":
    main()
