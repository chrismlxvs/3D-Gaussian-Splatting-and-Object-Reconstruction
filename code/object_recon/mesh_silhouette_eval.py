"""Evaluate two mesh silhouettes on a frozen set of validation cameras.

Uses CPU first-hit ray casting so candidate comparisons do not need the MILo
renderer or a GPU. Validation masks are read-only evaluation data and are never
used to build or modify either mesh.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


def load_gt(name: str, validation_dir: Path, corrected_frame52: Path) -> np.ndarray:
    stem = Path(name).stem
    if stem == "1979_000052":
        with Image.open(corrected_frame52) as image:
            return np.asarray(image.convert("L")) >= 128
    with Image.open(validation_dir / f"{stem}.png") as image:
        rgba = np.asarray(image.convert("RGBA"))
    return rgba[..., 3] >= 128


def evaluate_mesh(mesh_path: Path, views: list[dict], width: int, height: int,
                  intrinsics: tuple[float, float, float, float],
                  validation_dir: Path, corrected_frame52: Path, stride: int,
                  overlay_dir: Path | None) -> list[dict]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"empty mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    scene = o3d.t.geometry.RaycastingScene(nthreads=0, device=o3d.core.Device("CPU:0"))
    scene.add_triangles(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32),
    )
    fx, fy, cx, cy = intrinsics
    xs = np.arange(0, width, stride, dtype=np.float32)
    ys = np.arange(0, height, stride, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    camera_dirs = np.stack(((xx - cx) / fx, (yy - cy) / fy,
                            np.ones_like(xx)), axis=-1).reshape(-1, 3)
    results = []
    for view in views:
        gt = load_gt(view["name"], validation_dir, corrected_frame52)
        if gt.shape != (height, width):
            raise ValueError(f"validation mask size mismatch for {view['name']}: {gt.shape}")
        w2c = np.asarray(view["world_to_camera"], dtype=np.float64)
        rotation, translation = w2c[:3, :3], w2c[:3, 3]
        origin = -rotation.T @ translation
        world_dirs = camera_dirs @ rotation
        rays = np.concatenate((np.broadcast_to(origin, world_dirs.shape), world_dirs), axis=1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32),
                                              dtype=o3d.core.Dtype.Float32))
        predicted = (hit["primitive_ids"].numpy() < len(faces)).reshape(len(ys), len(xs))
        sampled_gt = gt[yy.astype(np.int64), xx.astype(np.int64)]
        intersection = int(np.count_nonzero(predicted & sampled_gt))
        union = int(np.count_nonzero(predicted | sampled_gt))
        results.append({
            "name": view["name"],
            "iou": float(intersection / union) if union else 1.0,
            "gt_sampled_pixels": int(sampled_gt.sum()),
            "mesh_sampled_pixels": int(predicted.sum()),
            "gt_pixels_without_mesh_hit": int(np.count_nonzero(sampled_gt & ~predicted)),
            "mesh_hits_outside_gt": int(np.count_nonzero(predicted & ~sampled_gt)),
        })
        if overlay_dir is not None:
            overlay = np.zeros((*predicted.shape, 3), dtype=np.uint8)
            overlay[sampled_gt] = (0, 190, 0)
            overlay[predicted] = (0, 0, 220)
            overlay[sampled_gt & predicted] = (0, 220, 220)
            overlay_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_dir / (Path(view["name"]).stem + "_overlay.png")),
                        cv2.resize(overlay, (width, height), interpolation=cv2.INTER_NEAREST))
    del scene, mesh
    gc.collect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--corrected-frame52", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--overlay-dir", type=Path)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    metadata = json.loads(args.cameras_json.read_text(encoding="utf-8"))
    intrinsics = tuple(float(metadata[k]) for k in ("fx", "fy", "cx", "cy"))
    views = metadata["views"]
    baseline = evaluate_mesh(args.baseline, views, metadata["width"], metadata["height"],
                             intrinsics, args.validation_dir, args.corrected_frame52,
                             args.stride, None)
    candidate = evaluate_mesh(args.candidate, views, metadata["width"], metadata["height"],
                              intrinsics, args.validation_dir, args.corrected_frame52,
                              args.stride, args.overlay_dir)
    report = {
        "validation_views": len(views),
        "stride": args.stride,
        "baseline_mesh": str(args.baseline),
        "candidate_mesh": str(args.candidate),
        "frame52_mask": str(args.corrected_frame52),
        "validation_used_for_geometry_or_cleanup": False,
        "mean_iou": {
            "baseline": float(np.mean([x["iou"] for x in baseline])),
            "candidate": float(np.mean([x["iou"] for x in candidate])),
        },
        "per_view": [{"name": a["name"], "baseline_iou": a["iou"],
                      "candidate_iou": b["iou"],
                      "delta": b["iou"] - a["iou"],
                      "candidate_gt_pixels_without_mesh_hit": b["gt_pixels_without_mesh_hit"],
                      "candidate_mesh_hits_outside_gt": b["mesh_hits_outside_gt"]}
                     for a, b in zip(baseline, candidate)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["mean_iou"], indent=2))


if __name__ == "__main__":
    main()
