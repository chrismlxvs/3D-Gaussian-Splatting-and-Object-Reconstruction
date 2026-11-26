"""Estimate a tabletop object placement from its axes and a Garden surface.

The candidate is derived from Garden COLMAP points projected into a selected
validation view, plus the fixed object validation poses. It is intended for
fusion preview and visual review, not as an unmeasured physical calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from utils.cameras import load_camera_json  # noqa: E402
from utils.colmap import load_points3d_binary  # noqa: E402


def _unit(v: np.ndarray, label: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-8:
        raise ValueError(f"cannot normalize {label}")
    return v / n


def _fit_table_plane(points: np.ndarray, camera, intrinsics, seed: int = 7):
    w2c = camera.world_to_camera
    q = (w2c[:3, :3] @ points.T + w2c[:3, 3:4]).T
    z = q[:, 2]
    u = intrinsics.fx * q[:, 0] / z + intrinsics.cx
    v = intrinsics.fy * q[:, 1] / z + intrinsics.cy
    # This ROI covers the visible tabletop in DSC07956. Depth bounds reject
    # the hedge behind the table and the patio beneath it.
    roi = ((z > 2.5) & (z < 5.0) & (u > 350) & (u < 990) &
           (v > 335) & (v < 465))
    candidates = points[roi]
    if len(candidates) < 100:
        raise ValueError("too few Garden COLMAP points in tabletop ROI")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(2000):
        a, b, c = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        offset = -float(normal @ a)
        mask = np.abs(candidates @ normal + offset) < 0.025
        if best is None or int(mask.sum()) > best[0]:
            best = (int(mask.sum()), normal, offset, mask)
    if best is None or best[0] < 1000:
        raise ValueError("could not robustly fit the Garden tabletop plane")
    _, normal, offset, inliers = best
    return normal, offset, candidates[inliers]


def estimate_placement(garden_points_path: Path, garden_cameras_path: Path,
                       object_cameras_path: Path, object_geometry_path: Path,
                       output_path: Path, view_name: str = "DSC07956.JPG",
                       target_pixel: tuple[float, float] = (470.0, 395.0),
                       size_ratio: float = 0.14,
                       support_clearance_scene_units: float = 0.0) -> dict:
    if size_ratio <= 0:
        raise ValueError("size_ratio must be positive")
    if support_clearance_scene_units < 0:
        raise ValueError("support clearance cannot be negative")
    garden = load_camera_json(garden_cameras_path)
    garden_view = next((v for v in garden.views if v.name == view_name), None)
    if garden_view is None:
        raise ValueError(f"Garden camera {view_name!r} was not found")
    points = load_points3d_binary(garden_points_path).xyz
    normal, offset, surface_points = _fit_table_plane(
        points, garden_view, garden.intrinsics)

    camera_up = np.mean([-np.linalg.inv(v.world_to_camera)[:3, 1]
                         for v in garden.views], axis=0)
    up = _unit(camera_up, "Garden camera up")
    if float(normal @ up) < 0:
        normal, offset = -normal, -offset
    garden_up = _unit(normal, "table normal")

    # Estimate the tabletop's major in-plane span with PCA, trimming edge
    # outliers before setting a conservative object height.
    centered = surface_points - np.median(surface_points, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:2]
    coords = centered @ axes.T
    lo, hi = np.quantile(coords, [0.01, 0.99], axis=0)
    table_span = float(np.max(hi - lo))

    px, py = target_pixel
    ray_camera = np.array([(px - garden.intrinsics.cx) / garden.intrinsics.fx,
                           (py - garden.intrinsics.cy) / garden.intrinsics.fy,
                           1.0])
    c2w = np.linalg.inv(garden_view.world_to_camera)
    ray_world = c2w[:3, :3] @ ray_camera
    camera_center = c2w[:3, 3]
    denom = float(garden_up @ ray_world)
    if abs(denom) < 1e-7:
        raise ValueError("target pixel ray is parallel to the fitted tabletop")
    distance = -(float(garden_up @ camera_center) + offset) / denom
    if distance <= 0:
        raise ValueError("tabletop intersection lies behind the selected camera")
    plane_target = camera_center + distance * ray_world
    target = plane_target + support_clearance_scene_units * garden_up

    object_meta = json.loads(object_geometry_path.read_text(encoding="utf-8"))
    object_up = _unit(np.asarray(object_meta["object_up"], dtype=np.float64),
                      "object up")
    object_cameras = load_camera_json(object_cameras_path)
    measured_object_up = _unit(np.mean(
        [-np.linalg.inv(v.world_to_camera)[:3, 1] for v in object_cameras.views],
        axis=0), "object validation camera up")
    if float(object_up @ measured_object_up) < 0.95:
        raise ValueError("stored object up axis disagrees with validation poses")
    object_front = np.asarray(object_meta["object_front_toward_cameras"],
                              dtype=np.float64)
    object_front = _unit(object_front - object_up * float(object_front @ object_up),
                         "object front projected onto its horizontal plane")
    object_right = _unit(np.cross(object_up, object_front), "object right")
    object_basis = np.column_stack([object_right, object_up, object_front])

    garden_front = camera_center - target
    garden_front -= garden_up * float(garden_front @ garden_up)
    garden_front = _unit(garden_front, "Garden front")
    garden_right = _unit(np.cross(garden_up, garden_front), "Garden right")
    garden_basis = np.column_stack([garden_right, garden_up, garden_front])
    rotation = garden_basis @ object_basis.T
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("estimated orientation is not rigid")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("estimated orientation is not a proper rotation")

    heights = object_meta["height_percentiles"]
    object_height = float(heights["0.99"] - heights["0.01"])
    if object_height <= 0 or table_span <= 0:
        raise ValueError("invalid object or tabletop extent")
    desired_height = table_span * size_ratio
    scale = desired_height / object_height
    support = np.asarray(object_meta["support_centroid"], dtype=np.float64)
    translation = target - scale * (rotation @ support)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation

    result = {
        "object_to_scene": transform.tolist(),
        "convention": "column vectors; object COLMAP world to Garden COLMAP world",
        "transform_type": "uniform-scale rigid transform",
        "object_axes": {"up": object_up.tolist(), "front": object_front.tolist()},
        "garden_axes": {"up": garden_up.tolist(), "front": garden_front.tolist()},
        "support_point_object": support.tolist(),
        "contact_point_scene": target.tolist(),
        "table_plane_point_scene": plane_target.tolist(),
        "support_clearance_scene_units": support_clearance_scene_units,
        "contact_plane": {"normal": garden_up.tolist(), "offset": offset,
                           "ransac_threshold": 0.025,
                           "inlier_count": int(len(surface_points))},
        "scale": scale,
        "target_pixel": {"view": view_name, "xy": [px, py]},
        "table_span_scene_units": table_span,
        "object_height_units": object_height,
        "desired_object_height_scene_units": desired_height,
        "size_ratio_to_table_span": size_ratio,
        "note": "candidate transform; small support clearance compensates for Gaussian splat thickness above the fitted tabletop plane",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--garden-points", type=Path, required=True)
    parser.add_argument("--garden-cameras", type=Path, required=True)
    parser.add_argument("--object-cameras", type=Path, required=True)
    parser.add_argument("--object-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", default="DSC07956.JPG")
    parser.add_argument("--target-pixel", nargs=2, type=float, default=(470.0, 395.0))
    parser.add_argument("--size-ratio", type=float, default=0.14)
    parser.add_argument("--support-clearance", type=float, default=0.0,
                        help="raise the object support point above the fitted plane to avoid splat interpenetration")
    args = parser.parse_args()
    result = estimate_placement(
        args.garden_points, args.garden_cameras, args.object_cameras,
        args.object_geometry, args.output, args.view,
        tuple(args.target_pixel), args.size_ratio, args.support_clearance)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
