"""Place an anchored plushie mesh on the Garden tabletop."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from estimate_placement import _fit_table_plane, _unit
from utils.cameras import load_camera_json
from utils.colmap import load_points3d_binary


def estimate(garden_points_path: Path, garden_cameras_path: Path,
             object_frame_path: Path, output_path: Path,
             view_name: str = "DSC07956.JPG",
             target_pixel: tuple[float, float] = (470.0, 365.0),
             size_ratio: float = 0.18,
             clearance: float = 0.005) -> dict:
    if size_ratio <= 0 or clearance < 0:
        raise ValueError("size ratio must be positive and clearance nonnegative")
    garden = load_camera_json(garden_cameras_path)
    view = next((v for v in garden.views if v.name == view_name), None)
    if view is None:
        raise ValueError(f"Garden view not found: {view_name}")
    points = load_points3d_binary(garden_points_path).xyz
    normal, offset, surface = _fit_table_plane(points, view, garden.intrinsics)
    camera_up = np.mean([-np.linalg.inv(v.world_to_camera)[:3, 1]
                         for v in garden.views], axis=0)
    if float(normal @ camera_up) < 0:
        normal, offset = -normal, -offset
    up = _unit(normal, "Garden table normal")

    center = np.median(surface, axis=0)
    _, _, vt = np.linalg.svd(surface - center, full_matrices=False)
    xy = (surface - center) @ vt[:2].T
    span = float(np.max(np.quantile(xy, .99, axis=0) -
                        np.quantile(xy, .01, axis=0)))

    obj = json.loads(object_frame_path.read_text(encoding="utf-8"))
    height = float(obj["height_99_percentile"])
    if height <= 0:
        raise ValueError("invalid plushie height")
    scale = span*size_ratio/height
    center_obj = np.asarray(obj.get("surface_area_centroid_object",
                                    (np.asarray(obj["aabb_object_frame"]["min"]) +
                                     np.asarray(obj["aabb_object_frame"]["max"])) / 2.0),
                            dtype=np.float64)
    if center_obj.shape != (3,) or not np.all(np.isfinite(center_obj)):
        raise ValueError("invalid mesh center in object frame")

    px, py = target_pixel
    intr = garden.intrinsics
    ray_cam = np.array([(px-intr.cx)/intr.fx, (py-intr.cy)/intr.fy, 1.0])
    c2w = np.linalg.inv(view.world_to_camera)
    ray_world = c2w[:3, :3] @ ray_cam
    denom = float(up @ ray_world)
    if abs(denom) < 1e-8:
        raise ValueError("target ray is parallel to the tabletop")
    # Target the mesh's measured center in the image. Intersect the pixel ray
    # with a parallel plane at that center's scaled height plus the requested
    # support clearance, then solve the support-origin translation from it.
    center_height_scene = scale * float(center_obj[1])
    target_plane_offset = offset - center_height_scene - clearance
    t = -(float(up @ c2w[:3, 3]) + target_plane_offset) / denom
    if t <= 0:
        raise ValueError("mesh-center target is behind the Garden camera")
    center_scene = c2w[:3, 3] + t * ray_world

    # Anchored object coordinates are right-handed: X across, Y up, Z front.
    obj_right = np.array([1., 0., 0.])
    obj_up = np.array([0., 1., 0.])
    obj_front = np.array([0., 0., 1.])
    camera_center = c2w[:3, 3]
    front = camera_center-center_scene
    front -= up*float(front@up)
    front = _unit(front, "Garden-facing plushie front")
    right = _unit(np.cross(up, front), "Garden-facing plushie right")
    front = _unit(np.cross(right, up), "orthogonal plushie front")
    rotation = np.column_stack((right, up, front)) @ np.column_stack((obj_right, obj_up, obj_front)).T
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("plushie placement is not a proper rotation")
    transform = np.eye(4)
    transform[:3, :3] = scale*rotation
    # The anchored frame origin is the support centroid at local y=0.
    support_scene = center_scene - scale * (rotation @ center_obj)
    actual_clearance = float(up @ support_scene + offset)
    if not np.isclose(actual_clearance, clearance, atol=1e-6):
        raise ValueError(f"support clearance {actual_clearance:.6g} differs from requested {clearance:.6g}")
    transform[:3, 3] = support_scene
    target_front = front.tolist()
    result = {
        "object_to_scene": transform.tolist(),
        "convention": "column vectors; anchored plushie object frame to Garden COLMAP world",
        "transform_type": "uniform-scale rigid transform",
        "object_kind": "bounded-Poisson plushie mesh",
        "object_axes": {"up": obj_up.tolist(), "front": obj_front.tolist()},
        "garden_axes": {"up": up.tolist(), "front": target_front},
        "support_point_object": [0., 0., 0.],
        "contact_point_scene": support_scene.tolist(),
        "mesh_center_object": center_obj.tolist(),
        "mesh_center_scene": center_scene.tolist(),
        "mesh_center_method": obj.get("mesh_center_method", "AABB center fallback"),
        "target_pixel_semantics": "surface-area centroid projected to target pixel; support origin solved on fitted table plane",
        "support_clearance_scene_units": clearance,
        "contact_plane": {"normal": up.tolist(), "offset": offset,
                          "ransac_threshold": 0.025,
                          "inlier_count": int(len(surface))},
        "scale": float(scale),
        "target_pixel": {"view": view_name, "xy": [float(px), float(py)]},
        "table_span_scene_units": span,
        "object_height_units": height,
        "desired_object_height_scene_units": span*size_ratio,
        "size_ratio_to_table_span": size_ratio,
        "note": "support-plane origin seated on fitted tabletop; front aligned toward the establishing camera",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--garden-points", type=Path, required=True)
    p.add_argument("--garden-cameras", type=Path, required=True)
    p.add_argument("--object-frame", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--view", default="DSC07956.JPG")
    p.add_argument("--target-pixel", type=float, nargs=2, default=(470.,365.))
    p.add_argument("--size-ratio", type=float, default=.18)
    p.add_argument("--clearance", type=float, default=.005)
    a = p.parse_args()
    print(json.dumps(estimate(a.garden_points, a.garden_cameras, a.object_frame,
                              a.output, a.view, tuple(a.target_pixel),
                              a.size_ratio, a.clearance), indent=2))

if __name__ == "__main__":
    main()
