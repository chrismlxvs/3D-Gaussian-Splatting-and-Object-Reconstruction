"""Place the source-view-validated upright anchor at the existing Garden target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unit(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length < 1e-10:
        raise ValueError("degenerate placement axis")
    return value / length


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--object-frame", type=Path, required=True)
    p.add_argument("--reference-object-frame", type=Path, required=True)
    p.add_argument("--reference-placement", type=Path, required=True)
    p.add_argument("--garden-cameras", type=Path, required=True)
    p.add_argument("--view", default="DSC07956.JPG")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size-ratio", type=float, default=0.15)
    p.add_argument("--clearance", type=float, default=0.005)
    p.add_argument("--target-pixel", type=float, nargs=2, default=(500.0, 365.0))
    args = p.parse_args()

    obj = json.loads(args.object_frame.read_text(encoding="utf-8"))
    ref_obj = json.loads(args.reference_object_frame.read_text(encoding="utf-8"))
    ref = json.loads(args.reference_placement.read_text(encoding="utf-8"))
    cameras = json.loads(args.garden_cameras.read_text(encoding="utf-8"))
    up = unit(np.asarray(ref["garden_axes"]["up"], dtype=np.float64))
    front = np.asarray(ref["garden_axes"]["front"], dtype=np.float64)
    front -= up * float(front @ up)
    front = unit(front)
    right = unit(np.cross(up, front))
    front = unit(np.cross(right, up))
    rotation = np.column_stack((right, up, front))
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("placement rotation is not right-handed")

    height = float(obj["height_99_percentile"])
    table_span = float(ref["table_span_scene_units"])
    scale = table_span * args.size_ratio / height
    if "surface_area_centroid_object" in obj:
        center_obj = np.asarray(obj["surface_area_centroid_object"], dtype=np.float64)
    else:
        old_center = np.asarray(ref["mesh_center_object"], dtype=np.float64)
        to_new_object = (np.linalg.inv(np.asarray(obj["T_colmap_from_object"], dtype=np.float64)) @
                         np.asarray(ref_obj["T_colmap_from_object"], dtype=np.float64))
        center_obj = to_new_object[:3, :3] @ old_center + to_new_object[:3, 3]
    min_y = float(obj["aabb_object_frame"]["min"][1])
    intr = {k: cameras[k] for k in ("width", "height", "fx", "fy", "cx", "cy")}
    source_view = next((v for v in cameras["views"] if v["name"] == args.view), None)
    if source_view is None:
        raise ValueError(f"Garden view not found: {args.view}")
    w2c = np.asarray(source_view["world_to_camera"], dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    px, py = args.target_pixel
    ray_camera = np.array([(px - intr["cx"]) / intr["fx"],
                           (py - intr["cy"]) / intr["fy"], 1.0])
    ray_world = c2w[:3, :3] @ ray_camera
    plane = ref["contact_plane"]
    offset = float(plane["offset"])
    center_height = scale * (float(center_obj[1]) - min_y) + args.clearance
    denom = float(up @ ray_world)
    if abs(denom) < 1e-8:
        raise ValueError("target pixel ray is parallel to tabletop")
    t = -(float(up @ c2w[:3, 3]) + offset - center_height) / denom
    if t <= 0:
        raise ValueError("target plane point is behind the camera")
    center_scene = c2w[:3, 3] + t * ray_world
    support_scene = center_scene - scale * (rotation @ center_obj)
    support_height = float(up @ support_scene + offset)
    expected_support_height = args.clearance - scale * min_y
    if not np.isclose(support_height, expected_support_height, atol=1e-6):
        raise ValueError("support origin height does not match mesh minimum clearance")

    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = support_scene
    target_view = args.view
    result = {
        "object_to_scene": transform.tolist(),
        "convention": "column vectors; source-view-validated upright object frame to Garden world",
        "transform_type": "uniform-scale rigid transform",
        "object_kind": "fresh MILo reconstructed plush mesh; no vertical inversion correction",
        "object_axes": {"up": [0.0, 1.0, 0.0], "front": [0.0, 0.0, 1.0]},
        "garden_axes": {"up": up.tolist(), "front": front.tolist()},
        "support_point_object": [0.0, 0.0, 0.0],
        "contact_point_scene": support_scene.tolist(),
        "mesh_center_object": center_obj.tolist(),
        "mesh_center_scene": center_scene.tolist(),
        "mesh_center_method": obj.get("mesh_center_method", ref["mesh_center_method"]),
        "target_pixel_semantics": "triangle-area centroid projects to the target; mesh minimum is seated above the table plane",
        "support_clearance_scene_units": float(args.clearance),
        "clearance_semantics": "minimum mesh vertex height above fitted Garden tabletop",
        "mesh_minimum_y_object_units": min_y,
        "anchor_origin_height_above_plane_scene_units": support_height,
        "contact_plane": plane,
        "scale": float(scale),
        "target_pixel": {"view": target_view, "xy": [float(px), float(py)]},
        "table_span_scene_units": table_span,
        "object_height_units": height,
        "desired_object_height_scene_units": table_span * args.size_ratio,
        "size_ratio_to_table_span": float(args.size_ratio),
        "note": "head-up orientation is taken from the source-view-validated support normal; camera and object transforms stay fixed across the clip",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scale": scale, "target_pixel": args.target_pixel,
                      "contact_point_scene": support_scene.tolist(),
                      "minimum_clearance": args.clearance,
                      "support_origin_height": support_height}, indent=2))


if __name__ == "__main__":
    main()
