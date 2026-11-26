"""Create a continuous Garden orbit that passes through a vase/object occlusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from estimate_placement import _unit
from estimate_placement import _fit_table_plane
from utils.cameras import load_camera_json
from utils.colmap import load_points3d_binary


def _ray_plane_hit(camera, intrinsics, plane_normal, plane_offset,
                   pixel: tuple[float, float]) -> np.ndarray:
    c2w = np.linalg.inv(camera.world_to_camera)
    px, py = pixel
    ray_camera = np.array([(px - intrinsics.cx) / intrinsics.fx,
                           (py - intrinsics.cy) / intrinsics.fy, 1.0])
    ray_world = c2w[:3, :3] @ ray_camera
    t = -(float(plane_normal @ c2w[:3, 3]) + plane_offset) / float(plane_normal @ ray_world)
    if t <= 0:
        raise ValueError("tabletop intersection is behind Garden camera")
    return c2w[:3, 3] + t * ray_world


def make_path(garden_points: Path, garden_cameras: Path, placement_path: Path,
              output_path: Path, frames: int = 240, fps: int = 24,
              resolution: tuple[int, int] = (1280, 720),
              vase_pixel: tuple[float, float] = (650.0, 360.0),
              establishing_frames: int = 36, final_hold_frames: int = 36,
              sweep_degrees: float = 150.0,
              elevation_lift: float = 0.0,
              return_degrees: float | None = None) -> dict:
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    garden = load_camera_json(garden_cameras)
    camera = next(v for v in garden.views
                  if v.name == placement["target_pixel"]["view"])
    points = load_points3d_binary(garden_points).xyz
    normal, offset, _ = _fit_table_plane(points, camera, garden.intrinsics)
    up = np.asarray(placement["garden_axes"]["up"], dtype=np.float64)
    if normal @ up < 0:
        normal, offset = -normal, -offset
    up = _unit(normal, "tabletop up")
    object_point = np.asarray(placement["contact_point_scene"], dtype=np.float64)
    vase_point = _ray_plane_hit(camera, garden.intrinsics, up, offset, vase_pixel)
    pivot = 0.5 * (object_point + vase_point)

    c2w_initial = np.linalg.inv(camera.world_to_camera)
    initial_center = c2w_initial[:3, 3]
    radial = initial_center - pivot
    camera_height = float(radial @ up)
    radial -= camera_height * up
    radius = float(np.linalg.norm(radial))
    radial = _unit(radial, "initial horizontal camera radius")
    axis_u = c2w_initial[:3, 0] - up * float(c2w_initial[:3, 0] @ up)
    axis_u = _unit(axis_u, "orbit axis u")
    axis_v = _unit(np.cross(up, axis_u), "orbit axis v")
    start_angle = float(np.arctan2(radial @ axis_v, radial @ axis_u))
    pair = vase_point - object_point
    pair -= up * float(pair @ up)
    pair = _unit(pair, "object-to-vase direction")
    vase_angle = float(np.arctan2(pair @ axis_v, pair @ axis_u))
    delta = (vase_angle - start_angle + np.pi) % (2 * np.pi) - np.pi
    if abs(delta) < np.deg2rad(12):
        raise ValueError("starting view already aligns the vase and object; choose another view")
    # Move through the object-vase alignment and continue far enough to show
    # the object clear again. A single partial arc keeps the narrative focused
    # and avoids spending most of the clip on a redundant full revolution.
    peak_angle = np.sign(delta) * np.deg2rad(abs(sweep_degrees))
    if return_degrees is None and abs(peak_angle) <= abs(delta) + np.deg2rad(15):
        raise ValueError("sweep must pass the vase alignment and continue at least 15 degrees")
    if return_degrees is not None and not 0 <= return_degrees < abs(sweep_degrees):
        raise ValueError("return angle must be nonnegative and smaller than the peak sweep")
    end_angle = (np.sign(delta) * np.deg2rad(return_degrees)
                 if return_degrees is not None else peak_angle)
    if establishing_frames + final_hold_frames >= frames:
        raise ValueError("holds leave no frames for camera motion")

    width, height = resolution
    focal = 1410.0 * width / 1280.0
    k = {"width": width, "height": height, "fx": focal, "fy": focal,
         "cx": width / 2.0, "cy": height / 2.0}
    views = []
    angle_rows = []
    motion_frames = frames - establishing_frames - final_hold_frames
    if return_degrees is None:
        align_fraction = float(np.clip(abs(delta / end_angle), 0.0, 1.0))
        predicted_alignment = int(establishing_frames + motion_frames * align_fraction)
    elif abs(delta) <= abs(peak_angle):
        align_fraction = float(np.clip(abs(delta / peak_angle), 0.0, 1.0))
        predicted_alignment = int(establishing_frames + motion_frames * 0.5 * align_fraction)
    else:
        predicted_alignment = None
    peak_overlap_frame = (establishing_frames + motion_frames // 2
                          if return_degrees is not None else predicted_alignment)
    target = pivot + up * 0.15
    for index in range(frames):
        if index < establishing_frames:
            t = 0.0
        elif index >= frames - final_hold_frames:
            t = 1.0
        else:
            raw = (index - establishing_frames) / max(motion_frames - 1, 1)
            if return_degrees is None:
                t = raw * raw * (3.0 - 2.0 * raw)
            else:
                t = raw
        if return_degrees is None:
            angle = start_angle + end_angle * t
        elif index < establishing_frames:
            angle = start_angle
        elif index >= frames - final_hold_frames:
            angle = start_angle + end_angle
        elif t <= 0.5:
            eased = (2*t) ** 2 * (3 - 2*(2*t))
            angle = start_angle + peak_angle*eased
        else:
            eased = (2*t-1) ** 2 * (3 - 2*(2*t-1))
            angle = start_angle + peak_angle + (end_angle-peak_angle)*eased
        direction = np.cos(angle) * axis_u + np.sin(angle) * axis_v
        center = (pivot + radius * direction + camera_height * up
                  + elevation_lift * t * up)
        forward = _unit(target - center, "camera forward")
        right = _unit(np.cross(forward, up), "camera right")
        down = _unit(np.cross(forward, right), "camera down")
        rotation = np.stack([right, down, forward])
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("generated camera rotation is not right-handed")
        w2c = np.eye(4)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = -rotation @ center
        views.append({"name": f"frame_{index:03d}.png",
                      "world_to_camera": w2c.tolist()})
        angle_rows.append(angle)

    result = {
        **k,
        "views": views,
        "fps": fps,
        "duration_seconds": frames / fps,
        "source": "Garden COLMAP frame; smooth orbit centered on the inserted camera and existing vase",
        "object_contact": object_point.tolist(),
        "vase_table_point": vase_point.tolist(),
        "orbit_radius_scene_units": radius,
        "camera_height_above_table_scene_units": camera_height,
        "end_elevation_lift_scene_units": elevation_lift,
        "orbit_motion": ("smooth out-and-back arc" if return_degrees is not None
                         else "smooth partial arc"),
        "peak_sweep_degrees": float(np.rad2deg(peak_angle)),
        "ending_offset_degrees": (float(np.rad2deg(end_angle))
                                   if return_degrees is not None else None),
        "establishing_frames": establishing_frames,
        "final_hold_frames": final_hold_frames,
        "predicted_vase_object_alignment_frame": predicted_alignment,
        "predicted_peak_overlap_frame": peak_overlap_frame,
        "start_angle_rad": start_angle,
        "end_angle_rad": start_angle + end_angle,
        "peak_angle_rad": start_angle + peak_angle,
        "peak_rotation_degrees": float(np.rad2deg(peak_angle)),
        "sweep_degrees": float(np.rad2deg(peak_angle)),
        "full_revolution": False,
        "note": ("out-and-back swing passes through vase occlusion and returns to a clear view; verify rendered frames"
                 if return_degrees is not None else
                 "partial arc establishes placement, passes vase occlusion, and continues to reveal; verify rendered frames"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--garden-points", type=Path, required=True)
    parser.add_argument("--garden-cameras", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--sweep-degrees", type=float, default=150.0)
    parser.add_argument("--elevation-lift", type=float, default=0.0,
                        help="smoothly raise the moving orbit camera toward the end")
    parser.add_argument("--return-degrees", type=float,
                        help="return toward the establishing view and end at this offset")
    args = parser.parse_args()
    print(json.dumps(make_path(args.garden_points, args.garden_cameras,
                               args.placement, args.output, frames=args.frames,
                               sweep_degrees=args.sweep_degrees,
                               elevation_lift=args.elevation_lift,
                               return_degrees=args.return_degrees),
                     indent=2))


if __name__ == "__main__":
    main()
