"""Audit mesh support against training silhouettes without changing geometry.

The audit uses CPU ray casting to count first-hit faces and compare rendered
silhouettes with matching training masks. It does not alter the input mesh.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def read_cameras(path: Path) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model = int(fields[0]), fields[1]
        width, height = map(int, fields[2:4])
        params = list(map(float, fields[4:]))
        if model == "PINHOLE" and len(params) == 4:
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE" and len(params) == 3:
            fx = fy = params[0]
            cx, cy = params[1:]
        elif model in {"SIMPLE_RADIAL", "RADIAL"} and len(params) >= 4:
            # This audit supports pinhole projections only; reject distortion
            # rather than silently mis-score support.
            if any(abs(x) > 1e-12 for x in params[3:]):
                raise ValueError(f"distorted camera {camera_id} requires undistorted inputs")
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(f"unsupported COLMAP camera model: {model}")
        result[camera_id] = {"width": width, "height": height,
                             "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return result


def read_images(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"malformed COLMAP image record: {line[:100]}")
        records.append({
            "name": fields[9],
            "q": np.asarray(list(map(float, fields[1:5])), dtype=np.float64),
            "t": np.asarray(list(map(float, fields[5:8])), dtype=np.float64),
            "camera_id": int(fields[8]),
        })
        # Each image record is followed by a possibly empty 2D-point line.
        if i < len(lines):
            i += 1
    return records


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def audit(mesh_path: Path, cameras_path: Path, images_path: Path,
          masks_dir: Path, csv_path: Path, report_path: Path,
          stride: int, min_views: int, preview_dir: Path | None,
          preview_images: set[str]) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"empty mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    cameras = read_cameras(cameras_path)
    images = read_images(images_path)
    scene = o3d.t.geometry.RaycastingScene(nthreads=0, device=o3d.core.Device("CPU:0"))
    scene.add_triangles(
        o3d.core.Tensor(vertices.astype(np.float32), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(faces.astype(np.uint32), dtype=o3d.core.Dtype.UInt32),
    )
    visible_views = np.zeros(len(faces), dtype=np.uint16)
    visible_rays = np.zeros(len(faces), dtype=np.uint32)
    foreground_rays = np.zeros(len(faces), dtype=np.uint32)
    views_used = []
    view_scores = []

    for image in images:
        mask_path = masks_dir / image["name"]
        if not mask_path.is_file():
            # Common mask pipelines change only the image suffix.
            alternatives = [masks_dir / (Path(image["name"]).stem + ext)
                            for ext in (".png", ".jpg", ".jpeg")]
            mask_path = next((p for p in alternatives if p.is_file()), mask_path)
        if not mask_path.is_file():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        camera = cameras[image["camera_id"]]
        h, w = mask.shape
        if (w, h) != (camera["width"], camera["height"]):
            raise ValueError(f"mask/intrinsics resolution mismatch: {image['name']}")
        rotation = qvec2rotmat(image["q"])
        xs = np.arange(0, w, stride, dtype=np.float32)
        ys = np.arange(0, h, stride, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        camera_dirs = np.stack(((xx - camera["cx"]) / camera["fx"],
                                (yy - camera["cy"]) / camera["fy"],
                                np.ones_like(xx)), axis=-1).reshape(-1, 3)
        world_dirs = camera_dirs @ rotation
        origin = -rotation.T @ image["t"]
        rays = np.concatenate((np.broadcast_to(origin, world_dirs.shape), world_dirs), axis=1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32),
                                              dtype=o3d.core.Dtype.Float32))
        primitive_ids = hit["primitive_ids"].numpy().astype(np.int64)
        valid = (primitive_ids >= 0) & (primitive_ids < len(faces))
        sampled_mask = mask[yy.astype(np.int64).ravel(), xx.astype(np.int64).ravel()] >= 128
        hit_mask = valid & sampled_mask
        hit_background = valid & ~sampled_mask
        counts = np.bincount(primitive_ids[valid], minlength=len(faces))
        foreground_counts = np.bincount(primitive_ids[hit_mask], minlength=len(faces))
        background_counts = np.bincount(primitive_ids[hit_background], minlength=len(faces))
        visible_rays += counts.astype(np.uint32)
        foreground_rays += foreground_counts.astype(np.uint32)
        visible_views += (counts > 0).astype(np.uint16)
        predicted = valid.reshape(len(ys), len(xs))
        gt = sampled_mask.reshape(len(ys), len(xs))
        intersection = int(np.count_nonzero(predicted & gt))
        union = int(np.count_nonzero(predicted | gt))
        view_scores.append({
            "image": image["name"],
            "sampled_silhouette_iou": float(intersection / union) if union else 1.0,
            "foreground_sampled_pixels": int(gt.sum()),
            "foreground_pixels_without_mesh_hit": int(np.count_nonzero(gt & ~predicted)),
            "mesh_hits_outside_mask": int(np.count_nonzero(predicted & ~gt)),
        })
        if preview_dir is not None and image["name"] in preview_images:
            overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
            overlay[gt] = (0, 190, 0)  # ground-truth foreground
            overlay[predicted] = (0, 0, 220)  # mesh hit
            overlay[gt & predicted] = (0, 220, 220)  # overlap
            preview_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_dir / (Path(image["name"]).stem + "_overlay.png")),
                        cv2.resize(overlay, (w, h), interpolation=cv2.INTER_NEAREST))
        views_used.append(image["name"])

    eligible = visible_views >= min_views
    outside_fraction = np.divide(
        visible_rays - foreground_rays, np.maximum(visible_rays, 1), dtype=np.float64
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["face_index", "v0", "v1", "v2", "visible_train_views",
                         "visible_training_rays", "visible_rays_in_mask",
                         "visible_ray_outside_fraction", "eligible_min_views"])
        for face_id, face in enumerate(faces):
            writer.writerow([face_id, *face.tolist(), int(visible_views[face_id]),
                             int(visible_rays[face_id]), int(foreground_rays[face_id]),
                             float(outside_fraction[face_id]), int(eligible[face_id])])

    report = {
        "mesh": str(mesh_path),
        "registered_training_views_used": len(views_used),
        "validation_or_test_views_used": 0,
        "faces": len(faces),
        "visibility_audit": {
            "method": "CPU first-hit ray casting at sampled pixel centers",
            "pixel_stride": stride,
            "minimum_projected_views_for_summary": min_views,
            "faces_visible_in_minimum_views": int(eligible.sum()),
            "faces_never_visible": int((visible_views == 0).sum()),
            "visible_faces_with_no_in_mask_rays": int(((visible_views > 0) & (foreground_rays == 0)).sum()),
            "visible_ray_outside_fraction_quantiles": np.quantile(
                outside_fraction[visible_rays > 0], [0, .05, .25, .5, .75, .95, 1]
            ).tolist() if np.any(visible_rays > 0) else [],
            "mean_sampled_silhouette_iou": float(np.mean([x["sampled_silhouette_iou"] for x in view_scores])) if view_scores else None,
            "per_view_silhouette_scores": view_scores,
        },
        "per_face_csv": str(csv_path),
        "interpretation": "A face never hit by these cameras is unsupported by the observed first surfaces, not necessarily wrong. Do not delete or fill faces from this report alone; hidden underside regions need independent evidence.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True, help="COLMAP cameras.txt")
    parser.add_argument("--images", type=Path, required=True, help="training-only COLMAP images.txt")
    parser.add_argument("--masks", type=Path, required=True, help="training-only mask directory")
    parser.add_argument("--csv", type=Path, required=True, help="per-face proxy output")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2, help="sample every Nth image pixel")
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--preview-dir", type=Path,
                        help="optional directory for mask/mesh overlay images")
    parser.add_argument("--preview-images", nargs="*", default=[],
                        help="image names for preview overlays, e.g. 1979_000020.png")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    report = audit(args.mesh, args.cameras, args.images, args.masks, args.csv,
                   args.report, args.stride, args.min_views, args.preview_dir,
                   set(args.preview_images))
    visibility = report["visibility_audit"]
    worst = sorted(visibility["per_view_silhouette_scores"],
                   key=lambda x: x["sampled_silhouette_iou"])[:5]
    print(json.dumps({"mesh": report["mesh"],
                      "registered_training_views_used": report["registered_training_views_used"],
                      "faces": report["faces"],
                      "visibility_summary": {k: v for k, v in visibility.items()
                                             if k != "per_view_silhouette_scores"},
                      "worst_views": worst,
                      "report": str(args.report), "per_face_csv": str(args.csv)}, indent=2))


if __name__ == "__main__":
    main()
