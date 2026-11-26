"""Cull SDF mesh faces that disagree with registered training silhouettes.

Uses only registered training cameras and training masks. A face receives a
vote in each camera where its centroid projects into the image; hidden surfaces
remain when their projections fall inside the foreground silhouette. The
reserved test clip is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ], dtype=np.float32)


def read_colmap(cameras_path: Path, images_path: Path):
    cameras = {}
    for line in cameras_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        cameras[int(f[0])] = (int(f[2]), int(f[3]), *map(float, f[4:8]))
    records = []
    lines = images_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line or line.startswith("#"):
            continue
        f = line.split()
        records.append({
            "name": f[-1], "q": np.asarray(list(map(float, f[1:5])), np.float32),
            "t": np.asarray(list(map(float, f[5:8])), np.float32),
            "camera": cameras[int(f[8])],
        })
    return records


def cull(source: Path, cameras_txt: Path, images_txt: Path, masks_dir: Path,
         report_path: Path, output: Path, tolerance_px: int = 3,
         outside_ratio: float = 0.10, min_projected_views: int = 3,
         plane_margin: float = 0.04) -> dict:
    prep = json.loads(report_path.read_text(encoding="utf-8"))
    plane = prep["support_plane"]
    if plane.get("status") != "ok":
        raise ValueError("support plane missing or invalid")
    mesh = o3d.io.read_triangle_mesh(str(source))
    if mesh.is_empty() or not mesh.has_vertex_colors():
        raise ValueError("source mesh is empty or has no vertex colors")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    n_faces = len(faces)
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz = torch.from_numpy(vertices).to(device)
    visible = np.zeros(n_faces, dtype=np.int64)
    outside = np.zeros(n_faces, dtype=np.int64)
    face_centers = xyz[torch.from_numpy(faces).to(device=device, dtype=torch.long)].mean(dim=1)
    views_used = 0
    for cam in read_colmap(cameras_txt, images_txt):
        mask_path = masks_dir / cam["name"]
        if not mask_path.is_file():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape
        dilated = cv2.dilate(mask, np.ones((2*tolerance_px+1, 2*tolerance_px+1), np.uint8)) > 0
        width, height, fx, fy, cx, cy = cam["camera"]
        if (w, h) != (width, height):
            raise ValueError(f"image/mask resolution mismatch for {cam['name']}: {(w,h)} vs {(width,height)}")
        R = qvec2rotmat(cam["q"])
        # Project each face centroid into this registered camera. Back-facing
        # surfaces are intentionally allowed to vote when their projection lies
        # inside the foreground silhouette; this preserves hidden surface faces.
        center_cam = face_centers @ torch.from_numpy(R.T.copy()).to(device) + torch.from_numpy(cam["t"]).to(device)
        cz = center_cam[:, 2]
        safe_cz = cz.clamp(min=1e-4)
        cu = fx * center_cam[:, 0] / safe_cz + cx
        cv = fy * center_cam[:, 1] / safe_cz + cy
        px = torch.round(cu).to(torch.int64).cpu().numpy()
        py = torch.round(cv).to(torch.int64).cpu().numpy()
        center_z = cz.cpu().numpy()
        in_bounds = ((px >= 0) & (px < width) & (py >= 0) & (py < height) &
                     (center_z > 0.01))
        ids = np.flatnonzero(in_bounds)
        visible[ids] += 1
        outside[ids] += ~dilated[py[ids], px[ids]]
        views_used += 1

    seen_enough = visible >= min_projected_views
    ratios = np.divide(outside, np.maximum(visible, 1), dtype=np.float64)
    silhouette_keep = ~(seen_enough & (ratios > outside_ratio))

    normal = np.asarray(plane["normal"], dtype=np.float32)
    offset = float(plane["offset"])
    signed = vertices @ normal + offset
    entirely_below = np.all(signed[faces] < -plane_margin, axis=1)
    keep = silhouette_keep & ~entirely_below
    before = n_faces
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()

    # Keep the main surface and any substantial attached component.
    labels, counts, areas = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    areas = np.asarray(areas)
    if len(counts):
        largest = int(counts.max())
        cluster_keep = np.where((counts >= max(50, int(largest * 0.001))) &
                                (areas >= max(float(areas.max()) * 1e-5, 1e-12)))[0]
        mesh.remove_triangles_by_mask(~np.isin(labels, cluster_keep))
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(output), mesh, write_vertex_colors=True,
                               write_vertex_normals=True)
    report = {
        "source": str(source), "output": str(output), "registered_training_views_used": views_used,
        "validation_and_test_masks_used": 0, "faces_before": before,
        "faces_culled_by_silhouette": int((~silhouette_keep).sum()),
        "faces_culled_below_support_plane": int((silhouette_keep & entirely_below).sum()),
        "faces_after_silhouette_plane": int(keep.sum()),
        "faces_after_component_cleanup": len(mesh.triangles),
        "vertices_after": len(mesh.vertices),
        "silhouette_rule": {"dilation_px": tolerance_px, "outside_fraction": outside_ratio,
                            "minimum_projected_views": min_projected_views},
        "support_plane_margin": plane_margin,
        "mask_paths": str(masks_dir),
        "mean_outside_fraction_projected_faces": float(ratios[seen_enough].mean()) if seen_enough.any() else None,
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cameras", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--colmap-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tolerance-px", type=int, default=3)
    p.add_argument("--outside-ratio", type=float, default=0.10)
    p.add_argument("--min-projected-views", type=int, default=3)
    p.add_argument("--plane-margin", type=float, default=0.04)
    a = p.parse_args()
    print(json.dumps(cull(a.source, a.cameras, a.images, a.masks, a.colmap_report,
                          a.output, a.tolerance_px, a.outside_ratio,
                          a.min_projected_views, a.plane_margin), indent=2))


if __name__ == "__main__":
    main()
