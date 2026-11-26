"""Orient the MILo mesh to a right-handed support-plane object frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy],
    ], dtype=np.float64)


def first_camera_center(images_txt: Path, image_name: str) -> np.ndarray:
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[-1] == image_name:
            q = np.asarray(list(map(float, fields[1:5])))
            t = np.asarray(list(map(float, fields[5:8])))
            r = qvec2rotmat(q)
            return -r.T @ t
    raise ValueError(f"could not find camera {image_name} in {images_txt}")


def triangle_area_centroid(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the triangle-area-weighted surface centroid and area.

    The extracted mesh is open, so a volume centroid is not defined. The
    surface centroid is stable under uneven vertex sampling and is reported
    explicitly for scene placement.
    """
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]), axis=1)
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 1e-12:
        raise ValueError("mesh has no measurable triangle surface area")
    centroids = triangles.mean(axis=1)
    center = (centroids * areas[:, None]).sum(axis=0) / total_area
    return center, total_area


def anchor(mesh_path: Path, colmap_report: Path, images_txt: Path,
           output_root: Path, front_image: str = "1978_000000.png",
           invert_vertical: bool = False) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values()
                      if isinstance(g, trimesh.Trimesh) and not g.is_empty]
        if not geometries:
            raise ValueError(f"scene contains no triangle mesh: {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"not a triangle mesh: {mesh_path}")
    prep = json.loads(colmap_report.read_text(encoding="utf-8"))
    plane = prep["support_plane"]
    if plane.get("status") != "ok":
        raise ValueError("a stable support plane was not found")
    y_axis = np.asarray(plane["normal"], dtype=np.float64)
    y_axis /= np.linalg.norm(y_axis)
    plane_d = float(plane["offset"])
    plane_point = -plane_d * y_axis
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    signed = verts @ y_axis + plane_d
    height = float(np.percentile(signed, 99.5) - np.percentile(signed, 0.5))
    contact_level = float(np.percentile(signed, 2.0))
    contact = verts[signed <= contact_level]
    contact_centroid = contact.mean(axis=0)
    origin = contact_centroid - (contact_centroid @ y_axis + plane_d) * y_axis

    center = np.asarray(prep["object_center_colmap"], dtype=np.float64)
    camera_center = first_camera_center(images_txt, front_image)
    z_axis = camera_center - center
    z_axis -= np.dot(z_axis, y_axis) * y_axis
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("front camera direction is parallel to the support plane normal")
    z_axis /= z_norm
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    basis = np.stack((x_axis, y_axis, z_axis), axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = basis
    transform[:3, 3] = -basis @ origin
    transform_inv = np.linalg.inv(transform)
    vertices_obj = (verts - origin) @ basis.T
    if invert_vertical:
        # The source model's visible head/feet direction is reversed relative
        # to the upright plushie in its development views. Rotate 180 degrees
        # around local Z, preserve the front axis, and move the opposite
        # support end to y=0. This is a rigid orientation correction only.
        correction = np.eye(4, dtype=np.float64)
        correction[0, 0] = -1.0
        correction[1, 1] = -1.0
        correction[1, 3] = float(vertices_obj[:, 1].max())
        transform = correction @ transform
        transform_inv = np.linalg.inv(transform)
        vertices_obj = vertices_obj @ correction[:3, :3].T + correction[:3, 3]
        basis = transform[:3, :3]
        x_axis, y_axis, z_axis = basis[0], basis[1], basis[2]
        origin = transform_inv[:3, 3]
        plane_d = -float(y_axis @ origin)
        plane = {**plane, "normal": y_axis.tolist(), "offset": plane_d,
                 "orientation_corrected": True}
        signed = vertices_obj[:, 1]
        contact_level = float(np.percentile(signed, 2.0))

    output_root.mkdir(parents=True)
    has_uv_texture = getattr(mesh.visual, "kind", None) == "texture"
    anchored = trimesh.Trimesh(vertices=vertices_obj, faces=np.asarray(mesh.faces),
                               visual=mesh.visual, process=False)
    clean_ply = output_root / "mesh_clean_object_frame.ply"
    glb = output_root / ("mesh_textured_object_frame.glb" if has_uv_texture
                         else "mesh_vertex_colored_object_frame.glb")
    anchored.export(clean_ply)
    anchored.export(glb)
    lo = vertices_obj.min(axis=0)
    hi = vertices_obj.max(axis=0)
    surface_centroid, surface_area = triangle_area_centroid(
        vertices_obj, np.asarray(mesh.faces, dtype=np.int64))
    contact_xz = vertices_obj[signed <= contact_level][:, [0, 2]]
    hull_area = None
    if len(contact_xz) >= 3:
        try:
            hull_area = float(ConvexHull(contact_xz).volume)
        except Exception:
            hull_area = 0.0
    report = {
        "source_mesh": str(mesh_path), "source_colmap_report": str(colmap_report),
        "front_reference_image": front_image,
        "orientation_correction": ("180-degree rotation about object Z; upright direction verified against development views"
                                   if invert_vertical else "none"),
        "T_object_from_colmap": transform.tolist(),
        "T_colmap_from_object": transform_inv.tolist(),
        "axes_colmap": {"X": x_axis.tolist(), "Y_support_normal": y_axis.tolist(),
                        "Z_front": z_axis.tolist()},
        "origin_colmap": origin.tolist(), "support_plane": plane,
        "units": "COLMAP units; metric scale not calibrated",
        "height_99_percentile": height,
        "aabb_object_frame": {"min": lo.tolist(), "max": hi.tolist(),
                               "dimensions": (hi-lo).tolist()},
        "aabb_center_object": ((lo+hi)*0.5).tolist(),
        "surface_area_centroid_object": surface_centroid.tolist(),
        "mesh_center_method": "triangle-area-weighted surface centroid; mesh is open, so no volume centroid is defined",
        "mesh_surface_area_object_units": surface_area,
        "contact_polygon_area_xz": hull_area,
        "vertex_count": len(vertices_obj), "face_count": len(mesh.faces),
        "surface_color_type": "UV texture" if has_uv_texture else "vertex colors",
        "outputs": [str(clean_ply), str(glb)],
    }
    (output_root / "object_frame.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--colmap-report", type=Path, required=True)
    parser.add_argument("--images-txt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--front-image", default="1978_000000.png")
    parser.add_argument("--invert-vertical", action="store_true",
                        help="rotate the mesh upright while keeping its front direction")
    args = parser.parse_args()
    print(json.dumps(anchor(args.mesh, args.colmap_report, args.images_txt,
                            args.output_root, args.front_image,
                            args.invert_vertical), indent=2))


if __name__ == "__main__":
    main()
