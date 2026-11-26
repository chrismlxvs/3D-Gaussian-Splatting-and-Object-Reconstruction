"""Transfer vertex colors to a revised mesh by closest-surface barycentrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def transfer(source: Path, target: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    color_mesh = o3d.io.read_triangle_mesh(str(source))
    target_mesh = o3d.io.read_triangle_mesh(str(target))
    if (color_mesh.is_empty() or not color_mesh.has_vertex_colors() or
            target_mesh.is_empty() or not target_mesh.has_triangles()):
        raise ValueError("source must have vertex colors and target must have triangles")

    color_vertices = np.asarray(color_mesh.vertices, dtype=np.float32)
    color_faces = np.asarray(color_mesh.triangles, dtype=np.int64)
    colors = np.asarray(color_mesh.vertex_colors, dtype=np.float32)
    target_vertices = np.asarray(target_mesh.vertices, dtype=np.float32)
    scene = o3d.t.geometry.RaycastingScene(nthreads=0,
                                           device=o3d.core.Device("CPU:0"))
    scene.add_triangles(
        o3d.core.Tensor(color_vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(color_faces, dtype=o3d.core.Dtype.UInt32),
    )
    query = o3d.core.Tensor(target_vertices, dtype=o3d.core.Dtype.Float32)
    nearest = scene.compute_closest_points(query)
    ids = nearest["primitive_ids"].numpy().astype(np.int64)
    uv = nearest["primitive_uvs"].numpy()
    distance = nearest["points"].numpy() - target_vertices
    distance = np.linalg.norm(distance, axis=1)
    valid = ids < len(color_faces)
    if not np.all(valid):
        raise ValueError(f"{int((~valid).sum())} target vertices have no color source")
    tri = color_faces[ids]
    w1, w2 = uv[:, 0:1], uv[:, 1:2]
    w0 = 1.0 - w1 - w2
    target_colors = (colors[tri[:, 0]] * w0 + colors[tri[:, 1]] * w1 +
                     colors[tri[:, 2]] * w2)
    target_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(target_colors, 0.0, 1.0))
    target_mesh.compute_vertex_normals()
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output), target_mesh, write_vertex_colors=True,
                               write_vertex_normals=True)
    report = {
        "color_source": str(source),
        "geometry_target": str(target),
        "output": str(output),
        "vertices_colored": int(len(target_vertices)),
        "nearest_surface_distance": {
            "median": float(np.median(distance)),
            "p95": float(np.percentile(distance, 95)),
            "max": float(np.max(distance)),
        },
        "method": "closest point on source triangles with barycentric RGB interpolation",
        "geometry_modified": False,
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                    encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-source", type=Path, required=True)
    parser.add_argument("--geometry-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(transfer(args.color_source, args.geometry_target, args.output),
                     indent=2))


if __name__ == "__main__":
    main()
