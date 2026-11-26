"""Create an isolated Poisson closure candidate from an oriented triangle mesh.

This is an experiment for open or ragged scan surfaces. It does not modify the
input mesh. Use training-view silhouette cleanup and held-out evaluation before
considering the output for delivery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def reconstruct(source: Path, output: Path, points: int = 350_000,
                depth: int = 9, density_percentile: float = 1.0,
                bbox_margin: float = 0.015) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    mesh = o3d.io.read_triangle_mesh(str(source))
    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError(f"empty triangle mesh: {source}")
    mesh.compute_vertex_normals()
    bounds = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bounds.get_extent())
    pad = float(np.linalg.norm(extent) * bbox_margin)
    bound_min = np.asarray(bounds.min_bound).copy() - pad
    bound_max = np.asarray(bounds.max_bound).copy() + pad

    cloud = mesh.sample_points_uniformly(number_of_points=points,
                                         use_triangle_normal=True)
    cloud.normalize_normals()
    reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth, scale=1.05, linear_fit=True)
    density = np.asarray(densities)
    threshold = float(np.percentile(density, density_percentile))
    reconstructed.remove_vertices_by_mask(density < threshold)
    # Clip low-confidence extrapolated geometry to a small margin around the
    # source extent before the training-only silhouette pass.
    vertices = np.asarray(reconstructed.vertices)
    faces = np.asarray(reconstructed.triangles)
    inside = np.all((vertices >= bound_min) & (vertices <= bound_max), axis=1)
    reconstructed.remove_triangles_by_mask(~inside[faces].all(axis=1))
    reconstructed.remove_unreferenced_vertices()
    reconstructed.remove_degenerate_triangles()
    reconstructed.compute_vertex_normals()

    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output), reconstructed, write_vertex_normals=True,
                               write_vertex_colors=False)
    report = {
        "source": str(source),
        "output": str(output),
        "source_vertices": len(mesh.vertices),
        "source_faces": len(mesh.triangles),
        "poisson_input_points": len(cloud.points),
        "poisson_depth": depth,
        "density_percentile": density_percentile,
        "density_threshold": threshold,
        "source_bbox_margin_fraction": bbox_margin,
        "output_vertices": len(reconstructed.vertices),
        "output_faces": len(reconstructed.triangles),
        "output_watertight": reconstructed.is_watertight(),
        "validation_or_test_data_used": False,
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                    encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=350_000)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--density-percentile", type=float, default=1.0)
    parser.add_argument("--bbox-margin", type=float, default=0.015)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.source, args.output, args.points, args.depth,
                                 args.density_percentile, args.bbox_margin), indent=2))


if __name__ == "__main__":
    main()
