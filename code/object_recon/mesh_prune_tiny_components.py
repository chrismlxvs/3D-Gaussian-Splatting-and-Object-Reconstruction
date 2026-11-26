"""Remove only detached tiny triangle islands from a copied mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def prune(source: Path, output: Path, minimum_faces: int = 10) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    mesh = o3d.io.read_triangle_mesh(str(source))
    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError(f"empty triangle mesh: {source}")
    before_vertices, before_faces = len(mesh.vertices), len(mesh.triangles)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    vertices_after_weld = len(mesh.vertices)
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    keep_components = np.flatnonzero(counts >= minimum_faces)
    remove_faces = ~np.isin(labels, keep_components)
    before = {"vertices": before_vertices, "faces": before_faces,
              "vertices_after_duplicate_weld": vertices_after_weld,
              "components_after_weld": int(len(counts)),
              "component_face_counts_after_weld": sorted(counts.astype(int).tolist(),
                                                          reverse=True)}
    removed_components = counts[counts < minimum_faces]
    mesh.remove_triangles_by_mask(remove_faces)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output), mesh, write_vertex_colors=True,
                               write_vertex_normals=True)
    _, counts_after, _ = mesh.cluster_connected_triangles()
    report = {
        "source": str(source), "output": str(output),
        "minimum_component_faces": minimum_faces,
        "before": before,
        "removed_component_face_counts": removed_components.astype(int).tolist(),
        "after": {"vertices": len(mesh.vertices), "faces": len(mesh.triangles),
                  "components": int(len(counts_after)),
                  "component_face_counts": sorted(np.asarray(counts_after).astype(int).tolist(),
                                                   reverse=True),
                  "non_manifold_edges": len(mesh.get_non_manifold_edges(
                      allow_boundary_edges=True)),
                  "watertight": mesh.is_watertight()},
        "source_unchanged": True,
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                    encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-faces", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(prune(args.source, args.output, args.minimum_faces), indent=2))


if __name__ == "__main__":
    main()
