"""Remove only edge-incidence defects from a copied mesh; do not cap holes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import open3d as o3d


def repair(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    mesh = o3d.io.read_triangle_mesh(str(source))
    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError(f"empty triangle mesh: {source}")
    before = {"vertices": len(mesh.vertices), "faces": len(mesh.triangles),
              "non_manifold_edges": len(mesh.get_non_manifold_edges(
                  allow_boundary_edges=True))}
    mesh.remove_non_manifold_edges()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output), mesh, write_vertex_normals=True,
                               write_vertex_colors=mesh.has_vertex_colors())
    report = {
        "source": str(source),
        "output": str(output),
        "before": before,
        "after": {"vertices": len(mesh.vertices), "faces": len(mesh.triangles),
                  "non_manifold_edges": len(mesh.get_non_manifold_edges(
                      allow_boundary_edges=True)),
                  "watertight": mesh.is_watertight()},
        "repair": "remove only excess edge-incidence faces; do not fill boundary loops",
        "source_unchanged": True,
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                    encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
