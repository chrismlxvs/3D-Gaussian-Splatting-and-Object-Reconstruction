"""Non-destructively simplify a mesh while preserving its broad surface shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import open3d as o3d


def decimate(source: Path, output: Path, target_faces: int) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    mesh = o3d.io.read_triangle_mesh(str(source))
    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError(f"empty triangle mesh: {source}")
    before = {"vertices": len(mesh.vertices), "faces": len(mesh.triangles)}
    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_faces)
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_unreferenced_vertices()
    simplified.compute_vertex_normals()
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output), simplified, write_vertex_normals=True,
                               write_vertex_colors=simplified.has_vertex_colors())
    report = {
        "source": str(source),
        "output": str(output),
        "target_faces": target_faces,
        "before": before,
        "after": {"vertices": len(simplified.vertices),
                  "faces": len(simplified.triangles)},
        "watertight": simplified.is_watertight(),
        "source_unchanged": True,
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                    encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=400_000)
    args = parser.parse_args()
    print(json.dumps(decimate(args.source, args.output, args.target_faces), indent=2))


if __name__ == "__main__":
    main()
