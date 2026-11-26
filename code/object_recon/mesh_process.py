"""Clean and export MILo's vertex-colored plush surface without replacing raw SDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def process(source: Path, output_root: Path, target_faces: int = 120_000) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    mesh = o3d.io.read_triangle_mesh(str(source))
    if mesh.is_empty() or not mesh.has_vertex_colors():
        raise ValueError("MILo SDF mesh is empty or has no vertex colors")
    before = {"vertices": len(mesh.vertices), "faces": len(mesh.triangles)}
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    labels, counts, areas = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    areas = np.asarray(areas)
    if len(counts):
        main_count = int(counts.max())
        keep_clusters = np.where((counts >= max(100, int(main_count * 0.005))) &
                                 (areas >= max(float(areas.max()) * 1e-4, 1e-12)))[0]
        mesh.remove_triangles_by_mask(~np.isin(labels, keep_clusters))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    output_root.mkdir(parents=True)
    clean_ply = output_root / "mesh_clean_colmap.ply"
    o3d.io.write_triangle_mesh(str(clean_ply), mesh, write_vertex_colors=True,
                               write_vertex_normals=True, compressed=False)
    cleaned_faces = len(mesh.triangles)

    reduced = mesh
    if cleaned_faces > target_faces:
        reduced = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        reduced.remove_degenerate_triangles()
        reduced.remove_duplicated_triangles()
        reduced.remove_unreferenced_vertices()
        reduced.compute_vertex_normals()
    decimated_ply = output_root / "mesh_vertex_colored_colmap.ply"
    o3d.io.write_triangle_mesh(str(decimated_ply), reduced,
                               write_vertex_colors=True, write_vertex_normals=True)
    tri = trimesh.Trimesh(vertices=np.asarray(reduced.vertices),
                          faces=np.asarray(reduced.triangles),
                          vertex_colors=np.clip(np.asarray(reduced.vertex_colors) * 255,
                                                0, 255).astype(np.uint8),
                          vertex_normals=np.asarray(reduced.vertex_normals),
                          process=False)
    glb_path = output_root / "mesh_vertex_colored_colmap.glb"
    tri.export(glb_path)
    report = {
        "source_mesh": str(source), "before": before,
        "cleaned": {"vertices": len(mesh.vertices), "faces": cleaned_faces},
        "decimated": {"vertices": len(reduced.vertices), "faces": len(reduced.triangles)},
        "components_kept": int(len(keep_clusters)) if len(counts) else 0,
        "component_filter": "retain components >= max(100 faces, 0.5% of largest) and >=1e-4 largest area",
        "simplification": "Open3D quadric decimation preserving vertex colors",
        "surface_color_type": "MILo vertex colors; no UV texture was baked",
        "outputs": [str(clean_ply), str(decimated_ply), str(glb_path)],
    }
    (output_root / "mesh_process_report.json").write_text(
        __import__("json").dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=120_000)
    args = parser.parse_args()
    import json
    print(json.dumps(process(args.source, args.output_root, args.target_faces), indent=2))


if __name__ == "__main__":
    main()
