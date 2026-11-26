"""Read-only report of open boundary chains and loops in triangle meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def audit(path: Path) -> dict:
    mesh = trimesh.load(path, force="mesh", process=True)
    vertices = np.asarray(mesh.vertices)
    unique_edges = np.asarray(mesh.edges_unique)
    counts = np.bincount(np.asarray(mesh.edges_unique_inverse),
                         minlength=len(unique_edges))
    boundary = unique_edges[counts == 1]
    overused = unique_edges[counts > 2]
    if len(boundary) == 0:
        components = []
    else:
        nodes = np.unique(boundary)
        local = np.full(len(vertices), -1, dtype=np.int64)
        local[nodes] = np.arange(len(nodes))
        edge_ids = local[boundary]
        rows = np.r_[edge_ids[:, 0], edge_ids[:, 1]]
        cols = np.r_[edge_ids[:, 1], edge_ids[:, 0]]
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)),
                           shape=(len(nodes), len(nodes))).tocsr()
        count, labels = connected_components(graph, directed=False)
        components = []
        for label in range(count):
            vertex_ids = np.flatnonzero(labels == label)
            mask = np.isin(edge_ids[:, 0], vertex_ids)
            edges = boundary[mask]
            points = vertices[nodes[vertex_ids]]
            degree = np.asarray(graph[vertex_ids][:, vertex_ids].sum(axis=1)).ravel()
            components.append({
                "boundary_vertices": int(len(vertex_ids)),
                "boundary_edges": int(len(edges)),
                "closed_loop": bool(np.all(degree == 2)),
                "degree_histogram": {str(k): int((degree == k).sum())
                                     for k in np.unique(degree)},
                "perimeter": float(np.linalg.norm(
                    vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).sum()),
                "bounds_min": points.min(axis=0).tolist(),
                "bounds_max": points.max(axis=0).tolist(),
                "centroid": points.mean(axis=0).tolist(),
            })
        components.sort(key=lambda row: row["boundary_edges"], reverse=True)
    return {
        "mesh": str(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "boundary_edges": int(len(boundary)),
        "non_manifold_edges": int(len(overused)),
        "boundary_components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"meshes": [audit(path) for path in args.meshes]}
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
