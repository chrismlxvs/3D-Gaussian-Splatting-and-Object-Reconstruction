"""Test caps on only tiny, closed boundary loops; leave all chains untouched."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def edge_stats(faces: np.ndarray) -> tuple[int, int]:
    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return int((counts == 1).sum()), int((counts > 2).sum())


def cap(source: Path, output: Path, max_perimeter: float,
        max_extent: float) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    mesh = trimesh.load(source, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    boundary = directed[counts[inverse] == 1]
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for a, b in boundary:
        outgoing.setdefault(int(a), []).append(int(b))
        incoming.setdefault(int(b), []).append(int(a))

    candidates = []
    visited: set[int] = set()
    for start in outgoing:
        if start in visited:
            continue
        loop = []
        current = start
        local: set[int] = set()
        closed = False
        while current not in local and len(loop) <= len(boundary):
            local.add(current)
            loop.append(current)
            if len(outgoing.get(current, [])) != 1 or len(incoming.get(current, [])) != 1:
                break
            current = outgoing[current][0]
            if current == start:
                closed = True
                break
        visited.update(local)
        if not closed or len(loop) < 3:
            continue
        points = vertices[np.asarray(loop)]
        perimeter = float(np.linalg.norm(np.roll(points, -1, axis=0) - points,
                                         axis=1).sum())
        extent = np.ptp(points, axis=0)
        if perimeter <= max_perimeter and float(extent.max()) <= max_extent:
            candidates.append({"loop": loop, "perimeter": perimeter,
                               "bbox_extent": extent.tolist(),
                               "centroid": points.mean(axis=0).tolist()})

    new_vertices, new_faces, new_colors = [], [], []
    colors = np.asarray(mesh.visual.vertex_colors) if mesh.visual.kind == "vertex" else None
    for candidate in candidates:
        loop = candidate["loop"]
        center_index = len(vertices) + len(new_vertices)
        new_vertices.append(vertices[loop].mean(axis=0))
        for i, a in enumerate(loop):
            b = loop[(i + 1) % len(loop)]
            new_faces.append([b, a, center_index])
        if colors is not None:
            new_colors.append(np.rint(colors[loop].astype(float).mean(axis=0)).astype(np.uint8))

    out_vertices = np.vstack((vertices, np.asarray(new_vertices))) if new_vertices else vertices.copy()
    out_faces = np.vstack((faces, np.asarray(new_faces, dtype=np.int64))) if new_faces else faces.copy()
    visuals = {"vertex_colors": np.vstack((colors, np.asarray(new_colors)))
               if colors is not None and new_colors else colors}
    visuals = {key: value for key, value in visuals.items() if value is not None}
    result = trimesh.Trimesh(vertices=out_vertices, faces=out_faces,
                             process=False, **visuals)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.export(output)
    boundary_edges, nonmanifold_edges = edge_stats(out_faces)
    report = {
        "source": str(source), "output": str(output),
        "thresholds": {"max_loop_perimeter": max_perimeter,
                       "max_loop_bbox_extent": max_extent},
        "loops_capped": len(candidates), "faces_added": len(new_faces),
        "vertices_added": len(new_vertices),
        "filled_loops": [{key: row[key] for key in ("perimeter", "bbox_extent", "centroid")}
                         for row in candidates],
        "before": {"vertices": len(vertices), "faces": len(faces),
                   "boundary_edges": int((counts == 1).sum()),
                   "nonmanifold_edges": int((counts > 2).sum())},
        "after": {"vertices": len(out_vertices), "faces": len(out_faces),
                  "boundary_edges": boundary_edges,
                  "nonmanifold_edges": nonmanifold_edges},
        "method": "centroid fan caps only on small directed closed loops; no original vertex moved",
        "source_unchanged": True,
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                    encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-perimeter", type=float, default=0.06)
    parser.add_argument("--max-extent", type=float, default=0.025)
    args = parser.parse_args()
    print(json.dumps(cap(args.source, args.output, args.max_perimeter,
                         args.max_extent), indent=2))


if __name__ == "__main__":
    main()
