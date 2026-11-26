"""Align one mesh into another COLMAP frame using shared camera poses."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def read_rows(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {re.sub(r"_frame_", "_", row["img_name"]): row for row in rows}


def fit_alignment(source_cameras: Path, target_cameras: Path):
    source, target = read_rows(source_cameras), read_rows(target_cameras)
    names = sorted(source.keys() & target.keys())
    if len(names) < 6:
        raise ValueError(f"too few shared camera poses: {len(names)}")
    x = np.asarray([source[n]["position"] for n in names], dtype=np.float64)
    y = np.asarray([target[n]["position"] for n in names], dtype=np.float64)
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    u, singular, vt = np.linalg.svd(xc.T @ yc)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / np.sum(xc * xc))
    translation = my - scale * (mx @ rotation)
    predicted = scale * (x @ rotation) + translation
    center_error = np.linalg.norm(predicted - y, axis=1)
    angular_error = []
    for name in names:
        rot_source = np.asarray(source[name]["rotation"], dtype=np.float64)
        rot_target = np.asarray(target[name]["rotation"], dtype=np.float64)
        relative = rot_target @ (rotation.T @ rot_source).T
        angular_error.append(np.degrees(Rotation.from_matrix(relative).magnitude()))
    alignment = {
        "shared_cameras": len(names),
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "transform": "p_target = scale * (p_source @ rotation) + translation",
        "camera_center_error": {
            "median": float(np.median(center_error)),
            "p95": float(np.percentile(center_error, 95)),
            "max": float(center_error.max()),
        },
        "rotation_error_degrees": {
            "median": float(np.median(angular_error)),
            "p95": float(np.percentile(angular_error, 95)),
            "max": float(np.max(angular_error)),
        },
        "shared_name_examples": names[:8],
    }
    return alignment


def align_mesh(source_mesh: Path, output: Path, alignment: dict) -> dict:
    # Weld exact duplicate vertices (common at GLB UV seams) before writing a
    # geometry-only PLY, so edge/component statistics describe the surface.
    mesh = trimesh.load(source_mesh, force="mesh", process=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    rotation = np.asarray(alignment["rotation"])
    transformed = alignment["scale"] * (vertices @ rotation) + alignment["translation"]
    visuals = {}
    if mesh.visual.kind == "vertex":
        visuals["vertex_colors"] = np.asarray(mesh.visual.vertex_colors).copy()
    aligned = trimesh.Trimesh(vertices=transformed, faces=faces,
                              process=True, **visuals)
    output.parent.mkdir(parents=True, exist_ok=True)
    aligned.export(output)
    return {"source_mesh": str(source_mesh), "aligned_mesh": str(output),
            "vertices": len(vertices), "faces": len(faces),
            "source_unchanged": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cameras", type=Path, required=True)
    parser.add_argument("--target-cameras", type=Path, required=True)
    parser.add_argument("--source-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    alignment = fit_alignment(args.source_cameras, args.target_cameras)
    mesh_info = align_mesh(args.source_mesh, args.output, alignment)
    report = {"alignment": alignment, **mesh_info}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
