"""Render a vertex-colored triangle mesh on a clean white background."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh


def render(mesh_path: Path, output_path: Path, size: int = 1100) -> None:
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in loaded.geometry.values()
                                                if isinstance(g, trimesh.Trimesh)))
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"not a triangle mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    colors = np.asarray(mesh.visual.vertex_colors)
    if colors.ndim != 2 or len(colors) != len(vertices):
        colors = np.tile(np.array([[220, 220, 220, 255]], dtype=np.uint8),
                         (len(vertices), 1))
    colors = colors[:, :3].astype(np.float32)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    size_world = float(max(extent)) * 1.22
    camera_offset = np.array([0.42, 0.28, 1.0], dtype=np.float64)
    camera_offset /= np.linalg.norm(camera_offset)
    view = -camera_offset
    right = np.cross(view, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    rel = vertices - center
    projected_x = rel @ right
    projected_y = rel @ up
    depth = rel @ view
    scale = (size - 100) / size_world
    px = np.rint(size * 0.5 + projected_x * scale).astype(np.int32)
    py = np.rint(size * 0.56 - projected_y * scale).astype(np.int32)
    screen = np.column_stack((px, py))

    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(norms[:, None], 1e-12)
    light = np.array([-0.4, 0.8, 0.7], dtype=np.float64)
    light /= np.linalg.norm(light)
    shade = 0.46 + 0.54 * np.maximum(0.0, normals @ light)
    face_colors = colors[faces].mean(axis=1) * shade[:, None]
    face_depth = depth[faces].mean(axis=1)
    order = np.argsort(face_depth)[::-1]  # back to front for a painter's pass

    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    # Soft grounding shadow under the contact band.
    shadow = np.zeros((size, size), dtype=np.uint8)
    sx = int(round(size * 0.5 - np.dot(center, right) * 0.0))
    sy = int(round(size * 0.87))
    cv2.ellipse(shadow, (sx, sy), (int(size * 0.17), int(size * 0.035)),
                0, 0, 360, 45, -1, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(shadow, (0, 0), size * 0.012)
    canvas = np.clip(canvas.astype(np.float32) - shadow[..., None] * 0.45,
                     0, 255).astype(np.uint8)
    for fi in order:
        poly = screen[faces[fi]]
        if (poly[:, 0].max() < 0 or poly[:, 0].min() >= size or
                poly[:, 1].max() < 0 or poly[:, 1].min() >= size):
            continue
        col = np.clip(face_colors[fi], 0, 255).astype(np.uint8).tolist()
        cv2.fillConvexPoly(canvas, poly, col, lineType=cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not write preview {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1100)
    args = parser.parse_args()
    render(args.mesh, args.output, args.size)


if __name__ == "__main__":
    main()
