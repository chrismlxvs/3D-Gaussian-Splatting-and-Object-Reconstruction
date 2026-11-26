"""Render user-owned plush mesh iterations side by side in neutral gray."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

VIEWS = {
    "Front (-Z)": (np.array([0., 0., -1.]), np.array([0., 1., 0.])),
    "Right (+X)": (np.array([1., 0., 0.]), np.array([0., 1., 0.])),
    "Rear (+Z)": (np.array([0., 0., 1.]), np.array([0., 1., 0.])),
    "Underside (-Y)": (np.array([0., -1., 0.]), np.array([0., 0., -1.])),
}


def read_mesh(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if mesh.is_empty():
        raise ValueError(f"empty mesh: {path}")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.78, 0.79, 0.82])
    return mesh


def render(mesh, direction, up, center, distance, width, height):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1., 1., 1., 1.])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_roughness = 0.9
    renderer.scene.add_geometry("mesh", mesh, material)
    renderer.setup_camera(38.0, center, center + direction * distance, up)
    image = np.asarray(renderer.render_to_image()).copy()
    renderer.scene.clear_geometry()
    del renderer
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", action="append", required=True,
                        help="repeat as LABEL=/absolute/path/to/mesh.ply")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=440)
    parser.add_argument("--height", type=int, default=560)
    args = parser.parse_args()
    entries = []
    for spec in args.mesh:
        if "=" not in spec:
            parser.error("--mesh must be LABEL=PATH")
        label, raw_path = spec.split("=", 1)
        if not label:
            parser.error("mesh label cannot be empty")
        entries.append((label, Path(raw_path)))
    meshes = [(label, path, read_mesh(path)) for label, path in entries]
    boxes = [mesh.get_axis_aligned_bounding_box() for _, _, mesh in meshes]
    bounds = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.min([box.min_bound for box in boxes], axis=0),
        max_bound=np.max([box.max_bound for box in boxes], axis=0),
    )
    center = bounds.get_center()
    distance = float(np.linalg.norm(bounds.get_extent()) * 2.5)
    gap, header = 8, 42
    cell_w, cell_h = args.width, args.height + header
    sheet = Image.new("RGB", (cell_w * len(meshes), cell_h * len(VIEWS)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (view_name, (direction, up)) in enumerate(VIEWS.items()):
        y0 = row * cell_h
        draw.text((gap, y0 + 4), view_name, fill="black")
        for col, (label, _, mesh) in enumerate(meshes):
            x0 = col * cell_w
            draw.text((x0 + gap, y0 + 21), label, fill="black")
            image = render(mesh, direction, up, center, distance,
                           args.width, args.height).convert("RGB")
            sheet.paste(image, (x0, y0 + header))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
