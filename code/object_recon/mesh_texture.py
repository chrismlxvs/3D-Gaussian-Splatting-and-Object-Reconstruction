"""Bake MILo vertex colors into a UV atlas and export a textured GLB/OBJ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh
from scipy import ndimage
import xatlas
import nvdiffrast.torch as dr


def read_camera_records(cameras_txt: Path, images_txt: Path):
    cameras = {}
    for line in cameras_txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[1] != "PINHOLE":
            raise ValueError(f"expected undistorted PINHOLE camera, got {fields[1]}")
        cameras[int(fields[0])] = (int(fields[2]), int(fields[3]),
                                   *map(float, fields[4:8]))
    records = []
    for line in images_txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        q = np.asarray(list(map(float, f[1:5])), dtype=np.float32)
        qw, qx, qy, qz = q
        R = np.array([
            [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
            [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
            [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy],
        ], dtype=np.float32)
        t = np.asarray(list(map(float, f[5:8])), dtype=np.float32)
        records.append((f[-1], R, t, cameras[int(f[8])]))
    return records


def projective_texture(atlas_vertices, atlas_faces, uv, source_faces,
                       images_dir: Path, cameras_txt: Path, images_txt: Path,
                       resolution: int, device: torch.device):
    """Photo-project masked training images into the UV atlas with z visibility."""
    H = W = resolution
    ctx = dr.RasterizeCudaContext(device=device)
    face_t = torch.from_numpy(atlas_faces.astype(np.int32)).to(device=device)
    # UV-space rasterization provides one 3D sample for each covered atlas texel.
    uv_clip = np.column_stack((uv * 2.0 - 1.0,
                               np.zeros(len(uv), np.float32),
                               np.ones(len(uv), np.float32)))
    uv_pos = torch.from_numpy(uv_clip).to(device=device)[None]
    uv_rast, _ = dr.rasterize(ctx, uv_pos, face_t,
                              resolution=(H, W), grad_db=False)
    uv_rast = uv_rast[0]
    active = uv_rast[..., 3] > 0
    pix = active.nonzero(as_tuple=False)
    rast_values = uv_rast[active]
    tri_id = rast_values[:, 3].long() - 1
    bary = torch.stack((1.0 - rast_values[:, 0] - rast_values[:, 1],
                        rast_values[:, 0], rast_values[:, 1]), dim=-1)
    face_idx = face_t[tri_id].long()
    tri_vertices = torch.from_numpy(atlas_vertices.astype(np.float32)).to(device=device)[face_idx]
    points = (tri_vertices * bary[..., None]).sum(dim=1)
    face_normals = F.normalize(torch.cross(tri_vertices[:, 1]-tri_vertices[:, 0],
                                           tri_vertices[:, 2]-tri_vertices[:, 0], dim=-1),
                               dim=-1, eps=1e-8)

    original_mesh_vertices = torch.from_numpy(atlas_vertices.astype(np.float32)).to(device=device)
    original_triangles = torch.from_numpy(atlas_faces.astype(np.int32)).to(device=device)
    accum = torch.zeros((len(points), 3), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((len(points),), dtype=torch.float32, device=device)
    near = 0.01
    used_views = 0
    for name, R_np, t_np, intr in read_camera_records(cameras_txt, images_txt):
        image_path = images_dir / name
        if not image_path.is_file():
            continue
        im = np.asarray(Image.open(image_path).convert("RGBA"), dtype=np.float32) / 255.0
        h, w = im.shape[:2]
        iw, ih, fx, fy, cx, cy = intr
        if (w, h) != (iw, ih):
            raise ValueError(f"camera/image resolution mismatch for {name}")
        R = torch.from_numpy(R_np).to(device=device)
        t = torch.from_numpy(t_np).to(device=device)
        cam_vertices = original_mesh_vertices @ R.T + t
        z = cam_vertices[:, 2]
        safe_z = z.clamp(min=near)
        ndc_x = 2.0 * (fx * cam_vertices[:, 0] / safe_z + cx) / w - 1.0
        ndc_y = 1.0 - 2.0 * (fy * cam_vertices[:, 1] / safe_z + cy) / h
        clip = torch.stack((ndc_x * safe_z, ndc_y * safe_z,
                            safe_z-near, safe_z), dim=-1)
        clip = torch.where((z > near)[:, None], clip,
                           torch.tensor([0., 0., 0., -1.], device=device))
        depth_rast, _ = dr.rasterize(ctx, clip[None], original_triangles,
                                     resolution=(h, w), grad_db=False)
        depth = depth_rast[0, :, :, 2][None, None]

        cam_points = points @ R.T + t
        pz = cam_points[:, 2]
        safe_pz = pz.clamp(min=near)
        u = fx * cam_points[:, 0] / safe_pz + cx
        v = fy * cam_points[:, 1] / safe_pz + cy
        gx = 2.0 * u / max(w-1, 1) - 1.0
        gy = 2.0 * v / max(h-1, 1) - 1.0
        grid = torch.stack((gx, gy), dim=-1)[None, :, None, :]
        source = torch.from_numpy(im).to(device=device).permute(2, 0, 1)[None]
        sampled = F.grid_sample(source, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)[0, :, :, 0].T
        depth_sample = F.grid_sample(depth, grid, mode="nearest",
                                     padding_mode="zeros", align_corners=True)[0, 0, :, 0]
        p_ndc_z = 1.0 - near / safe_pz
        in_bounds = ((u >= 0) & (u < w) & (v >= 0) & (v < h) & (pz > near))
        is_visible = (in_bounds & (sampled[:, 3] > 0.5) &
                      (depth_sample > 0) & (p_ndc_z <= depth_sample + 0.002))
        camera_center = -R.T @ t
        view_dir = F.normalize(camera_center[None] - points, dim=-1, eps=1e-8)
        cosine = torch.abs((face_normals * view_dir).sum(dim=-1)).clamp(0, 1)
        weights = cosine.pow(4) * is_visible.float()
        accum += sampled[:, :3] * weights[:, None]
        weight_sum += weights
        used_views += 1

    tex_colors = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    valid = weight_sum > 1e-8
    colors = accum / weight_sum.clamp(min=1e-8)[:, None]
    tex_colors[pix[:, 0], pix[:, 1]] = colors
    valid_map = torch.zeros((H, W), dtype=torch.bool, device=device)
    valid_map[pix[:, 0], pix[:, 1]] = valid
    occupied = active.detach().cpu().numpy()
    valid_np = valid_map.detach().cpu().numpy()
    tex = tex_colors.detach().cpu().numpy()
    if valid_np.any():
        nearest = ndimage.distance_transform_edt(~valid_np, return_distances=False,
                                                  return_indices=True)
        tex[occupied & ~valid_np] = tex[nearest[0][occupied & ~valid_np],
                                         nearest[1][occupied & ~valid_np]]
    else:
        raise RuntimeError("no UV texels received a visible masked photo sample")
    tex = np.flipud(tex)
    return (np.clip(tex, 0.0, 1.0), occupied,
            float(valid_np.sum() / max(occupied.sum(), 1)), used_views)


def bake(source: Path, output_root: Path, resolution: int = 2048,
         images_dir: Path | None = None, cameras_txt: Path | None = None,
         images_txt: Path | None = None) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    mesh = trimesh.load(source, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()
                                               if isinstance(g, trimesh.Trimesh)))
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"not a triangle mesh: {source}")
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vmapping, atlas_faces, uv = xatlas.parametrize(vertices, faces)
    atlas_vertices = vertices[vmapping]
    atlas_faces = np.asarray(atlas_faces, dtype=np.int32)
    uv = np.asarray(uv, dtype=np.float32)

    device = torch.device("cuda")
    if images_dir and cameras_txt and images_txt:
        tex, occupied, observed_fraction, views_used = projective_texture(
            atlas_vertices, atlas_faces, uv, faces, images_dir, cameras_txt,
            images_txt, resolution, device)
        color_source = "masked training photos; z-buffer visible; cosine^4 view weighting"
    else:
        # Fallback remains useful for color-only assets without camera inputs.
        atlas_colors = colors[vmapping]
        clip = np.concatenate((uv * 2.0 - 1.0,
                               np.zeros((len(uv), 1), np.float32),
                               np.ones((len(uv), 1), np.float32)), axis=1)
        clip_t = torch.from_numpy(clip).to(device=device)[None]
        face_t = torch.from_numpy(atlas_faces).to(device=device, dtype=torch.int32)
        color_t = torch.from_numpy(atlas_colors).to(device=device)[None]
        ctx = dr.RasterizeCudaContext(device=device)
        rast, _ = dr.rasterize(ctx, clip_t, face_t, resolution=(resolution, resolution))
        texture_t, _ = dr.interpolate(color_t, rast, face_t)
        texture_t = dr.antialias(texture_t, rast, clip_t, face_t)
        tex = np.clip(texture_t[0].detach().cpu().numpy(), 0.0, 1.0)
        occupied0 = rast[0, :, :, 3].detach().cpu().numpy() > 0
        nearest = ndimage.distance_transform_edt(~occupied0, return_distances=False,
                                                  return_indices=True)
        tex[~occupied0] = tex[nearest[0][~occupied0], nearest[1][~occupied0]]
        tex = np.flipud(tex)
        occupied = occupied0
        observed_fraction, views_used = 0.0, 0
        color_source = "MILo vertex colors interpolated into an xatlas UV layout"
    albedo = (tex * 255.0 + 0.5).astype(np.uint8)
    texture_img = Image.fromarray(albedo, mode="RGB")
    output_root.mkdir(parents=True)
    texture_path = output_root / "albedo.png"
    texture_img.save(texture_path)
    Image.new("L", (resolution, resolution), 230).save(output_root / "roughness.png")
    Image.new("L", (resolution, resolution), 0).save(output_root / "metallic.png")

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_img, metallicFactor=0.0, roughnessFactor=0.9,
        alphaMode="OPAQUE")
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    textured = trimesh.Trimesh(vertices=atlas_vertices, faces=atlas_faces,
                                visual=visual, process=False)
    glb_path = output_root / "mesh_textured.glb"
    obj_path = output_root / "mesh_textured.obj"
    textured.export(glb_path)
    normals = np.asarray(textured.vertex_normals, dtype=np.float64)
    with obj_path.open("w", encoding="utf-8", newline="\n") as obj:
        obj.write("mtllib mesh_textured.mtl\no white_plush\nusemtl plush\n")
        obj.write("\n".join(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}" for v in atlas_vertices))
        obj.write("\n")
        obj.write("\n".join(f"vt {uvp[0]:.9g} {uvp[1]:.9g}" for uvp in uv))
        obj.write("\n")
        obj.write("\n".join(f"vn {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}" for n in normals))
        obj.write("\n")
        obj.write("\n".join(
            "f " + " ".join(f"{int(i)+1}/{int(i)+1}/{int(i)+1}" for i in face)
            for face in atlas_faces))
        obj.write("\n")
    (output_root / "mesh_textured.mtl").write_text(
        "newmtl plush\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 20\n"
        "d 1.0\nillum 2\nmap_Kd albedo.png\n", encoding="utf-8")
    report = {
        "source_mesh": str(source), "texture_size": [resolution, resolution],
        "atlas_vertices": int(len(atlas_vertices)), "faces": int(len(atlas_faces)),
        "uv_occupied_fraction": float(occupied.mean()),
        "color_source": color_source,
        "visible_photo_views_used": views_used,
        "observed_uv_fraction": observed_fraction,
        "material_approximations": {"roughness": 0.9, "metallic": 0.0},
        "texture_is_de_lit_albedo": False,
        "outputs": [str(glb_path), str(obj_path), str(texture_path),
                    str(output_root / "mesh_textured.mtl"),
                    str(output_root / "roughness.png"), str(output_root / "metallic.png")],
    }
    (output_root / "texture_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--cameras-txt", type=Path)
    parser.add_argument("--images-txt", type=Path)
    args = parser.parse_args()
    print(json.dumps(bake(args.source, args.output_root, args.resolution,
                          args.images_dir, args.cameras_txt, args.images_txt), indent=2))


if __name__ == "__main__":
    main()
