"""Create an isolated, undistorted MILo dataset from the existing COLMAP map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import cv2
import numpy as np
from PIL import Image


def read_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    with path.open("rb") as f:
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            cid, model, width, height = struct.unpack("<iiQQ", f.read(24))
            if model != 2:  # COLMAP SIMPLE_RADIAL
                raise ValueError(f"expected SIMPLE_RADIAL camera, got model id {model}")
            params = struct.unpack("<4d", f.read(32))
            cameras[cid] = {"id": cid, "width": width, "height": height,
                            "f": params[0], "cx": params[1], "cy": params[2],
                            "k": params[3]}
    return cameras


def read_images(path: Path) -> dict[int, dict]:
    images = {}
    with path.open("rb") as f:
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            raw = f.read(64)
            iid, *pose_and_cam = struct.unpack("<idddddddi", raw)
            qvec = np.asarray(pose_and_cam[:4], dtype=np.float64)
            tvec = np.asarray(pose_and_cam[4:7], dtype=np.float64)
            camera_id = int(pose_and_cam[7])
            name_bytes = bytearray()
            while True:
                char = f.read(1)
                if not char or char == b"\0":
                    break
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8")
            (npoints,) = struct.unpack("<Q", f.read(8))
            raw_points = f.read(npoints * 24)
            data = np.frombuffer(raw_points, dtype=np.dtype([
                ("x", "<f8"), ("y", "<f8"), ("point_id", "<i8")]))
            xys = np.column_stack((data["x"], data["y"]))
            point_ids = data["point_id"].astype(np.int64)
            images[iid] = {"id": iid, "qvec": qvec, "tvec": tvec,
                           "camera_id": camera_id, "name": name,
                           "xys": xys, "point_ids": point_ids}
    return images


def read_points(path: Path) -> dict[int, dict]:
    points = {}
    with path.open("rb") as f:
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            header = f.read(43)
            pid = struct.unpack_from("<Q", header, 0)[0]
            xyz = np.asarray(struct.unpack_from("<3d", header, 8), dtype=np.float64)
            rgb = np.asarray(struct.unpack_from("<3B", header, 32), dtype=np.uint8)
            error = struct.unpack_from("<d", header, 35)[0]
            (track_len,) = struct.unpack("<Q", f.read(8))
            track = np.frombuffer(f.read(track_len * 8), dtype="<u4").reshape(-1, 2)
            points[pid] = {"id": pid, "xyz": xyz, "rgb": rgb,
                           "error": error, "track": track}
    return points


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy],
    ], dtype=np.float64)


def mask_at(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as im:
        mask = np.asarray(im.convert("L"), dtype=np.uint8) >= 128
    if mask.shape != (height, width):
        raise ValueError(f"mask size mismatch for {path}: {mask.shape}")
    return mask


def resolve_mask_path(name: str, masks_dir: Path,
                      fallback_masks_dir: Path | None = None) -> tuple[Path, bool]:
    path = masks_dir / f"{Path(name).stem}.png"
    if path.is_file():
        return path, False
    if fallback_masks_dir is not None:
        fallback = fallback_masks_dir / f"{Path(name).stem}.png"
        if fallback.is_file():
            return fallback, True
    raise FileNotFoundError(f"no mask for {name} in {masks_dir}"
                            + (f" or {fallback_masks_dir}" if fallback_masks_dir else ""))


def project(xyz: np.ndarray, image: dict, camera: dict) -> tuple[float, float] | None:
    r = qvec2rotmat(image["qvec"])
    xc = r @ xyz + image["tvec"]
    if xc[2] <= 1e-6:
        return None
    x, y = xc[0] / xc[2], xc[1] / xc[2]
    radial = 1.0 + camera["k"] * (x*x + y*y)
    return (camera["f"] * x * radial + camera["cx"],
            camera["f"] * y * radial + camera["cy"])


def make_dataset(manifest_path: Path, images_dir: Path, masks_dir: Path,
                 sparse_dir: Path, output_root: Path,
                 fallback_masks_dir: Path | None = None) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["images"]
    if any(Path(r["image"]).stem.startswith("1980_") for r in rows):
        raise ValueError("reserved test clip leaked into the development manifest")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    by_name = {r["image"]: r for r in train_rows}
    cameras = read_cameras(sparse_dir / "cameras.bin")
    images = read_images(sparse_dir / "images.bin")
    points = read_points(sparse_dir / "points3D.bin")
    if len(images) != len(train_rows):
        raise ValueError(f"COLMAP has {len(images)} images but manifest has {len(train_rows)} train images")
    if set(im["name"] for im in images.values()) != set(by_name):
        raise ValueError("COLMAP and train manifest image inventories differ")
    camera = next(iter(cameras.values()))
    if len(cameras) != 1:
        raise ValueError("expected one shared camera model")
    width, height = int(camera["width"]), int(camera["height"])
    if (width, height) != (406, 720):
        raise ValueError(f"unexpected image size {width}x{height}")

    # Classify sparse points by their observed 2D locations in the refined masks.
    fg_scores: dict[int, list[bool]] = {pid: [] for pid in points}
    id_to_name = {iid: im["name"] for iid, im in images.items()}
    mask_cache = {}
    fallback_mask_names = set()

    def get_mask(name: str) -> np.ndarray:
        if name not in mask_cache:
            path, used_fallback = resolve_mask_path(name, masks_dir, fallback_masks_dir)
            mask_cache[name] = mask_at(path, width, height)
            if used_fallback:
                fallback_mask_names.add(name)
        return mask_cache[name]

    for iid, im in images.items():
        name = im["name"]
        mask = get_mask(name)
        for (x, y), pid in zip(im["xys"], im["point_ids"]):
            if pid < 0 or int(pid) not in points:
                continue
            ix, iy = int(round(x)), int(round(y))
            inside = 0 <= ix < width and 0 <= iy < height and mask[iy, ix]
            fg_scores[int(pid)].append(bool(inside))
    fg_fraction = {pid: float(np.mean(v)) for pid, v in fg_scores.items() if v}
    n_obs = {pid: len(v) for pid, v in fg_scores.items()}
    object_points = [p for pid, p in points.items()
                     if n_obs[pid] >= 3 and fg_fraction.get(pid, 0.0) >= 0.65
                     and p["error"] <= 3.0]
    if len(object_points) < 100:
        raise ValueError(f"only {len(object_points)} sparse object points passed mask voting")
    object_xyz = np.stack([p["xyz"] for p in object_points])
    obj_center = np.median(object_xyz, axis=0)
    obj_radius = float(np.percentile(np.linalg.norm(object_xyz - obj_center, axis=1), 95))

    # Cross-clip static-object check: project object points into held-out views of
    # the other development clip, using only masks as a silhouette consistency test.
    check = {}
    for source_clip, target_clip in (("1978.mp4", "1979.mp4"), ("1979.mp4", "1978.mp4")):
        source_ids = {iid for iid, im in images.items()
                      if by_name[im["name"]]["frame_id"].startswith(source_clip + ":")}
        target_images = [(iid, im) for iid, im in images.items()
                         if by_name[im["name"]]["frame_id"].startswith(target_clip + ":")]
        source_points = {pid for pid, p in points.items()
                         if any(int(obs_iid) in source_ids and fg_fraction.get(pid, 0) >= 0.65
                                for obs_iid, _ in p["track"])}
        trials = hits = 0
        for pid in source_points:
            xyz = points[pid]["xyz"]
            for _, im in target_images[::max(1, len(target_images)//16)]:
                uv = project(xyz, im, camera)
                if uv is None:
                    continue
                x, y = map(lambda a: int(round(a)), uv)
                if 0 <= x < width and 0 <= y < height:
                    mask = get_mask(im["name"])
                    trials += 1
                    hits += bool(mask[y, x])
        check[f"{source_clip}_object_points_projected_to_{target_clip}"] = {
            "points": len(source_points), "projection_trials": trials,
            "inside_target_mask_fraction": hits / trials if trials else None}

    # RANSAC support plane from nearby background points, with normal oriented to
    # the average camera position.
    background = [p["xyz"] for pid, p in points.items()
                  if n_obs[pid] >= 3 and fg_fraction.get(pid, 0.0) <= 0.25
                  and p["error"] <= 3.0
                  and np.linalg.norm(p["xyz"] - obj_center) <= 2.5 * obj_radius]
    bg = np.asarray(background, dtype=np.float64).reshape(-1, 3)
    if len(bg) < 20:
        plane = {"status": "insufficient_nearby_background_points", "candidate_count": len(bg)}
    else:
        rng = np.random.default_rng(0)
        threshold = 0.02 * obj_radius
        best = np.zeros(len(bg), dtype=bool)
        for _ in range(5000):
            chosen = bg[rng.choice(len(bg), 3, replace=False)]
            n = np.cross(chosen[1] - chosen[0], chosen[2] - chosen[0])
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            n /= norm
            inliers = np.abs((bg - chosen[0]) @ n) <= threshold
            if inliers.sum() > best.sum():
                best = inliers
        if best.sum() < 20:
            plane = {"status": "ransac_found_no_stable_plane", "candidate_count": len(bg)}
        else:
            pts = bg[best]
            origin = pts.mean(axis=0)
            _, _, vh = np.linalg.svd(pts - origin, full_matrices=False)
            normal = vh[-1]
            camera_centers = []
            for im in images.values():
                r = qvec2rotmat(im["qvec"])
                camera_centers.append(-r.T @ im["tvec"])
            if np.dot(normal, np.mean(camera_centers, axis=0) - origin) < 0:
                normal = -normal
            plane = {"status": "ok", "normal": normal.tolist(),
                     "offset": float(-normal @ origin), "inliers": int(best.sum()),
                     "candidate_count": int(len(bg)), "threshold": threshold}

    # New, independent training dataset. Use masked RGB plus alpha for MILo's SDF.
    output_root.mkdir(parents=True)
    image_out = output_root / "images"
    sparse_out = output_root / "sparse" / "0"
    val_out = output_root / "validation"
    masks_out = output_root / "masks"
    image_out.mkdir()
    sparse_out.mkdir(parents=True)
    val_out.mkdir()
    masks_out.mkdir()
    f = float(camera["f"])
    k = np.array([[f, 0, camera["cx"]], [0, f, camera["cy"]], [0, 0, 1]], dtype=np.float64)
    dist = np.array([camera["k"], 0, 0, 0, 0], dtype=np.float64)
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (width, height), 0.0, (width, height))
    map_x, map_y = cv2.initUndistortRectifyMap(
        k, dist, None, new_k, (width, height), cv2.CV_32FC1)
    out_w, out_h = width - 2, height - 2
    cam_lines = [f"{camera['id']} PINHOLE {out_w} {out_h} {new_k[0,0]:.12g} {new_k[1,1]:.12g} {new_k[0,2]-1:.12g} {new_k[1,2]-1:.12g}"]
    (sparse_out / "cameras.txt").write_text("# Camera list\n" + "\n".join(cam_lines) + "\n", encoding="utf-8")
    image_lines = ["# Image list with two lines of data per image:"]
    coverages = []
    for iid, im in sorted(images.items(), key=lambda item: item[1]["name"]):
        name = im["name"]
        stem = Path(name).stem
        with Image.open(images_dir / name) as pil:
            rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        mask = get_mask(name)
        undistorted = cv2.undistort(rgb, k, dist, None, new_k)[1:-1, 1:-1]
        undistorted_mask = cv2.remap(mask.astype(np.uint8), map_x, map_y,
                                     cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)[1:-1, 1:-1] > 0
        rgba = np.dstack((undistorted * undistorted_mask[..., None],
                          undistorted_mask.astype(np.uint8) * 255))
        Image.fromarray(rgba, mode="RGBA").save(image_out / f"{stem}.png")
        Image.fromarray(undistorted_mask.astype(np.uint8) * 255, mode="L").save(
            masks_out / f"{stem}.png")
        coverages.append(float(undistorted_mask.mean()))
        q = im["qvec"]
        t = im["tvec"]
        image_lines.append(f"{iid} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
                           f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {im['camera_id']} {stem}.png")
        image_lines.append("")
    (sparse_out / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    point_lines = ["# Object-only points: point3D_id X Y Z R G B error track[]"]
    for p in object_points:
        x, y, z = p["xyz"]
        r, g, b = p["rgb"]
        point_lines.append(f"{p['id']} {x:.17g} {y:.17g} {z:.17g} {r} {g} {b} {p['error']:.12g}")
    (sparse_out / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    with (sparse_out / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(object_points)))
        for p in object_points:
            f.write(struct.pack("<Q3d3BdQ", int(p["id"]), *p["xyz"],
                                *map(int, p["rgb"]), float(p["error"]), 0))
    # Validation views and masks are staged separately; MILo never reads them.
    for row in val_rows:
        name = row["image"]
        stem = Path(name).stem
        with Image.open(images_dir / name) as pil:
            rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        mask = get_mask(name)
        undistorted = cv2.undistort(rgb, k, dist, None, new_k)[1:-1, 1:-1]
        undistorted_mask = cv2.remap(mask.astype(np.uint8), map_x, map_y,
                                     cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)[1:-1, 1:-1] > 0
        rgba = np.dstack((undistorted * undistorted_mask[..., None],
                          undistorted_mask.astype(np.uint8) * 255))
        Image.fromarray(rgba, mode="RGBA").save(val_out / f"{stem}.png")
        Image.fromarray(undistorted_mask.astype(np.uint8) * 255, mode="L").save(
            masks_out / f"{stem}.png")

    report = {
        "source_map": str(sparse_dir), "source_manifest": str(manifest_path),
        "train_images": len(train_rows), "val_images_separate": len(val_rows),
        "colmap_registered_images": len(images), "camera_model_in": "SIMPLE_RADIAL",
        "camera_model_out": "PINHOLE", "output_resolution": [out_w, out_h],
        "undistortion_crop": "one pixel on each border",
        "object_point_rule": ">=3 observations, >=65% inside refined plush masks, reprojection error <=3px",
        "object_sparse_points": len(object_points), "object_center_colmap": obj_center.tolist(),
        "object_radius_p95": obj_radius, "clip_static_silhouette_checks": check,
        "support_plane": plane,
        "train_alpha_masks": "SAM2 image-refined masks, undistorted and stored as RGBA alpha",
        "primary_mask_dir": str(masks_dir),
        "fallback_mask_dir": str(fallback_masks_dir) if fallback_masks_dir else None,
        "fallback_mask_frames": sorted(fallback_mask_names),
        "val_used_for_training": False, "reserved_test_clip_used": False,
        "mean_undistorted_train_coverage": float(np.mean(coverages)),
    }
    (output_root / "colmap_preparation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "images-dir", "masks-dir", "sparse-dir", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fallback-masks-dir", type=Path,
                        help="optional source masks used only when a primary mask is absent")
    args = parser.parse_args()
    print(json.dumps(make_dataset(args.manifest, args.images_dir, args.masks_dir,
                                  args.sparse_dir, args.output_root,
                                  args.fallback_masks_dir), indent=2))


if __name__ == "__main__":
    main()
