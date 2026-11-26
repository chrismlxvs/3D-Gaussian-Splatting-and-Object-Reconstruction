"""Build an isolated MILo dataset with reviewed white-plush training masks.

Only the three supplied hand masks and the known 1978/118 background-leak mask
are changed. Source RGB is undistorted from the original training JPGs, and the
object-only COLMAP initialization points are re-voted against the revised masks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import struct
import sys

import cv2
import numpy as np
from PIL import Image


def write_points(points: list[dict], sparse_out: Path) -> None:
    with (sparse_out / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# Object-only points: point3D_id X Y Z R G B error track[]\n")
        for p in points:
            xyz = p["xyz"]
            rgb = p["rgb"]
            f.write(f"{p['id']} {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g} "
                    f"{rgb[0]} {rgb[1]} {rgb[2]} {p['error']:.12g}\n")
    with (sparse_out / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for p in points:
            f.write(struct.pack("<Q3d3BdQ", int(p["id"]), *p["xyz"],
                                *map(int, p["rgb"]), float(p["error"]), 0))


def run(base_data: Path, source_images: Path, source_sparse: Path,
        manual_masks: Path, cloth_mask_rgba: Path, output_data: Path) -> dict:
    if output_data.exists():
        raise FileExistsError(f"refusing to overwrite {output_data}")
    # Import the binary COLMAP readers used by the original dataset builder.
    code_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(code_root))
    from colmap_poses import read_cameras, read_images, read_points  # noqa: E402

    shutil.copytree(base_data, output_data)
    image_out = output_data / "images"
    mask_out = output_data / "masks"
    sparse_out = output_data / "sparse" / "0"

    cameras = read_cameras(source_sparse / "cameras.bin")
    images = read_images(source_sparse / "images.bin")
    points = read_points(source_sparse / "points3D.bin")
    if len(cameras) != 1:
        raise ValueError(f"expected one shared COLMAP camera; got {len(cameras)}")
    camera = next(iter(cameras.values()))
    width, height = int(camera["width"]), int(camera["height"])
    if (width, height) != (406, 720):
        raise ValueError(f"unexpected source size {width}x{height}")
    k = np.array([[camera["f"], 0, camera["cx"]],
                  [0, camera["f"], camera["cy"]], [0, 0, 1]], dtype=np.float64)
    dist = np.array([camera["k"], 0, 0, 0, 0], dtype=np.float64)
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (width, height), 0.0,
                                             (width, height))

    overrides: dict[str, np.ndarray] = {}
    for stem in ("1978_000074", "1978_000076", "1978_000100"):
        p = manual_masks / f"{stem}.png"
        if not p.is_file():
            raise FileNotFoundError(p)
        with Image.open(p) as im:
            m = np.asarray(im.convert("L"), dtype=np.uint8)
        if m.shape != (718, 404):
            raise ValueError(f"{p} must be 404x718; got {m.shape[1]}x{m.shape[0]}")
        overrides[stem] = m >= 128

    with Image.open(cloth_mask_rgba) as im:
        cloth = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    if cloth.shape[:2] != (718, 404):
        raise ValueError(f"cloth-mask image must be 404x718; got {cloth.shape[1]}x{cloth.shape[0]}")
    overrides["1978_000118"] = cloth[:, :, 3] >= 128

    source_names = {im["name"] for im in images.values()}
    required_names = {f"{stem}.jpg" for stem in overrides}
    if not required_names.issubset(source_names):
        raise ValueError(f"missing source COLMAP views: {sorted(required_names - source_names)}")

    # Rebuild the four affected RGBA images from full RGB source pixels. This
    # keeps newly included edge pixels from inheriting black from the old alpha.
    for stem, mask in overrides.items():
        raw_path = source_images / f"{stem}.jpg"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        with Image.open(raw_path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != (height, width):
            raise ValueError(f"unexpected raw RGB size in {raw_path}: {rgb.shape[1]}x{rgb.shape[0]}")
        rgb_u = cv2.undistort(rgb, k, dist, None, new_k)[1:-1, 1:-1]
        if rgb_u.shape[:2] != mask.shape:
            raise ValueError(f"undistorted RGB/mask size mismatch for {stem}")
        alpha = mask.astype(np.uint8) * 255
        rgba = np.dstack((rgb_u * mask[:, :, None], alpha))
        Image.fromarray(rgba, mode="RGBA").save(image_out / f"{stem}.png")
        Image.fromarray(alpha, mode="L").save(mask_out / f"{stem}.png")

    # Re-vote original COLMAP points in the revised undistorted masks.
    undistort_k = np.array([[new_k[0, 0], 0, new_k[0, 2] - 1],
                            [0, new_k[1, 1], new_k[1, 2] - 1],
                            [0, 0, 1]], dtype=np.float64)
    del undistort_k  # xys are mapped with OpenCV's exact newK then cropped below.
    votes: dict[int, list[bool]] = {pid: [] for pid in points}
    masks: dict[str, np.ndarray] = {}
    for im in images.values():
        stem = Path(im["name"]).stem
        if stem not in masks:
            with Image.open(mask_out / f"{stem}.png") as p:
                masks[stem] = np.asarray(p.convert("L"), dtype=np.uint8) >= 128
        xy = im["xys"]
        und_xy = cv2.undistortPoints(xy.reshape(-1, 1, 2), k, dist,
                                     P=new_k).reshape(-1, 2) - 1.0
        mask = masks[stem]
        for (x, y), pid in zip(und_xy, im["point_ids"]):
            if pid < 0 or int(pid) not in points:
                continue
            ix, iy = int(round(x)), int(round(y))
            inside = 0 <= ix < 404 and 0 <= iy < 718 and bool(mask[iy, ix])
            votes[int(pid)].append(inside)
    selected = [p for pid, p in points.items()
                if len(votes[pid]) >= 3 and np.mean(votes[pid]) >= 0.65
                and p["error"] <= 3.0]
    if len(selected) < 100:
        raise ValueError(f"only {len(selected)} sparse points passed revised-mask voting")
    write_points(selected, sparse_out)

    report_path = output_data / "colmap_preparation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["train_alpha_masks"] = "image-v2 masks with three user-refined masks and one corrected cloth-leak mask"
    report["manual_mask_frames"] = ["1978_000074", "1978_000076", "1978_000100"]
    report["corrected_known_leak_frames"] = ["1978_000118"]
    report["object_sparse_points"] = len(selected)
    report["val_used_for_training"] = False
    report["reserved_test_clip_used"] = False
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"train_images": len(images), "overrides": sorted(overrides),
            "object_sparse_points": len(selected), "output_data": str(output_data),
            "validation_used_for_training": False, "reserved_test_used": False}


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("base-data", "source-images", "source-sparse", "manual-masks",
                 "cloth-mask-rgba", "output-data"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.base_data, a.source_images, a.source_sparse,
                         a.manual_masks, a.cloth_mask_rgba, a.output_data), indent=2))


if __name__ == "__main__":
    main()
