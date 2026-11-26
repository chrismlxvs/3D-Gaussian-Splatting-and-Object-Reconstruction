"""Refine training-view plush masks with per-frame SAM 2 image prompts.

Each training image uses its existing binary silhouette only to derive a loose
box and interior point. Validation images are never sent to SAM as prompts and
their archived masks are copied unchanged for held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
import torch


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    if mask.shape != shape or not mask.any():
        raise ValueError(f"invalid source mask {path}: {mask.shape}, expected {shape}")
    return mask


def clean(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if not count:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    keep = np.flatnonzero(sizes >= max(1, int(sizes[largest] * 0.02)))
    keep = keep[keep != 0]
    out = np.isin(labels, keep)
    return ndimage.binary_closing(
        out, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))


def run(manifest_path: Path, image_dir: Path, source_masks: Path,
        output_root: Path, checkpoint: Path, sam2_package_root: Path,
        prompts_path: Path, physical_gpu: int) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("bind this run to exactly one visible CUDA GPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES={physical_gpu}, got {visible!r}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["images"]
    if any(Path(r["image"]).stem.startswith("1980_") for r in rows):
        raise ValueError("reserved test clip leaked into the development manifest")
    settings = json.loads(prompts_path.read_text(encoding="utf-8"))
    manual_points = settings["manual_positive_points_xy"]
    sys.path.insert(0, str(sam2_package_root.parent.resolve()))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = checkpoint.resolve()
    package_root = sam2_package_root.resolve()
    original_cwd = Path.cwd()
    try:
        os.chdir(package_root)
        model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
                           str(checkpoint), device="cuda:0",
                           apply_postprocessing=False)
    finally:
        os.chdir(original_cwd)
    predictor = SAM2ImagePredictor(model)

    masks_dir = output_root / "masks"
    alpha_dir = output_root / "alpha"
    output_root.mkdir(parents=True, exist_ok=False)
    masks_dir.mkdir()
    alpha_dir.mkdir()
    report_rows = []
    for row in rows:
        stem = Path(row["image"]).stem
        mask_path = source_masks / f"{stem}.png"
        if row["split"] == "val":
            with Image.open(mask_path) as im:
                source = np.asarray(im.convert("L"), dtype=np.uint8)
            refined = source >= 128
            method = "held_out_source_mask_unchanged"
            score = None
        else:
            image_path = image_dir / row["image"]
            with Image.open(image_path) as im:
                rgb = np.asarray(im.convert("RGB"))
            h, w = rgb.shape[:2]
            source = load_mask(mask_path, (h, w))
            ys, xs = np.where(source)
            padding = int(settings["box_padding_px"])
            box = np.asarray([
                max(0, xs.min() - padding), max(0, ys.min() - padding),
                min(w - 1, xs.max() + padding), min(h - 1, ys.max() + padding),
            ], dtype=np.float32)
            center = np.asarray([xs[np.argmax(ndimage.distance_transform_edt(source)[ys, xs])],
                                 ys[np.argmax(ndimage.distance_transform_edt(source)[ys, xs])]],
                                dtype=np.float32)
            neg_y = min(h - 2, max(int(box[3]) + 2,
                                   int(box[3]) + int(settings["negative_offset_px"])))
            points = [center.tolist(), [float((box[0] + box[2]) / 2), float(neg_y)]]
            labels = [1, 0]
            points.extend([[float(x), float(y)] for x, y in manual_points.get(stem, [])])
            labels.extend([1] * len(manual_points.get(stem, [])))
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray(points, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32),
                box=box, multimask_output=False, normalize_coords=True)
            refined = clean(np.asarray(masks[0], dtype=bool))
            coverage = float(refined.mean())
            method = "sam2_image_prompt"
            score = float(scores[0])
            if not refined.any() or not 0.02 <= coverage <= 0.55:
                refined = clean(source)
                method = "source_mask_fallback_invalid_sam"
        coverage = float(refined.mean())
        if not 0.02 <= coverage <= 0.55:
            raise ValueError(f"implausible mask coverage {coverage:.3f} for {stem}")
        binary = refined.astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(masks_dir / f"{stem}.png")
        Image.fromarray(binary, mode="L").save(alpha_dir / f"{stem}.png")
        report_rows.append({"image": row["image"], "split": row["split"],
                            "area_fraction": coverage, "method": method,
                            "sam_score": score})
    report = {
        "method": "SAM 2.1 Hiera Large image predictor; per-frame weak prompts on train only",
        "checkpoint": str(checkpoint), "physical_gpu": physical_gpu,
        "input_manifest": str(manifest_path), "source_mask_dir": str(source_masks),
        "validation_prompts_used": False, "validation_masks_modified": False,
        "reserved_test_clip_used": False, "frame_count": len(report_rows),
        "sam_train_count": sum(r["split"] == "train" and r["method"] == "sam2_image_prompt"
                                for r in report_rows),
        "fallback_count": sum(r["method"].startswith("source_mask_fallback")
                              for r in report_rows),
        "per_frame": report_rows,
    }
    (output_root / "mask_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in report.items() if k != "per_frame"}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "image-dir", "source-masks", "output-root",
                 "checkpoint", "sam2-package-root", "prompts"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.image_dir, args.source_masks, args.output_root,
                 args.checkpoint, args.sam2_package_root, args.prompts,
                 args.physical_gpu)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
