"""Generate development-only plush masks with SAM 2.1 video propagation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import torch


def _load_mask(path: Path, expected: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    if mask.shape != expected or not mask.any():
        raise ValueError(f"invalid prompt mask {path}: {mask.shape}, expected {expected}")
    return mask


def _largest_interior(mask: np.ndarray) -> tuple[int, int]:
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return int(x), int(y)


def _box(mask: np.ndarray, padding: int) -> list[float]:
    ys, xs = np.where(mask)
    h, w = mask.shape
    return [float(max(0, xs.min() - padding)),
            float(max(0, ys.min() - padding)),
            float(min(w - 1, xs.max() + padding)),
            float(min(h - 1, ys.max() + padding))]


def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("SAM 2 returned an empty mask")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    threshold = max(1, int(sizes[largest] * 0.02))
    keep = [idx for idx in range(1, count + 1) if sizes[idx] >= threshold]
    cleaned = np.isin(labels, keep)
    # Fill internal holes and close only 4-neighbour gaps; do not erode soft edges.
    cleaned = ndimage.binary_fill_holes(cleaned)
    cleaned = ndimage.binary_closing(
        cleaned, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        iterations=1)
    return cleaned, count


def _decode_video(video_path: Path, output_dir: Path) -> tuple[int, tuple[int, int]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"could not open development clip {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shape: tuple[int, int] | None = None
    for idx in range(count):
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"short decode in {video_path.name} at frame {idx}")
        current = bgr.shape[:2]
        if shape is None:
            shape = current
        if current != shape or current != (720, 406):
            cap.release()
            raise ValueError(f"unexpected dimensions in {video_path.name}: {current}")
        target = output_dir / f"{idx:06d}.jpg"
        if not cv2.imwrite(str(target), bgr, [cv2.IMWRITE_JPEG_QUALITY, 98]):
            cap.release()
            raise OSError(f"failed to write {target}")
    cap.release()
    if shape is None:
        raise ValueError(f"no frames in {video_path}")
    return count, shape


def _prompts_for_clip(rows: list[dict], source_masks: Path, shape: tuple[int, int],
                      stride: int, padding: int, negative_offset: int,
                      manual_points: dict[str, list[list[int]]]) -> list[dict]:
    selected = [row for row in rows if row["split"] == "train" and
                int(Path(row["image"]).stem.split("_")[1]) % stride == 0]
    if rows:
        first_train = min((row for row in rows if row["split"] == "train"),
                          key=lambda row: row["image"])
        if first_train not in selected:
            selected.append(first_train)
    for stem in manual_points:
        row = next((r for r in rows if Path(r["image"]).stem == stem), None)
        if row is not None and row not in selected:
            selected.append(row)
    selected.sort(key=lambda row: int(Path(row["image"]).stem.split("_")[1]))

    prompts = []
    for row in selected:
        stem = Path(row["image"]).stem
        frame_idx = int(stem.split("_")[1])
        mask = _load_mask(source_masks / f"{stem}.png", shape)
        box = _box(mask, padding)
        center = _largest_interior(mask)
        neg_x = int(round((box[0] + box[2]) / 2))
        neg_y = int(min(shape[0] - 3, max(box[3] + 2, box[3] + negative_offset)))
        points = [[center[0], center[1], 1], [neg_x, neg_y, 0]]
        points += [[int(x), int(y), 1] for x, y in manual_points.get(stem, [])]
        prompts.append({"stem": stem, "frame_idx": frame_idx, "box": box,
                        "points": points})
    return prompts


def run(manifest_path: Path, video_dir: Path, source_masks: Path, output_root: Path,
        checkpoint: Path, sam2_package_root: Path, prompts_path: Path,
        config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda:0", physical_gpu: int | None = None) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM 2 propagation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("bind this run to exactly one physical GPU with CUDA_VISIBLE_DEVICES")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if physical_gpu is not None and visible != str(physical_gpu):
        raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES={physical_gpu}, got {visible!r}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["images"]
    if any(Path(row["image"]).stem.startswith("1980_") for row in rows):
        raise ValueError("reserved 1980 test clip leaked into the development manifest")
    settings = json.loads(prompts_path.read_text(encoding="utf-8"))
    stride = int(settings["keyframe_stride"])
    manual_points = settings["manual_positive_points_xy"]

    # Import SAM 2 only after the training-data/test-split guard has passed.
    sys.path.insert(0, str(sam2_package_root.parent))
    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = checkpoint.resolve()
    sam2_package_root = sam2_package_root.resolve()
    binary_root = output_root / "masks"
    alpha_root = output_root / "alpha"
    scratch_root = output_root / "_video_frames"
    original_cwd = Path.cwd()
    try:
        os.chdir(sam2_package_root)
        model = build_sam2_video_predictor(
            config, str(checkpoint), device=device, apply_postprocessing=False)
    finally:
        os.chdir(original_cwd)
    output_root.mkdir(parents=True, exist_ok=False)
    binary_root.mkdir()
    alpha_root.mkdir()
    scratch_root.mkdir()

    output_rows = []
    clip_summaries = {}
    for clip in ("1978.mp4", "1979.mp4"):
        clip_rows = [row for row in rows if row["frame_id"].split(":", 1)[0] == clip]
        if not clip_rows:
            continue
        video_path = video_dir / clip
        temp_frames = scratch_root / Path(clip).stem
        num_frames, shape = _decode_video(video_path, temp_frames)
        if any(int(Path(row["image"]).stem.split("_")[1]) >= num_frames
               for row in clip_rows):
            raise ValueError(f"manifest frame index exceeds {clip}")
        prompt_rows = _prompts_for_clip(
            clip_rows, source_masks, shape, stride,
            int(settings["box_padding_px"]), int(settings["negative_offset_px"]),
            manual_points,
        )
        state = model.init_state(
            video_path=str(temp_frames), offload_video_to_cpu=True,
            offload_state_to_cpu=True, async_loading_frames=False)
        for row in prompt_rows:
            points = np.asarray([[p[0], p[1]] for p in row["points"]], dtype=np.float32)
            labels = np.asarray([p[2] for p in row["points"]], dtype=np.int32)
            model.add_new_points_or_box(
                state, frame_idx=row["frame_idx"], obj_id=1,
                points=points, labels=labels, box=np.asarray(row["box"], dtype=np.float32),
                clear_old_points=True, normalize_coords=False)

        predictions: dict[int, np.ndarray] = {}
        start_idx = min(row["frame_idx"] for row in prompt_rows)
        for frame_idx, object_ids, logits in model.propagate_in_video(
                state, start_frame_idx=start_idx, reverse=False):
            if 1 in object_ids:
                pred_idx = object_ids.index(1)
                predictions[frame_idx] = logits[pred_idx, 0].detach().float().cpu().numpy()
        if start_idx > 0:
            for frame_idx, object_ids, logits in model.propagate_in_video(
                    state, start_frame_idx=start_idx, reverse=True):
                if 1 in object_ids:
                    pred_idx = object_ids.index(1)
                    predictions[frame_idx] = logits[pred_idx, 0].detach().float().cpu().numpy()
        model.reset_state(state)

        wanted = {int(Path(row["image"]).stem.split("_")[1]): row for row in clip_rows}
        missing = sorted(set(wanted) - set(predictions))
        if missing:
            raise RuntimeError(f"SAM 2 omitted {clip} frames {missing[:10]}")
        areas = []
        for frame_idx, row in sorted(wanted.items()):
            stem = Path(row["image"]).stem
            logits = predictions[frame_idx]
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            raw = logits > 0.0
            fallback = False
            # Video propagation can occasionally collapse to an empty mask on fast
            # motion. Keep the annotated U2Net silhouette for that frame rather
            # than silently dropping it or aborting the whole sequence.
            source_mask = None
            if not raw.any():
                source_mask = _load_mask(source_masks / f"{stem}.png", shape)
                raw = source_mask.copy()
                prob = np.maximum(prob, source_mask.astype(np.float32))
                fallback = True
            cleaned, component_count = _components(raw)
            area = float(cleaned.mean())
            if not 0.02 <= area <= 0.55:
                if source_mask is None:
                    source_mask = _load_mask(source_masks / f"{stem}.png", shape)
                cleaned, component_count = _components(source_mask)
                area = float(cleaned.mean())
                fallback = True
                if not 0.02 <= area <= 0.55:
                    raise ValueError(f"implausible SAM 2/source mask coverage {area:.3f} for {stem}")
            out_mask = (cleaned.astype(np.uint8) * 255)
            out_alpha = (np.clip(prob, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            out_alpha[~cleaned] = 0
            Image.fromarray(out_mask, mode="L").save(binary_root / f"{stem}.png")
            Image.fromarray(out_alpha, mode="L").save(alpha_root / f"{stem}.png")
            border = bool(cleaned[0].any() or cleaned[-1].any() or
                          cleaned[:, 0].any() or cleaned[:, -1].any())
            output_rows.append({"image": row["image"], "split": row["split"],
                                "area_fraction": area, "raw_component_count": component_count,
                                "border_touch": border,
                                "source_mask_fallback": fallback,
                                "prompted_frame": any(p["frame_idx"] == frame_idx
                                                      for p in prompt_rows)})
            areas.append((frame_idx, cleaned, area))

        adjacent_ious = []
        by_idx = {idx: mask for idx, mask, _ in areas}
        for left in sorted(by_idx):
            right = left + 1
            if right not in by_idx:
                continue
            a, b = by_idx[left], by_idx[right]
            union = np.logical_or(a, b).sum()
            adjacent_ious.append(float(np.logical_and(a, b).sum() / max(int(union), 1)))
        clip_summaries[clip] = {
            "decoded_source_frames": num_frames,
            "development_manifest_frames": len(clip_rows),
            "keyframe_prompt_count": len(prompt_rows),
            "prompt_frames": [row["frame_idx"] for row in prompt_rows],
            "mean_mask_area": float(np.mean([area for _, _, area in areas])),
            "min_mask_area": float(np.min([area for _, _, area in areas])),
            "max_mask_area": float(np.max([area for _, _, area in areas])),
            "adjacent_iou_mean": (float(np.mean(adjacent_ious)) if adjacent_ious else None),
            "adjacent_iou_min": (float(np.min(adjacent_ious)) if adjacent_ious else None),
        }
        shutil.rmtree(temp_frames)

    manifest_out = {
        "method": "SAM 2.1 Hiera Large, prompted every 10 train frames; binary plus soft alpha",
        "checkpoint": str(checkpoint), "config": config, "device": device,
        "physical_gpu": physical_gpu,
        "input_manifest": str(manifest_path), "training_prompt_masks_source": str(source_masks),
        "manual_limb_prompts": settings["manual_positive_points_xy"],
        "validation_frames_used_as_prompts": False,
        "reserved_test_clip_used": False,
        "frame_count": len(output_rows), "clips": clip_summaries,
        "per_frame": output_rows,
    }
    (output_root / "mask_report.json").write_text(
        json.dumps(manifest_out, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in manifest_out.items() if key != "per_frame"}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "video-dir", "source-masks", "output-root",
                 "checkpoint", "sam2-package-root", "prompts"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.video_dir, args.source_masks,
                         args.output_root, args.checkpoint,
                         args.sam2_package_root, args.prompts,
                         args.config, args.device, args.physical_gpu), indent=2))


if __name__ == "__main__":
    main()
