"""Garden 3DGS trainer using gsplat's rasterizer and DefaultStrategy.

The assignment camera JSON is authoritative for image poses and intrinsics.
All supplied Garden training views are included; this script has no automatic
holdout. A camera JSON split may only be used for render/evaluation, never as
training input.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from utils.cameras import CameraSet, load_camera_json  # noqa: E402
from utils.colmap import load_points3d_binary  # noqa: E402


def ssim_loss(render: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Differentiable DSSIM loss with the standard 3DGS constants."""
    x = render.permute(0, 3, 1, 2)
    y = target.permute(0, 3, 1, 2)
    mx = F.avg_pool2d(x, window, stride=1)
    my = F.avg_pool2d(y, window, stride=1)
    vx = F.avg_pool2d(x * x, window, stride=1) - mx * mx
    vy = F.avg_pool2d(y * y, window, stride=1) - my * my
    vxy = F.avg_pool2d(x * y, window, stride=1) - mx * my
    score = ((2 * mx * my + 0.01**2) * (2 * vxy + 0.03**2)) / (
        (mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2)
    )
    return 1.0 - score.mean()


def _load_images(camera_set: CameraSet, image_dir: Path) -> tuple[list[str], torch.Tensor]:
    names: list[str] = []
    images: list[np.ndarray] = []
    expected = (camera_set.intrinsics.width, camera_set.intrinsics.height)
    for view in camera_set.views:
        path = image_dir / view.name
        if not path.is_file():
            raise FileNotFoundError(f"camera {view.name} has no training image at {path}")
        with Image.open(path) as image:
            if image.size != expected:
                raise ValueError(f"{path} is {image.size}, expected {expected}")
            images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        names.append(view.name)
    if len(set(names)) != len(names):
        raise ValueError("camera JSON contains duplicate view names")
    return names, torch.from_numpy(np.stack(images))


def _load_masks(names: list[str], mask_dir: Path, width: int, height: int) -> torch.Tensor:
    masks: list[np.ndarray] = []
    for name in names:
        path = mask_dir / (Path(name).stem + ".png")
        if not path.is_file():
            raise FileNotFoundError(f"training image {name} has no foreground mask at {path}")
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f"{path} is {image.size}, expected {(width, height)}")
            masks.append(np.asarray(image.convert("L"), dtype=np.uint8) > 127)
    return torch.from_numpy(np.stack(masks))


def _make_parameters(points_xyz: np.ndarray, points_rgb: np.ndarray,
                     device: torch.device, seed: int) -> torch.nn.ParameterDict:
    xyz = np.asarray(points_xyz, dtype=np.float32)
    rgb = np.asarray(points_rgb, dtype=np.float32) / 255.0
    distances, _ = cKDTree(xyz).query(xyz, k=4, workers=-1)
    scale = np.sqrt(np.mean(np.square(distances[:, 1:]), axis=1)).clip(1e-6)
    means = torch.as_tensor(xyz, device=device)
    sh = torch.zeros((len(xyz), 16, 3), dtype=torch.float32, device=device)
    sh[:, 0, :] = (torch.as_tensor(rgb, device=device) - 0.5) / 0.28209479177387814
    generator = torch.Generator(device=device).manual_seed(seed)
    quats = torch.randn((len(xyz), 4), generator=generator, device=device)
    quats = F.normalize(quats, dim=-1)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.as_tensor(scale, dtype=torch.float32,
                                                      device=device).log()[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(torch.full((len(xyz),), -2.1972246, device=device)),
        "sh0": torch.nn.Parameter(sh[:, :1, :].clone()),
        "shN": torch.nn.Parameter(sh[:, 1:, :].clone()),
    })


def _camera_tensors(camera_set: CameraSet, device: torch.device):
    w2cs = torch.as_tensor(np.stack([v.world_to_camera_gsplat for v in camera_set.views]),
                           dtype=torch.float32, device=device)
    k = camera_set.intrinsics
    intrinsics = torch.tensor([[k.fx, 0.0, k.cx], [0.0, k.fy, k.cy],
                               [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    return w2cs, intrinsics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Garden data directory with train/ and train/sparse/0/")
    parser.add_argument("--camera-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--loss-mask-dir", type=Path, default=None,
                        help="optional directory of per-view foreground masks")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gsplat-commit", default="937e29912570c372bed6747a5c9bf85fed877bae")
    args = parser.parse_args()
    if args.max_steps < 1 or args.log_every < 1 or args.checkpoint_every < 1:
        raise ValueError("step and logging intervals must be positive")

    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("run with CUDA_VISIBLE_DEVICES bound to exactly one physical GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    camera_json = args.camera_json or (args.data_root / "train/cameras.json")
    camera_set = load_camera_json(camera_json)
    names, cpu_images = _load_images(camera_set, args.data_root / "train/images")
    images = cpu_images.to(device=device, dtype=torch.float32).div_(255.0)
    masks = None
    if args.loss_mask_dir is not None:
        masks = _load_masks(names, args.loss_mask_dir, camera_set.intrinsics.width,
                            camera_set.intrinsics.height).to(device=device, dtype=torch.float32)
    w2cs, intrinsic = _camera_tensors(camera_set, device)

    points = load_points3d_binary(args.data_root / "train/sparse/0/points3D.bin")
    if not len(points.xyz):
        raise ValueError("supplied COLMAP point cloud is empty")
    params = _make_parameters(points.xyz, points.rgb, device, args.seed)
    camera_centers = torch.linalg.inv(w2cs)[:, :3, 3]
    scene_center = camera_centers.mean(0)
    scene_scale = float(torch.linalg.vector_norm(camera_centers - scene_center, dim=-1).max())

    initial_points = len(params["means"])
    optimizer_lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3,
                     "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3,
                     "shN": 1.25e-4}
    optimizers = {key: torch.optim.Adam([value], lr=optimizer_lrs[key])
                  for key, value in params.items()}
    strategy = DefaultStrategy(verbose=False)
    strategy.refine_stop_iter = min(strategy.refine_stop_iter, args.max_steps)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    strategy.check_sanity(params, optimizers)
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / args.max_steps))

    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": args.seed, "max_steps": args.max_steps,
        "training_views": names, "training_view_count": len(names),
        "automatic_holdout": False,
        "camera_json": str(camera_json), "width": camera_set.intrinsics.width,
        "height": camera_set.intrinsics.height, "fx": camera_set.intrinsics.fx,
        "fy": camera_set.intrinsics.fy, "cx": camera_set.intrinsics.cx,
        "cy": camera_set.intrinsics.cy,
        "colmap_point_count": len(points.xyz), "initial_gaussians": initial_points,
        "learning_rates": optimizer_lrs, "ssim_weight": args.ssim_weight,
        "device": torch.cuda.get_device_name(device), "physical_gpu": args.physical_gpu,
        "visible_cuda_devices": torch.cuda.device_count(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gsplat": __import__("importlib.metadata", fromlist=["version"]).version("gsplat"),
        "gsplat_commit": args.gsplat_commit,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    loss_path = args.output / "loss.csv"
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(names)).tolist()
    cursor = 0
    rows = []
    tic = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        if cursor == len(order):
            order = rng.permutation(len(names)).tolist()
            cursor = 0
        camera_idx = order[cursor]
        cursor += 1
        target = images[camera_idx:camera_idx + 1]
        loss_mask = masks[camera_idx:camera_idx + 1, ..., None] if masks is not None else None
        w2c = w2cs[camera_idx:camera_idx + 1]
        k = intrinsic[None]

        sh_degree = min((step - 1) // 1000, 3)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        rendered, _, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colors,
            viewmats=w2c, Ks=k, width=camera_set.intrinsics.width,
            height=camera_set.intrinsics.height, sh_degree=sh_degree,
            packed=True,
        )
        if loss_mask is None:
            l1 = F.l1_loss(rendered, target)
            ssim = ssim_loss(rendered, target)
        else:
            l1 = ((rendered - target).abs() * loss_mask).sum() / (loss_mask.sum() * 3).clamp_min(1)
            x = rendered.permute(0, 3, 1, 2)
            y = target.permute(0, 3, 1, 2)
            mx = F.avg_pool2d(x, 11, stride=1)
            my = F.avg_pool2d(y, 11, stride=1)
            vx = F.avg_pool2d(x * x, 11, stride=1) - mx * mx
            vy = F.avg_pool2d(y * y, 11, stride=1) - my * my
            vxy = F.avg_pool2d(x * y, 11, stride=1) - mx * my
            score = ((2 * mx * my + 0.01**2) * (2 * vxy + 0.03**2)) / (
                (mx * mx + my * my + 0.01**2) * (vx + vy + 0.03**2)
            )
            valid = F.avg_pool2d(loss_mask.permute(0, 3, 1, 2), 11, stride=1) > 0.99
            ssim = 1.0 - score.masked_select(valid.expand_as(score)).mean()
        loss = (1.0 - args.ssim_weight) * l1 + args.ssim_weight * ssim
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step} for {names[camera_idx]}")
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        means_scheduler.step()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)

        row = {"step": step, "image": names[camera_idx],
               "loss": float(loss.detach()), "l1": float(l1.detach()),
               "ssim_loss": float(ssim.detach()), "gaussians": len(params["means"])}
        rows.append(row)
        if step % args.log_every == 0 or step == 1 or step == args.max_steps:
            print(f"step={step}/{args.max_steps} image={names[camera_idx]} "
                  f"loss={row['loss']:.6f} l1={row['l1']:.6f} "
                  f"ssim_loss={row['ssim_loss']:.6f} gaussians={row['gaussians']}",
                  flush=True)
        if step % args.checkpoint_every == 0 or step == args.max_steps:
            checkpoint = {key: value.detach().cpu() for key, value in params.items()}
            torch.save({"step": step, "params": checkpoint, "config": config},
                       args.output / f"step_{step:06d}.pt")
        if step % args.log_every == 0 or step == args.max_steps:
            with loss_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=("step", "image", "loss", "l1",
                                                       "ssim_loss", "gaussians"))
                writer.writeheader()
                writer.writerows(rows)

    torch.cuda.synchronize(device)
    summary = {
        "step_count": args.max_steps, "elapsed_seconds": time.perf_counter() - tic,
        "physical_gpu": args.physical_gpu, "visible_cuda_devices": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(device), "training_views": len(names),
        "initial_gaussians": initial_points, "final_gaussians": len(params["means"]),
        "first_loss": rows[0]["loss"], "last_loss": rows[-1]["loss"],
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
