"""Tiny one-GPU integration smoke test; not a quality/training run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from utils.cameras import load_camera_json  # noqa: E402
from utils.colmap import load_points3d_binary  # noqa: E402


def render(means, quats, scales, opacities, colors, view, intr, device):
    w2c = torch.as_tensor(view.world_to_camera_gsplat, dtype=torch.float32,
                          device=device)[None]
    k = torch.tensor([[intr.fx, 0.0, intr.cx],
                      [0.0, intr.fy, intr.cy],
                      [0.0, 0.0, 1.0]], dtype=torch.float32,
                     device=device)[None]
    image, alpha, meta = rasterization(
        means, quats, scales, opacities, colors, w2c, k,
        width=intr.width, height=intr.height, packed=True,
    )
    return image, alpha, meta


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = (tensor[0].detach().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Garden root containing train/, val/ and sparse/ under train/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    if args.train_steps < 1 or args.max_points < 1:
        raise ValueError("train steps and point limit must be positive")

    device = torch.device("cuda:0")  # physical GPU is isolated by CUDA_VISIBLE_DEVICES
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA GPU")
    torch.manual_seed(20261111)
    train_set = load_camera_json(args.data_root / "train/cameras.json")
    val_set = load_camera_json(args.data_root / "val/cameras.json")
    train_view = next(v for v in train_set.views if v.name == "DSC07957.JPG")
    val_view = val_set.views[0]
    target_path = args.data_root / "train/images" / train_view.name
    with Image.open(target_path) as im:
        target = torch.from_numpy(np.array(im.convert("RGB"), copy=True)).to(device=device,
                    dtype=torch.float32).div_(255.0)[None]
    if target.shape[1:3] != (train_set.intrinsics.height, train_set.intrinsics.width):
        raise ValueError(f"training image size mismatch: {tuple(target.shape)}")

    sparse = load_points3d_binary(args.data_root / "train/sparse/0/points3D.bin")
    rng = np.random.default_rng(20261111)
    count = min(args.max_points, len(sparse.xyz))
    selected = rng.choice(len(sparse.xyz), count, replace=False)
    means = torch.nn.Parameter(torch.as_tensor(sparse.xyz[selected], dtype=torch.float32,
                                                device=device))
    colors = torch.nn.Parameter(torch.as_tensor(sparse.rgb[selected], dtype=torch.float32,
                                                 device=device).div_(255.0))
    scales = torch.nn.Parameter(torch.full_like(means, 0.01))
    quats = torch.zeros((count, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # gsplat quaternion order is w,x,y,z
    opacities = torch.nn.Parameter(torch.full((count,), 0.5, device=device))
    optimizer = torch.optim.Adam([means, colors, scales, opacities], lr=1e-4)

    # Positive-Z identity-camera probe catches convention/backend mismatches
    # before the supplied scene cameras are used.
    probe_mean = torch.tensor([[0.0, 0.0, 2.0]], device=device)
    probe = rasterization(
        probe_mean, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        torch.full((1, 3), 0.1, device=device), torch.ones(1, device=device),
        torch.ones((1, 3), device=device), torch.eye(4, device=device)[None],
        torch.tensor([[[100.0, 0.0, 64.0], [0.0, 100.0, 32.0],
                       [0.0, 0.0, 1.0]]], device=device),
        width=128, height=64,
    )
    probe_alpha = float(probe[1].sum().item())
    if not np.isfinite(probe_alpha) or probe_alpha <= 0:
        raise RuntimeError("positive-Z rasterizer convention probe rendered no alpha")

    losses: list[float] = []
    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction, _, _ = render(means, quats, scales, opacities, colors,
                                  train_view, train_set.intrinsics, device)
        loss = F.l1_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite smoke loss at step {step}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"step={step}/{args.train_steps} l1={losses[-1]:.7f}", flush=True)

    val_image, val_alpha, _ = render(means, quats, scales, opacities, colors,
                                     val_view, val_set.intrinsics, device)
    args.output.mkdir(parents=True, exist_ok=True)
    render_path = args.output / (Path(val_view.name).stem + ".png")
    save_rgb(render_path, val_image)
    with (args.output / "loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("step", "l1"))
        writer.writerows(enumerate(losses, start=1))
    summary = {
        "physical_gpu": args.physical_gpu,
        "visible_cuda_devices": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gsplat": __import__("importlib.metadata", fromlist=["version"]).version("gsplat"),
        "seed": 20261111,
        "train_image": train_view.name,
        "render_camera": val_view.name,
        "render_size": [val_set.intrinsics.width, val_set.intrinsics.height],
        "colmap_points_total": len(sparse.xyz),
        "gaussians_used": count,
        "train_steps": args.train_steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "positive_z_probe_alpha_sum": probe_alpha,
        "val_alpha_sum": float(val_alpha.sum().item()),
        "max_gpu_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
        "render_png": str(render_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
