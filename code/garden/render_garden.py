"""Render a GSplat checkpoint from any supplied camera JSON split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from gsplat import rasterization

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from utils.cameras import load_camera_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("run with CUDA_VISIBLE_DEVICES bound to exactly one physical GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = {key: value.to(device) for key, value in state["params"].items()}
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    camera_set = load_camera_json(args.camera_json)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for index, view in enumerate(camera_set.views):
            k = camera_set.intrinsics
            Ks = torch.tensor([[[k.fx, 0.0, k.cx], [0.0, k.fy, k.cy],
                                [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
            w2c = torch.as_tensor(view.world_to_camera_gsplat, dtype=torch.float32,
                                  device=device)[None]
            image, alpha, _ = rasterization(
                means=params["means"], quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]), colors=colors,
                viewmats=w2c, Ks=Ks, width=k.width, height=k.height,
                sh_degree=args.sh_degree, packed=True,
            )
            pixels = (image[0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            target = args.output / (Path(view.name).stem + ".png")
            Image.fromarray(pixels, mode="RGB").save(target)
            rows.append({"index": index, "name": view.name, "png": target.name,
                         "width": k.width, "height": k.height,
                         "alpha_sum": float(alpha.sum().cpu())})
            print(f"rendered {index + 1}/{len(camera_set.views)} {target.name}", flush=True)
    summary = {"checkpoint": str(args.checkpoint), "camera_json": str(args.camera_json),
               "physical_gpu": args.physical_gpu, "visible_cuda_devices": torch.cuda.device_count(),
               "device": torch.cuda.get_device_name(device), "views": rows}
    (args.output / "render_summary.json").write_text(json.dumps(summary, indent=2),
                                                       encoding="utf-8")


if __name__ == "__main__":
    main()
