"""Render Garden RGB and expected depth for mesh/scene compositing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from utils.cameras import load_camera_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--camera-json", type=Path, required=True)
    p.add_argument("--rgb-output", type=Path, required=True)
    p.add_argument("--depth-output", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()
    if a.stride < 1:
        raise ValueError("stride must be at least one")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("bind exactly one CUDA device with CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    params = {k:v.to(device) for k,v in state["params"].items()}
    colors = torch.cat((params["sh0"],params["shN"]),dim=1)
    cameras = load_camera_json(a.camera_json)
    a.rgb_output.mkdir(parents=True,exist_ok=True)
    a.depth_output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.inference_mode():
        for i in range(0,len(cameras.views),a.stride):
            view=cameras.views[i]
            k=cameras.intrinsics
            Ks=torch.tensor([[[k.fx,0.,k.cx],[0.,k.fy,k.cy],[0.,0.,1.]]],
                            dtype=torch.float32,device=device)
            w2c=torch.as_tensor(view.world_to_camera_gsplat,
                                dtype=torch.float32,device=device)[None]
            rendered,alpha,_=rasterization(
                means=params["means"],quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),colors=colors,
                viewmats=w2c,Ks=Ks,width=k.width,height=k.height,
                sh_degree=3,packed=True,render_mode="RGB+ED")
            rgb=(rendered[0,...,:3].clamp(0,1).cpu().numpy()*255+.5).astype(np.uint8)
            depth=rendered[0,...,3].clamp_min(0).cpu().numpy().astype(np.float16)
            alpha_np=alpha[0,...,0].clamp(0,1).cpu().numpy().astype(np.float16)
            name=Path(view.name).stem
            Image.fromarray(rgb,"RGB").save(a.rgb_output/f"{name}.png")
            np.savez_compressed(a.depth_output/f"{name}.npz",depth=depth,alpha=alpha_np)
            rows.append({"index":i,"name":view.name,"width":k.width,
                         "height":k.height,"positive_depth_fraction":float((depth>0).mean()),
                         "alpha_mean":float(alpha_np.mean())})
            print(f"rendered RGB+depth {i+1}/{len(cameras.views)} {view.name}",flush=True)
    summary={"checkpoint":str(a.checkpoint),"camera_json":str(a.camera_json),
             "physical_gpu":a.physical_gpu,"device":torch.cuda.get_device_name(device),
             "stride":a.stride,"views":rows}
    (a.depth_output/"render_summary.json").write_text(json.dumps(summary,indent=2)+"\n")

if __name__=="__main__":
    main()
