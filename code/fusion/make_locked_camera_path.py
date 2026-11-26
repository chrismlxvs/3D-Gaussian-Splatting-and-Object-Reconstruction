"""Repeat one Garden view for a screen-space locked fusion clip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--garden-cameras", type=Path, required=True)
    p.add_argument("--view", default="DSC07956.JPG")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()
    source = json.loads(args.garden_cameras.read_text(encoding="utf-8"))
    source_view = next((v for v in source["views"] if v["name"] == args.view), None)
    if source_view is None:
        raise ValueError(f"Garden view not found: {args.view}")
    first = source_view["world_to_camera"]
    result = {
        **{k: source[k] for k in ("width", "height", "fx", "fy", "cx", "cy")},
        "views": [{"name": f"frame_{i:03d}.png", "world_to_camera": first}
                  for i in range(args.frames)],
        "fps": args.fps,
        "duration_seconds": args.frames / args.fps,
        "orbit_motion": "none; locked camera",
        "camera_pose_policy": "identical world_to_camera matrix for every frame",
        "object_transform_policy": "one static object_to_scene transform for every frame",
        "source_view": args.view,
        "note": "Static camera eliminates parallax, apparent spin, and screen-space drift relative to the tabletop.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(result["views"]), "duration_seconds": result["duration_seconds"],
                      "identical_camera_poses": len({json.dumps(v["world_to_camera"]) for v in result["views"]}) == 1,
                      "orbit_motion": result["orbit_motion"]}, indent=2))


if __name__ == "__main__":
    main()
