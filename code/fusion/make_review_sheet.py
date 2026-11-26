"""Build a compact labeled contact sheet for a rendered fusion orbit."""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()
    paths = sorted(args.frames.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"no rendered frames in {args.frames}")
    frame_ids = [int(path.stem.rsplit("_",1)[1]) for path in paths]
    if args.start_frame is not None or args.end_frame is not None:
        start = frame_ids[0] if args.start_frame is None else args.start_frame
        end = frame_ids[-1] if args.end_frame is None else args.end_frame
        selected = [(i,n) for i,n in enumerate(frame_ids) if start <= n <= end]
        if not selected:
            raise ValueError("review frame interval is outside the rendered sequence")
    else:
        selected = list(enumerate(frame_ids))
    chosen = sorted(set(round(i * (len(selected)-1) / (args.count-1))
                        for i in range(args.count)))
    font = ImageFont.load_default()
    thumb_w, thumb_h, label_h = 480, 270, 24
    cols = 3
    rows = (len(chosen)+cols-1)//cols
    sheet = Image.new("RGB", (cols*thumb_w, rows*(thumb_h+label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, selection_index in enumerate(chosen):
        path_index, idx = selected[selection_index]
        path = paths[path_index]
        with Image.open(path) as source:
            thumb = source.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x, y = (slot%cols)*thumb_w, (slot//cols)*(thumb_h+label_h)
        draw.text((x+8, y+5), f"frame {idx:03d}  |  {idx/24:.2f} s", fill="black", font=font)
        sheet.paste(thumb, (x, y+label_h))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92, optimize=True)

if __name__ == "__main__":
    main()
