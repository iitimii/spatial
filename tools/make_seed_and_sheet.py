#!/usr/bin/env python3
"""Helper: prepare a 512x256 seed PPM and render rollout PPMs into a PNG contact sheet + GIF."""
import argparse
import glob
import os
import numpy as np
from PIL import Image, ImageDraw


def make_seed(out_ppm: str, src_image: str | None = None):
    W, H = 512, 256
    if src_image and os.path.exists(src_image):
        img = Image.open(src_image).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    else:
        # Rich first-person outdoor corridor/road scene so motion & perspective are unmistakable!
        img = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)
        # Sky gradient
        for y in range(H // 2):
            t = y / (H // 2)
            r = int(40 + 130 * t)
            g = int(90 + 115 * t)
            b = int(185 + 60 * t)
            draw.line([(0, y), (W, y)], fill=(r, g, b))
        # Distant mountains
        draw.polygon([(0, 128), (90, 72), (190, 118), (310, 62), (430, 108), (512, 82), (512, 128)],
                     fill=(72, 88, 112))
        # Ground grass
        for y in range(H // 2, H):
            t = (y - H // 2) / (H // 2)
            draw.line([(0, y), (W, y)], fill=(int(58 + 30 * t), int(112 + 40 * t), int(48 + 18 * t)))
        # Perspective road leading toward horizon center
        draw.polygon([(140, H), (236, 128), (276, 128), (372, H)], fill=(88, 86, 84))
        draw.polygon([(252, 128), (260, 128), (266, H), (246, H)], fill=(225, 200, 95))
        # Roadside pillars/trees for parallax cues
        for depth, (lx, rx, top, bot) in enumerate([
            (190, 322, 105, 145),
            (145, 367, 88, 175),
            (75, 437, 55, 225),
        ]):
            w_t = 8 + depth * 6
            draw.rectangle([lx - w_t, top, lx + w_t, bot], fill=(95, 62, 42))
            draw.ellipse([lx - w_t * 3, top - w_t * 3, lx + w_t * 3, top + w_t], fill=(38, 105, 46))
            draw.rectangle([rx - w_t, top, rx + w_t, bot], fill=(95, 62, 42))
            draw.ellipse([rx - w_t * 3, top - w_t * 3, rx + w_t * 3, top + w_t], fill=(38, 105, 46))
        # Sun
        draw.ellipse([375, 22, 415, 62], fill=(255, 240, 175))

    os.makedirs(os.path.dirname(os.path.abspath(out_ppm)), exist_ok=True)
    with open(out_ppm, "wb") as f:
        f.write(f"P6\n{W} {H}\n255\n".encode("ascii"))
        f.write(np.asarray(img, dtype=np.uint8).tobytes())
    print(f"Wrote seed PPM: {out_ppm} ({W}x{H})")


def make_sheet(frames_dir: str, out_png: str, out_gif: str | None = None):
    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.ppm")))
    assert paths, f"No frames in {frames_dir}"
    imgs = [Image.open(p).convert("RGB") for p in paths]
    # Pick up to 9 evenly spaced frames for a crisp 3x3 contact sheet
    idxs = np.linspace(0, len(imgs) - 1, min(9, len(imgs))).round().astype(int)
    picked = [imgs[i] for i in idxs]
    w, h = picked[0].size
    cols = 3
    rows = (len(picked) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (15, 15, 18))
    draw = ImageDraw.Draw(sheet)
    for k, (idx, im) in enumerate(zip(idxs, picked)):
        r, c = divmod(k, cols)
        sheet.paste(im, (c * w, r * h))
        draw.rectangle([c * w + 6, r * h + 6, c * w + 106, r * h + 26], fill=(0, 0, 0))
        draw.text((c * w + 12, r * h + 10), f"frame {idx:02d}", fill=(255, 255, 255))
    sheet.save(out_png)
    print(f"Saved contact sheet: {out_png} ({len(picked)} panels from {len(imgs)} frames)")
    if out_gif:
        imgs[0].save(out_gif, save_all=True, append_images=imgs[1:], duration=67, loop=0)
        print(f"Saved animated GIF: {out_gif}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("seed")
    p1.add_argument("--out", required=True)
    p1.add_argument("--src", default=None)
    p2 = sub.add_parser("sheet")
    p2.add_argument("--dir", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--gif", default=None)
    args = ap.parse_args()
    if args.cmd == "seed":
        make_seed(args.out, args.src)
    else:
        make_sheet(args.dir, args.out, args.gif)


if __name__ == "__main__":
    main()
