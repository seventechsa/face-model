#!/usr/bin/env python3
"""Generate a sequential aging progression from input face image(s).

Examples:
    # all age groups for one image, using the best weights
    python scripts/generate.py --config configs/utkface_456.yaml \
        --weights outputs/utkface_456/weights/best_G.pth \
        --input my_face.jpg --output results/

    # only older groups (e.g. 40-49 and 50+) for a whole folder, with face alignment
    python scripts/generate.py --config configs/utkface_456.yaml \
        --weights outputs/utkface_456/weights/best_G.pth \
        --input ./faces/ --output results/ --targets 3,4 --align
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.data import group_labels, list_images  # noqa: E402
from src.identity import load_mtcnn  # noqa: E402
from src.models import Generator  # noqa: E402
from src.utils import denorm, get_device, label2onehot  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    p = argparse.ArgumentParser(description="Sequential face-aging inference")
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True, help="path to a G state_dict (e.g. best_G.pth)")
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--output", default="results", help="output folder")
    p.add_argument("--targets", default="all",
                   help="'all' or comma list of group indices, e.g. '3,4'")
    p.add_argument("--align", action="store_true", help="MTCNN face crop before aging")
    p.add_argument("--no-strip", action="store_true", help="skip the labeled progression strip")
    return p.parse_args()


def gather_inputs(path):
    if os.path.isdir(path):
        return list_images(path)
    if os.path.splitext(path)[1].lower() in IMG_EXTS:
        return [path]
    raise SystemExit(f"[error] not an image or folder: {path}")


def load_face(path, image_size, mtcnn):
    img = Image.open(path).convert("RGB")
    if mtcnn is not None:
        boxes, _ = mtcnn.detect(img)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]
            m = 0.2 * max(x2 - x1, y2 - y1)
            crop = (max(0, x1 - m), max(0, y1 - m),
                    min(img.width, x2 + m), min(img.height, y2 + m))
            img = img.crop(crop)
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 127.5 - 1.0  # -> [-1,1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def tensor_to_pil(t):
    arr = (denorm(t).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def _font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def make_strip(pils, headers):
    pad, header_h = 8, 30
    w, h = pils[0].size
    n = len(pils)
    canvas = Image.new("RGB", (n * w + (n + 1) * pad, h + header_h + 2 * pad), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font(max(14, w // 18))
    for i, (im, txt) in enumerate(zip(pils, headers)):
        x = pad + i * (w + pad)
        canvas.paste(im, (x, header_h + pad))
        tb = draw.textbbox((0, 0), txt, font=font)
        draw.text((x + (w - (tb[2] - tb[0])) // 2, pad), txt, fill=(0, 0, 0), font=font)
    return canvas


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()
    num_groups = len(cfg.age_bins) - 1
    labels = group_labels(cfg.age_bins)

    if args.targets.strip().lower() == "all":
        targets = list(range(num_groups))
    else:
        targets = [int(t) for t in args.targets.split(",") if t.strip() != ""]
        for t in targets:
            if not 0 <= t < num_groups:
                raise SystemExit(f"[error] target group {t} out of range 0..{num_groups - 1}")

    G = Generator(num_groups, cfg.g_conv_dim, cfg.g_res_blocks, cfg.g_downsample).to(device)
    state = torch.load(args.weights, map_location=device)
    G.load_state_dict(state)
    G.eval()
    print(f"[load] G weights: {args.weights}")

    mtcnn = load_mtcnn(device, cfg.image_size) if args.align else None
    os.makedirs(args.output, exist_ok=True)

    inputs = gather_inputs(args.input)
    print(f"[gen] {len(inputs)} image(s) -> groups {[labels[t] for t in targets]}")

    for path in inputs:
        stem = os.path.splitext(os.path.basename(path))[0]
        x = load_face(path, cfg.image_size, mtcnn).to(device)
        pils, headers = [tensor_to_pil(x[0])], ["input"]
        with torch.no_grad():
            for t in targets:
                c = label2onehot(torch.tensor([t], device=device), num_groups)
                out = G(x, c)[0]
                pil = tensor_to_pil(out)
                pil.save(os.path.join(args.output, f"{stem}_to_{labels[t]}.png"))
                pils.append(pil)
                headers.append(labels[t])
        if not args.no_strip:
            make_strip(pils, headers).save(
                os.path.join(args.output, f"{stem}_aging_strip.png"))
        print(f"  {stem}: saved {len(targets)} aged image(s) + strip")

    print(f"[done] results in {args.output}")


if __name__ == "__main__":
    main()
