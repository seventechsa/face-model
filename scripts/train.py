#!/usr/bin/env python3
"""Entry point: train the face-aging GAN.

Examples:
    python scripts/train.py --config configs/utkface_456.yaml
    python scripts/train.py --config configs/utkface_456.yaml --data-root /path/UTKFace
    python scripts/train.py --config configs/utkface_456.yaml --resume
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.train import main  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train StarGAN-style face-aging model")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--data-root", default=None, help="override data_root")
    p.add_argument("--output-root", default=None, help="override output_root")
    p.add_argument("--run-name", default=None, help="override run_name")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="resume from latest_ckpt.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {
        "data_root": args.data_root,
        "output_root": args.output_root,
        "run_name": args.run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
    }
    cfg = load_config(args.config, overrides)
    cfg.resume = args.resume
    print(cfg)
    main(cfg)
