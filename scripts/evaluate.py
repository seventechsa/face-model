#!/usr/bin/env python3
"""Evaluate a trained generator with FID, SSIM, PSNR, LPIPS, CSIM.

Protocol:
  * SSIM / PSNR / LPIPS / CSIM  -> paired: aged output vs. the input face,
    averaged over every (test image x every target age group). Measure how much
    structure / identity is preserved under aging.
  * FID -> distributional, per age group: {real test faces in group g} vs.
    {all test faces translated to group g}. We report per-group FID and the mean.

Example:
    python scripts/evaluate.py --config configs/utkface_456.yaml \
        --weights outputs/utkface_456/weights/best_G.pth
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.data import build_dataloaders, group_labels  # noqa: E402
from src.identity import IdentityNet  # noqa: E402
from src.metrics import MetricBank  # noqa: E402
from src.models import Generator  # noqa: E402
from src.utils import denorm, get_device, label2onehot  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate face-aging GAN")
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True, help="G state_dict, e.g. best_G.pth")
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--max-images", type=int, default=None, help="cap per metric pass")
    p.add_argument("--output", default=None, help="dir for metrics.json (default outputs/<run>/eval)")
    return p.parse_args()


def onehot_const(value, n, num_groups, device):
    return label2onehot(torch.full((n,), value, dtype=torch.long, device=device), num_groups)


@torch.no_grad()
def paired_metrics(G, loader, num_groups, device, bank, idnet, max_images):
    """SSIM/PSNR/LPIPS/CSIM: aged output vs input, averaged over all target groups."""
    bank.paired_reset()
    csim_sum, csim_n = 0.0, 0
    seen = 0
    for x, _ in tqdm(loader, desc="paired (ssim/psnr/lpips/csim)"):
        x = x.to(device)
        for g in range(num_groups):
            out = G(x, onehot_const(g, x.size(0), num_groups, device))
            bank.paired_update(denorm(out), denorm(x))
            if getattr(idnet, "available", False):
                cs = idnet.csim(x, out)
                if cs == cs:  # not NaN
                    csim_sum += cs * x.size(0)
                    csim_n += x.size(0)
        seen += x.size(0)
        if max_images and seen >= max_images:
            break
    res = bank.paired_compute()
    res["csim"] = (csim_sum / csim_n) if csim_n else float("nan")
    return res


@torch.no_grad()
def fid_per_group(G, loader, num_groups, device, bank, max_images):
    """Per group g: FID(real faces of group g, all faces translated to g)."""
    if bank.fid is None:
        return {"per_group": [float("nan")] * num_groups, "mean": float("nan")}
    per_group = []
    for g in range(num_groups):
        bank.fid_reset()
        seen, n_real = 0, 0
        for x, lab in tqdm(loader, desc=f"FID group {g}"):
            x = x.to(device)
            mask = lab == g
            if mask.any():
                bank.fid_update(denorm(x[mask.to(x.device)]), real=True)
                n_real += int(mask.sum())
            bank.fid_update(denorm(G(x, onehot_const(g, x.size(0), num_groups, device))), real=False)
            seen += x.size(0)
            if max_images and seen >= max_images:
                break
        fid_g = bank.fid_compute() if n_real >= 2 else float("nan")
        per_group.append(fid_g)
    valid = [v for v in per_group if v == v]
    return {"per_group": per_group, "mean": (sum(valid) / len(valid)) if valid else float("nan")}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()
    num_groups = len(cfg.age_bins) - 1
    labels = group_labels(cfg.age_bins)

    train_loader, val_loader, test_loader, _, _ = build_dataloaders(cfg)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    G = Generator(num_groups, cfg.g_conv_dim, cfg.g_res_blocks, cfg.g_downsample).to(device)
    G.load_state_dict(torch.load(args.weights, map_location=device))
    G.eval()
    print(f"[load] {args.weights}  split={args.split}")

    bank = MetricBank(device)
    idnet = IdentityNet(device) if cfg.use_identity else type("X", (), {"available": False})()

    paired = paired_metrics(G, loader, num_groups, device, bank, idnet, args.max_images)
    fid = fid_per_group(G, loader, num_groups, device, bank, args.max_images)

    print("\n================ RESULTS ================")
    print(f"  FID (mean over groups) : {fid['mean']:.3f}")
    for lab, v in zip(labels, fid["per_group"]):
        print(f"      FID[{lab:>6}]        : {v:.3f}")
    print(f"  SSIM  (out vs input)   : {paired['ssim']:.4f}   (higher = more preserved)")
    print(f"  PSNR  (out vs input)   : {paired['psnr']:.3f} dB")
    print(f"  LPIPS (out vs input)   : {paired['lpips']:.4f}   (lower = closer)")
    print(f"  CSIM  (identity)       : {paired['csim']:.4f}   (higher = identity kept)")
    print("=========================================")

    out_dir = args.output or os.path.join(cfg.output_root, cfg.run_name, "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"metrics_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump({
            "weights": args.weights, "split": args.split, "image_size": cfg.image_size,
            "group_labels": labels,
            "fid_mean": fid["mean"], "fid_per_group": fid["per_group"],
            "ssim": paired["ssim"], "psnr": paired["psnr"],
            "lpips": paired["lpips"], "csim": paired["csim"],
        }, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
