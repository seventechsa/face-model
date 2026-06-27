"""Small shared helpers: seeding, device, normalization, one-hot, grids, AMP."""
import os
import random

import numpy as np
import torch
import torchvision.utils as vutils


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(verbose=True):
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"[device] using: {dev}")
    return dev


def denorm(x):
    """[-1, 1] -> [0, 1] (clamped)."""
    return (x.clamp(-1, 1) + 1) / 2


def label2onehot(labels, dim):
    """labels: (B,) long  ->  (B, dim) float one-hot."""
    labels = labels.long()
    out = torch.zeros(labels.size(0), dim, device=labels.device)
    out[torch.arange(labels.size(0)), labels] = 1.0
    return out


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def amp_autocast(enabled):
    """Mixed-precision context (no-op when disabled / non-CUDA)."""
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled):
    """AMP gradient scaler, new API with fallback for older torch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def save_image_grid(tensor, path, nrow=8):
    """Save a tensor batch in [-1, 1] as an image grid."""
    grid = vutils.make_grid(denorm(tensor.detach().cpu().float()), nrow=nrow, padding=2)
    vutils.save_image(grid, path)


def count_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6  # in millions
