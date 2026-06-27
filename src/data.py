"""UTKFace dataset, age-group bucketing, and train/val/test dataloaders.

UTKFace filename format:  [age]_[gender]_[race]_[date&time].jpg
We parse `age` from the filename and bucket it into a group via `age_bins`.
"""
import glob
import os
import random

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def list_images(root):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(set(files))


def parse_age_utkface(path):
    """Return age int from a UTKFace filename, or None if malformed."""
    name = os.path.basename(path)
    try:
        age = int(name.split("_")[0])
    except (ValueError, IndexError):
        return None
    return age if 0 <= age <= 120 else None


def age_to_group(age, bins):
    """bins=[0,20,30,40,50,200] -> group index in 0..len(bins)-2."""
    for i in range(len(bins) - 1):
        if bins[i] <= age < bins[i + 1]:
            return i
    return len(bins) - 2


def group_labels(bins):
    """Human-readable age-range labels per group, e.g. '0-19', '50+'."""
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        labels.append(f"{lo}+" if hi >= 200 else f"{lo}-{hi - 1}")
    return labels


def build_samples(root, age_bins):
    """Scan `root`, return list of (path, group_idx) for valid images."""
    samples = []
    for f in list_images(root):
        age = parse_age_utkface(f)
        if age is None:
            continue
        samples.append((f, age_to_group(age, age_bins)))
    return samples


def make_transform(image_size, train):
    tf = [T.Resize((image_size, image_size))]
    if train:
        tf.append(T.RandomHorizontalFlip())
    tf += [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]  # -> [-1, 1]
    return T.Compose(tf)


class FaceDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, group = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), group


def build_dataloaders(cfg):
    """Returns (train_loader, val_loader, test_loader, num_groups, splits)."""
    samples = build_samples(cfg.data_root, cfg.age_bins)
    num_groups = len(cfg.age_bins) - 1
    if len(samples) == 0:
        raise RuntimeError(
            f"No valid UTKFace images under '{cfg.data_root}'. "
            f"Expected files named like '25_0_0_20170116.jpg'."
        )

    rng = random.Random(cfg.seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(n * cfg.test_fraction)
    n_val = int(n * cfg.val_fraction)
    test_idx = set(idx[:n_test])
    val_idx = set(idx[n_test : n_test + n_val])
    train_idx = idx[n_test + n_val :]

    train_s = [samples[i] for i in train_idx]
    val_s = [samples[i] for i in sorted(val_idx)]
    test_s = [samples[i] for i in sorted(test_idx)]

    tr = make_transform(cfg.image_size, train=True)
    ev = make_transform(cfg.image_size, train=False)

    common = dict(num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(
        FaceDataset(train_s, tr), batch_size=cfg.batch_size, shuffle=True,
        drop_last=True, **common,
    )
    val_loader = DataLoader(
        FaceDataset(val_s, ev), batch_size=cfg.batch_size, shuffle=False, **common,
    )
    test_loader = DataLoader(
        FaceDataset(test_s, ev), batch_size=cfg.batch_size, shuffle=False, **common,
    )
    return train_loader, val_loader, test_loader, num_groups, (train_s, val_s, test_s)


def group_distribution(samples, num_groups):
    counts = [0] * num_groups
    for _, g in samples:
        counts[g] += 1
    return counts


def get_fixed_batch(loader, n, device):
    """Grab a fixed batch of up to n images for consistent per-epoch sample grids."""
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        if sum(t.size(0) for t in xs) >= n:
            break
    if not xs:
        return None, None
    x = torch.cat(xs, 0)[:n].to(device)
    y = torch.cat(ys, 0)[:n].to(device)
    return x, y
