"""Training loop for StarGAN-style face-aging.

Per epoch it writes:
  outputs/<run>/weights/epoch_XXX/{G.pth,D.pth}   (weights for every epoch)
  outputs/<run>/weights/latest_ckpt.pth           (full state for --resume)
  outputs/<run>/weights/best_G.pth                 (lowest val FID seen)
  outputs/<run>/samples/epoch_XXX.png             (input -> each age group)
  outputs/<run>/logs/loss_log.csv                 (per-epoch losses + val FID)
"""
import csv
import json
import os
import shutil
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import (build_dataloaders, get_fixed_batch, group_distribution,
                   group_labels)
from .identity import IdentityNet
from .losses import (classification_loss, d_hinge, d_wgan, g_hinge, g_wgan,
                     gradient_penalty, r1_penalty)
from .metrics import MetricBank
from .models import Discriminator, Generator
from .utils import (amp_autocast, count_params, denorm, get_device, label2onehot,
                    make_dirs, make_grad_scaler, save_image_grid, set_seed,
                    AverageMeter)


@torch.no_grad()
def save_aging_grid(G, x_fixed, num_groups, path, device):
    """Each row = one person: [input, group_0, group_1, ... group_{K-1}]."""
    was_training = G.training
    G.eval()
    cols = [x_fixed]
    for g in range(num_groups):
        c = label2onehot(torch.full((x_fixed.size(0),), g, dtype=torch.long, device=device), num_groups)
        cols.append(G(x_fixed, c))
    k = num_groups + 1
    tiles = []
    for i in range(x_fixed.size(0)):
        for j in range(k):
            tiles.append(cols[j][i])
    save_image_grid(torch.stack(tiles, 0), path, nrow=k)
    if was_training:
        G.train()


@torch.no_grad()
def quick_val_fid(G, val_loader, num_groups, device, bank, max_images):
    """Proxy FID: real val faces vs. their aged translations (random target group)."""
    if bank.fid is None:
        return float("nan")
    was_training = G.training
    G.eval()
    bank.fid_reset()
    seen = 0
    for x, _ in val_loader:
        x = x.to(device)
        trg = torch.randint(0, num_groups, (x.size(0),), device=device)
        x_fake = G(x, label2onehot(trg, num_groups))
        bank.fid_update(denorm(x), real=True)
        bank.fid_update(denorm(x_fake), real=False)
        seen += x.size(0)
        if seen >= max_images:
            break
    if was_training:
        G.train()
    return bank.fid_compute()


def _save_epoch_weights(G, D, weights_dir, epoch, keep_last_n):
    ep_dir = os.path.join(weights_dir, f"epoch_{epoch:03d}")
    make_dirs(ep_dir)
    torch.save(G.state_dict(), os.path.join(ep_dir, "G.pth"))
    torch.save(D.state_dict(), os.path.join(ep_dir, "D.pth"))
    if keep_last_n and keep_last_n > 0:
        kept = sorted(
            d for d in os.listdir(weights_dir)
            if d.startswith("epoch_") and os.path.isdir(os.path.join(weights_dir, d))
        )
        for old in kept[:-keep_last_n]:
            shutil.rmtree(os.path.join(weights_dir, old), ignore_errors=True)


def main(cfg):
    set_seed(cfg.seed)
    device = get_device()

    out_dir = os.path.join(cfg.output_root, cfg.run_name)
    weights_dir = os.path.join(out_dir, "weights")
    samples_dir = os.path.join(out_dir, "samples")
    logs_dir = os.path.join(out_dir, "logs")
    make_dirs(out_dir, weights_dir, samples_dir, logs_dir)

    # data
    train_loader, val_loader, test_loader, num_groups, splits = build_dataloaders(cfg)
    labels = group_labels(cfg.age_bins)
    dist = group_distribution(splits[0], num_groups)
    print(f"[data] groups={num_groups} {labels}")
    print(f"[data] train={len(splits[0])} val={len(splits[1])} test={len(splits[2])}")
    print(f"[data] train group distribution: {dict(zip(labels, dist))}")

    # save run config + label map
    with open(os.path.join(logs_dir, "config.json"), "w") as f:
        json.dump({**cfg.to_dict(), "group_labels": labels}, f, indent=2)

    # models
    G = Generator(num_groups, cfg.g_conv_dim, cfg.g_res_blocks, cfg.g_downsample).to(device)
    D = Discriminator(cfg.image_size, num_groups, cfg.d_conv_dim, cfg.d_layers).to(device)
    print(f"[model] G={count_params(G):.2f}M params  D={count_params(D):.2f}M params")

    g_opt = torch.optim.Adam(G.parameters(), cfg.g_lr, (cfg.beta1, cfg.beta2))
    d_opt = torch.optim.Adam(D.parameters(), cfg.d_lr, (cfg.beta1, cfg.beta2))

    idnet = IdentityNet(device) if cfg.use_identity else type("X", (), {"available": False})()

    # AMP: must be off when using double-backward penalties (gp / R1)
    penalty_on = (cfg.adv_loss == "wgan-gp") or (cfg.r1_gamma > 0)
    use_amp = bool(cfg.use_amp) and device.type == "cuda" and not penalty_on
    if cfg.use_amp and penalty_on:
        print("[amp] disabled because gradient penalty / R1 is enabled.")
    scaler_g = make_grad_scaler(use_amp)
    scaler_d = make_grad_scaler(use_amp)

    # FID bank (for best tracking) — paired metrics computed in evaluate.py
    bank = MetricBank(device)

    x_fixed, _ = get_fixed_batch(val_loader, n=min(8, max(2, cfg.batch_size)), device=device)

    # resume
    start_epoch, best_fid, global_step = 0, float("inf"), 0
    ckpt_path = os.path.join(weights_dir, "latest_ckpt.pth")
    if getattr(cfg, "resume", False) and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        g_opt.load_state_dict(ck["g_opt"]); d_opt.load_state_dict(ck["d_opt"])
        scaler_g.load_state_dict(ck["scaler_g"]); scaler_d.load_state_dict(ck["scaler_d"])
        start_epoch = ck["epoch"] + 1
        best_fid = ck.get("best_fid", float("inf"))
        global_step = ck.get("global_step", 0)
        print(f"[resume] from epoch {start_epoch} (best_fid={best_fid:.3f})")

    csv_path = os.path.join(logs_dir, "loss_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "d_loss", "g_loss", "g_adv", "g_cls", "g_rec", "g_id", "val_fid", "sec"])

    for epoch in range(start_epoch, cfg.epochs):
        G.train(); D.train()
        t0 = time.time()
        m = {k: AverageMeter() for k in ["d", "g", "adv", "cls", "rec", "id"]}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs - 1}")

        for x_real, label_org in pbar:
            x_real = x_real.to(device, non_blocking=True)
            label_org = label_org.to(device).long()
            bs = x_real.size(0)
            label_trg = torch.randint(0, num_groups, (bs,), device=device)
            c_org = label2onehot(label_org, num_groups)
            c_trg = label2onehot(label_trg, num_groups)

            # ---------------- Discriminator ----------------
            with amp_autocast(use_amp):
                x_fake = G(x_real, c_trg)
                real_src, real_cls = D(x_real)
                fake_src, _ = D(x_fake.detach())
                d_adv = d_hinge(real_src, fake_src) if cfg.adv_loss == "hinge" \
                    else d_wgan(real_src, fake_src)
                d_cls = classification_loss(real_cls, label_org)
                d_loss = d_adv + cfg.lambda_cls * d_cls
            if cfg.adv_loss == "wgan-gp":
                d_loss = d_loss + cfg.lambda_gp * gradient_penalty(D, x_real, x_fake.detach(), device)
            if cfg.r1_gamma > 0 and (global_step % cfg.r1_every == 0):
                d_loss = d_loss + 0.5 * cfg.r1_gamma * r1_penalty(D, x_real)
            d_opt.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(d_opt)
            scaler_d.update()
            m["d"].update(d_loss.item(), bs)

            # ---------------- Generator ----------------
            if global_step % cfg.n_critic == 0:
                with amp_autocast(use_amp):
                    x_fake = G(x_real, c_trg)
                    fake_src, fake_cls = D(x_fake)
                    g_adv = g_hinge(fake_src) if cfg.adv_loss == "hinge" else g_wgan(fake_src)
                    g_cls = classification_loss(fake_cls, label_trg)
                    x_rec = G(x_fake, c_org)
                    g_rec = F.l1_loss(x_rec, x_real)
                    g_loss = g_adv + cfg.lambda_cls * g_cls + cfg.lambda_rec * g_rec
                g_id_val = 0.0
                if getattr(idnet, "available", False) and cfg.lambda_id > 0:
                    with amp_autocast(False):
                        g_id = idnet.id_loss(x_real, x_fake.float())
                    g_loss = g_loss + cfg.lambda_id * g_id
                    g_id_val = float(g_id.item())
                g_opt.zero_grad(set_to_none=True)
                scaler_g.scale(g_loss).backward()
                scaler_g.step(g_opt)
                scaler_g.update()
                m["g"].update(g_loss.item(), bs); m["adv"].update(g_adv.item(), bs)
                m["cls"].update(g_cls.item(), bs); m["rec"].update(g_rec.item(), bs)
                m["id"].update(g_id_val, bs)

            global_step += 1
            if global_step % cfg.log_every_iters == 0:
                pbar.set_postfix(d=f"{m['d'].avg:.2f}", g=f"{m['g'].avg:.2f}",
                                 rec=f"{m['rec'].avg:.2f}", idl=f"{m['id'].avg:.3f}")

        # ---------------- end of epoch ----------------
        val_fid = float("nan")
        if cfg.eval_every_epochs > 0 and ((epoch + 1) % cfg.eval_every_epochs == 0 or epoch == 0):
            val_fid = quick_val_fid(G, val_loader, num_groups, device, bank, cfg.max_eval_images)
            if val_fid == val_fid and val_fid < best_fid:  # not NaN and improved
                best_fid = val_fid
                torch.save(G.state_dict(), os.path.join(weights_dir, "best_G.pth"))
                torch.save(D.state_dict(), os.path.join(weights_dir, "best_D.pth"))
                print(f"[best] new best val FID={best_fid:.3f} -> best_G.pth")

        if x_fixed is not None and (epoch % cfg.sample_every_epochs == 0):
            save_aging_grid(G, x_fixed, num_groups,
                            os.path.join(samples_dir, f"epoch_{epoch:03d}.png"), device)

        if epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs - 1:
            _save_epoch_weights(G, D, weights_dir, epoch, cfg.keep_last_n_epochs)
        torch.save({
            "epoch": epoch, "best_fid": best_fid, "global_step": global_step,
            "G": G.state_dict(), "D": D.state_dict(),
            "g_opt": g_opt.state_dict(), "d_opt": d_opt.state_dict(),
            "scaler_g": scaler_g.state_dict(), "scaler_d": scaler_d.state_dict(),
        }, ckpt_path)

        sec = time.time() - t0
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{m['d'].avg:.4f}", f"{m['g'].avg:.4f}", f"{m['adv'].avg:.4f}",
                f"{m['cls'].avg:.4f}", f"{m['rec'].avg:.4f}", f"{m['id'].avg:.4f}",
                f"{val_fid:.4f}", f"{sec:.1f}"])
        print(f"[epoch {epoch}] d={m['d'].avg:.3f} g={m['g'].avg:.3f} "
              f"rec={m['rec'].avg:.3f} id={m['id'].avg:.3f} "
              f"val_fid={val_fid:.3f} ({sec:.0f}s)")

    print(f"[done] best val FID={best_fid:.3f}. Weights in {weights_dir}")
