"""Face aging progression with StarGAN-style multi-domain GAN.

Modules:
    config    - YAML config loader with defaults
    utils     - seeding, device, denorm, one-hot, checkpoint/grid helpers
    data      - UTKFace dataset, age grouping, dataloaders
    models    - Generator (encoder/resblocks/decoder) + Discriminator (src + cls heads)
    identity  - FaceNet embedding for identity loss + CSIM (optional, degrades gracefully)
    losses    - adversarial (hinge / wgan-gp), classification, gradient penalty, R1
    metrics   - FID / SSIM / PSNR / LPIPS / CSIM
    train     - training loop with per-epoch weights + sample grids + best tracking
"""

__version__ = "1.0.0"
