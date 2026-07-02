"""StarGAN-style generator and discriminator for multi-domain (age-group) translation.

Generator G(x, c): encoder -> residual blocks -> decoder, conditioned on a target
age-group one-hot `c` that is spatially broadcast and concatenated to the input.

Discriminator D(x): PatchGAN backbone with two heads:
    out_src  -> patch real/fake logits
    out_cls  -> age-group classification logits (global avg-pool + linear; robust to
                non-power-of-two input sizes like 456).
"""
import torch
import torch.nn as nn


def _inorm(dim):
    return nn.InstanceNorm2d(dim, affine=True, track_running_stats=False)


def weights_init(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False), _inorm(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False), _inorm(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, c_dim, conv_dim=64, res_blocks=6, downsample=2):
        super().__init__()
        layers = [
            nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3, bias=False),
            _inorm(conv_dim), nn.ReLU(inplace=True),
        ]
        curr = conv_dim
        for _ in range(downsample):
            layers += [
                nn.Conv2d(curr, curr * 2, 4, 2, 1, bias=False),
                _inorm(curr * 2), nn.ReLU(inplace=True),
            ]
            curr *= 2
        for _ in range(res_blocks):
            layers.append(ResidualBlock(curr))
        for _ in range(downsample):
            layers += [
                nn.ConvTranspose2d(curr, curr // 2, 4, 2, 1, bias=False),
                _inorm(curr // 2), nn.ReLU(inplace=True),
            ]
            curr //= 2
        layers += [nn.Conv2d(curr, 3, 7, 1, 3), nn.Tanh()]
        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        return self.main(torch.cat([x, c], dim=1))


class Discriminator(nn.Module):
    def __init__(self, image_size, c_dim, conv_dim=64, n_layers=6, max_channels=512):
        super().__init__()
        layers = [nn.Conv2d(3, conv_dim, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        curr = conv_dim
        for _ in range(1, n_layers):
            nxt = min(curr * 2, max_channels)
            layers += [nn.Conv2d(curr, nxt, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            curr = nxt
        self.backbone = nn.Sequential(*layers)
        self.conv_src = nn.Conv2d(curr, 1, 3, 1, 1, bias=False)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(curr, c_dim),
        )
        self.apply(weights_init)

    def forward(self, x):
        h = self.backbone(x)
        return self.conv_src(h), self.cls_head(h)
