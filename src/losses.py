"""Adversarial / classification losses and gradient regularizers."""
import torch
import torch.nn.functional as F


def d_hinge(real_src, fake_src):
    """Discriminator hinge loss."""
    return F.relu(1.0 - real_src).mean() + F.relu(1.0 + fake_src).mean()


def g_hinge(fake_src):
    """Generator hinge loss."""
    return -fake_src.mean()


def d_wgan(real_src, fake_src):
    return -real_src.mean() + fake_src.mean()


def g_wgan(fake_src):
    return -fake_src.mean()


def classification_loss(logits, target):
    """Softmax cross-entropy over mutually-exclusive age groups."""
    return F.cross_entropy(logits, target)


def gradient_penalty(D, real, fake, device):
    """WGAN-GP penalty on interpolated samples (needs create_graph -> AMP off)."""
    b = real.size(0)
    alpha = torch.rand(b, 1, 1, 1, device=device)
    x_hat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    out_src, _ = D(x_hat)
    grad = torch.autograd.grad(
        outputs=out_src.sum(), inputs=x_hat, create_graph=True, retain_graph=True
    )[0]
    grad = grad.view(b, -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def r1_penalty(D, real):
    """R1 gradient penalty on real samples (needs create_graph -> AMP off)."""
    real = real.detach().requires_grad_(True)
    out_src, _ = D(real)
    grad = torch.autograd.grad(outputs=out_src.sum(), inputs=real, create_graph=True)[0]
    return grad.view(grad.size(0), -1).pow(2).sum(1).mean()
