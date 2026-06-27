"""Face-identity embedding (FaceNet / InceptionResnetV1, VGGFace2 weights).

Used for:
  * identity-preservation loss during training (1 - cosine similarity)
  * the CSIM evaluation metric

Degrades gracefully: if `facenet-pytorch` is missing, identity loss returns 0 and
CSIM returns NaN, with a one-time warning, so training still runs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.available = False
        self.net = None
        try:
            from facenet_pytorch import InceptionResnetV1

            self.net = InceptionResnetV1(pretrained="vggface2").eval().to(device)
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.available = True
            print("[identity] FaceNet (vggface2) loaded for identity loss + CSIM.")
        except Exception as e:  # noqa: BLE001
            print(f"[identity] facenet-pytorch unavailable ({e}); identity loss & CSIM disabled.")

    def embed(self, x):
        """x in [-1,1], shape (B,3,H,W) -> L2-normalized 512-d embeddings."""
        x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        return F.normalize(self.net(x), dim=1)

    def id_loss(self, real, fake):
        """1 - cosine(embed(real), embed(fake)); grad flows through `fake` only."""
        if not self.available:
            return torch.zeros((), device=fake.device)
        real_e = self.embed(real).detach()
        fake_e = self.embed(fake)
        return (1.0 - (real_e * fake_e).sum(dim=1)).mean()

    @torch.no_grad()
    def csim(self, real, fake):
        """Mean cosine similarity (CSIM). Returns float (NaN if unavailable)."""
        if not self.available:
            return float("nan")
        real_e = self.embed(real)
        fake_e = self.embed(fake)
        return (real_e * fake_e).sum(dim=1).mean().item()


def load_mtcnn(device, image_size):
    """Optional MTCNN detector for face cropping in inference. None if unavailable."""
    try:
        from facenet_pytorch import MTCNN

        return MTCNN(image_size=image_size, margin=int(0.2 * image_size),
                     post_process=False, select_largest=True, device=device)
    except Exception as e:  # noqa: BLE001
        print(f"[align] MTCNN unavailable ({e}); proceeding without face alignment.")
        return None
