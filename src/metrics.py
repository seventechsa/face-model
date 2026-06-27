"""Image-quality / fidelity metrics: FID, SSIM, PSNR, LPIPS.

CSIM (identity cosine similarity) lives in identity.IdentityNet.csim.

All metric backends are optional and guarded: a missing dependency disables only
that metric (logged once) instead of crashing the run. Inputs to every method are
expected in [0, 1] float, shape (B, 3, H, W).
"""
import torch


def _try(fn, name):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"[metrics] {name} unavailable ({e}); it will be skipped.")
        return None


class MetricBank:
    def __init__(self, device, lpips_net="alex", fid_features=2048):
        self.device = device
        # FID uses float64 internally (sqrtm), which MPS does not support -> run it on CPU.
        self.fid_device = torch.device("cpu") if device.type == "mps" else device
        self.fid = _try(
            lambda: _make_fid(fid_features).to(self.fid_device), "FID")
        self.ssim = _try(
            lambda: _make_ssim().to(device), "SSIM")
        self.psnr = _try(
            lambda: _make_psnr().to(device), "PSNR")
        self.lpips = _try(
            lambda: _make_lpips(lpips_net).to(device), "LPIPS")

    # ---- FID (distributional: real set vs fake set) ----
    def fid_reset(self):
        if self.fid is not None:
            self.fid.reset()

    def fid_update(self, imgs01, real):
        if self.fid is not None:
            self.fid.update(imgs01.clamp(0, 1).to(self.fid_device), real=real)

    def fid_compute(self):
        if self.fid is None:
            return float("nan")
        try:
            return float(self.fid.compute())
        except Exception as e:  # noqa: BLE001
            print(f"[metrics] FID compute failed ({e}).")
            return float("nan")

    # ---- paired metrics (output vs reference), accumulate then compute ----
    def paired_reset(self):
        for m in (self.ssim, self.psnr, self.lpips):
            if m is not None:
                m.reset()

    def paired_update(self, pred01, ref01):
        pred01 = pred01.clamp(0, 1)
        ref01 = ref01.clamp(0, 1)
        if self.ssim is not None:
            self.ssim.update(pred01, ref01)
        if self.psnr is not None:
            self.psnr.update(pred01, ref01)
        if self.lpips is not None:
            self.lpips.update(pred01, ref01)

    def paired_compute(self):
        def safe(m):
            if m is None:
                return float("nan")
            try:
                return float(m.compute())
            except Exception:  # noqa: BLE001
                return float("nan")
        return {"ssim": safe(self.ssim), "psnr": safe(self.psnr), "lpips": safe(self.lpips)}


# --- backend factories (import lazily so a missing extra disables just that metric) ---
def _make_fid(features):
    from torchmetrics.image.fid import FrechetInceptionDistance
    return FrechetInceptionDistance(feature=features, normalize=True)


def _make_ssim():
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    return StructuralSimilarityIndexMeasure(data_range=1.0)


def _make_psnr():
    from torchmetrics.image import PeakSignalNoiseRatio
    return PeakSignalNoiseRatio(data_range=1.0)


def _make_lpips(net):
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except Exception:  # older/newer layout
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
    return LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True)
