"""Config loader: merges YAML over defaults, with optional CLI overrides."""
import yaml


# All knobs live here with sane defaults; YAML files only need to override what changes.
DEFAULTS = {
    # --- run / data ---
    "run_name": "utkface_456",
    "seed": 42,
    "data_root": "./data/UTKFace",        # folder with UTKFace jpgs (recursive ok)
    "output_root": "./outputs",           # weights/ samples/ logs/ eval/ go under output_root/run_name
    "image_size": 456,
    # age_bins -> groups [bins[i], bins[i+1]); 6 edges => 5 groups
    "age_bins": [0, 20, 30, 40, 50, 200],
    "num_workers": 4,
    "val_fraction": 0.05,
    "test_fraction": 0.05,

    # --- model ---
    "g_conv_dim": 64,
    "g_res_blocks": 6,
    "g_downsample": 2,
    "d_conv_dim": 64,
    "d_layers": 6,

    # --- training ---
    "epochs": 100,
    "batch_size": 4,
    "g_lr": 1e-4,
    "d_lr": 1e-4,
    "beta1": 0.5,
    "beta2": 0.999,
    "n_critic": 1,                # D updates per G update
    "adv_loss": "hinge",         # "hinge" | "wgan-gp"
    "lambda_cls": 1.0,           # age-group classification
    "lambda_rec": 10.0,          # cycle reconstruction
    "lambda_id": 1.0,            # identity preservation (FaceNet)
    "lambda_gp": 10.0,           # used only if adv_loss == "wgan-gp"
    "r1_gamma": 0.0,             # R1 reg on D (0 = off). NOTE: enabling gp/r1 forces AMP off.
    "r1_every": 16,
    "label_smoothing": 0.1,      # one-sided label smoothing for D (0 = off)
    "use_amp": True,             # mixed precision (CUDA only)

    # --- identity / inference ---
    "use_identity": True,        # FaceNet identity loss + CSIM metric
    "align_faces": False,        # MTCNN face crop in generate.py

    # --- logging / checkpoints / eval ---
    "sample_every_epochs": 1,
    "save_every_epochs": 1,
    "eval_every_epochs": 5,      # quick val FID to pick "best" weights
    "log_every_iters": 50,
    "max_eval_images": 500,      # cap eval set for speed
    "keep_last_n_epochs": -1,    # -1 keeps every epoch's weights; N keeps only last N
}


class Config:
    """Attribute-style access to the merged config dict."""

    def __init__(self, d):
        self.__dict__.update(d)

    def to_dict(self):
        return dict(self.__dict__)

    def __repr__(self):
        items = "\n".join(f"  {k}: {v}" for k, v in sorted(self.__dict__.items()))
        return f"Config(\n{items}\n)"


def load_config(path, overrides=None):
    """Load YAML at `path`, layer it over DEFAULTS, then apply non-None overrides."""
    with open(path, "r") as f:
        user = yaml.safe_load(f) or {}
    merged = dict(DEFAULTS)
    merged.update(user)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return Config(merged)
