"""
ConformerFlow — Config Validator
Loads a YAML config, merges with defaults, validates required keys.
Provides set_nested() used by train.py CLI overrides.
"""

import copy
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULTS = {
    "representation": {
        "structure": "se3_frames",
        "sequence":  "onehot",
    },
    "model": {
        "d_model":          256,
        "d_latent":         16,
        "n_encoder_layers": 4,
        "n_flow_layers":    8,
        "n_heads":          8,
        "n_points":         4,
        "d_ff":             512,
        "dropout":          0.1,
    },
    "generative_model": {
        "type":              "flow_matching",
        "sigma_min":         0.01,
        "ode_method":        "heun",
        "n_inference_steps": 20,
    },
    "ensemble_stats": {
        "covariance":  "full_covariance",
        "free_bits":   0.5,
        "n_pca_modes": 10,
    },
    "training": {
        "batch_size":        4,
        "max_epochs":        100,
        "learning_rate":     1e-4,
        "weight_decay":      1e-4,
        "warmup_steps":      1000,
        "lr_decay_steps":    100000,
        "checkpoint_dir":    "checkpoints",
        "n_gen_conformers":  10,
        "val_every":         500,
        "save_every":        2000,
        "max_grad_norm":     1.0,
        "loss_schedule":     "fixed",
        "multi_gpu":         False,
        "use_bf16":          False,
    },
    "loss": {
        "lambda_flow":      1.0,
        "lambda_ensemble":  1.0,
        "lambda_kl":        0.01,
        "lambda_diversity": 0.5,
        "lambda_geometry":  0.1,
        "min_spread":       0.5,
    },
    "logging": {
        "use_wandb":    False,
        "run_name":     "conformerflow",
        "log_every":    50,
        "project_name": "conformerflow",
    },
    "data": {
        "max_residues": 800,
        "pdb_data_dir": "pdb_data",
        "max_workers":  4,
    },
}

REQUIRED_KEYS = [
    "representation.structure",
    "model.d_model",
    "training.batch_size",
    "training.max_epochs",
    "training.learning_rate",
    "training.weight_decay",
    "generative_model.type",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (in place). Returns base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def set_nested(cfg: dict, path: str, value) -> None:
    """
    Set a config value using a dot-separated path.
    Creates intermediate dicts as needed.
    Example: set_nested(cfg, "training.batch_size", 64)
    """
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_nested(cfg: dict, path: str, default=None):
    """Get a config value using a dot-separated path."""
    d = cfg
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_validate(config_path: str,
                      auto_batch:  bool = True,
                      verbose:     bool = False) -> dict:
    """
    Load YAML config, apply defaults for missing keys, validate.

    Args:
        config_path: path to YAML file (missing file falls back to all defaults)
        auto_batch:  scale batch_size up based on available GPU VRAM
        verbose:     log config summary

    Returns:
        cfg: nested dict ready for Trainer and model_factory
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")

    cfg = copy.deepcopy(DEFAULTS)

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
        if verbose:
            logger.info(f"Config loaded from {config_path}")
    elif config_path:
        logger.warning(f"Config file not found: {config_path} — using defaults")

    # ── Validate required keys ──
    missing = [p for p in REQUIRED_KEYS if get_nested(cfg, p) is None]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    # ── Auto batch scaling ──
    if auto_batch:
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb >= 80:
                    cfg["training"]["batch_size"] = max(cfg["training"]["batch_size"], 64)
                elif vram_gb >= 24:
                    cfg["training"]["batch_size"] = max(cfg["training"]["batch_size"], 16)
                elif vram_gb >= 8:
                    cfg["training"]["batch_size"] = max(cfg["training"]["batch_size"], 4)
        except Exception:
            pass

    if verbose:
        _log_config(cfg)

    return cfg


def _log_config(cfg: dict, prefix: str = "") -> None:
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _log_config(v, prefix=key)
        else:
            logger.info(f"  cfg  {key}: {v}")
