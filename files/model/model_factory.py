"""
ConformerFlow — Model Factory

Reads a resolved config dict and builds the complete model
with the correct encoder, generative model, and stats module.
This is the single entry point for model construction.

Usage:
    from model.model_factory import build_model
    model = build_model(cfg)
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# ENCODER FACTORY
# ─────────────────────────────────────────────────────────

def build_encoder(cfg: dict) -> nn.Module:
    """Build the structure encoder from config."""
    struct  = cfg.get("representation", {}).get("structure", "se3_frames")
    seq_enc = cfg.get("representation", {}).get("sequence",  "onehot")
    logger.info(f"Building encoder: structure={struct} sequence={seq_enc}")
    from model.encoders import build_encoder as _build
    return _build(cfg)


# ─────────────────────────────────────────────────────────
# ENSEMBLE STATS FACTORY
# ─────────────────────────────────────────────────────────

def build_ensemble_stats(cfg: dict) -> tuple:
    """Build the ensemble statistics module and sampler."""
    arch = cfg["model"]
    es   = cfg["ensemble_stats"]
    cov  = es["covariance"]

    logger.info(f"Building ensemble stats: covariance={cov}")

    from model.ensemble_stats import EnsembleStatisticsModule, DistributionSampler

    stats = EnsembleStatisticsModule(
        d_model   = arch["d_model"],
        d_latent  = arch["d_latent"],
        n_heads   = arch["n_heads"],
        dropout   = arch["dropout"],
    )
    sampler = DistributionSampler(d_latent=arch["d_latent"])

    # Note: full_covariance is used by default in EnsembleStatisticsModule.
    # diagonal and pca_topk variants modify how Sigma is used in sampling.
    stats._covariance_mode = cov
    stats._n_pca_modes     = es.get("n_pca_modes", 10)

    return stats, sampler


# ─────────────────────────────────────────────────────────
# GENERATIVE MODEL FACTORY
# ─────────────────────────────────────────────────────────

def build_generative_model(cfg: dict) -> nn.Module:
    """Build the generative model from config."""
    arch = cfg["model"]
    gm   = cfg["generative_model"]
    gtype= gm["type"]

    logger.info(f"Building generative model: type={gtype}")

    from model.generative_models import build_generative_model as _build
    return _build(cfg)


# ─────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────

class ConformerFlowModel(nn.Module):
    """
    Full ConformerFlow model built from a YAML config.
    Supports all encoder and generative model combinations.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.encoder        = build_encoder(cfg)
        self.ensemble_stats, self.sampler = build_ensemble_stats(cfg)
        self.generative     = build_generative_model(cfg)

        # Store key config values for convenience
        self.d_latent    = cfg["model"]["d_latent"]
        self.n_gen_train = cfg["training"].get("n_gen_conformers", 10)
        self.gen_type    = cfg["generative_model"]["type"]
        self.n_steps_inf = cfg["generative_model"].get("n_inference_steps", 20)
        self.ode_method  = cfg["generative_model"].get("ode_method", "heun")

    def encode(self, coords, one_hot, mask, seq_mask, conf_mask):
        """Encode structure → distribution parameters."""
        h     = self.encoder(coords, one_hot, mask, seq_mask)
        stats = self.ensemble_stats(h, conf_mask, seq_mask)
        return stats

    def forward(self, batch: dict, n_gen: int = None) -> dict:
        """
        Full forward pass for training.
        Works with any encoder + generative model combination.
        """
        coords    = batch["coords"]
        one_hot   = batch["one_hot"]
        mask      = batch["mask"]
        seq_mask  = batch["seq_mask"]
        conf_mask = batch["conformer_mask"]
        n_gen     = n_gen or self.n_gen_train

        B, M, L, _, _ = coords.shape

        # ── Encode ──
        stats   = self.encode(coords, one_hot, mask, seq_mask, conf_mask)
        theta   = stats["theta"]
        mu      = stats["mu"]
        log_var = stats["log_var"]

        # ── Sample one latent for training step ──
        z_single = self.sampler(mu, log_var, n_samples=1,
                                use_full_cov=False).squeeze(1)

        # ── Pick random target conformer ──
        n_valid    = conf_mask.float().sum(dim=-1).long()
        target_idx = torch.zeros(B, dtype=torch.long, device=coords.device)
        for b in range(B):
            target_idx[b] = torch.randint(0, max(n_valid[b].item(), 1), (1,))

        target_coords = coords[torch.arange(B), target_idx]   # (B, L, 4, 3)

        # Get backbone frames for flow-based models
        from model.frames import build_backbone_frames
        R1_5d, t1_5d, _ = build_backbone_frames(
            target_coords.unsqueeze(1), mask
        )
        R1 = R1_5d.squeeze(1)   # (B, L, 3, 3)
        t1 = t1_5d.squeeze(1)   # (B, L, 3) CA positions

        # ── Generative model training step ──
        # training_step() samples its own flow path (t, x0, x_t) internally
        # and returns BOTH the prediction AND the target from that same draw.
        # This is critical: v_R_pred and u_R must correspond to the same t.
        gen_out = self.generative.training_step(R1, t1, theta, z_single, seq_mask)

        # u_R / u_t are always present in gen_out — every generative model
        # (flow matching, DDPM, DDIM, VAE, score matching) stores them.
        u_R = gen_out["u_R"]
        u_t = gen_out["u_t"]

        # ── Generate ensemble for ensemble loss ──
        z_gen = self.sampler(mu, log_var, n_samples=n_gen, use_full_cov=False)
        gen_coords = self.generative.generate(
            theta, z_gen, seq_mask,
            n_conformers = n_gen,
            n_steps      = min(10, self.n_steps_inf),
            method       = self.ode_method if self.gen_type == "flow_matching"
                           else self.gen_type,
        )

        return {
            "v_R_pred":      gen_out["v_R_pred"],
            "v_t_pred":      gen_out["v_t_pred"],
            "u_R":           u_R,
            "u_t":           u_t,
            "gen_coords":    gen_coords,
            "nmr_coords":    coords[:, :, :, 1, :],
            "mu":            mu,
            "log_var":       log_var,
            "mask":          seq_mask,
            "t_flow":        gen_out.get("t_flow", 0.5),
            # For chirality loss: full backbone of selected target conformer
            "target_coords": target_coords,   # (B, L, 4, 3)
            "atom_mask":     mask,            # (B, L, 4)
        }

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        enc   = sum(p.numel() for p in self.encoder.parameters())
        es    = sum(p.numel() for p in self.ensemble_stats.parameters())
        gen   = sum(p.numel() for p in self.generative.parameters())
        return {"total": total, "encoder": enc,
                "ensemble_stats": es, "generative": gen}

    @torch.no_grad()
    def predict_ensemble(self, batch: dict,
                          n_conformers: int,
                          n_steps: int = None,
                          method: str  = None) -> torch.Tensor:
        """
        Inference: generate N conformers for input structures.
        Returns: (B, N, L, 3) CA coordinates
        """
        coords    = batch["coords"]
        one_hot   = batch["one_hot"]
        mask      = batch["mask"]
        seq_mask  = batch["seq_mask"]
        conf_mask = batch["conformer_mask"]

        stats  = self.encode(coords, one_hot, mask, seq_mask, conf_mask)
        theta  = stats["theta"]
        mu     = stats["mu"]
        log_var= stats["log_var"]

        z = self.sampler(mu, log_var, n_samples=n_conformers,
                         use_full_cov=False)

        return self.generative.generate(
            theta, z, seq_mask,
            n_conformers = n_conformers,
            n_steps      = n_steps or self.n_steps_inf,
            method       = method or self.ode_method,
        )


# ─────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────

def build_model(cfg: dict) -> ConformerFlowModel:
    """
    Build ConformerFlow model from a resolved config dict.
    This is the single entry point for all model construction.

    Args:
        cfg: resolved config dict from config_validator.load_and_validate()

    Returns:
        ConformerFlowModel ready for training or inference
    """
    model = ConformerFlowModel(cfg)
    params = model.count_parameters()
    logger.info(
        f"ConformerFlow built — "
        f"encoder={cfg['representation']['structure']} "
        f"gen={cfg['generative_model']['type']} "
        f"cov={cfg['ensemble_stats']['covariance']}"
    )
    logger.info(
        f"Parameters: total={params['total']:,} "
        f"encoder={params['encoder']:,} "
        f"ensemble_stats={params['ensemble_stats']:,} "
        f"generative={params['generative']:,}"
    )
    return model


def load_model(checkpoint_path: str,
               device: str = "auto") -> tuple:
    """
    Load a trained model from checkpoint.
    Returns (model, cfg)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["config"]

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    return model, cfg
