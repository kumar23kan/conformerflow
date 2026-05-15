"""
ConformerFlow — Generative Model Backends
Each class implements the same interface:

    training_step(R1, t1, theta, z, seq_mask) -> dict
        R1:       (B, L, 3, 3)  ground-truth rotation matrices  (unused by coord models)
        t1:       (B, L, 3)     ground-truth CA coordinates
        theta:    (B, L, d_model)  sequence/structure context
        z:        (B, d_latent)    latent sample
        seq_mask: (B, L)        bool — True where residue is valid
        Returns dict with keys: v_R_pred, v_t_pred, u_R, u_t, t_flow
            (losses.py ConformerFlowLoss consumes these keys)

    generate(theta, z_samples, seq_mask, n_conformers, n_steps, method) -> (B, N, L, 3)
        theta:     (B, L, d_model)
        z_samples: (B, N, d_latent)   pre-drawn latent samples
        Returns CA coordinates for N conformers.

After setup (step 1) this file lives at model/generative_models.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── Shared building blocks ────────────────────────────────────────────────────

class _FFN(nn.Module):
    def __init__(self, d: int, d_ff: int, dropout: float):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x + self.net(x))


class _CrossAttnLayer(nn.Module):
    """Self-attention on query, cross-attention to context (theta), then FFN."""
    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                                 batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                                 batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = _FFN(d, d_ff, dropout)

    def forward(self, x, context, key_padding_mask=None):
        a, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + a)
        a, _ = self.cross_attn(x, context, context,
                                key_padding_mask=key_padding_mask)
        x = self.norm2(x + a)
        return self.ffn(x)


class _DenoisingTransformer(nn.Module):
    """
    Shared backbone for DDPM / DDIM / Score Matching.
    Input:  noisy CA coords + time embedding + theta context + latent z
    Output: predicted quantity at each residue (B, L, 3)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        m   = cfg.get("model", {})
        d   = m.get("d_model",      256)
        d_z = m.get("d_latent",     128)
        nh  = m.get("n_heads",        8)
        n_l = m.get("n_flow_layers",  8)
        dff = m.get("d_ff",         512)
        dp  = m.get("dropout",       0.1)

        # Embed time scalar → d
        self.time_embed = nn.Sequential(
            nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d)
        )
        # Embed latent z → d
        self.z_proj = nn.Linear(d_z, d)
        # Project noisy coords (3) + theta (d) + t_emb (d) + z_emb (d) → d
        self.in_proj = nn.Linear(3 + d + d + d, d)

        self.layers = nn.ModuleList([
            _CrossAttnLayer(d, nh, dff, dp) for _ in range(n_l)
        ])
        self.out_proj = nn.Linear(d, 3)

    def forward(self, noisy_coords, t, theta, z, seq_mask):
        # noisy_coords: (B, L, 3)
        # t:            (B,) scalar in [0, 1]
        # theta:        (B, L, d)
        # z:            (B, d_z)
        # seq_mask:     (B, L) bool
        B, L, _ = noisy_coords.shape

        t_emb = self.time_embed(t[:, None].float())   # (B, d)
        t_emb = t_emb.unsqueeze(1).expand(-1, L, -1)  # (B, L, d)
        z_emb = self.z_proj(z)                         # (B, L, d)  — z is per-residue

        x = torch.cat([noisy_coords, theta, t_emb, z_emb], dim=-1)
        x = self.in_proj(x)

        pad_key = ~seq_mask
        for layer in self.layers:
            x = layer(x, theta, key_padding_mask=pad_key)

        pred = self.out_proj(x)   # (B, L, 3)
        return pred * seq_mask.unsqueeze(-1).float()


class _VAEDecoder(nn.Module):
    """Direct decoder: theta + z → CA coords. No time embedding."""
    def __init__(self, cfg: dict):
        super().__init__()
        m   = cfg.get("model", {})
        d   = m.get("d_model",      256)
        d_z = m.get("d_latent",     128)
        nh  = m.get("n_heads",        8)
        n_l = m.get("n_flow_layers",  8)
        dff = m.get("d_ff",         512)
        dp  = m.get("dropout",       0.1)

        self.z_proj  = nn.Linear(d_z, d)
        self.in_proj = nn.Linear(d + d, d)

        self.layers = nn.ModuleList([
            _CrossAttnLayer(d, nh, dff, dp) for _ in range(n_l)
        ])
        self.out_proj = nn.Linear(d, 3)

    def forward(self, theta, z, seq_mask):
        B, L, _ = theta.shape
        z_emb = self.z_proj(z)                            # (B, L, d) — per-residue
        x = self.in_proj(torch.cat([theta, z_emb], dim=-1))

        pad_key = ~seq_mask
        for layer in self.layers:
            x = layer(x, theta, key_padding_mask=pad_key)

        pred = self.out_proj(x)
        return pred * seq_mask.unsqueeze(-1).float()


def _identity_R(B, L, device):
    """Return (B, L, 3, 3) identity matrices — used when rotation loss should be 0."""
    I = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, L, -1, -1)
    return I.clone()


# ── Flow Matching ─────────────────────────────────────────────────────────────

class FlowMatchingGenerativeModel(nn.Module):
    """
    Thin wrapper around FlowMatchingModule from model.flow_matching.
    Uses SE(3) frames — full rotation + translation prediction.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m   = cfg.get("model", {})
        gm  = cfg.get("generative_model", {})
        from model.flow_matching import FlowMatchingModule
        self.fm = FlowMatchingModule(
            d_model   = m.get("d_model",         256),
            d_latent  = m.get("d_latent",        128),
            n_layers  = m.get("n_flow_layers",     8),
            n_heads   = m.get("n_heads",            8),
            d_ff      = m.get("d_ff",             512),
            dropout   = m.get("dropout",          0.1),
            sigma_min = gm.get("sigma_min",      0.01),
        )

    def training_step(self, R1, t1, theta, z, seq_mask):
        return self.fm.training_step(R1, t1, theta, z, seq_mask)

    def generate(self, theta, z_samples, seq_mask, n_conformers, n_steps, method):
        return self.fm.generate(theta, z_samples, seq_mask,
                                n_conformers=n_conformers,
                                n_steps=n_steps, method=method)


# ── Optimal Transport CFM ─────────────────────────────────────────────────────

class OTCFMGenerativeModel(nn.Module):
    """
    OT-CFM: matches noise→data pairs within each mini-batch using the
    linear assignment algorithm (scipy.optimize.linear_sum_assignment) to
    minimise transport cost, reducing gradient variance vs. vanilla CFM.
    Coordinate-based (not SE(3) equivariant).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m            = cfg.get("model", {})
        self.sigma_min = cfg.get("generative_model", {}).get("sigma_min", 0.01)
        self.denoiser  = _DenoisingTransformer(cfg)

    def _ot_match(self, x0, x1, seq_mask):
        """
        x0, x1: (B, L, 3) noise / data
        Returns permuted x0 so that ||x0[perm[i]] - x1[i]|| is minimised.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return x0   # fall back to random matching if scipy unavailable

        B, L, _ = x0.shape
        x0_perm = x0.clone()
        for b in range(B):
            valid = seq_mask[b]   # (L,) bool
            x0_v  = x0[b][valid].detach().cpu().numpy()   # (V, 3)
            x1_v  = x1[b][valid].detach().cpu().numpy()

            # Cost matrix: squared distances (V, V)
            diff  = x0_v[:, None] - x1_v[None, :]        # (V, V, 3)
            cost  = (diff ** 2).sum(-1)
            row_ind, col_ind = linear_sum_assignment(cost)

            perm_full = torch.arange(L, device=x0.device)
            valid_idx = valid.nonzero(as_tuple=True)[0]
            perm_full[valid_idx[col_ind]] = valid_idx[row_ind]
            x0_perm[b] = x0[b][perm_full]

        return x0_perm

    def training_step(self, R1, t1, theta, z, seq_mask):
        B, L, _ = t1.shape
        x0 = torch.randn_like(t1)
        x0_matched = self._ot_match(x0, t1, seq_mask)

        t_flow = torch.rand(B, device=t1.device)
        xt = (1 - t_flow[:, None, None]) * x0_matched + t_flow[:, None, None] * t1
        xt = xt + self.sigma_min * torch.randn_like(xt)

        u_t     = t1 - x0_matched                        # ground-truth velocity
        v_t_pred = self.denoiser(xt, t_flow, theta, z, seq_mask)

        I = _identity_R(B, L, t1.device)
        return {
            "v_R_pred": I, "u_R": I,
            "v_t_pred": v_t_pred,
            "u_t":      u_t,
            "t_flow":   t_flow,
        }

    @torch.no_grad()
    def generate(self, theta, z_samples, seq_mask, n_conformers, n_steps, method):
        B, L, _ = theta.shape
        device   = theta.device
        results  = []

        for n in range(n_conformers):
            z = z_samples[:, n]                          # (B, d_z)
            xt = torch.randn(B, L, 3, device=device)

            dt = 1.0 / n_steps
            for i in range(n_steps):
                t = torch.full((B,), i * dt, device=device)
                v = self.denoiser(xt, t, theta, z, seq_mask)
                if method == "euler":
                    xt = xt + dt * v
                else:   # heun
                    xt2 = xt + dt * v
                    t2  = torch.full((B,), (i + 1) * dt, device=device)
                    v2  = self.denoiser(xt2, t2, theta, z, seq_mask)
                    xt  = xt + 0.5 * dt * (v + v2)

            results.append(xt)

        return torch.stack(results, dim=1)   # (B, N, L, 3)


# ── DDPM ──────────────────────────────────────────────────────────────────────

class DDPMGenerativeModel(nn.Module):
    """
    Denoising Diffusion Probabilistic Model on Cα coordinates.
    Cosine noise schedule. Predicts noise ε. Reverse diffusion at inference.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        gm = cfg.get("generative_model", {})
        self.T       = gm.get("n_inference_steps", 1000)
        self.denoiser = _DenoisingTransformer(cfg)
        self._build_schedule()

    def _build_schedule(self):
        T = self.T
        t_vals = torch.arange(T + 1).float() / T
        f      = torch.cos((t_vals + 0.008) / 1.008 * math.pi / 2) ** 2
        betas  = torch.clamp(1 - f[1:] / f[:-1], 1e-4, 0.9999)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, 0)

        self.register_buffer = lambda name, t: setattr(self, name, nn.Parameter(t, requires_grad=False))
        # Store as plain tensors (registered via nn.Buffer pattern below)
        self._betas          = betas
        self._alphas         = alphas
        self._alpha_bar      = alpha_bar

    def _buffers_to(self, device):
        if self._betas.device != device:
            self._betas      = self._betas.to(device)
            self._alphas     = self._alphas.to(device)
            self._alpha_bar  = self._alpha_bar.to(device)

    def training_step(self, R1, t1, theta, z, seq_mask):
        B, L, _ = t1.shape
        device   = t1.device
        self._buffers_to(device)

        # Uniform random timestep
        step = torch.randint(0, self.T, (B,), device=device)
        ab   = self._alpha_bar[step][:, None, None]   # (B, 1, 1)
        eps  = torch.randn_like(t1)
        xt   = ab.sqrt() * t1 + (1 - ab).sqrt() * eps

        t_norm    = step.float() / self.T
        eps_pred  = self.denoiser(xt, t_norm, theta, z, seq_mask)

        I = _identity_R(B, L, device)
        return {
            "v_R_pred": I,
            "u_R":      I,
            "v_t_pred": eps_pred,
            "u_t":      eps,
            "t_flow":   t_norm,
        }

    @torch.no_grad()
    def generate(self, theta, z_samples, seq_mask, n_conformers, n_steps, method):
        B, L, _ = theta.shape
        device   = theta.device
        self._buffers_to(device)

        # Use n_steps as sub-sampled DDPM steps
        skip = max(1, self.T // n_steps)
        steps = list(range(self.T - 1, -1, -skip))

        results = []
        for n in range(n_conformers):
            z  = z_samples[:, n]
            xt = torch.randn(B, L, 3, device=device)

            for step in steps:
                t_norm = torch.full((B,), step / self.T, device=device)
                eps_pred = self.denoiser(xt, t_norm, theta, z, seq_mask)

                ab  = self._alpha_bar[step]
                ab_prev = self._alpha_bar[step - skip] if step - skip >= 0 \
                          else torch.ones(1, device=device)

                x0_pred = (xt - (1 - ab).sqrt() * eps_pred) / ab.sqrt().clamp(min=1e-8)
                x0_pred = x0_pred.clamp(-5, 5)

                if step > 0:
                    noise = torch.randn_like(xt)
                    sigma = ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
                    xt = ab_prev.sqrt() * x0_pred \
                         + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_pred \
                         + sigma * noise
                else:
                    xt = x0_pred

            results.append(xt)

        return torch.stack(results, dim=1)


# ── DDIM ──────────────────────────────────────────────────────────────────────

class DDIMGenerativeModel(DDPMGenerativeModel):
    """
    DDIM: same training as DDPM (predicts ε), but deterministic inference
    — uses fewer steps without quality loss.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)

    @torch.no_grad()
    def generate(self, theta, z_samples, seq_mask, n_conformers, n_steps, method):
        B, L, _ = theta.shape
        device   = theta.device
        self._buffers_to(device)

        skip  = max(1, self.T // n_steps)
        steps = list(range(self.T - 1, -1, -skip))

        results = []
        for n in range(n_conformers):
            z  = z_samples[:, n]
            xt = torch.randn(B, L, 3, device=device)

            for i, step in enumerate(steps):
                t_norm   = torch.full((B,), step / self.T, device=device)
                eps_pred = self.denoiser(xt, t_norm, theta, z, seq_mask)

                ab  = self._alpha_bar[step]
                ab_prev = self._alpha_bar[steps[i + 1]] if i + 1 < len(steps) \
                          else torch.ones(1, device=device)

                x0_pred = (xt - (1 - ab).sqrt() * eps_pred) / ab.sqrt().clamp(min=1e-8)
                x0_pred = x0_pred.clamp(-5, 5)
                # Deterministic step (eta=0)
                xt = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_pred

            results.append(xt)

        return torch.stack(results, dim=1)


# ── VAE ───────────────────────────────────────────────────────────────────────

class VAEGenerativeModel(nn.Module):
    """
    Variational Auto-Encoder decoder.
    Training: decode z+theta → x0, loss = reconstruction MSE + KL.
    Inference: single forward pass per conformer — fastest generator.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.decoder = _VAEDecoder(cfg)

    def training_step(self, R1, t1, theta, z, seq_mask):
        B, L, _ = t1.shape
        x0_pred = self.decoder(theta, z, seq_mask)

        I = _identity_R(B, L, t1.device)
        # u_t = t1 (ground truth), v_t_pred = x0_pred (reconstruction)
        # losses.py flow_matching_loss will compute ||x0_pred - t1||^2 weighted by mask
        # t_flow = 1 so the schedule factor is neutral
        return {
            "v_R_pred": I,
            "u_R":      I,
            "v_t_pred": x0_pred,
            "u_t":      t1,
            "t_flow":   torch.ones(B, device=t1.device),
        }

    @torch.no_grad()
    def generate(self, theta, z_samples, seq_mask, n_conformers, n_steps, method):
        # n_steps and method are ignored for VAE (single-pass)
        B = theta.shape[0]
        results = []
        for n in range(n_conformers):
            z    = z_samples[:, n]
            x0   = self.decoder(theta, z, seq_mask)
            results.append(x0)
        return torch.stack(results, dim=1)   # (B, N, L, 3)


# ── Score Matching ────────────────────────────────────────────────────────────

class ScoreMatchingGenerativeModel(nn.Module):
    """
    Score-based generative model (Song & Ermon, 2020) with geometric sigma schedule.
    Trains score network s_θ ≈ -ε/σ. Langevin SDE inference.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        gm = cfg.get("generative_model", {})
        self.sigma_min = gm.get("sigma_min",  0.01)
        self.sigma_max = gm.get("sigma_max",  50.0)
        self.denoiser  = _DenoisingTransformer(cfg)

    def _sigma(self, t):
        """Geometric schedule: sigma(t) = sigma_min * (sigma_max/sigma_min)^t"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def training_step(self, R1, t1, theta, z, seq_mask):
        B, L, _ = t1.shape
        device   = t1.device

        t_flow = torch.rand(B, device=device)
        sigma  = self._sigma(t_flow)[:, None, None]   # (B, 1, 1)
        eps    = torch.randn_like(t1)
        xt     = t1 + sigma * eps

        # Score network predicts s = -eps/sigma, re-arranged: predict eps
        eps_pred = self.denoiser(xt, t_flow, theta, z, seq_mask)

        I = _identity_R(B, L, device)
        return {
            "v_R_pred": I,
            "u_R":      I,
            "v_t_pred": eps_pred,
            "u_t":      eps,
            "t_flow":   t_flow,
        }

    @torch.no_grad()
    def generate(self, theta, z_samples, seq_mask, n_conformers, n_steps, method):
        B, L, _ = theta.shape
        device   = theta.device

        results = []
        for n in range(n_conformers):
            z  = z_samples[:, n]
            # Start from high-noise sample
            xt = torch.randn(B, L, 3, device=device) * self.sigma_max

            dt = 1.0 / n_steps
            for i in range(n_steps, 0, -1):
                t    = torch.full((B,), i / n_steps, device=device)
                t_p  = torch.full((B,), (i - 1) / n_steps, device=device)
                sig  = self._sigma(t[:, None, None].float())
                sig_p = self._sigma(t_p[:, None, None].float())

                eps_pred = self.denoiser(xt, t, theta, z, seq_mask)
                score    = -eps_pred / sig.clamp(min=1e-8)

                # Euler-Maruyama SDE step
                dx   = -sig ** 2 * score * dt
                noise_scale = (sig ** 2 - sig_p ** 2).clamp(min=0).sqrt()
                xt   = xt + dx + noise_scale * torch.randn_like(xt)

            results.append(xt)

        return torch.stack(results, dim=1)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_generative_model(cfg: dict) -> nn.Module:
    kind = cfg.get("generative_model", {}).get("type", "flow_matching")
    if kind == "flow_matching":
        return FlowMatchingGenerativeModel(cfg)
    elif kind == "ot_cfm":
        return OTCFMGenerativeModel(cfg)
    elif kind == "ddpm":
        return DDPMGenerativeModel(cfg)
    elif kind == "ddim":
        return DDIMGenerativeModel(cfg)
    elif kind == "vae":
        return VAEGenerativeModel(cfg)
    elif kind == "score_matching":
        return ScoreMatchingGenerativeModel(cfg)
    else:
        raise ValueError(
            f"Unknown generative model type '{kind}'. "
            f"Choose from: flow_matching, ot_cfm, ddpm, ddim, vae, score_matching"
        )
