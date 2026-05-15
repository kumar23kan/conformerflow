"""
ConformerFlow — Phase 2: Ensemble Statistics Module
Computes distribution parameters Θ = (μ, Σ) across all M conformers.

This is the core of what makes ConformerFlow unique:
  - It sees ALL M conformers simultaneously
  - Learns the full covariance structure (correlated motions)
  - Outputs a distribution that the flow matching head samples from

The full covariance matrix captures things like:
  - Hinge-bending (residues i and j move together)
  - Loop-helix coupling (flexibility in one region affects another)
  - Correlated side-chain orientations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EnsembleStatisticsModule(nn.Module):
    """
    Aggregates M conformer embeddings into a distribution Θ = (μ, Σ).

    Architecture:
      1. Compute per-residue mean and variance across conformers
      2. Compute cross-residue covariance via attention
      3. Project to latent distribution parameters

    Input:  h (B, M, L, d_model)  — per-conformer embeddings
    Output:
      mu:    (B, L, d_latent)       — mean of latent distribution
      Sigma: (B, L, L, d_latent)    — full covariance matrix
      theta: (B, L, d_model)        — aggregated context for flow matching
    """

    def __init__(self,
                 d_model:   int = 256,
                 d_latent:  int = 128,
                 n_heads:   int = 8,
                 dropout:   float = 0.1):
        super().__init__()
        self.d_model  = d_model
        self.d_latent = d_latent
        self.n_heads  = n_heads

        # ── Per-residue statistics ──
        # Project mean and variance of conformer embeddings
        self.mean_proj = nn.Linear(d_model, d_model)
        self.var_proj  = nn.Linear(d_model, d_model)

        # ── Covariance attention ──
        # Cross-residue attention to capture correlated motions
        self.cov_q = nn.Linear(d_model, d_model)
        self.cov_k = nn.Linear(d_model, d_model)

        # ── Distribution parameter heads ──
        self.mu_head    = nn.Linear(d_model, d_latent)
        self.logvar_head= nn.Linear(d_model, d_latent)

        # ── Covariance head ──
        # Projects pair-wise features to covariance entries
        self.cov_head = nn.Sequential(
            nn.Linear(d_model * 2 + n_heads, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_latent),
        )

        # ── Context projection ──
        # Aggregated context θ for conditioning the flow matching head
        self.context_proj = nn.Sequential(
            nn.Linear(d_model * 2 + n_heads, d_model),  # mean + var + cov_summary
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                h:             torch.Tensor,
                conformer_mask: torch.Tensor,
                seq_mask:      torch.Tensor) -> dict:
        """
        Args:
            h:              (B, M, L, d_model)  — per-conformer embeddings
            conformer_mask: (B, M)              — True for real conformers
            seq_mask:       (B, L)              — True for real residues

        Returns dict:
            mu:       (B, L, d_latent)       — mean
            log_var:  (B, L, d_latent)       — log variance
            Sigma:    (B, L, L, d_latent)    — full covariance
            theta:    (B, L, d_model)        — conditioning context
            h_mean:   (B, L, d_model)        — mean embedding (for visualization)
        """
        B, M, L, D = h.shape

        # ── Step 1: Per-residue statistics across M conformers ──

        # Masked mean: ignore padded conformers
        # conformer_mask: (B, M) → (B, M, 1, 1)
        c_mask = conformer_mask.float().unsqueeze(-1).unsqueeze(-1)  # (B, M, 1, 1)
        n_valid = c_mask.sum(dim=1).clamp(min=1)                     # (B, 1, 1)

        h_masked = h * c_mask                                         # (B, M, L, D)
        h_mean   = h_masked.sum(dim=1) / n_valid                     # (B, L, D)

        # Variance across conformers
        h_var = ((h - h_mean.unsqueeze(1)) ** 2 * c_mask).sum(dim=1) \
                / n_valid.clamp(min=1)                                # (B, L, D)
        h_std = h_var.sqrt()

        # Project mean and variance
        mean_feat = self.mean_proj(h_mean)   # (B, L, D)
        var_feat  = self.var_proj(h_std)     # (B, L, D)

        # ── Step 2: Cross-residue covariance attention ──
        # Captures correlated flexibility between residue pairs

        # Use mean embedding as queries and keys for covariance
        Q_cov = self.cov_q(h_mean)   # (B, L, D)
        K_cov = self.cov_k(h_mean)   # (B, L, D)

        # Multi-head covariance attention
        d_head = self.d_model // self.n_heads
        Q_mh = Q_cov.view(B, L, self.n_heads, d_head)  # (B, L, H, d)
        K_mh = K_cov.view(B, L, self.n_heads, d_head)

        # Covariance scores: (B, H, L, L)
        cov_scores = torch.einsum("blhd,bjhd->bhlj", Q_mh, K_mh) / (d_head ** 0.5)

        # Mask padding
        if seq_mask is not None:
            pad_mask    = (~seq_mask).float() * -1e9     # (B, L)
            cov_scores  = cov_scores + pad_mask.unsqueeze(1).unsqueeze(2)

        cov_attn = F.softmax(cov_scores, dim=-1)         # (B, H, L, L)
        cov_attn = self.dropout(cov_attn)

        # Covariance summary per residue: (B, L, H) — how much each residue
        # attends to others (captures global flexibility coupling)
        cov_summary = cov_attn.mean(dim=-1).permute(0, 2, 1)  # (B, L, H)

        # ── Step 3: Full covariance matrix Σ ──
        # For each pair (i,j), compute covariance features
        # Shape: (B, L, L, d_latent)

        mean_i = h_mean.unsqueeze(2).expand(B, L, L, D)   # (B, L, L, D)
        mean_j = h_mean.unsqueeze(1).expand(B, L, L, D)   # (B, L, L, D)
        attn_ij= cov_attn.mean(dim=1)                     # (B, L, L) — average over heads

        # Pairwise covariance features
        pair_feat = torch.cat([
            mean_i,
            mean_j,
            attn_ij.unsqueeze(-1).expand(B, L, L, self.n_heads)
        ], dim=-1)  # (B, L, L, 2D + H)

        Sigma = self.cov_head(pair_feat)     # (B, L, L, d_latent)

        # Symmetrize: Σ_ij = (Σ_ij + Σ_ji) / 2
        Sigma = (Sigma + Sigma.permute(0, 2, 1, 3)) * 0.5

        # ── Step 4: Distribution parameters μ, log_var ──
        mu      = self.mu_head(mean_feat)      # (B, L, d_latent)
        log_var = self.logvar_head(var_feat)   # (B, L, d_latent)
        # Clamp log_var for stability
        log_var = log_var.clamp(-10, 4)

        # ── Step 5: Context θ for flow matching ──
        context_input = torch.cat([mean_feat, var_feat, cov_summary], dim=-1)
        theta = self.context_proj(context_input)   # (B, L, d_model)

        return {
            "mu":      mu,        # (B, L, d_latent)
            "log_var": log_var,   # (B, L, d_latent)
            "Sigma":   Sigma,     # (B, L, L, d_latent)
            "theta":   theta,     # (B, L, d_model) — flow conditioning context
            "h_mean":  h_mean,    # (B, L, d_model) — for auxiliary losses
            "h_std":   h_std,     # (B, L, d_model) — for auxiliary losses
        }


class DistributionSampler(nn.Module):
    """
    Samples from the learned distribution Θ = (μ, Σ).

    For training: reparameterization trick z = μ + ε * σ
    For inference: sample N times to generate N conformers
    """

    def __init__(self, d_latent: int = 128):
        super().__init__()
        self.d_latent = d_latent

    def reparameterize(self,
                       mu:      torch.Tensor,
                       log_var: torch.Tensor,
                       n_samples: int = 1) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + ε * σ
        where ε ~ N(0, I)

        Args:
            mu:       (B, L, d_latent)
            log_var:  (B, L, d_latent)
            n_samples: number of samples to draw

        Returns:
            z: (B, n_samples, L, d_latent)
        """
        std = torch.exp(0.5 * log_var)   # (B, L, d_latent)

        # Sample n_samples noise vectors
        eps = torch.randn(
            mu.shape[0], n_samples, mu.shape[1], self.d_latent,
            device=mu.device, dtype=mu.dtype
        )   # (B, n_samples, L, d_latent)

        z = mu.unsqueeze(1) + eps * std.unsqueeze(1)   # (B, n_samples, L, d_latent)
        return z

    def sample_with_covariance(self,
                                mu:    torch.Tensor,
                                Sigma: torch.Tensor,
                                n_samples: int = 1,
                                temperature: float = 1.0) -> torch.Tensor:
        """
        Sample from the full multivariate distribution using Sigma.
        Uses Cholesky decomposition: z = μ + L @ ε

        Args:
            mu:    (B, L, d_latent)
            Sigma: (B, L, L, d_latent)  — full covariance
            n_samples: number of conformers to sample
            temperature: scales the noise (>1 = more diverse, <1 = tighter)

        Returns:
            z: (B, n_samples, L, d_latent)
        """
        B, L, d = mu.shape

        # For efficiency, process each latent dimension independently
        # Sigma_d: (B, L, L) for each d
        samples = []

        for d_idx in range(d):
            S = Sigma[..., d_idx]   # (B, L, L)

            # Add small diagonal for numerical stability
            S = S + torch.eye(L, device=S.device).unsqueeze(0) * 1e-4

            # Cholesky decomposition: S = L @ L^T
            try:
                chol = torch.linalg.cholesky(S)   # (B, L, L)
            except Exception:
                # Fallback: diagonal approximation
                chol = torch.diag_embed(S.diagonal(dim1=-2, dim2=-1).sqrt())

            # Sample: z_d = mu_d + temperature * L @ eps
            eps = torch.randn(B, n_samples, L, 1, device=mu.device)
            z_d = mu[:, :, d_idx].unsqueeze(1) \
                  + temperature * (chol.unsqueeze(1) @ eps).squeeze(-1)  # (B, n_samples, L)
            samples.append(z_d)

        # Stack along latent dimension
        z = torch.stack(samples, dim=-1)   # (B, n_samples, L, d_latent)
        return z

    def forward(self,
                mu:          torch.Tensor,
                log_var:     torch.Tensor,
                Sigma:       torch.Tensor = None,
                n_samples:   int = 1,
                use_full_cov: bool = True,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Sample latent vectors.

        Training: use_full_cov=False (faster, reparameterization trick)
        Inference: use_full_cov=True (full covariance, more diverse)
        """
        if use_full_cov and Sigma is not None:
            return self.sample_with_covariance(mu, Sigma, n_samples, temperature)
        else:
            return self.reparameterize(mu, log_var, n_samples)
