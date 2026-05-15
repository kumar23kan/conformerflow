"""
ConformerFlow — Phase 4: Loss Functions

All five loss components that train the AI to learn
conformational distributions from NMR ensembles.

Each loss teaches the model something specific:
  L_flow:      Learn the SE(3) vector field (core flow matching)
  L_ensemble:  Generated conformers should match NMR statistics
  L_kl:        Regularize the latent distribution
  L_diversity: Prevent mode collapse
  L_geometry:  Generate physically valid structures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# 1. FLOW MATCHING LOSS
#    Already implemented in flow_matching.py
#    Wrapped here for unified interface
# ─────────────────────────────────────────────────────────

def flow_matching_loss(v_R_pred:  torch.Tensor,
                        v_t_pred:  torch.Tensor,
                        u_R:       torch.Tensor,
                        u_t:       torch.Tensor,
                        mask:      torch.Tensor) -> dict:
    """
    Core flow matching loss on SE(3) frames.

    Translation: MSE between predicted and target displacement
    Rotation:    Frobenius norm of (R_pred^T @ R_target - I)

    Args:
        v_R_pred: (B, L, 3, 3)  predicted rotation velocity
        v_t_pred: (B, L, 3)     predicted translation velocity
        u_R:      (B, L, 3, 3)  target rotation (NMR conformer)
        u_t:      (B, L, 3)     target translation velocity
        mask:     (B, L)        True for real residues

    Returns dict: loss, translation_loss, rotation_loss
    """
    m = mask.float().unsqueeze(-1)    # (B, L, 1)
    n = m.sum().clamp(min=1)

    # Translation MSE
    t_loss = ((v_t_pred - u_t) ** 2 * m).sum() / (n * 3)

    # Rotation geodesic loss
    I      = torch.eye(3, device=v_R_pred.device)
    R_diff = torch.matmul(v_R_pred.transpose(-1, -2), u_R) - I
    r_loss = (R_diff ** 2 * m.unsqueeze(-1)).sum() / (n * 9)

    return {
        "flow_loss":         t_loss + r_loss,
        "flow_t_loss":       t_loss,
        "flow_r_loss":       r_loss,
    }


# ─────────────────────────────────────────────────────────
# 2. ENSEMBLE RECONSTRUCTION LOSS
#    Generated conformers should reproduce NMR ensemble stats
# ─────────────────────────────────────────────────────────

def ensemble_reconstruction_loss(gen_coords:  torch.Tensor,
                                   nmr_coords:  torch.Tensor,
                                   mask:        torch.Tensor) -> dict:
    """
    Measures how well generated conformers reproduce
    the statistical properties of the NMR ensemble.

    Does NOT require a 1-to-1 correspondence between
    generated and NMR conformers — compares DISTRIBUTIONS.

    Three sub-components:
      1. Mean structure RMSD: generated mean ≈ NMR mean
      2. Per-residue variance: generated flexibility ≈ NMR flexibility
      3. Covariance similarity: correlated motions preserved

    Args:
        gen_coords: (B, N_gen, L, 3)  generated CA coordinates
        nmr_coords: (B, M,     L, 3)  NMR ensemble CA coordinates
        mask:       (B, L)            True for real residues

    Returns dict: ensemble_loss + components
    """
    m = mask.float().unsqueeze(-1)   # (B, L, 1)
    n = m.sum(dim=1).clamp(min=1)    # (B, 1)

    # ── Mean structure loss ──
    gen_mean = gen_coords.mean(dim=1)   # (B, L, 3)
    nmr_mean = nmr_coords.mean(dim=1)   # (B, L, 3)
    mean_diff= (gen_mean - nmr_mean) ** 2 * m
    mean_loss= mean_diff.sum() / (m.sum() * 3 + 1e-8)

    # ── Per-residue variance loss ──
    # Generated and NMR should have similar per-residue flexibility
    gen_var = gen_coords.var(dim=1)     # (B, L, 3)
    nmr_var = nmr_coords.var(dim=1)     # (B, L, 3)
    var_loss= F.mse_loss(
        gen_var * m, nmr_var * m
    )

    # ── Pairwise distance distribution loss ──
    # Compares the distribution of inter-CA distances between
    # generated and NMR ensembles (SE(3)-invariant)
    def pairwise_dists(coords):
        # coords: (B, N, L, 3) → (B, N, L, L)
        diff  = coords.unsqueeze(-2) - coords.unsqueeze(-3)
        return diff.norm(dim=-1)

    gen_dists = pairwise_dists(gen_coords)  # (B, N_gen, L, L)
    nmr_dists = pairwise_dists(nmr_coords)  # (B, M,     L, L)

    # Compare mean and std of distance distributions
    gen_dist_mean = gen_dists.mean(dim=1)   # (B, L, L)
    nmr_dist_mean = nmr_dists.mean(dim=1)
    gen_dist_std  = gen_dists.std(dim=1)
    nmr_dist_std  = nmr_dists.std(dim=1)

    dist_mean_loss = F.mse_loss(gen_dist_mean, nmr_dist_mean)
    dist_std_loss  = F.mse_loss(gen_dist_std,  nmr_dist_std)
    dist_loss      = dist_mean_loss + dist_std_loss

    ensemble_loss = mean_loss + var_loss + dist_loss

    return {
        "ensemble_loss":      ensemble_loss,
        "ensemble_mean_loss": mean_loss,
        "ensemble_var_loss":  var_loss,
        "ensemble_dist_loss": dist_loss,
    }


# ─────────────────────────────────────────────────────────
# 3. KL DIVERGENCE LOSS
#    Keep the learned latent distribution close to N(0,I)
#    This is what makes the latent space well-structured
#    and enables smooth interpolation between conformers
# ─────────────────────────────────────────────────────────

def kl_divergence_loss(mu:      torch.Tensor,
                        log_var: torch.Tensor,
                        mask:    torch.Tensor,
                        free_bits: float = 0.5) -> dict:
    """
    KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0,I).

    Analytic formula: KL = -0.5 * sum(1 + log_var - mu² - exp(log_var))

    Free bits trick: don't penalize KL below threshold per dimension.
    This prevents posterior collapse (latent z being ignored).

    Args:
        mu:        (B, L, d_latent)
        log_var:   (B, L, d_latent)
        mask:      (B, L)
        free_bits: minimum KL per dimension (prevents collapse)

    Returns dict: kl_loss
    """
    # Per-element KL: (B, L, d_latent)
    kl_elem = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

    # Free bits: clamp per latent dimension
    # Average over batch and sequence, then clamp
    kl_per_dim = kl_elem.mean(dim=[0, 1])             # (d_latent,)
    kl_clamped = kl_per_dim.clamp(min=free_bits)

    # Mask: only count real residues
    m      = mask.float().unsqueeze(-1)                # (B, L, 1)
    kl_sum = (kl_elem * m).sum() / (m.sum() + 1e-8)

    return {
        "kl_loss":     kl_clamped.mean(),
        "kl_raw":      kl_sum,
        "kl_per_dim":  kl_per_dim.detach(),
    }


# ─────────────────────────────────────────────────────────
# 4. DIVERSITY LOSS
#    Prevent mode collapse: all N conformers being identical.
#    The model MUST generate diverse conformers to capture
#    the full flexibility space seen in NMR ensembles.
# ─────────────────────────────────────────────────────────

def diversity_loss(gen_coords: torch.Tensor,
                    mask:       torch.Tensor,
                    min_spread: float = 0.5) -> dict:
    """
    Encourages diversity across generated conformers.

    Strategy: maximize pairwise RMSD between conformers,
    but only up to a minimum spread threshold.
    (We don't want infinite diversity — just enough to cover
     the real NMR ensemble spread.)

    Args:
        gen_coords: (B, N, L, 3)  generated CA coordinates
        mask:       (B, L)
        min_spread: target minimum CA spread in Angstrom

    Returns dict: diversity_loss
    """
    B, N, L, _ = gen_coords.shape
    m = mask.float().unsqueeze(-1)    # (B, L, 1)

    # Per-residue std across N conformers: measures spread
    # (B, L, 3) — how much each residue moves across conformers
    per_res_std = gen_coords.std(dim=1)                   # (B, L, 3)
    ca_spread   = (per_res_std * m).norm(dim=-1)          # (B, L) CA spread per residue

    # Mean spread across sequence
    mean_spread = (ca_spread * mask.float()).sum(dim=-1) \
                  / mask.float().sum(dim=-1).clamp(min=1) # (B,)

    # Penalize when spread is below the minimum threshold
    # (hinge loss: only kicks in when too collapsed)
    div_loss = F.relu(min_spread - mean_spread).mean()

    # Also compute pairwise RMSD for monitoring
    with torch.no_grad():
        ca_flat = gen_coords.reshape(B, N, -1)            # (B, N, L*3)
        pdist   = torch.cdist(ca_flat, ca_flat)           # (B, N, N)
        # Average over upper triangle (pairwise distances)
        triu_mask = torch.triu(torch.ones(N, N, device=gen_coords.device), diagonal=1).bool()
        mean_pairwise_rmsd = pdist[:, triu_mask].mean() / (L ** 0.5)

    return {
        "diversity_loss":      div_loss,
        "mean_spread":         mean_spread.mean().detach(),
        "mean_pairwise_rmsd":  mean_pairwise_rmsd.detach(),
    }


# ─────────────────────────────────────────────────────────
# 5. GEOMETRY LOSS
#    Generated structures should be physically valid:
#    - CA-CA virtual bond lengths ≈ 3.8 Å
#    - No severe clashes (CA-CA < 3.5 Å for non-bonded)
# ─────────────────────────────────────────────────────────

def geometry_loss(gen_coords: torch.Tensor,
                   mask:       torch.Tensor) -> dict:
    """
    Enforces physical validity of generated backbone.

    Virtual CA-CA bond: consecutive CA atoms should be ~3.8 Å apart.
    Clashes: non-bonded CA atoms should be > 3.5 Å apart.

    Args:
        gen_coords: (B, N, L, 3)  generated CA coordinates
        mask:       (B, L)

    Returns dict: geometry_loss + components
    """
    B, N, L, _ = gen_coords.shape

    # Flatten over batch and conformers for efficiency
    ca = gen_coords.reshape(B * N, L, 3)

    # ── Virtual bond loss (CA_i → CA_{i+1} ≈ 3.8 Å) ──
    bond_ideal = 3.8  # Angstrom
    bond_vec   = ca[:, 1:] - ca[:, :-1]               # (B*N, L-1, 3)
    bond_len   = bond_vec.norm(dim=-1)                 # (B*N, L-1)

    # Mask: only count bonds where both residues are real
    m_seq = mask.float()                               # (B, L)
    m_seq_flat = m_seq.unsqueeze(1).expand(B, N, L)
    m_seq_flat = m_seq_flat.reshape(B * N, L)
    bond_mask  = m_seq_flat[:, 1:] * m_seq_flat[:, :-1]  # (B*N, L-1)

    bond_loss = (F.mse_loss(bond_len * bond_mask,
                             torch.full_like(bond_len, bond_ideal) * bond_mask))

    # ── Clash loss (non-bonded CA-CA > 3.5 Å) ──
    clash_thresh = 3.5  # Å
    # Pairwise distances: (B*N, L, L)
    diff   = ca.unsqueeze(2) - ca.unsqueeze(1)         # (B*N, L, L, 3)
    dists  = diff.norm(dim=-1)                         # (B*N, L, L)

    # Mask out bonded pairs (i, i±1) and self (i, i)
    pair_mask = torch.ones(L, L, device=ca.device, dtype=torch.bool)
    pair_mask.fill_diagonal_(False)
    idx = torch.arange(L - 1, device=ca.device)
    pair_mask[idx, idx + 1] = False
    pair_mask[idx + 1, idx] = False

    # Clash: penalize if non-bonded CA distance < threshold
    clash_penalty = F.relu(clash_thresh - dists) * pair_mask.unsqueeze(0)
    clash_loss    = clash_penalty.pow(2).mean()

    geometry_loss_val = bond_loss + 0.1 * clash_loss

    return {
        "geometry_loss":   geometry_loss_val,
        "bond_loss":       bond_loss,
        "clash_loss":      clash_loss,
        "mean_bond_len":   (bond_len * bond_mask).sum() \
                            / bond_mask.sum().clamp(min=1),
    }


# ─────────────────────────────────────────────────────────
# 6. COMBINED LOSS
# ─────────────────────────────────────────────────────────

class ConformerFlowLoss(nn.Module):
    """
    Combined loss function for ConformerFlow.

    L_total = λ_flow      * L_flow
            + λ_ensemble  * L_ensemble
            + λ_kl        * L_kl
            + λ_diversity * L_diversity
            + λ_geometry  * L_geometry

    Lambda weights are tunable — we start with equal weighting
    and can anneal them during training based on validation performance.
    """

    def __init__(self,
                 lambda_flow:      float = 1.0,
                 lambda_ensemble:  float = 1.0,
                 lambda_kl:        float = 0.01,   # small: don't over-regularize
                 lambda_diversity: float = 0.5,
                 lambda_geometry:  float = 0.1,
                 free_bits:        float = 0.5,
                 min_spread:       float = 0.5):
        super().__init__()
        self.lambda_flow      = lambda_flow
        self.lambda_ensemble  = lambda_ensemble
        self.lambda_kl        = lambda_kl
        self.lambda_diversity = lambda_diversity
        self.lambda_geometry  = lambda_geometry
        self.free_bits        = free_bits
        self.min_spread       = min_spread

    def forward(self,
                # Flow matching outputs (per-conformer training step)
                v_R_pred:   torch.Tensor,
                v_t_pred:   torch.Tensor,
                u_R:        torch.Tensor,
                u_t:        torch.Tensor,
                # Generated ensemble (for ensemble + diversity + geometry)
                gen_coords: torch.Tensor,
                nmr_coords: torch.Tensor,
                # Distribution parameters (for KL)
                mu:         torch.Tensor,
                log_var:    torch.Tensor,
                # Masks
                mask:       torch.Tensor) -> dict:
        """
        Args:
            v_R_pred:   (B, L, 3, 3)    predicted rotation velocity
            v_t_pred:   (B, L, 3)       predicted translation velocity
            u_R:        (B, L, 3, 3)    target rotation
            u_t:        (B, L, 3)       target translation velocity
            gen_coords: (B, N, L, 3)    generated CA coordinates
            nmr_coords: (B, M, L, 3)    NMR ensemble CA coordinates
            mu:         (B, L, d_lat)   latent mean
            log_var:    (B, L, d_lat)   latent log variance
            mask:       (B, L)          True for real residues

        Returns:
            dict with total_loss + all components for logging
        """
        # Compute all loss components
        flow_losses  = flow_matching_loss(v_R_pred, v_t_pred, u_R, u_t, mask)
        ens_losses   = ensemble_reconstruction_loss(gen_coords, nmr_coords, mask)
        kl_losses    = kl_divergence_loss(mu, log_var, mask, self.free_bits)
        div_losses   = diversity_loss(gen_coords, mask, self.min_spread)
        geom_losses  = geometry_loss(gen_coords, mask)

        # Weighted sum
        total = (
            self.lambda_flow      * flow_losses["flow_loss"]      +
            self.lambda_ensemble  * ens_losses["ensemble_loss"]   +
            self.lambda_kl        * kl_losses["kl_loss"]          +
            self.lambda_diversity * div_losses["diversity_loss"]   +
            self.lambda_geometry  * geom_losses["geometry_loss"]
        )

        return {
            "total_loss": total,
            **flow_losses,
            **ens_losses,
            **kl_losses,
            **div_losses,
            **geom_losses,
        }
