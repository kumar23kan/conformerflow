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
from typing import Optional


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
                    min_spread: float = 2.0) -> dict:
    """
    Encourages diversity across generated conformers.

    Measures mean pairwise RMSD between conformers after centring
    each conformer by its CA centroid (translation-invariant).
    The hinge loss fires only when conformers collapse toward each
    other (mean pairwise RMSD < min_spread).

    Using centred pairwise RMSD instead of uncentred per-residue std
    prevents a random-noise model from trivially satisfying the
    constraint: for a well-trained model generating physically correct
    coordinates, pairwise RMSD directly measures structural diversity.

    Args:
        gen_coords: (B, N, L, 3)  generated CA coordinates
        mask:       (B, L)        True for real residues
        min_spread: hinge threshold in Angstrom (default 2.0 Å —
                    flexible proteins span 1–10 Å, rigid < 0.5 Å)

    Returns dict with diversity_loss, mean_pairwise_rmsd, mean_spread
    """
    B, N, L, _ = gen_coords.shape
    mf = mask.float()                              # (B, L)
    n_valid = mf.sum(-1, keepdim=True).clamp(min=1)  # (B, 1)

    # Centre each conformer by its masked CA centroid
    # (removes global translation, making RMSD rotation-robust for
    # a well-trained model that generates conformers in a consistent frame)
    centroid  = (gen_coords * mf[:, None, :, None]).sum(dim=2, keepdim=True) \
                / n_valid[:, :, None, None]         # (B, N, 1, 3)
    gen_c = gen_coords - centroid                   # (B, N, L, 3)

    # Zero-out padding positions
    gen_c = gen_c * mf[:, None, :, None]

    # Pairwise RMSD between all N*(N-1)/2 conformer pairs
    ca_flat  = gen_c.reshape(B, N, -1)             # (B, N, L*3)
    pdist    = torch.cdist(ca_flat, ca_flat)        # (B, N, N)  L2 of flattened
    # pdist[b,i,j] = ||c_i - c_j||_F  over all residues
    # RMSD = ||...||_F / sqrt(n_valid)
    rmsd_mat = pdist / n_valid.unsqueeze(-1).clamp(min=1).sqrt()  # (B, N, N)

    triu = torch.triu(torch.ones(N, N, device=gen_coords.device,
                                 dtype=torch.bool), diagonal=1)
    if triu.sum() == 0 or N < 2:
        mean_pairwise_rmsd = torch.zeros(B, device=gen_coords.device)
    else:
        mean_pairwise_rmsd = rmsd_mat[:, triu].mean(dim=-1)  # (B,)

    # Hinge: penalise when conformers are too similar
    div_loss = F.relu(min_spread - mean_pairwise_rmsd).mean()

    # Keep old per-residue spread for monitoring (unaffected by the fix)
    with torch.no_grad():
        per_res_std = gen_coords.std(dim=1)                      # (B, L, 3)
        ca_spread   = (per_res_std * mf[:, :, None]).norm(dim=-1)  # (B, L)
        mean_spread = (ca_spread * mf).sum(-1) / n_valid.squeeze(-1)

    return {
        "diversity_loss":      div_loss,
        "mean_pairwise_rmsd":  mean_pairwise_rmsd.mean().detach(),
        "mean_spread":         mean_spread.mean().detach(),
    }


# ─────────────────────────────────────────────────────────
# 5. CHIRALITY LOSS
#    For non-SE3 encoders (Cartesian, Distance) the encoder
#    is mirror-symmetric: it cannot distinguish L- from
#    D-amino acids.  The generative model may consequently
#    produce D-amino acid backbone configurations.
#    We penalise them using the scalar triple product:
#    det([N-CA, C-CA, CB-CA]) > 0  ⟺  L-amino acid.
# ─────────────────────────────────────────────────────────

def chirality_loss(backbone_coords: torch.Tensor,
                   atom_mask:       torch.Tensor) -> dict:
    """
    Penalise D-amino acid backbone configurations in the training target.

    For L-amino acids the triple product (N-CA)×(C-CA)·(CB-CA) > 0.
    Any negative value indicates a mirror-image (D-amino acid) frame
    and is penalised linearly.

    Useful during training as a data-quality guard and, for non-SE3
    encoders, as a regulariser that keeps the model away from mirror-image
    solutions.  The loss is identically 0 for clean NMR data (all L-amino
    acids), so it adds no gradient noise when the data is correct.

    Args:
        backbone_coords: (B, L, 4, 3)  N=0, CA=1, C=2, CB=3
        atom_mask:       (B, L, 4)     True where atom exists

    Returns dict: chirality_loss, n_chiral_violations
    """
    n_ca = backbone_coords[..., 0, :] - backbone_coords[..., 1, :]  # N - CA
    c_ca = backbone_coords[..., 2, :] - backbone_coords[..., 1, :]  # C - CA
    cb_ca= backbone_coords[..., 3, :] - backbone_coords[..., 1, :]  # CB- CA

    # Triple product = (N-CA) × (C-CA) · (CB-CA)
    # Positive for L-amino acids, negative for D-amino acids.
    cross   = torch.linalg.cross(n_ca, c_ca, dim=-1)   # (B, L, 3)
    triple  = (cross * cb_ca).sum(-1)                   # (B, L)

    # Only count residues where all 4 backbone atoms are present
    has_all = atom_mask[..., 0] & atom_mask[..., 1] \
            & atom_mask[..., 2] & atom_mask[..., 3]    # (B, L)
    m = has_all.float()

    n_total = m.sum().clamp(min=1)
    penalty = F.relu(-triple) * m                       # 0 for L-, ‖triple‖ for D-

    with torch.no_grad():
        n_violations = (triple < 0).float() * m

    return {
        "chirality_loss":      penalty.sum() / n_total,
        "n_chiral_violations": n_violations.sum().detach(),
    }


# ─────────────────────────────────────────────────────────
# 6. GEOMETRY LOSS
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
# 7. COMBINED LOSS
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
                 lambda_kl:        float = 0.01,
                 lambda_diversity: float = 0.5,
                 lambda_geometry:  float = 0.1,
                 lambda_chirality: float = 0.1,
                 free_bits:        float = 0.5,
                 min_spread:       float = 0.5):
        super().__init__()
        self.lambda_flow      = lambda_flow
        self.lambda_ensemble  = lambda_ensemble
        self.lambda_kl        = lambda_kl
        self.lambda_diversity = lambda_diversity
        self.lambda_geometry  = lambda_geometry
        self.lambda_chirality = lambda_chirality
        self.free_bits        = free_bits
        self.min_spread       = min_spread

    def forward(self,
                # Flow matching outputs (per-conformer training step)
                v_R_pred:        torch.Tensor,
                v_t_pred:        torch.Tensor,
                u_R:             torch.Tensor,
                u_t:             torch.Tensor,
                # Generated ensemble (for ensemble + diversity + geometry)
                gen_coords:      torch.Tensor,
                nmr_coords:      torch.Tensor,
                # Distribution parameters (for KL)
                mu:              torch.Tensor,
                log_var:         torch.Tensor,
                # Masks
                mask:            torch.Tensor,
                # Optional: full backbone for chirality check
                backbone_coords: Optional[torch.Tensor] = None,
                atom_mask:       Optional[torch.Tensor] = None) -> dict:
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

        # Chirality loss — only when full backbone is provided
        chir_losses: dict = {}
        chir_term = torch.tensor(0.0, device=mask.device)
        if backbone_coords is not None and atom_mask is not None:
            chir_losses = chirality_loss(backbone_coords, atom_mask)
            chir_term   = chir_losses["chirality_loss"]

        # Weighted sum
        total = (
            self.lambda_flow      * flow_losses["flow_loss"]      +
            self.lambda_ensemble  * ens_losses["ensemble_loss"]   +
            self.lambda_kl        * kl_losses["kl_loss"]          +
            self.lambda_diversity * div_losses["diversity_loss"]   +
            self.lambda_geometry  * geom_losses["geometry_loss"]  +
            self.lambda_chirality * chir_term
        )

        return {
            "total_loss": total,
            **flow_losses,
            **ens_losses,
            **kl_losses,
            **div_losses,
            **geom_losses,
            **chir_losses,
        }
