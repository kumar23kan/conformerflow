"""
ConformerFlow — Phase 6: Evaluation Metrics

All metrics for comparing predicted conformational ensembles
against NMR ground truth. Organized into 4 validation levels.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────

def center_coords(coords: np.ndarray,
                  mask:   np.ndarray = None) -> np.ndarray:
    """
    Remove global translation by centering on centroid.
    coords: (N, L, 3) or (L, 3)
    mask:   (L,) bool, True for real residues
    """
    if mask is not None:
        if coords.ndim == 3:
            mu = coords[:, mask, :].mean(axis=1, keepdims=True)  # (N,1,3)
        else:
            mu = coords[mask, :].mean(axis=0, keepdims=True)     # (1,3)
    else:
        if coords.ndim == 3:
            mu = coords.mean(axis=1, keepdims=True)
        else:
            mu = coords.mean(axis=0, keepdims=True)
    return coords - mu


def kabsch_align(mobile: np.ndarray,
                  target: np.ndarray) -> tuple:
    """
    Optimal superposition via Kabsch algorithm.
    mobile, target: (L, 3) CA coordinates
    Returns: (aligned_mobile, rotation_matrix, rmsd)
    """
    # Center both
    m_center = mobile.mean(axis=0)
    t_center = target.mean(axis=0)
    m = mobile - m_center
    t = target - t_center

    # SVD of covariance matrix
    H  = m.T @ t
    U, S, Vt = np.linalg.svd(H)
    d  = np.linalg.det(Vt.T @ U.T)
    D  = np.diag([1, 1, d])
    R  = Vt.T @ D @ U.T

    # Apply rotation
    aligned = (m @ R.T) + t_center
    rmsd    = np.sqrt(((aligned - target) ** 2).sum(axis=-1).mean())
    return aligned, R, float(rmsd)


# ─────────────────────────────────────────────────────────
# LEVEL 1 — STRUCTURAL METRICS
# ─────────────────────────────────────────────────────────

def compute_rmsd(coords_a: np.ndarray,
                  coords_b: np.ndarray,
                  align:    bool = True) -> float:
    """
    CA-RMSD between two structures after optional Kabsch alignment.
    coords_a, coords_b: (L, 3)
    """
    if align:
        _, _, rmsd = kabsch_align(coords_a, coords_b)
        return rmsd
    diff = coords_a - coords_b
    return float(np.sqrt((diff ** 2).sum(axis=-1).mean()))


def compute_ensemble_rmsd(pred: np.ndarray,
                           true: np.ndarray,
                           mode: str = "min_avg") -> dict:
    """
    Compare two ensembles structurally.

    mode options:
      'min_avg':   for each true conformer, find the closest predicted one
                   (measures coverage of true ensemble)
      'pairwise':  average RMSD across all pred × true pairs
      'mean_struct': RMSD between the two mean structures

    pred: (N_pred, L, 3)
    true: (N_true, L, 3)
    """
    N_pred, L, _ = pred.shape
    N_true       = true.shape[0]

    # Align all structures to first true conformer
    ref = true[0]
    pred_aligned = np.array([kabsch_align(p, ref)[0] for p in pred])
    true_aligned = np.array([kabsch_align(t, ref)[0] for t in true])

    # Pairwise RMSD matrix: (N_pred, N_true)
    pairwise = np.zeros((N_pred, N_true))
    for i in range(N_pred):
        for j in range(N_true):
            diff = pred_aligned[i] - true_aligned[j]
            pairwise[i, j] = np.sqrt((diff**2).sum(axis=-1).mean())

    # Coverage: for each true conformer, closest predicted
    coverage_rmsd = pairwise.min(axis=0).mean()      # mean of min over pred

    # Precision: for each predicted, closest true
    precision_rmsd = pairwise.min(axis=1).mean()     # mean of min over true

    # Mean pairwise
    mean_pairwise  = pairwise.mean()

    # Mean structure RMSD
    pred_mean = pred_aligned.mean(axis=0)
    true_mean = true_aligned.mean(axis=0)
    mean_struct_rmsd = compute_rmsd(pred_mean, true_mean, align=False)

    return {
        "coverage_rmsd":    float(coverage_rmsd),    # lower = better coverage
        "precision_rmsd":   float(precision_rmsd),   # lower = better precision
        "mean_pairwise_rmsd": float(mean_pairwise),
        "mean_struct_rmsd": float(mean_struct_rmsd),
        "pairwise_matrix":  pairwise,
    }


def ensemble_coverage_at_threshold(gen_coords:  np.ndarray,
                                    nmr_coords:  np.ndarray,
                                    mask:        np.ndarray = None,
                                    thresholds:  tuple = (1.0, 2.0, 5.0)) -> dict:
    """
    NMR recall and generation precision at RMSD thresholds.

    NMR recall @ t:        fraction of NMR conformers with at least one
                           generated conformer within t Å RMSD after alignment.
    Generation precision @ t: fraction of generated conformers within t Å RMSD
                           of at least one NMR conformer.

    gen_coords: (N_gen, L, 3)  CA coordinates
    nmr_coords: (N_nmr, L, 3)  CA coordinates
    mask:       (L,) bool
    thresholds: iterable of RMSD cutoffs in Angstroms
    """
    ref         = nmr_coords[0]
    gen_aligned = np.array([kabsch_align(g, ref)[0] for g in gen_coords])
    nmr_aligned = np.array([kabsch_align(n, ref)[0] for n in nmr_coords])

    if mask is not None:
        gen_m = gen_aligned[:, mask, :]
        nmr_m = nmr_aligned[:, mask, :]
    else:
        gen_m, nmr_m = gen_aligned, nmr_aligned

    N_gen, N_nmr = len(gen_m), len(nmr_m)

    # Pairwise RMSD matrix (N_gen, N_nmr)
    pairwise = np.zeros((N_gen, N_nmr))
    for i in range(N_gen):
        for j in range(N_nmr):
            diff = gen_m[i] - nmr_m[j]
            pairwise[i, j] = np.sqrt((diff ** 2).sum(axis=-1).mean())

    results = {}
    for t in thresholds:
        key = f"{t:.0f}A"
        results[f"nmr_recall_{key}"]    = float((pairwise.min(axis=0) <= t).mean())
        results[f"gen_precision_{key}"] = float((pairwise.min(axis=1) <= t).mean())

    results["coverage_pairwise_rmsd"] = pairwise
    return results


def compute_tm_score(mobile: np.ndarray,
                      target: np.ndarray) -> float:
    """
    TM-score between two CA structures.
    Normalized by target length — invariant to protein size.
    mobile, target: (L, 3)
    Score ∈ (0,1]: >0.5 = same fold
    """
    L  = len(target)
    d0 = 1.24 * (L - 15) ** (1/3) - 1.8 if L > 15 else 0.5
    d0 = max(d0, 0.5)

    aligned, _, _ = kabsch_align(mobile, target)
    di2 = ((aligned - target) ** 2).sum(axis=-1)  # (L,)
    tm  = (1 / (1 + di2 / d0**2)).mean()
    return float(tm)


def compute_gdt_ts(mobile: np.ndarray,
                    target: np.ndarray) -> float:
    """
    GDT-TS score (Global Distance Test — Total Score).
    Fraction of CA atoms within 1, 2, 4, 8 Å cutoffs, averaged.
    mobile, target: (L, 3)
    Score ∈ [0,1]: 1 = perfect
    """
    aligned, _, _ = kabsch_align(mobile, target)
    dists = np.sqrt(((aligned - target) ** 2).sum(axis=-1))  # (L,)

    p1 = (dists <= 1.0).mean()
    p2 = (dists <= 2.0).mean()
    p4 = (dists <= 4.0).mean()
    p8 = (dists <= 8.0).mean()
    return float((p1 + p2 + p4 + p8) / 4)


# ─────────────────────────────────────────────────────────
# LEVEL 2 — CONFORMATIONAL DISTRIBUTION METRICS
# ─────────────────────────────────────────────────────────

def compute_rmsf(coords: np.ndarray,
                  mask:   np.ndarray = None) -> np.ndarray:
    """
    Root Mean Square Fluctuation — per-residue flexibility.
    coords: (N, L, 3)
    Returns: (L,) RMSF values
    """
    mean = coords.mean(axis=0)           # (L, 3)
    diff = coords - mean[np.newaxis]     # (N, L, 3)
    rmsf = np.sqrt((diff**2).sum(axis=-1).mean(axis=0))  # (L,)
    if mask is not None:
        rmsf[~mask] = 0.0
    return rmsf


def compute_rmsf_correlation(pred_coords: np.ndarray,
                               true_coords: np.ndarray,
                               mask: np.ndarray = None) -> dict:
    """
    Compare per-residue flexibility (RMSF) between predicted
    and NMR ensembles.

    High correlation = model correctly identifies flexible regions.

    Returns:
        pearson_r:   Pearson correlation of RMSF profiles
        spearman_r:  Spearman rank correlation (order of flexibility)
        rmsf_pred:   (L,) predicted RMSF
        rmsf_true:   (L,) NMR RMSF
    """
    rmsf_pred = compute_rmsf(pred_coords, mask)
    rmsf_true = compute_rmsf(true_coords, mask)

    # Only compare real residues
    if mask is not None:
        rp = rmsf_pred[mask]
        rt = rmsf_true[mask]
    else:
        rp, rt = rmsf_pred, rmsf_true

    pearson_r,  p_pval = stats.pearsonr(rp, rt)
    spearman_r, s_pval = stats.spearmanr(rp, rt)

    return {
        "rmsf_pearson_r":  float(pearson_r),
        "rmsf_spearman_r": float(spearman_r),
        "rmsf_pearson_p":  float(p_pval),
        "rmsf_pred":       rmsf_pred,
        "rmsf_true":       rmsf_true,
        "rmsf_mae":        float(np.abs(rp - rt).mean()),
    }


def compute_backbone_torsions(ca_coords: np.ndarray) -> dict:
    """
    Approximate backbone torsion angles from CA coordinates only.
    Uses pseudo-dihedral angles between consecutive CA atoms.

    ca_coords: (N, L, 3)  or  (L, 3) for single structure

    Returns:
        pseudo_phi: (N, L-3) pseudo-phi angles in radians
        pseudo_psi: (N, L-3) pseudo-psi angles in radians
    """
    if ca_coords.ndim == 2:
        ca_coords = ca_coords[np.newaxis]

    N, L, _ = ca_coords.shape
    angles   = []

    for i in range(L - 3):
        a = ca_coords[:, i,   :]   # (N, 3)
        b = ca_coords[:, i+1, :]
        c = ca_coords[:, i+2, :]
        d = ca_coords[:, i+3, :]

        b1 = b - a
        b2 = c - b
        b3 = d - c

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = n1 / (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-8)
        n2_norm = n2 / (np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-8)

        cos_a = (n1_norm * n2_norm).sum(axis=-1).clip(-1, 1)
        angle = np.arccos(cos_a)
        angles.append(angle)

    if angles:
        pseudo_angles = np.stack(angles, axis=1)  # (N, L-3)
    else:
        pseudo_angles = np.zeros((N, 0))

    return {"pseudo_torsions": pseudo_angles}


def compute_torsion_distribution_overlap(pred_coords: np.ndarray,
                                          true_coords: np.ndarray,
                                          n_bins:      int = 36) -> dict:
    """
    Compare the distribution of pseudo-torsion angles between
    predicted and NMR ensembles using histogram overlap.

    A score of 1.0 = identical distributions.
    A score of 0.0 = completely different distributions.
    """
    pred_tors = compute_backbone_torsions(pred_coords)["pseudo_torsions"]
    true_tors = compute_backbone_torsions(true_coords)["pseudo_torsions"]

    if pred_tors.shape[1] == 0 or true_tors.shape[1] == 0:
        return {"torsion_overlap": 0.0, "torsion_js_div": 1.0}

    # Histogram overlap per torsion position
    bins   = np.linspace(-np.pi, np.pi, n_bins + 1)
    overlaps = []
    js_divs  = []

    for pos in range(min(pred_tors.shape[1], true_tors.shape[1])):
        p_hist, _ = np.histogram(pred_tors[:, pos], bins=bins, density=True)
        t_hist, _ = np.histogram(true_tors[:, pos], bins=bins, density=True)

        # Normalize to probability distributions
        p_norm = p_hist / (p_hist.sum() + 1e-8)
        t_norm = t_hist / (t_hist.sum() + 1e-8)

        # Histogram intersection (overlap)
        overlap = np.minimum(p_norm, t_norm).sum()
        overlaps.append(overlap)

        # Jensen-Shannon divergence
        m = 0.5 * (p_norm + t_norm)
        eps = 1e-10
        js = 0.5 * (
            np.where(p_norm > 0, p_norm * np.log(p_norm / (m + eps) + eps), 0).sum() +
            np.where(t_norm > 0, t_norm * np.log(t_norm / (m + eps) + eps), 0).sum()
        )
        js_divs.append(float(js))

    return {
        "torsion_overlap":  float(np.mean(overlaps)),
        "torsion_js_div":   float(np.mean(js_divs)),
        "per_pos_overlap":  overlaps,
    }


def compute_rg_distribution(coords: np.ndarray) -> np.ndarray:
    """
    Radius of gyration distribution across conformers.
    coords: (N, L, 3)
    Returns: (N,) Rg values
    """
    center = coords.mean(axis=1, keepdims=True)   # (N, 1, 3)
    rg     = np.sqrt(((coords - center)**2).sum(axis=-1).mean(axis=-1))
    return rg


# ─────────────────────────────────────────────────────────
# LEVEL 3 — CORRELATED MOTION METRICS
# ─────────────────────────────────────────────────────────

def compute_covariance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute CA covariance matrix across ensemble conformers.
    Captures how residues move together.

    coords: (N, L, 3)
    Returns: (L, L) covariance matrix (using mean displacement magnitude)
    """
    N, L, _ = coords.shape
    mean     = coords.mean(axis=0)             # (L, 3)
    delta    = coords - mean[np.newaxis]       # (N, L, 3)

    # Scalar covariance: C_ij = mean over conformers of (|delta_i| * |delta_j|)
    # Signed by dot product direction
    delta_mag = np.sqrt((delta**2).sum(axis=-1))              # (N, L)
    dots  = np.einsum("nid,njd->nij", delta, delta)            # (N, L, L)
    signs = np.where(dots >= 0, 1.0, -1.0)                    # (N, L, L)
    outer = delta_mag[:, :, None] * delta_mag[:, None, :]     # (N, L, L)
    cov   = (signs * outer).mean(axis=0)                       # (L, L)
    return cov


def compute_covariance_similarity(pred_coords: np.ndarray,
                                   true_coords: np.ndarray) -> dict:
    """
    Compare covariance matrices of predicted and NMR ensembles.

    Metrics:
      frobenius_sim:    1 - normalized Frobenius distance (higher = better)
      pearson_r:        correlation of matrix elements
      top_modes_overlap: overlap of top eigenvectors (principal motions)
    """
    # Efficient covariance via numpy (vectorized)
    def fast_cov(coords):
        N, L, _ = coords.shape
        mean  = coords.mean(axis=0)
        delta = coords - mean[np.newaxis]
        # Flatten to (N, L*3) for covariance
        flat  = delta.reshape(N, L * 3)
        C_full= flat.T @ flat / N           # (L*3, L*3)
        # Extract scalar per-residue covariance (L, L)
        # by summing 3x3 blocks
        C = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                block = C_full[i*3:(i+1)*3, j*3:(j+1)*3]
                C[i, j] = np.trace(block)
        return C

    L = min(pred_coords.shape[1], true_coords.shape[1])
    pred_coords = pred_coords[:, :L, :]
    true_coords = true_coords[:, :L, :]

    C_pred = fast_cov(pred_coords)
    C_true = fast_cov(true_coords)

    # Frobenius similarity
    frob_pred = np.linalg.norm(C_pred, 'fro')
    frob_true = np.linalg.norm(C_true, 'fro')
    frob_diff = np.linalg.norm(C_pred - C_true, 'fro')
    frob_sim  = 1 - frob_diff / (frob_pred + frob_true + 1e-8)

    # Pearson correlation of matrix elements
    pearson_r, _ = stats.pearsonr(C_pred.flatten(), C_true.flatten())

    # Top eigenvectors (principal motion directions)
    k = min(5, L)
    evals_p, evecs_p = np.linalg.eigh(C_pred)
    evals_t, evecs_t = np.linalg.eigh(C_true)
    # Top-k eigenvectors (highest eigenvalues)
    top_p = evecs_p[:, -k:]   # (L, k)
    top_t = evecs_t[:, -k:]   # (L, k)
    # Overlap: squared projection
    overlap = np.abs(top_p.T @ top_t)   # (k, k)
    mode_overlap = float(overlap.max(axis=1).mean())

    return {
        "covariance_frobenius_sim":  float(frob_sim),
        "covariance_pearson_r":      float(pearson_r),
        "top_mode_overlap":          mode_overlap,
        "C_pred":                    C_pred,
        "C_true":                    C_true,
    }


def compute_dcc(coords: np.ndarray) -> np.ndarray:
    """
    Dynamic Cross-Correlation matrix.
    DCC_ij measures how correlated the motions of residues i and j are.
    DCC ∈ [-1, 1]: +1=perfectly correlated, -1=anti-correlated

    coords: (N, L, 3)
    Returns: (L, L) DCC matrix
    """
    N, L, _ = coords.shape
    mean  = coords.mean(axis=0)
    delta = coords - mean[np.newaxis]               # (N, L, 3)

    # Magnitude of displacement
    mag   = np.sqrt((delta**2).sum(axis=-1))        # (N, L)
    mag_mean = mag.mean(axis=0)                      # (L,)

    dcc = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            num = (delta[:, i, :] * delta[:, j, :]).sum(axis=-1).mean()
            den = mag_mean[i] * mag_mean[j] + 1e-8
            dcc[i, j] = num / den

    return dcc


# ─────────────────────────────────────────────────────────
# LEVEL 4 — PHYSICAL VALIDITY METRICS
# ─────────────────────────────────────────────────────────

def _dihedral_deg(p1: np.ndarray, p2: np.ndarray,
                   p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """
    Dihedral angle (degrees) for batched 4-point input.
    p1..p4: (..., 3)
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    b2_norm = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    m1 = np.cross(n1, b2_norm)

    x = (n1 * n2).sum(axis=-1)
    y = (m1 * n2).sum(axis=-1)
    return np.degrees(np.arctan2(y, x))


# Richardson et al. 2003 hard-coded rectangular regions (degrees)
# Each entry: (phi_lo, phi_hi, psi_lo, psi_hi)
_RAMA_FAVORED = [
    (-80,  -45,  -60,  -10),  # α-helix core
    (-155, -60,  100,  180),  # β-sheet (positive ψ)
    (-155, -60, -180, -150),  # β-sheet (ψ wraps at ±180)
    (  40,  80,   20,   65),  # left-handed α
    ( -90, -50,  120,  165),  # PPII
]
_RAMA_ALLOWED = [
    (-100, -25,  -75,    5),  # α-helix expanded
    (-175, -45,   85,  180),  # β-sheet expanded (positive ψ)
    (-175, -45, -180, -140),  # β-sheet expanded (wrap)
    (  25, 100,    5,   80),  # left-handed α expanded
    (-130, -50,   40,   90),  # α/β bridge
]


def _in_regions(phi: np.ndarray, psi: np.ndarray, regions: list) -> np.ndarray:
    inside = np.zeros(phi.shape, dtype=bool)
    for phi_lo, phi_hi, psi_lo, psi_hi in regions:
        inside |= ((phi >= phi_lo) & (phi <= phi_hi) &
                   (psi >= psi_lo) & (psi <= psi_hi))
    return inside


def compute_ramachandran_quality(backbone_coords: np.ndarray,
                                  atom_mask: np.ndarray = None) -> dict:
    """
    Ramachandran quality from true φ/ψ backbone torsion angles.

    φ_i = dihedral(C_{i-1}, N_i, CA_i, C_i)
    ψ_i = dihedral(N_i, CA_i, C_i, N_{i+1})

    backbone_coords: (N, L, 4, 3) or (L, 4, 3), atoms ordered [N, CA, C, CB]
    atom_mask:       (N, L, 4) or (L, 4) bool

    Returns fractions in favored / allowed / outlier per Richardson 2003.
    """
    if backbone_coords.ndim == 3:
        backbone_coords = backbone_coords[np.newaxis]
        if atom_mask is not None:
            atom_mask = atom_mask[np.newaxis]

    N, L, _, _ = backbone_coords.shape
    n_coords  = backbone_coords[:, :, 0, :]   # (N, L, 3)
    ca_coords = backbone_coords[:, :, 1, :]
    c_coords  = backbone_coords[:, :, 2, :]

    # atom_mask for N (idx 0) and C (idx 2) presence
    if atom_mask is not None:
        has_n  = atom_mask[:, :, 0].astype(bool)   # (N, L)
        has_ca = atom_mask[:, :, 1].astype(bool)
        has_c  = atom_mask[:, :, 2].astype(bool)
    else:
        has_n  = has_ca = has_c = np.ones((N, L), dtype=bool)

    all_phi, all_psi = [], []

    for conf in range(N):
        n_c   = n_coords[conf]    # (L, 3)
        ca_c  = ca_coords[conf]
        c_c   = c_coords[conf]
        hn    = has_n[conf]
        hca   = has_ca[conf]
        hc    = has_c[conf]

        for i in range(1, L - 1):
            # φ_i: C_{i-1}, N_i, CA_i, C_i
            if hc[i-1] and hn[i] and hca[i] and hc[i]:
                phi = _dihedral_deg(c_c[i-1], n_c[i], ca_c[i], c_c[i])
            else:
                continue

            # ψ_i: N_i, CA_i, C_i, N_{i+1}
            if hn[i] and hca[i] and hc[i] and hn[i+1]:
                psi = _dihedral_deg(n_c[i], ca_c[i], c_c[i], n_c[i+1])
            else:
                continue

            all_phi.append(phi)
            all_psi.append(psi)

    if not all_phi:
        return {
            "rama_favored_frac":  1.0,
            "rama_allowed_frac":  1.0,
            "rama_outlier_frac":  0.0,
            "rama_n_residues":    0,
        }

    phi_arr = np.array(all_phi)
    psi_arr = np.array(all_psi)

    favored = _in_regions(phi_arr, psi_arr, _RAMA_FAVORED)
    allowed = _in_regions(phi_arr, psi_arr, _RAMA_ALLOWED) & ~favored
    outlier = ~favored & ~allowed

    n = len(phi_arr)
    return {
        "rama_favored_frac": float(favored.mean()),
        "rama_allowed_frac": float((favored | allowed).mean()),
        "rama_outlier_frac": float(outlier.mean()),
        "rama_n_residues":   n,
    }


def compute_bond_length_quality(ca_coords: np.ndarray) -> dict:
    """
    Check virtual CA-CA bond lengths.
    Ideal: 3.8 Å ± 0.1 Å

    ca_coords: (N, L, 3)
    """
    N, L, _ = ca_coords.shape
    bond_vecs = ca_coords[:, 1:] - ca_coords[:, :-1]  # (N, L-1, 3)
    bond_lens = np.linalg.norm(bond_vecs, axis=-1)     # (N, L-1)

    ideal     = 3.8
    deviation = np.abs(bond_lens - ideal)

    return {
        "bond_len_mean":    float(bond_lens.mean()),
        "bond_len_std":     float(bond_lens.std()),
        "bond_len_mae":     float(deviation.mean()),
        "bond_len_outliers": float((deviation > 0.5).mean()),
                                   # fraction > 0.5 Å from ideal
    }


def compute_clash_score(ca_coords: np.ndarray,
                          clash_threshold: float = 3.5) -> dict:
    """
    Count severe CA-CA clashes (non-bonded distance < threshold).

    Reports per-100-residues (MolProbity convention) averaged across
    all conformers, as well as raw counts and fraction.

    ca_coords: (N, L, 3)
    """
    N, L, _ = ca_coords.shape
    total_clashes = 0
    total_pairs   = 0

    for n in range(N):
        ca   = ca_coords[n]        # (L, 3)
        dmat = cdist(ca, ca)       # (L, L)

        # Exclude diagonal and sequential bonded pairs
        nonbonded = np.ones((L, L), dtype=bool)
        np.fill_diagonal(nonbonded, False)
        idx = np.arange(L - 1)
        nonbonded[idx, idx + 1] = False
        nonbonded[idx + 1, idx] = False

        total_clashes += int((dmat[nonbonded] < clash_threshold).sum())
        total_pairs   += int(nonbonded.sum())

    # Per-100-residues: total_clashes / N_conformers / L * 100
    clash_per100 = total_clashes / N / L * 100 if L > 0 else 0.0

    return {
        "clash_score_per100": float(clash_per100),           # MolProbity convention
        "clash_fraction":     float(total_clashes / (total_pairs + 1)),
        "total_clashes":      int(total_clashes),
    }


# ─────────────────────────────────────────────────────────
# COMBINED EVALUATION
# ─────────────────────────────────────────────────────────

def evaluate_ensemble(pred_coords: np.ndarray,
                       true_coords: np.ndarray,
                       mask:        np.ndarray = None,
                       pdb_id:      str = "") -> dict:
    """
    Run full evaluation suite comparing predicted vs NMR ensemble.

    Args:
        pred_coords: (N_pred, L, 3)  generated CA coordinates
        true_coords: (N_true, L, 3)  NMR CA coordinates
        mask:        (L,) bool       True for real residues
        pdb_id:      str             for logging

    Returns:
        dict with all metrics organized by level
    """
    # Align all to same reference
    ref = true_coords[0]
    pred_aligned = np.array([kabsch_align(p, ref)[0] for p in pred_coords])
    true_aligned = np.array([kabsch_align(t, ref)[0] for t in true_coords])

    if mask is not None:
        pred_m = pred_aligned[:, mask, :]
        true_m = true_aligned[:, mask, :]
    else:
        pred_m = pred_aligned
        true_m = true_aligned

    results = {"pdb_id": pdb_id}

    # ── Level 1: Structural ──
    struct = compute_ensemble_rmsd(pred_m, true_m)
    results.update({
        "coverage_rmsd":      struct["coverage_rmsd"],
        "precision_rmsd":     struct["precision_rmsd"],
        "mean_pairwise_rmsd": struct["mean_pairwise_rmsd"],
        "mean_struct_rmsd":   struct["mean_struct_rmsd"],
    })

    # Coverage recall/precision at explicit RMSD thresholds (P2-2)
    cov = ensemble_coverage_at_threshold(pred_m, true_m)
    for k, v in cov.items():
        if k != "coverage_pairwise_rmsd":
            results[k] = v

    # TM-score and GDT on mean structures
    pred_mean = pred_m.mean(axis=0)
    true_mean = true_m.mean(axis=0)
    results["tm_score"]   = compute_tm_score(pred_mean, true_mean)
    results["gdt_ts"]     = compute_gdt_ts(pred_mean, true_mean)

    # ── Level 2: Conformational distribution ──
    rmsf = compute_rmsf_correlation(pred_m, true_m)
    results.update({
        "rmsf_pearson_r":  rmsf["rmsf_pearson_r"],
        "rmsf_spearman_r": rmsf["rmsf_spearman_r"],
        "rmsf_mae":        rmsf["rmsf_mae"],
        "rmsf_pred":       rmsf["rmsf_pred"],
        "rmsf_true":       rmsf["rmsf_true"],
    })

    tors = compute_torsion_distribution_overlap(pred_m, true_m)
    results.update({
        "torsion_overlap": tors["torsion_overlap"],
        "torsion_js_div":  tors["torsion_js_div"],
    })

    rg_pred = compute_rg_distribution(pred_m)
    rg_true = compute_rg_distribution(true_m)
    results["rg_mean_pred"] = float(rg_pred.mean())
    results["rg_mean_true"] = float(rg_true.mean())
    results["rg_std_pred"]  = float(rg_pred.std())
    results["rg_std_true"]  = float(rg_true.std())

    # ── Level 3: Correlated motions ──
    cov = compute_covariance_similarity(pred_m, true_m)
    results.update({
        "covariance_frobenius_sim": cov["covariance_frobenius_sim"],
        "covariance_pearson_r":     cov["covariance_pearson_r"],
        "top_mode_overlap":         cov["top_mode_overlap"],
    })

    # ── Level 4: Physical validity ──
    bond = compute_bond_length_quality(pred_aligned)
    results.update({
        "bond_len_mean":    bond["bond_len_mean"],
        "bond_len_mae":     bond["bond_len_mae"],
        "bond_len_outliers":bond["bond_len_outliers"],
    })

    clash = compute_clash_score(pred_aligned)
    results.update({
        "clash_score_per100": clash["clash_score_per100"],
        "clash_fraction":     clash["clash_fraction"],
        "total_clashes":      clash["total_clashes"],
    })

    return results
