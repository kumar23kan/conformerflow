"""
ConformerFlow — Phase 2: Backbone Frame Builder
Constructs SE(3)-invariant local reference frames from N, CA, C, CB atoms.

Each residue gets a rotation matrix R (3x3) and translation t (3,)
defining its local coordinate frame. All downstream geometry is
expressed relative to this frame, making the model invariant to
global rotations and translations.

Frame construction follows AlphaFold2 convention:
  - Origin: CA position
  - x-axis: CA → C direction (normalized)
  - y-axis: in the N-CA-C plane, perpendicular to x
  - z-axis: x cross y (completes right-handed frame)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Atom index constants (matches dataset.py ordering)
IDX_N  = 0
IDX_CA = 1
IDX_C  = 2
IDX_CB = 3


# ──────────────────────────────────────────────
# Frame Construction
# ──────────────────────────────────────────────

def build_backbone_frames(coords: torch.Tensor,
                          mask:   torch.Tensor) -> tuple:
    """
    Build SE(3)-invariant local backbone frames from N, CA, C atoms.

    Args:
        coords: (B, M, L, 4, 3)  — N/CA/C/CB coordinates
        mask:   (B, L, 4)        — atom existence mask

    Returns:
        R: (B, M, L, 3, 3)  — rotation matrices (local frame axes as rows)
        t: (B, M, L, 3)     — translations (CA positions)
        frame_mask: (B, M, L) — True where frame is valid (all 3 backbone atoms present)
    """
    B, M, L, _, _ = coords.shape

    # Extract backbone atoms
    N_pos  = coords[..., IDX_N,  :]   # (B, M, L, 3)
    CA_pos = coords[..., IDX_CA, :]   # (B, M, L, 3)
    C_pos  = coords[..., IDX_C,  :]   # (B, M, L, 3)

    # Translation = CA position
    t = CA_pos  # (B, M, L, 3)

    # Build orthonormal frame
    # v1: CA → C
    v1 = C_pos - CA_pos                          # (B, M, L, 3)
    # v2: CA → N
    v2 = N_pos - CA_pos                          # (B, M, L, 3)

    # Gram-Schmidt orthonormalization
    e1 = F.normalize(v1, dim=-1, eps=1e-8)       # x-axis
    u2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1, eps=1e-8)       # y-axis
    e3 = torch.cross(e1, e2, dim=-1)             # z-axis (right-handed)
    e3 = F.normalize(e3, dim=-1, eps=1e-8)

    # Stack into rotation matrix R: rows are frame axes
    # R shape: (B, M, L, 3, 3)
    R = torch.stack([e1, e2, e3], dim=-2)        # (B, M, L, 3, 3)

    # Frame validity: need N, CA, C all present
    # mask shape: (B, L, 4)
    backbone_mask = mask[..., IDX_N] & mask[..., IDX_CA] & mask[..., IDX_C]  # (B, L)
    frame_mask    = backbone_mask.unsqueeze(1).expand(B, M, L)                 # (B, M, L)

    return R, t, frame_mask


def coords_to_local_frame(coords:    torch.Tensor,
                           R:        torch.Tensor,
                           t:        torch.Tensor) -> torch.Tensor:
    """
    Express all atom coordinates in the local frame of each residue.
    Used to compute SE(3)-invariant atom position features.

    Args:
        coords: (B, M, L, 4, 3)  — global atom coordinates
        R:      (B, M, L, 3, 3)  — rotation matrices
        t:      (B, M, L, 3)     — translations (CA positions)

    Returns:
        local_coords: (B, M, L, L, 4, 3)
            For each residue i, the coordinates of all residues j
            expressed in residue i's local frame.
            This is the core of invariant point attention.
    """
    B, M, L, A, _ = coords.shape

    # Translate all coords relative to each residue's CA
    # t_i: (B, M, L, 3) → broadcast to (B, M, L, L, 3)
    t_expand  = t.unsqueeze(3).expand(B, M, L, L, 3)       # (B, M, L_i, L_j, 3)
    c_expand  = coords.unsqueeze(2).expand(B, M, L, L, A, 3) # (B, M, L_i, L_j, 4, 3)

    # Translate: subtract residue i's CA from residue j's atoms
    c_centered = c_expand - t_expand.unsqueeze(-2)           # (B, M, L, L, 4, 3)

    # Rotate into local frame using R_i
    # R: (B, M, L, 3, 3) → (B, M, L_i, 1, 1, 3, 3)
    R_expand = R.unsqueeze(3).unsqueeze(4).expand(B, M, L, L, A, 3, 3)

    # Apply rotation: local = R @ (global - t)
    # c_centered: (B, M, L, L, A, 3) → unsqueeze for matmul
    c_vec    = c_centered.unsqueeze(-1)                      # (B, M, L, L, A, 3, 1)
    local    = (R_expand @ c_vec).squeeze(-1)                # (B, M, L, L, A, 3)

    return local


def interresidue_distances(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute Cα-Cα pairwise distance matrix.
    Works with any leading dims (..., L, 4, 3).
    Returns: (..., L, L)
    """
    ca = coords[..., IDX_CA, :]                              # (..., L, 3)
    diff = ca.unsqueeze(-2) - ca.unsqueeze(-3)               # (B, M, L, L, 3)
    dists = torch.norm(diff, dim=-1)                         # (B, M, L, L)
    return dists


def cb_direction_features(coords: torch.Tensor,
                           R:     torch.Tensor,
                           mask:  torch.Tensor) -> torch.Tensor:
    """
    Compute CB direction in the local frame.
    Works with any leading dims: coords (..., L, 4, 3), R (..., L, 3, 3)

    Returns: (..., L, 3)
    """
    CA_pos = coords[..., IDX_CA, :]              # (..., L, 3)
    CB_pos = coords[..., IDX_CB, :]              # (..., L, 3)

    cb_vec  = CB_pos - CA_pos                    # (..., L, 3)
    cb_vec_expanded = cb_vec.unsqueeze(-1)        # (..., L, 3, 1)
    cb_local        = (R @ cb_vec_expanded).squeeze(-1)   # (..., L, 3)
    cb_dir          = F.normalize(cb_local, dim=-1, eps=1e-8)

    # Zero out where CB missing — mask[..., IDX_CB]: (..., L)
    cb_mask = mask[..., IDX_CB].unsqueeze(-1).float()  # (..., L, 1)
    cb_dir  = cb_dir * cb_mask

    return cb_dir


# ──────────────────────────────────────────────
# Relative Frame Encoding
# ──────────────────────────────────────────────

def relative_frame_encoding(R: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
    """
    Encode relative transformation between all residue pairs.
    Works with either (B, L, 3, 3) or (B, M, L, 3, 3) inputs.

    Returns: (..., L, L, 12)
    """
    # Flatten to (..., L, 3, 3) — works for both 4D and 5D inputs
    shape = R.shape  # (..., L, 3, 3)
    L = shape[-3]

    R_i_T = R.transpose(-1, -2)                            # (..., L, 3, 3)

    # Pairwise expansion
    R_i_T_exp = R_i_T.unsqueeze(-3).expand(*shape[:-3], L, L, 3, 3)
    R_j_exp   = R.unsqueeze(-4).expand(*shape[:-3], L, L, 3, 3)
    t_i_exp   = t.unsqueeze(-2).expand(*t.shape[:-2], L, L, 3)
    t_j_exp   = t.unsqueeze(-3).expand(*t.shape[:-2], L, L, 3)

    R_rel      = torch.matmul(R_i_T_exp, R_j_exp)          # (..., L, L, 3, 3)
    R_rel_flat = R_rel.reshape(*shape[:-3], L, L, 9)        # (..., L, L, 9)

    dt    = (t_j_exp - t_i_exp).unsqueeze(-1)               # (..., L, L, 3, 1)
    t_rel = torch.matmul(R_i_T_exp, dt).squeeze(-1)         # (..., L, L, 3)

    return torch.cat([R_rel_flat, t_rel], dim=-1)           # (..., L, L, 12)


# ──────────────────────────────────────────────
# Frame Module (nn.Module wrapper)
# ──────────────────────────────────────────────

class BackboneFrameModule(nn.Module):
    """
    Wraps all frame computation into a single nn.Module.
    Computes all SE(3)-invariant geometric features from raw coordinates.
    """

    def __init__(self,
                 d_model:       int = 256,
                 max_dist:      float = 32.0,
                 n_dist_bins:   int = 64):
        super().__init__()
        self.max_dist    = max_dist
        self.n_dist_bins = n_dist_bins

        # Distance bin edges for RBF encoding
        self.register_buffer(
            "dist_bins",
            torch.linspace(0, max_dist, n_dist_bins)
        )

        # Project geometric features to d_model
        # Input dim: 3 (CB dir) + 12 (rel frame, only diagonal used) + n_dist_bins
        self.geom_proj = nn.Linear(3 + n_dist_bins, d_model)

    def rbf_encode_distances(self, dists: torch.Tensor) -> torch.Tensor:
        """
        Radial basis function encoding of distances.
        Args:
            dists: (B, M, L, L)
        Returns:
            rbf: (B, M, L, L, n_dist_bins)
        """
        gamma = 2.0 / (self.max_dist / self.n_dist_bins) ** 2
        # bins broadcast works for any leading dims: dists (..., L, L) -> rbf (..., L, L, K)
        bins  = self.dist_bins.view(*([1] * dists.dim()), -1)  # dynamic broadcast
        d_exp = dists.unsqueeze(-1)                             # (..., L, L, 1)
        rbf   = torch.exp(-gamma * (d_exp - bins) ** 2)        # (..., L, L, K)
        return rbf

    def forward(self, coords: torch.Tensor,
                      mask:   torch.Tensor) -> dict:
        """
        Args:
            coords: (N, L, 4, 3)  — N = B*M flattened
            mask:   (N, L, 4)

        Returns dict with SE(3)-invariant geometric features.
        """
        N, L, _, _ = coords.shape

        # Build frames directly from 4D coords
        # Temporarily add fake M dim for build_backbone_frames
        coords_5d = coords.unsqueeze(1)                    # (N, 1, L, 4, 3)
        mask_4d   = mask                                   # (N, L, 4)

        R5, t5, fm5 = build_backbone_frames(coords_5d, mask_4d)
        R          = R5.squeeze(1)                         # (N, L, 3, 3)
        t          = t5.squeeze(1)                         # (N, L, 3)
        frame_mask = fm5.squeeze(1)                        # (N, L)

        # CB direction in local frame: (N, L, 3)
        cb_dir = cb_direction_features(coords, R, mask)

        # Pairwise CA distances → RBF encoding: (N, L, L, K)
        dists    = interresidue_distances(coords)
        dist_rbf = self.rbf_encode_distances(dists)

        # Relative frame encodings: (N, L, L, 12)
        rel_frames = relative_frame_encoding(R, t)

        # Per-residue node geometry: CB dir (3) + mean distance profile (K)
        mean_dist_profile = dist_rbf.mean(dim=-2)          # (N, L, K)
        node_geom = torch.cat([cb_dir, mean_dist_profile], dim=-1)  # (N, L, 3+K)
        node_geom = self.geom_proj(node_geom)              # (N, L, d_model)

        return {
            "R":          R,
            "t":          t,
            "frame_mask": frame_mask,
            "cb_dir":     cb_dir,
            "dist_rbf":   dist_rbf,
            "rel_frames": rel_frames,
            "node_geom":  node_geom,
        }
