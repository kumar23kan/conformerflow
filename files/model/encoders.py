"""
ConformerFlow — Structure Encoders
All encoders share the same forward signature:
    forward(coords, one_hot, mask, seq_mask) -> (B, M, L, d_model)

Where M = number of reference conformers in the batch item (usually 1 for
inference, variable for training on NMR ensembles).

After setup (step 1) this file lives at model/encoders.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── Shared building blocks ────────────────────────────────────────────────────

class _FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))


class _AttnLayer(nn.Module):
    """Standard multi-head self-attention + FFN transformer layer."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = _FFN(d_model, d_ff, dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (B, L, d_model)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        return self.ffn(x)


class _TransformerStack(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            _AttnLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


# ── SE(3) Frames Encoder ─────────────────────────────────────────────────────

class SE3FramesEncoder(nn.Module):
    """
    Wraps the existing StructureEncoder (IPA-based, SE(3)-equivariant).
    StructureEncoder already handles the (B, M, L, 4, 3) shape internally,
    so this is a thin cfg-unpacking wrapper.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get("model", {})
        seq_enc = cfg.get("representation", {}).get("sequence", "onehot")
        use_esm2 = seq_enc in ("esm2_650m", "esm2_3b", "prot_t5")
        from model.encoder import StructureEncoder
        self.enc = StructureEncoder(
            d_model  = m.get("d_model",          256),
            n_layers = m.get("n_encoder_layers",   4),
            n_heads  = m.get("n_heads",             8),
            n_points = m.get("n_points",            4),
            d_ff     = m.get("d_ff",              512),
            dropout  = m.get("dropout",           0.1),
            use_esm2 = use_esm2,
        )

    def forward(self,
                coords:   torch.Tensor,   # (B, M, L, 4, 3)
                one_hot:  torch.Tensor,   # (B, L, 20)
                mask:     torch.Tensor,   # (B, L, 4)  bool
                seq_mask: torch.Tensor,   # (B, L)     bool
                **_kw) -> torch.Tensor:   # → (B, M, L, d_model)
        # StructureEncoder.forward already handles (B, M, L, 4, 3) → (B, M, L, d)
        return self.enc(coords, one_hot, mask, seq_mask)


# ── Cartesian Encoder ─────────────────────────────────────────────────────────

class CartesianEncoder(nn.Module):
    """
    Cα Cartesian coordinates (x, y, z) centred by mean, sinusoidal positional
    encoding added, then standard transformer. NOT equivariant — orientation
    matters. Simple, fast, no geometric priors.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m   = cfg.get("model", {})
        d   = m.get("d_model",      256)
        n_l = m.get("n_encoder_layers", 4)
        nh  = m.get("n_heads",       8)
        dff = m.get("d_ff",         512)
        dp  = m.get("dropout",       0.1)

        seq_dim = 20
        self.in_proj = nn.Linear(3 + seq_dim, d)
        self.pos_enc = _SinusoidalPE(d)
        self.transformer = _TransformerStack(n_l, d, nh, dff, dp)

    def forward(self, coords, one_hot, mask, seq_mask, **_kw):
        B, M, L, _, _ = coords.shape
        ca = coords[:, :, :, 1, :]          # (B, M, L, 3)  — Cα only

        # Centre each conformer independently
        ca_valid = ca * seq_mask[:, None, :, None]
        n_valid  = seq_mask.sum(-1, keepdim=True).float().clamp(min=1)  # (B, 1)
        centre   = ca_valid.sum(-2) / n_valid.unsqueeze(-1)              # (B, M, 3)
        ca       = ca - centre.unsqueeze(2)

        oh_exp = one_hot.unsqueeze(1).expand(-1, M, -1, -1)  # (B, M, L, 20)
        x = torch.cat([ca, oh_exp], dim=-1)                   # (B, M, L, 23)

        x_flat  = x.view(B * M, L, x.shape[-1])
        pad_key = ~seq_mask.unsqueeze(1).expand(-1, M, -1).reshape(B * M, L)

        h = self.in_proj(x_flat)
        h = self.pos_enc(h)
        h = self.transformer(h, key_padding_mask=pad_key)     # (B*M, L, d)

        return h.view(B, M, L, -1)


# ── Distance Matrix Encoder ───────────────────────────────────────────────────

class DistanceEncoder(nn.Module):
    """
    Pairwise Cα distance matrix encoded with radial basis functions → row-wise
    summary pooled into per-residue features → transformer.
    SE(3)-invariant but loses chirality information.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m   = cfg.get("model", {})
        d   = m.get("d_model",      256)
        n_l = m.get("n_encoder_layers", 4)
        nh  = m.get("n_heads",       8)
        dff = m.get("d_ff",         512)
        dp  = m.get("dropout",       0.1)

        self.n_rbf     = 64
        self.d_max     = 40.0    # Å — beyond this all RBF responses ≈ 0
        self.rbf_means = nn.Parameter(
            torch.linspace(0, self.d_max, self.n_rbf), requires_grad=False)
        self.rbf_gamma = (self.d_max / self.n_rbf) ** 2

        # Row-summary: max over j of RBF(d_ij) → per residue feature
        seq_dim = 20
        self.in_proj = nn.Linear(self.n_rbf + seq_dim, d)
        self.pos_enc = _SinusoidalPE(d)
        self.transformer = _TransformerStack(n_l, d, nh, dff, dp)

    def _rbf(self, dists):
        # dists: (...), means: (n_rbf,)
        return torch.exp(-((dists.unsqueeze(-1) - self.rbf_means) ** 2) / self.rbf_gamma)

    def forward(self, coords, one_hot, mask, seq_mask, **_kw):
        B, M, L, _, _ = coords.shape
        ca = coords[:, :, :, 1, :]          # (B, M, L, 3)

        # Pairwise distances (B, M, L, L)
        diff  = ca.unsqueeze(3) - ca.unsqueeze(2)       # (B, M, L, L, 3)
        dists = diff.norm(dim=-1)                        # (B, M, L, L)

        # RBF encode → (B, M, L, L, n_rbf)
        rbf = self._rbf(dists)

        # Mask padding positions then take max over neighbour dim (j)
        valid = seq_mask[:, None, :, None] & seq_mask[:, None, None, :]  # (B, M, L, L)
        rbf   = rbf * valid.unsqueeze(-1).float()

        row_feat = rbf.max(dim=3).values   # (B, M, L, n_rbf)  — row-max summary

        oh_exp = one_hot.unsqueeze(1).expand(-1, M, -1, -1)
        x      = torch.cat([row_feat, oh_exp], dim=-1)   # (B, M, L, n_rbf+20)

        x_flat  = x.view(B * M, L, x.shape[-1])
        pad_key = ~seq_mask.unsqueeze(1).expand(-1, M, -1).reshape(B * M, L)

        h = self.in_proj(x_flat)
        h = self.pos_enc(h)
        h = self.transformer(h, key_padding_mask=pad_key)

        return h.view(B, M, L, -1)


# ── Torsion Encoder ───────────────────────────────────────────────────────────

class TorsionEncoder(nn.Module):
    """
    Backbone torsion angles (φ, ψ, ω) encoded as (sin, cos) pairs.
    Invariant to global rotation/translation. Very compact input.
    Vectorised — no Python loops over residues.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m   = cfg.get("model", {})
        d   = m.get("d_model",      256)
        n_l = m.get("n_encoder_layers", 4)
        nh  = m.get("n_heads",       8)
        dff = m.get("d_ff",         512)
        dp  = m.get("dropout",       0.1)

        # 3 angles × 2 (sin/cos) + 20 one-hot = 26 features per residue
        torsion_dim = 6
        seq_dim     = 20
        self.in_proj = nn.Linear(torsion_dim + seq_dim, d)
        self.pos_enc = _SinusoidalPE(d)
        self.transformer = _TransformerStack(n_l, d, nh, dff, dp)

    @staticmethod
    def _dihedral(a, b, c, d_):
        """
        Vectorised dihedral angle for atom sequences a-b-c-d.
        All inputs: (..., 3). Returns: (...,) in [-pi, pi].
        """
        b1 = b - a
        b2 = c - b
        b3 = d_ - c
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)
        b2n = F.normalize(b2, dim=-1)
        m1  = torch.linalg.cross(n1, b2n)
        x   = (n1 * n2).sum(-1)
        y   = (m1 * n2).sum(-1)
        return torch.atan2(y, x)

    def _compute_torsions(self, coords, seq_mask):
        """
        coords:   (B, M, L, 4, 3)
        seq_mask: (B, L)
        Returns:  (B, M, L, 6)  — sin/cos of phi, psi, omega; 0 at boundaries
        """
        B, M, L, _, _ = coords.shape

        N  = coords[:, :, :, 0, :]   # (B, M, L, 3)
        CA = coords[:, :, :, 1, :]
        C  = coords[:, :, :, 2, :]

        # phi:   C(i-1)–N(i)–CA(i)–C(i)   — defined for residues 1..L-1
        phi = self._dihedral(C[:, :, :-1], N[:, :, 1:], CA[:, :, 1:], C[:, :, 1:])
        # psi:   N(i)–CA(i)–C(i)–N(i+1)  — defined for residues 0..L-2
        psi = self._dihedral(N[:, :, :-1], CA[:, :, :-1], C[:, :, :-1], N[:, :, 1:])
        # omega: CA(i-1)–C(i-1)–N(i)–CA(i) — defined for residues 1..L-1
        omega = self._dihedral(CA[:, :, :-1], C[:, :, :-1], N[:, :, 1:], CA[:, :, 1:])

        # Pad so each has length L (first residue has no phi/omega, last has no psi)
        pad = torch.zeros(B, M, 1, device=coords.device, dtype=coords.dtype)
        phi   = torch.cat([pad, phi],   dim=2)   # (B, M, L)
        omega = torch.cat([pad, omega], dim=2)
        psi   = torch.cat([psi,   pad], dim=2)

        torsions = torch.stack([
            phi.sin(), phi.cos(),
            psi.sin(), psi.cos(),
            omega.sin(), omega.cos(),
        ], dim=-1)   # (B, M, L, 6)

        # Zero out invalid residues
        valid = seq_mask[:, None, :, None].float()  # (B, 1, L, 1)
        return torsions * valid

    def forward(self, coords, one_hot, mask, seq_mask, **_kw):
        B, M, L, _, _ = coords.shape

        torsions = self._compute_torsions(coords, seq_mask)   # (B, M, L, 6)
        oh_exp   = one_hot.unsqueeze(1).expand(-1, M, -1, -1)
        x        = torch.cat([torsions, oh_exp], dim=-1)      # (B, M, L, 26)

        x_flat  = x.view(B * M, L, x.shape[-1])
        pad_key = ~seq_mask.unsqueeze(1).expand(-1, M, -1).reshape(B * M, L)

        h = self.in_proj(x_flat)
        h = self.pos_enc(h)
        h = self.transformer(h, key_padding_mask=pad_key)

        return h.view(B, M, L, -1)


# ── Positional encoding ───────────────────────────────────────────────────────

class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_encoder(cfg: dict) -> nn.Module:
    kind = cfg.get("representation", {}).get("structure", "se3_frames")
    if kind == "se3_frames":
        return SE3FramesEncoder(cfg)
    elif kind == "cartesian":
        return CartesianEncoder(cfg)
    elif kind == "distance_matrix":
        return DistanceEncoder(cfg)
    elif kind == "torsion_angles":
        return TorsionEncoder(cfg)
    else:
        raise ValueError(
            f"Unknown structure representation '{kind}'. "
            f"Choose from: se3_frames, cartesian, distance_matrix, torsion_angles"
        )
