"""
ConformerFlow — Phase 2: SE(3)-Invariant Encoder
Combines backbone frames + sequence embeddings + IPA-style attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.frames import BackboneFrameModule


# ──────────────────────────────────────────────
# Sequence Encoder
# ──────────────────────────────────────────────

class SequenceEncoder(nn.Module):
    """One-hot + optional ESM-2 sequence encoder."""

    ESM2_DIM = 1280

    def __init__(self, d_model: int = 256, use_esm2: bool = True):
        super().__init__()
        self.use_esm2 = use_esm2
        self.d_model  = d_model

        self.onehot_proj = nn.Sequential(
            nn.Linear(20, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        if use_esm2:
            self.esm2_proj = nn.Sequential(
                nn.Linear(self.ESM2_DIM, d_model), nn.LayerNorm(d_model), nn.GELU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
            )
            self._esm_model = self._esm_alphabet = None

    def load_esm2(self):
        if self._esm_model is None:
            try:
                import esm
                self._esm_model, self._esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self._esm_model.eval()
                for p in self._esm_model.parameters():
                    p.requires_grad = False
            except ImportError:
                self.use_esm2 = False

    @torch.no_grad()
    def get_esm2_embeddings(self, sequences, device):
        self.load_esm2()
        if not self.use_esm2:
            return None
        batch_converter = self._esm_alphabet.get_batch_converter()
        data = [(f"s{i}", s) for i, s in enumerate(sequences)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        esm_model = self._esm_model.to(device)
        results   = esm_model(tokens, repr_layers=[33])
        return results["representations"][33][:, 1:-1, :]

    def forward(self, one_hot, sequences=None, esm2_emb=None):
        oh = self.onehot_proj(one_hot)
        if not self.use_esm2:
            return oh
        if esm2_emb is None and sequences is not None:
            esm2_emb = self.get_esm2_embeddings(sequences, one_hot.device)
        if esm2_emb is not None:
            L = one_hot.shape[1]
            if esm2_emb.shape[1] > L:
                esm2_emb = esm2_emb[:, :L]
            elif esm2_emb.shape[1] < L:
                pad = torch.zeros(*esm2_emb.shape[:-2], L - esm2_emb.shape[1],
                                  self.ESM2_DIM, device=one_hot.device)
                esm2_emb = torch.cat([esm2_emb, pad], dim=-2)
            esm = self.esm2_proj(esm2_emb)
            return self.fusion(torch.cat([oh, esm], dim=-1))
        return oh


# ──────────────────────────────────────────────
# Invariant Point Attention
# ──────────────────────────────────────────────

class InvariantPointAttention(nn.Module):
    """
    Simplified SE(3)-invariant attention.
    Computes attention from scalar features + 3D point distances in local frames.
    Both contributions are rotation/translation invariant.
    """

    def __init__(self, d_model=256, n_heads=8, n_points=4, dropout=0.1):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_points = n_points
        self.d_head   = d_model // n_heads

        # Scalar attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Point projections: each head gets n_points 3D query/key points
        self.q_pt = nn.Linear(d_model, n_heads * n_points * 3, bias=False)
        self.k_pt = nn.Linear(d_model, n_heads * n_points * 3, bias=False)
        self.v_pt = nn.Linear(d_model, n_heads * n_points * 3, bias=False)

        # Pair bias from relative frame features
        self.pair_bias = nn.Linear(12, n_heads, bias=False)

        # Learnable point weight
        self.log_pt_w = nn.Parameter(torch.zeros(n_heads))

        # Output: scalar values + point values (norms) + point coords
        out_dim = d_model + n_heads * n_points + n_heads * n_points * 3
        self.out_proj = nn.Linear(out_dim, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, h, R, t, rel_frames, seq_mask):
        """
        h:          (N, L, d_model)
        R:          (N, L, 3, 3)
        t:          (N, L, 3)
        rel_frames: (N, L, L, 12)
        seq_mask:   (N, L) bool
        Returns:    (N, L, d_model)
        """
        N, L, D = h.shape
        H = self.n_heads
        P = self.n_points
        d = self.d_head

        # ── Scalar QKV ──
        Q = self.q_proj(h).view(N, L, H, d)   # (N, L, H, d)
        K = self.k_proj(h).view(N, L, H, d)
        V = self.v_proj(h).view(N, L, H, d)

        # Scalar attention logits: (N, H, L, L)
        a_s = torch.einsum("nihd,njhd->nhij", Q, K) / (d ** 0.5)

        # ── Point QKV in global frame ──
        # Project to 3D points and rotate into global frame
        Qp = self.q_pt(h).view(N, L, H, P, 3)   # (N, L, H, P, 3)
        Kp = self.k_pt(h).view(N, L, H, P, 3)
        Vp = self.v_pt(h).view(N, L, H, P, 3)

        # Global frame: p_global = R @ p_local + t
        # R: (N, L, 3, 3), p: (N, L, H, P, 3)
        # Use einsum: global = sum_d R[n,l,d1,d2] * p[n,l,h,p,d2]
        def to_global(pts, R, t):
            # pts: (N, L, H, P, 3), R: (N, L, 3, 3), t: (N, L, 3)
            g = torch.einsum("nlij,nlhpj->nlhpi", R, pts)   # (N, L, H, P, 3)
            g = g + t[:, :, None, None, :]                   # broadcast t
            return g

        Qg = to_global(Qp, R, t)   # (N, L, H, P, 3)
        Kg = to_global(Kp, R, t)
        Vg = to_global(Vp, R, t)

        # Pairwise point distances: (N, H, L, L)
        # Qg[i] - Kg[j] for all i,j
        Qg_e = Qg.permute(0,2,1,3,4).unsqueeze(3)   # (N, H, L, 1, P, 3)
        Kg_e = Kg.permute(0,2,1,3,4).unsqueeze(2)   # (N, H, 1, L, P, 3)
        pt_d2 = ((Qg_e - Kg_e)**2).sum(dim=-1).sum(dim=-1)  # (N, H, L, L)

        w_pt = F.softplus(self.log_pt_w).view(1, H, 1, 1)
        a_pt = -0.5 * w_pt * pt_d2                           # (N, H, L, L)

        # ── Pair bias ──
        pb = self.pair_bias(rel_frames)              # (N, L, L, H)
        pb = pb.permute(0, 3, 1, 2)                  # (N, H, L, L)

        # ── Combined logits + mask ──
        logits = a_s + a_pt + pb                     # (N, H, L, L)
        if seq_mask is not None:
            pad = (~seq_mask).float() * -1e9         # (N, L)
            logits = logits + pad[:, None, None, :]  # broadcast over H, L_i

        attn = F.softmax(logits, dim=-1)             # (N, H, L, L)
        attn = self.dropout(attn)

        # ── Aggregate scalar values ──
        out_s = torch.einsum("nhij,njhd->nihd", attn, V)   # (N, L, H, d)
        out_s = out_s.reshape(N, L, D)                      # (N, L, D)

        # ── Aggregate point values in global frame ──
        # Vg: (N, L, H, P, 3) → permute → (N, H, L, P, 3)
        Vg_p  = Vg.permute(0, 2, 1, 3, 4)                  # (N, H, L, P, 3)
        attn_e= attn.unsqueeze(-1).unsqueeze(-1)            # (N, H, L, L, 1, 1)
        Vg_agg= (attn_e * Vg_p.unsqueeze(2)).sum(dim=3)    # (N, H, L, P, 3)

        # Bring to local frame: p_local = R^T @ (p_global - t)
        Vg_agg_p = Vg_agg.permute(0, 2, 1, 3, 4)           # (N, L, H, P, 3)
        Vg_cent  = Vg_agg_p - t[:, :, None, None, :]        # subtract CA
        R_T      = R.transpose(-1, -2)                      # (N, L, 3, 3)
        Vl       = torch.einsum("nlij,nlhpj->nlhpi", R_T, Vg_cent)  # (N, L, H, P, 3)

        out_pt    = Vl.reshape(N, L, H * P * 3)             # (N, L, H*P*3)
        out_norms = Vl.norm(dim=-1).reshape(N, L, H * P)    # (N, L, H*P)

        out = torch.cat([out_s, out_pt, out_norms], dim=-1)  # (N, L, D+H*P*3+H*P)
        return self.out_proj(out)                            # (N, L, D)


# ──────────────────────────────────────────────
# Encoder Layer + Full Encoder
# ──────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_points=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.ipa   = InvariantPointAttention(d_model, n_heads, n_points, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, h, R, t, rel_frames, seq_mask):
        h = self.norm1(h + self.ipa(h, R, t, rel_frames, seq_mask))
        h = self.norm2(h + self.ffn(h))
        return h


class StructureEncoder(nn.Module):
    """
    Full SE(3)-invariant encoder.
    Processes all M conformers in parallel (flattened batch).

    Input:
        coords:   (B, M, L, 4, 3)
        one_hot:  (B, L, 20)
        mask:     (B, L, 4)
        seq_mask: (B, L)
    Output:
        h: (B, M, L, d_model)
    """

    def __init__(self, d_model=256, n_layers=4, n_heads=8,
                 n_points=4, d_ff=512, dropout=0.1, use_esm2=True):
        super().__init__()
        self.d_model = d_model

        self.frame_module = BackboneFrameModule(d_model=d_model)
        self.seq_encoder  = SequenceEncoder(d_model=d_model, use_esm2=use_esm2)

        self.input_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model),
        )
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, n_points, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, coords, one_hot, mask, seq_mask,
                sequences=None, esm2_emb=None):
        B, M, L, _, _ = coords.shape

        # Sequence features: same for all conformers
        seq_feat = self.seq_encoder(one_hot, sequences, esm2_emb)      # (B, L, d)
        seq_feat = seq_feat.unsqueeze(1).expand(B, M, L, self.d_model) # (B, M, L, d)

        # Flatten B*M for parallel processing
        N = B * M
        coords_f   = rearrange(coords,   "b m l a d -> (b m) l a d")
        mask_f     = mask.unsqueeze(1).expand(B, M, L, 4)
        mask_f     = rearrange(mask_f,   "b m l a -> (b m) l a")
        seq_mask_f = seq_mask.unsqueeze(1).expand(B, M, L)
        seq_mask_f = rearrange(seq_mask_f,"b m l -> (b m) l")
        seq_feat_f = rearrange(seq_feat, "b m l d -> (b m) l d")

        # Geometric features
        geo = self.frame_module(coords_f, mask_f)
        R, t, rel_frames = geo["R"], geo["t"], geo["rel_frames"]

        # Combine sequence + geometry
        h = self.input_proj(torch.cat([seq_feat_f, geo["node_geom"]], dim=-1))

        # IPA layers
        for layer in self.layers:
            h = layer(h, R, t, rel_frames, seq_mask_f)

        h = self.final_norm(h)
        return rearrange(h, "(b m) l d -> b m l d", b=B, m=M)
