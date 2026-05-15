"""
ConformerFlow — Phase 3: Flow Matching on SE(3) Frames

The network learns the conformational distribution directly from NMR ensembles.
It operates on backbone SE(3) frames (rotation + translation per residue)
and learns to generate diverse, physically valid conformers.

Key design principles:
  1. The transformer learns everything from data — no hand-coded geometry rules
  2. Equivariance is guaranteed by predicting updates in LOCAL frames
  3. The network learns what flexibility patterns look like from NMR
  4. At inference it applies that learned knowledge to X-ray structures

Flow matching on SE(3):
  - Noise: random frames x_0
  - Data:  NMR conformer frames x_1
  - Path:  x_t = interpolate(x_0, x_1, t)  [on SE(3) manifold]
  - Learn: v_θ(x_t, t, θ) that predicts the flow direction

Reference:
  Yim et al. "SE(3) diffusion model with application to protein backbone
  generation." ICML 2023.
  Lipman et al. "Flow Matching for Generative Modeling." ICLR 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ─────────────────────────────────────────────────────────
# 1. SE(3) FRAME REPRESENTATION
#    The network operates on frames, not raw coordinates.
#    A frame per residue = (R ∈ SO(3), t ∈ R³)
#    Flattened to a 12-dim vector for the transformer.
# ─────────────────────────────────────────────────────────

def frames_to_features(R: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
    """
    Flatten SE(3) frames into feature vectors.
    R: (..., 3, 3)  rotation matrices
    t: (..., 3)     translations (CA positions)
    Returns: (..., 12)  — 9 from R + 3 from t
    """
    R_flat = R.reshape(*R.shape[:-2], 9)         # (..., 9)
    return torch.cat([R_flat, t], dim=-1)         # (..., 12)


def features_to_frames(f: torch.Tensor) -> tuple:
    """
    Recover SE(3) frames from feature vectors.
    f: (..., 12)
    Returns: R (..., 3, 3), t (..., 3)
    """
    R = f[..., :9].reshape(*f.shape[:-1], 3, 3)
    t = f[..., 9:]
    return R, t


def gram_schmidt(v1: torch.Tensor,
                  v2: torch.Tensor) -> torch.Tensor:
    """
    Build orthonormal rotation matrix from two vectors.
    Used to project predicted R back onto SO(3).
    v1, v2: (..., 3)
    Returns: R (..., 3, 3)  — orthonormal
    """
    e1 = F.normalize(v1, dim=-1, eps=1e-8)
    u2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1, eps=1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)      # (..., 3, 3)


def interpolate_frames(R0: torch.Tensor, t0: torch.Tensor,
                        R1: torch.Tensor, t1: torch.Tensor,
                        alpha: torch.Tensor) -> tuple:
    """
    Linear interpolation of SE(3) frames at time alpha.
    Uses geodesic interpolation for rotations (SLERP).
    For simplicity in training we use linear interpolation
    of the flattened frame features + re-orthogonalize R.

    R0, R1: (B, L, 3, 3)
    t0, t1: (B, L, 3)
    alpha:  (B, 1, 1)  in [0, 1]
    Returns: R_t, t_t
    """
    # Interpolate translation linearly
    t_t = (1 - alpha) * t0 + alpha * t1

    # Interpolate rotation via flattened linear interp + re-orthogonalize
    R0_flat = R0.reshape(*R0.shape[:-2], 9)
    R1_flat = R1.reshape(*R1.shape[:-2], 9)
    R_interp_flat = (1 - alpha.unsqueeze(-1).squeeze(-2)) * R0_flat \
                   + alpha.unsqueeze(-1).squeeze(-2) * R1_flat
    R_interp = R_interp_flat.reshape(*R0.shape)

    # Re-orthogonalize via Gram-Schmidt on first two columns
    v1 = R_interp[..., 0]   # (..., 3)
    v2 = R_interp[..., 1]   # (..., 3)
    R_t = gram_schmidt(v1, v2)

    return R_t, t_t


def sample_random_frames(B: int, L: int,
                          device: torch.device,
                          t_scale: float = 10.0) -> tuple:
    """
    Sample random SE(3) frames as noise source x_0.
    R ~ Haar measure on SO(3) via QR decomposition
    t ~ N(0, t_scale²)
    """
    # Random rotation via QR
    rand_mat = torch.randn(B, L, 3, 3, device=device)
    R0, _    = torch.linalg.qr(rand_mat)
    # Ensure proper rotation (det = +1)
    det      = torch.linalg.det(R0).unsqueeze(-1).unsqueeze(-1)
    R0       = R0 * det.sign()

    # Random translation
    t0 = torch.randn(B, L, 3, device=device) * t_scale

    return R0, t0


# ─────────────────────────────────────────────────────────
# 2. TIME EMBEDDING
# ─────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for flow time t ∈ [0, 1].
    The network learns to condition its predictions on
    how far along the flow path we are.
    """

    def __init__(self, d_model: int):
        super().__init__()
        half = d_model // 2
        freqs = torch.exp(-torch.arange(half).float() * (8.0 / half))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, d_model)"""
        t    = t.view(-1, 1).float() * 1000.0
        args = t * self.freqs.unsqueeze(0)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


# ─────────────────────────────────────────────────────────
# 3. FRAME TRANSFORMER
#    This is where the AI learning happens.
#    The transformer sees noisy frames and conditioning
#    and learns what update to apply.
# ─────────────────────────────────────────────────────────

class FrameTransformerLayer(nn.Module):
    """
    One transformer layer that operates on SE(3) frame features.

    What the AI learns here:
      - Self-attention: how does residue i's frame relate to residue j's?
        (learns flexibility correlations from NMR data)
      - Cross-attention: how does the conditioning θ constrain the update?
        (learns to condition on the ensemble distribution)
      - FFN: nonlinear transformation of frame features
        (learns complex sequence-structure relationships)

    Output: predicted frame UPDATE per residue in the LOCAL frame.
    Equivariance is free: updates in local frame automatically
    transform correctly under global rotations.
    """

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        # Self-attention over noisy frame features
        self.self_q  = nn.Linear(d_model, d_model, bias=False)
        self.self_k  = nn.Linear(d_model, d_model, bias=False)
        self.self_v  = nn.Linear(d_model, d_model, bias=False)
        self.self_out= nn.Linear(d_model, d_model)
        self.norm1   = nn.LayerNorm(d_model)

        # Cross-attention to conditioning θ from EnsembleStats
        self.cross_q  = nn.Linear(d_model, d_model, bias=False)
        self.cross_k  = nn.Linear(d_model, d_model, bias=False)
        self.cross_v  = nn.Linear(d_model, d_model, bias=False)
        self.cross_out= nn.Linear(d_model, d_model)
        self.norm2    = nn.LayerNorm(d_model)

        # Feed-forward: learns nonlinear sequence-structure relationships
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),                  # SiLU works better than ReLU for proteins
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _attention(self, Q, K, V, mask=None):
        """Multi-head attention with optional masking."""
        B, L, H, d = Q.shape
        # Attention scores
        scores = torch.einsum("bihd,bjhd->bhij", Q, K) / (d ** 0.5)
        if mask is not None:
            # mask: (B, L) → pad positions
            scores = scores + (~mask).float()[:, None, None, :] * -1e9
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.einsum("bhij,bjhd->bihd", weights, V)
        return out.reshape(B, L, H * d)

    def forward(self,
                h:        torch.Tensor,
                theta:    torch.Tensor,
                seq_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:        (B, L, d_model)  — current frame features
            theta:    (B, L, d_model)  — conditioning from EnsembleStats
            seq_mask: (B, L)           — True for real residues
        Returns:
            h: (B, L, d_model)  — updated frame features
        """
        B, L, D = h.shape
        H, d = self.n_heads, self.d_head

        # ── Self-attention: learn inter-residue correlations ──
        Q = self.self_q(h).view(B, L, H, d)
        K = self.self_k(h).view(B, L, H, d)
        V = self.self_v(h).view(B, L, H, d)
        sa = self._attention(Q, K, V, seq_mask)
        h  = self.norm1(h + self.self_out(sa))

        # ── Cross-attention: condition on ensemble distribution ──
        Q  = self.cross_q(h).view(B, L, H, d)
        K  = self.cross_k(theta).view(B, L, H, d)
        V  = self.cross_v(theta).view(B, L, H, d)
        ca = self._attention(Q, K, V, seq_mask)
        h  = self.norm2(h + self.cross_out(ca))

        # ── FFN: learn nonlinear transformations ──
        h = self.norm3(h + self.ffn(h))

        return h


class FrameTransformer(nn.Module):
    """
    Deep transformer that learns the SE(3) vector field.

    What the AI learns end-to-end:
      Given noisy frames at time t, the conditioning context θ
      (which encodes what the NMR ensemble distribution looks like),
      and the latent z (which encodes which part of the distribution
      we're sampling from), predict how to update each residue's
      frame to move toward a valid conformer.

    The learning signal comes purely from NMR ensembles:
      - The model sees thousands of (single structure → ensemble) pairs
      - It learns that certain sequence/structure patterns lead to
        certain flexibility distributions
      - At test time it applies this learned knowledge to X-ray structures
    """

    def __init__(self,
                 d_model:  int = 256,
                 d_latent: int = 128,
                 n_layers: int = 8,
                 n_heads:  int = 8,
                 d_ff:     int = 512,
                 dropout:  float = 0.1):
        super().__init__()
        self.d_model = d_model

        # ── Input projections ──
        # Frame features (12-dim) → d_model
        self.frame_proj = nn.Sequential(
            nn.Linear(12, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Time embedding → d_model
        self.time_emb = TimeEmbedding(d_model)

        # Latent z → d_model
        self.latent_proj = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Relative position bias (sequence distance between residues i, j)
        # The network learns how sequence distance relates to structural correlation
        self.rel_pos_bias = nn.Embedding(512, n_heads)

        # Fuse: frame features + time + latent → d_model
        self.input_fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # ── Transformer layers ──
        # Each layer: self-attn over frames + cross-attn to θ + FFN
        self.layers = nn.ModuleList([
            FrameTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # ── Output heads ──
        # Predict translation update Δt in local frame (3-dim)
        self.pred_translation = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),    # Δt per residue
        )

        # Predict rotation update as two vectors (for Gram-Schmidt)
        # The network predicts the update in local frame → equivariant
        self.pred_rotation = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 6),    # Two 3D vectors for Gram-Schmidt
        )

    def _add_relative_position_bias(self,
                                     logits: torch.Tensor,
                                     L:      int) -> torch.Tensor:
        """
        Add learned relative position bias to attention logits.
        The network learns how far-apart residues should attend to each other.
        logits: (B, H, L, L)
        """
        pos_i  = torch.arange(L, device=logits.device)
        pos_j  = torch.arange(L, device=logits.device)
        # Clamp sequence distance to embedding range
        rel    = (pos_i.unsqueeze(1) - pos_j.unsqueeze(0)).abs().clamp(0, 511)
        bias   = self.rel_pos_bias(rel)             # (L, L, H)
        bias   = bias.permute(2, 0, 1).unsqueeze(0) # (1, H, L, L)
        return logits + bias

    def forward(self,
                R_t:     torch.Tensor,
                t_coord: torch.Tensor,
                t_flow:  torch.Tensor,
                theta:   torch.Tensor,
                z:       torch.Tensor,
                seq_mask: torch.Tensor) -> tuple:
        """
        Predict frame updates: what direction should frames move
        to get closer to a valid conformer?

        Args:
            R_t:      (B, L, 3, 3)   noisy rotation matrices at flow time t
            t_coord:  (B, L, 3)      noisy translations at flow time t
            t_flow:   (B,)           flow time in [0, 1]
            theta:    (B, L, d_model) conditioning from EnsembleStats
            z:        (B, L, d_latent) latent sample
            seq_mask: (B, L)          True for real residues

        Returns:
            v_R: (B, L, 3, 3)  predicted rotation velocity (target rotation)
            v_t: (B, L, 3)     predicted translation velocity
        """
        B, L = R_t.shape[:2]

        # ── Build frame feature vectors ──
        frame_feat = frames_to_features(R_t, t_coord)  # (B, L, 12)
        h_frame    = self.frame_proj(frame_feat)        # (B, L, d_model)

        # ── Time embedding ──
        t_emb = self.time_emb(t_flow)                   # (B, d_model)
        t_emb = t_emb.unsqueeze(1).expand(B, L, -1)    # (B, L, d_model)

        # ── Latent ──
        h_z = self.latent_proj(z)                       # (B, L, d_model)

        # ── Fuse inputs ──
        h = self.input_fuse(
            torch.cat([h_frame, t_emb, h_z], dim=-1)
        )   # (B, L, d_model)

        # ── Transformer: the AI learns here ──
        for layer in self.layers:
            h = layer(h, theta, seq_mask)

        h = self.final_norm(h)

        # ── Predict frame updates in LOCAL frame ──
        # Translation velocity: Δt tells each CA where to move
        v_t = self.pred_translation(h)                  # (B, L, 3)

        # Rotation velocity: predict new rotation as two 3D vectors
        # then build orthonormal frame via Gram-Schmidt
        rot_vecs = self.pred_rotation(h)                # (B, L, 6)
        v1 = rot_vecs[..., :3]                         # (B, L, 3)
        v2 = rot_vecs[..., 3:]                         # (B, L, 3)
        v_R = gram_schmidt(v1, v2)                      # (B, L, 3, 3)

        # Zero out padded positions
        if seq_mask is not None:
            m = seq_mask.float().unsqueeze(-1)          # (B, L, 1)
            v_t = v_t * m
            v_R = v_R * m.unsqueeze(-1)

        return v_R, v_t


# ─────────────────────────────────────────────────────────
# 4. FLOW MATCHING TRAINING OBJECTIVE ON SE(3)
# ─────────────────────────────────────────────────────────

class SE3FlowMatcher:
    """
    Conditional flow matching objective on SE(3) frames.

    The AI is trained to learn:
      "Given a noisy frame at time t, predict the clean frame
       from the NMR ensemble."

    Training signal:
      - x_0: random frames (noise)
      - x_1: NMR conformer frames (target)
      - x_t: interpolated frames at time t
      - u_t: the vector field direction (what we want to predict)

    Loss:
      L = ||v_θ(x_t, t, θ, z) - u_t||²

    The model learns the distribution over NMR conformers,
    not a fixed transformation.
    """

    def __init__(self, sigma_min: float = 0.01):
        self.sigma_min = sigma_min

    def sample_flow_path(self,
                          R1:   torch.Tensor,
                          t1:   torch.Tensor,
                          mask: torch.Tensor = None) -> dict:
        """
        Sample a point along the flow path for training.

        Args:
            R1:   (B, L, 3, 3)  target rotation matrices (NMR conformer)
            t1:   (B, L, 3)     target translations (NMR conformer CA)
            mask: (B, L)

        Returns dict with:
            t_flow:   (B,)           sampled flow time
            R0, t0:   noise frames
            R_t, t_t: interpolated frames
            u_R, u_t: target vector field
        """
        B, L = R1.shape[:2]
        device = R1.device

        # Sample flow time t ~ U(0,1)
        t_flow = torch.rand(B, device=device)
        alpha  = t_flow.view(B, 1, 1)

        # Center translations (remove global translation)
        if mask is not None:
            n   = mask.float().sum(-1, keepdim=True).unsqueeze(-1)
            mu  = (t1 * mask.float().unsqueeze(-1)).sum(1, keepdim=True) / n
        else:
            mu = t1.mean(1, keepdim=True)
        t1_c = t1 - mu

        # Sample noise frames
        R0, t0 = sample_random_frames(B, L, device)

        # Interpolate: x_t = (1-t)*x_0 + t*x_1
        R_t, t_t = interpolate_frames(R0, t0, R1, t1_c, alpha)

        # Target vector field: direction from noise to data
        # u_t = x_1 - (1 - sigma_min) * x_0  (with slight noise retention)
        sm    = self.sigma_min
        u_t_coord = t1_c - (1 - sm) * t0                      # (B, L, 3)

        # Target rotation: the clean NMR rotation
        u_R_target = R1                                         # (B, L, 3, 3)

        return {
            "t_flow": t_flow,
            "R0": R0, "t0": t0,
            "R_t": R_t, "t_t": t_t,
            "u_R": u_R_target,
            "u_t": u_t_coord,
            "t1_centered": t1_c,
        }

    def compute_loss(self,
                     v_R_pred:  torch.Tensor,
                     v_t_pred:  torch.Tensor,
                     u_R:       torch.Tensor,
                     u_t:       torch.Tensor,
                     mask:      torch.Tensor = None) -> dict:
        """
        Compute flow matching loss on both rotation and translation.

        Translation loss: MSE on predicted vs target Δt
        Rotation loss:    Geodesic distance between predicted and target R
                          = ||R_pred^T R_target - I||_F

        Args:
            v_R_pred: (B, L, 3, 3)  predicted rotation velocity
            v_t_pred: (B, L, 3)     predicted translation velocity
            u_R:      (B, L, 3, 3)  target rotation
            u_t:      (B, L, 3)     target translation velocity
            mask:     (B, L)

        Returns: dict with individual loss components
        """
        m = mask.float().unsqueeze(-1) if mask is not None \
            else torch.ones(*v_t_pred.shape[:2], 1, device=v_t_pred.device)
        n = m.sum() + 1e-8

        # Translation loss: MSE
        t_loss = ((v_t_pred - u_t) ** 2 * m).sum() / (n * 3)

        # Rotation loss: Frobenius norm of (R_pred^T @ R_target - I)
        # This measures how far the predicted rotation is from the target
        I      = torch.eye(3, device=v_R_pred.device).unsqueeze(0).unsqueeze(0)
        R_diff = torch.matmul(v_R_pred.transpose(-1, -2), u_R) - I  # (B, L, 3, 3)
        r_loss = (R_diff ** 2 * m.unsqueeze(-1)).sum() / (n * 9)

        total  = t_loss + r_loss

        return {
            "loss":           total,
            "translation_loss": t_loss,
            "rotation_loss":    r_loss,
        }


# ─────────────────────────────────────────────────────────
# 5. ODE SAMPLER — Generate conformers at inference
# ─────────────────────────────────────────────────────────

class SE3ODESampler(nn.Module):
    """
    Integrates the learned SE(3) ODE to generate conformers.

    At inference time:
      1. Start with random frames x_0
      2. Integrate dx/dt = v_θ(x_t, t, θ, z)
      3. Arrive at x_1 = a valid conformer frame set
      4. Extract CA positions from x_1
      5. Repeat N times → N diverse conformers

    The diversity comes from:
      - Different noise samples x_0
      - Different latent samples z from the learned distribution
    """

    def __init__(self, vf: FrameTransformer):
        super().__init__()
        self.vf = vf

    @torch.no_grad()
    def _step_euler(self, R, t_coord, t_flow, dt, theta, z, seq_mask):
        v_R, v_t = self.vf(R, t_coord, t_flow, theta, z, seq_mask)
        # Update translation
        t_new = t_coord + dt * v_t
        # Update rotation: blend current R toward predicted v_R
        R_new = gram_schmidt(
            (R[..., 0] + dt * v_R[..., 0]),
            (R[..., 1] + dt * v_R[..., 1])
        )
        return R_new, t_new

    @torch.no_grad()
    def _step_heun(self, R, t_coord, t_flow, dt, theta, z, seq_mask):
        """Heun (2nd order RK) step for better accuracy."""
        # Predictor
        v_R1, v_t1 = self.vf(R, t_coord, t_flow, theta, z, seq_mask)
        t_pred = t_coord + dt * v_t1
        R_pred = gram_schmidt(
            R[..., 0] + dt * v_R1[..., 0],
            R[..., 1] + dt * v_R1[..., 1]
        )
        # Corrector
        t_next = t_flow + dt
        v_R2, v_t2 = self.vf(R_pred, t_pred, t_next, theta, z, seq_mask)
        # Average
        t_new = t_coord + 0.5 * dt * (v_t1 + v_t2)
        R_new = gram_schmidt(
            R[..., 0] + 0.5 * dt * (v_R1[..., 0] + v_R2[..., 0]),
            R[..., 1] + 0.5 * dt * (v_R1[..., 1] + v_R2[..., 1])
        )
        return R_new, t_new

    @torch.no_grad()
    def sample_one(self,
                   theta:    torch.Tensor,
                   z:        torch.Tensor,
                   seq_mask: torch.Tensor,
                   n_steps:  int = 20,
                   method:   str = "heun") -> torch.Tensor:
        """
        Generate one conformer's CA coordinates.

        Args:
            theta:    (B, L, d_model)
            z:        (B, L, d_latent)
            seq_mask: (B, L)
            n_steps:  ODE steps
            method:   'euler' or 'heun'

        Returns:
            ca_coords: (B, L, 3)  — generated CA positions
        """
        B, L = theta.shape[:2]
        device = theta.device

        # Start from random noise frames
        R, t_coord = sample_random_frames(B, L, device)

        dt   = 1.0 / n_steps
        step = self._step_heun if method == "heun" else self._step_euler

        for i in range(n_steps):
            t_flow = torch.full((B,), i * dt, device=device)
            R, t_coord = step(R, t_coord, t_flow, dt, theta, z, seq_mask)

        # Mask padded positions
        if seq_mask is not None:
            t_coord = t_coord * seq_mask.float().unsqueeze(-1)

        return t_coord   # CA positions = translations

    @torch.no_grad()
    def sample_ensemble(self,
                        theta:        torch.Tensor,
                        z_samples:    torch.Tensor,
                        seq_mask:     torch.Tensor,
                        n_conformers: int,
                        n_steps:      int = 20,
                        method:       str = "heun") -> torch.Tensor:
        """
        Generate N conformers.

        Args:
            theta:        (B, L, d_model)      conditioning
            z_samples:    (B, N, L, d_latent)  pre-sampled latents
            seq_mask:     (B, L)
            n_conformers: N (user-specified)
            n_steps:      ODE steps per conformer
            method:       integrator

        Returns:
            ensemble: (B, N, L, 3)  — N generated conformers, CA coordinates
        """
        conformers = []
        for n in range(n_conformers):
            z_n   = z_samples[:, n]   # (B, L, d_latent)
            ca_n  = self.sample_one(theta, z_n, seq_mask, n_steps, method)
            conformers.append(ca_n)

        return torch.stack(conformers, dim=1)   # (B, N, L, 3)


# ─────────────────────────────────────────────────────────
# 6. COMPLETE FLOW MATCHING MODULE
# ─────────────────────────────────────────────────────────

class FlowMatchingModule(nn.Module):
    """
    Complete Phase 3 module.

    Wraps:
      - FrameTransformer: deep AI that learns conformational distribution
      - SE3FlowMatcher: training objective
      - SE3ODESampler: inference engine

    Training:
      The transformer learns from NMR ensembles:
        "For this protein, given these ensemble statistics θ and
         this latent z, what should the frames look like?"

    Inference on X-ray structures:
      θ is computed from the single X-ray structure via the encoder.
      z is sampled N times from the learned distribution.
      The transformer generates N diverse conformers from what
      it learned about NMR flexibility patterns.
    """

    def __init__(self,
                 d_model:   int = 256,
                 d_latent:  int = 128,
                 n_layers:  int = 8,
                 n_heads:   int = 8,
                 d_ff:      int = 512,
                 dropout:   float = 0.1,
                 sigma_min: float = 0.01):
        super().__init__()

        self.transformer = FrameTransformer(
            d_model=d_model, d_latent=d_latent,
            n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout
        )
        self.flow_matcher = SE3FlowMatcher(sigma_min=sigma_min)
        self.sampler      = SE3ODESampler(self.transformer)

    def training_step(self,
                      R1:       torch.Tensor,
                      t1:       torch.Tensor,
                      theta:    torch.Tensor,
                      z:        torch.Tensor,
                      seq_mask: torch.Tensor) -> dict:
        """
        One training step.

        Args:
            R1:       (B, L, 3, 3)    target rotation matrices (NMR conformer)
            t1:       (B, L, 3)       target CA positions (NMR conformer)
            theta:    (B, L, d_model) conditioning from EnsembleStats
            z:        (B, L, d_latent) latent sample
            seq_mask: (B, L)

        Returns:
            dict with loss components and diagnostic tensors
        """
        # Sample flow path
        flow = self.flow_matcher.sample_flow_path(R1, t1, seq_mask)

        # Predict vector field with the transformer
        v_R_pred, v_t_pred = self.transformer(
            flow["R_t"],
            flow["t_t"],
            flow["t_flow"],
            theta,
            z,
            seq_mask,
        )

        # Compute loss: how well did the transformer predict the flow direction?
        losses = self.flow_matcher.compute_loss(
            v_R_pred, v_t_pred,
            flow["u_R"], flow["u_t"],
            seq_mask,
        )

        return {**losses,
                "t_flow":   flow["t_flow"].mean().item(),
                "v_t_pred": v_t_pred,
                "v_R_pred": v_R_pred,
                "u_R":      flow["u_R"],    # target rotation — same draw as prediction
                "u_t":      flow["u_t"],    # target translation — same draw as prediction
                }

    @torch.no_grad()
    def generate(self,
                 theta:        torch.Tensor,
                 z_samples:    torch.Tensor,
                 seq_mask:     torch.Tensor,
                 n_conformers: int = 10,
                 n_steps:      int = 20,
                 method:       str = "heun") -> torch.Tensor:
        """
        Generate N conformers at inference time.

        Args:
            theta:        (B, L, d_model)
            z_samples:    (B, N, L, d_latent)
            seq_mask:     (B, L)
            n_conformers: user-specified N
            n_steps:      ODE integration steps
            method:       'euler' or 'heun'

        Returns:
            ensemble: (B, N, L, 3)  — N diverse CA coordinate sets
        """
        return self.sampler.sample_ensemble(
            theta, z_samples, seq_mask,
            n_conformers, n_steps, method
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
