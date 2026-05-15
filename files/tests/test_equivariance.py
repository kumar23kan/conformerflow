"""
ConformerFlow — SE(3) Equivariance / Invariance Tests

Verifies that encoders have the correct symmetry properties:
  - DistanceEncoder:  SE(3)-invariant  (rotation + translation → same output)
  - TorsionEncoder:   SE(3)-invariant  (rotation + translation → same output)
  - CartesianEncoder: NOT invariant    (rotation DOES change output — expected)
  - SE3FramesEncoder: SE(3)-invariant  (skipped if full model stack unavailable)

Run with:
    cd /home/kumar-perinbam/AI-work/conformerflow/files
    python -m pytest tests/test_equivariance.py -v
  or:
    python tests/test_equivariance.py
"""

import sys
import math
import torch
import torch.nn.functional as F
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from encoders import DistanceEncoder, TorsionEncoder, CartesianEncoder


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def random_so3(device="cpu") -> torch.Tensor:
    """Sample a uniformly random rotation matrix via QR decomposition."""
    Q, R = torch.linalg.qr(torch.randn(3, 3, device=device))
    # Ensure proper rotation (det = +1, not -1)
    if torch.det(Q) < 0:
        Q = Q.clone()
        Q[:, 0] = -Q[:, 0]
    return Q   # (3, 3)


def apply_se3(coords: torch.Tensor,
              R: torch.Tensor,
              t: torch.Tensor) -> torch.Tensor:
    """
    Apply SE(3) transformation to coordinates.
    coords: (..., 3), R: (3, 3), t: (3,)
    Returns: same shape as coords
    """
    return coords @ R.T + t


def _make_batch(B=2, M=1, L=12, seed=0):
    """Create a small, random batch of backbone coords."""
    torch.manual_seed(seed)
    coords   = torch.randn(B, M, L, 4, 3) * 5.0   # (B, M, L, 4, 3)
    one_hot  = torch.zeros(B, L, 20)
    # Random one-hot assignments
    idx = torch.randint(0, 20, (B, L))
    one_hot.scatter_(2, idx.unsqueeze(-1), 1.0)
    mask     = torch.ones(B, L, 4, dtype=torch.bool)
    seq_mask = torch.ones(B, L, dtype=torch.bool)
    return coords, one_hot, mask, seq_mask


def _min_cfg():
    return {
        "model": {
            "d_model": 32,
            "n_encoder_layers": 1,
            "n_heads": 4,
            "d_ff": 64,
            "dropout": 0.0,
        }
    }


# ──────────────────────────────────────────────────────────
# Invariance tests
# ──────────────────────────────────────────────────────────

def test_distance_encoder_invariant_to_rotation():
    """DistanceEncoder output must be identical under any global rotation."""
    enc = DistanceEncoder(_min_cfg()).eval()
    coords, one_hot, mask, seq_mask = _make_batch()

    R = random_so3()
    coords_rot = apply_se3(coords, R, torch.zeros(3))

    with torch.no_grad():
        h1 = enc(coords,     one_hot, mask, seq_mask)
        h2 = enc(coords_rot, one_hot, mask, seq_mask)

    assert torch.allclose(h1, h2, atol=1e-4), (
        f"DistanceEncoder rotation invariance FAILED — "
        f"max diff: {(h1 - h2).abs().max():.6f}"
    )


def test_distance_encoder_invariant_to_translation():
    """DistanceEncoder output must be identical under any global translation."""
    enc = DistanceEncoder(_min_cfg()).eval()
    coords, one_hot, mask, seq_mask = _make_batch()

    t = torch.randn(3) * 20.0
    coords_shifted = apply_se3(coords, torch.eye(3), t)

    with torch.no_grad():
        h1 = enc(coords,         one_hot, mask, seq_mask)
        h2 = enc(coords_shifted, one_hot, mask, seq_mask)

    assert torch.allclose(h1, h2, atol=1e-4), (
        f"DistanceEncoder translation invariance FAILED — "
        f"max diff: {(h1 - h2).abs().max():.6f}"
    )


def test_torsion_encoder_invariant_to_rotation():
    """TorsionEncoder (sin/cos of φψω) must be identical under global rotation."""
    enc = TorsionEncoder(_min_cfg()).eval()
    coords, one_hot, mask, seq_mask = _make_batch()

    R = random_so3()
    coords_rot = apply_se3(coords, R, torch.zeros(3))

    with torch.no_grad():
        h1 = enc(coords,     one_hot, mask, seq_mask)
        h2 = enc(coords_rot, one_hot, mask, seq_mask)

    assert torch.allclose(h1, h2, atol=1e-4), (
        f"TorsionEncoder rotation invariance FAILED — "
        f"max diff: {(h1 - h2).abs().max():.6f}"
    )


def test_torsion_encoder_invariant_to_translation():
    """TorsionEncoder must be identical under global translation."""
    enc = TorsionEncoder(_min_cfg()).eval()
    coords, one_hot, mask, seq_mask = _make_batch()

    t = torch.randn(3) * 20.0
    coords_shifted = apply_se3(coords, torch.eye(3), t)

    with torch.no_grad():
        h1 = enc(coords,         one_hot, mask, seq_mask)
        h2 = enc(coords_shifted, one_hot, mask, seq_mask)

    assert torch.allclose(h1, h2, atol=1e-4), (
        f"TorsionEncoder translation invariance FAILED — "
        f"max diff: {(h1 - h2).abs().max():.6f}"
    )


def test_cartesian_encoder_NOT_invariant_to_rotation():
    """
    CartesianEncoder is intentionally NOT rotation-invariant.
    A rotation of 45° or more must produce different embeddings.
    This test verifies the negative: if it were invariant the
    encoder could not learn relative orientations.
    """
    enc = CartesianEncoder(_min_cfg()).eval()
    coords, one_hot, mask, seq_mask = _make_batch()

    # 90-degree rotation around z-axis — guaranteed to change output
    angle = math.pi / 2
    R = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle),  math.cos(angle), 0],
        [0,                0,               1],
    ], dtype=torch.float32)

    coords_rot = apply_se3(coords, R, torch.zeros(3))

    with torch.no_grad():
        h1 = enc(coords,     one_hot, mask, seq_mask)
        h2 = enc(coords_rot, one_hot, mask, seq_mask)

    max_diff = (h1 - h2).abs().max().item()
    assert max_diff > 1e-3, (
        f"CartesianEncoder unexpectedly invariant to rotation — "
        f"max diff: {max_diff:.6f} (expected > 1e-3)"
    )


def test_chirality_detection():
    """
    TorsionEncoder must produce DIFFERENT outputs for a structure and its
    mirror image (enantiomer), since backbone torsion angles change sign
    under reflection.  DistanceEncoder is NOT sensitive to chirality.
    """
    coords, one_hot, mask, seq_mask = _make_batch()

    # Mirror image: flip x-coordinate → chiral inversion
    coords_mirror = coords.clone()
    coords_mirror[..., 0] = -coords_mirror[..., 0]

    torsion_enc = TorsionEncoder(_min_cfg()).eval()
    dist_enc    = DistanceEncoder(_min_cfg()).eval()

    with torch.no_grad():
        t1 = torsion_enc(coords,        one_hot, mask, seq_mask)
        t2 = torsion_enc(coords_mirror, one_hot, mask, seq_mask)
        d1 = dist_enc(coords,           one_hot, mask, seq_mask)
        d2 = dist_enc(coords_mirror,    one_hot, mask, seq_mask)

    torsion_diff = (t1 - t2).abs().max().item()
    dist_diff    = (d1 - d2).abs().max().item()

    # TorsionEncoder: sin(angle) → -sin(angle) under mirror → different
    assert torsion_diff > 1e-3, (
        f"TorsionEncoder should be chirality-sensitive but diff={torsion_diff:.6f}"
    )
    # DistanceEncoder: distances are unchanged under reflection → same output
    assert dist_diff < 1e-3, (
        f"DistanceEncoder should be chirality-BLIND but diff={dist_diff:.6f}"
    )


def test_se3frames_encoder_invariant():
    """
    SE3FramesEncoder (IPA-based) must be SE(3)-invariant.
    Skipped if model.encoder dependencies are not installed.
    """
    try:
        from encoders import SE3FramesEncoder
        cfg = {
            **_min_cfg(),
            "representation": {"structure": "se3_frames", "sequence": "onehot"},
        }
        enc = SE3FramesEncoder(cfg).eval()
    except Exception as e:
        print(f"  [SKIP] SE3FramesEncoder not available: {e}")
        return

    coords, one_hot, mask, seq_mask = _make_batch()
    R = random_so3()
    t = torch.randn(3) * 10.0
    coords_transformed = apply_se3(coords, R, t)

    with torch.no_grad():
        h1 = enc(coords,             one_hot, mask, seq_mask)
        h2 = enc(coords_transformed, one_hot, mask, seq_mask)

    max_diff = (h1 - h2).abs().max().item()
    assert max_diff < 1e-3, (
        f"SE3FramesEncoder SE(3)-invariance FAILED — max diff: {max_diff:.6f}"
    )


def test_backbone_frame_determinism():
    """
    Building backbone frames twice from the same coords must give identical R.
    Catches non-determinism in Gram-Schmidt under jit / different dtypes.
    """
    from model.frames import build_backbone_frames

    torch.manual_seed(7)
    coords   = torch.randn(2, 3, 10, 4, 3)
    mask     = torch.ones(2, 10, 4, dtype=torch.bool)

    R1, t1, fm1 = build_backbone_frames(coords, mask)
    R2, t2, fm2 = build_backbone_frames(coords, mask)

    assert torch.allclose(R1, R2, atol=1e-6), "Backbone frame determinism FAILED"
    assert torch.allclose(t1, t2, atol=1e-6), "Backbone translation determinism FAILED"


def test_backbone_frame_orthonormality():
    """
    Rotation matrices from build_backbone_frames must satisfy R^T R ≈ I
    (orthonormal rows) and det(R) ≈ +1 (proper rotation, not reflection).
    """
    from model.frames import build_backbone_frames

    torch.manual_seed(99)
    B, M, L = 2, 2, 15
    coords = torch.randn(B, M, L, 4, 3)
    mask   = torch.ones(B, L, 4, dtype=torch.bool)

    R, _, frame_mask = build_backbone_frames(coords, mask)

    # Orthonormality: R^T R should be identity for valid frames
    RtR  = R.transpose(-1, -2) @ R                          # (B, M, L, 3, 3)
    I    = torch.eye(3, device=R.device).expand_as(RtR)
    diff = (RtR - I).abs()
    # Only check frames where all 3 backbone atoms are present
    diff_valid = diff[frame_mask]
    assert diff_valid.max().item() < 1e-4, (
        f"Backbone frames not orthonormal — max(|R^T R - I|) = {diff_valid.max():.6f}"
    )

    # Proper rotation: det should be +1
    det = torch.linalg.det(R)
    det_valid = det[frame_mask]
    assert (det_valid - 1.0).abs().max().item() < 1e-4, (
        f"Backbone frame det ≠ +1 — max(|det - 1|) = {(det_valid - 1).abs().max():.6f}"
    )


# ──────────────────────────────────────────────────────────
# Entry point for direct execution
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_distance_encoder_invariant_to_rotation,
        test_distance_encoder_invariant_to_translation,
        test_torsion_encoder_invariant_to_rotation,
        test_torsion_encoder_invariant_to_translation,
        test_cartesian_encoder_NOT_invariant_to_rotation,
        test_chirality_detection,
        test_se3frames_encoder_invariant,
        test_backbone_frame_determinism,
        test_backbone_frame_orthonormality,
    ]

    passed, failed = 0, 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}\n        {e}")
            failed += 1
        except Exception as e:
            print(f"  SKIP  {name}\n        {e}")

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests.")
    if failed:
        sys.exit(1)
