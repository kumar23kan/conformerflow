# How ConformerFlow Works

ConformerFlow is a deep generative model that takes a single protein structure (typically from X-ray crystallography) and generates a diverse ensemble of conformations that approximates the protein's natural flexibility as measured by NMR spectroscopy.

---

## The Core Problem

X-ray crystal structures capture a single, averaged snapshot of a protein in a rigid lattice environment. Real proteins in solution flex continuously — loops shift, helices breathe, domains rotate. NMR spectroscopy captures this flexibility as an ensemble of ~10–50 conformations deposited as a single PDB entry with multiple models.

ConformerFlow bridges this gap: given a rigid X-ray structure, it learns (from NMR training data) to generate a set of conformers that matches the statistical distribution of real solution-phase flexibility.

---

## High-Level Architecture

```
Input PDB (X-ray)
       │
       ▼
┌─────────────┐
│   Encoder   │  SE(3)-invariant per-residue embeddings
│  (IPA attn) │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Ensemble Stats  │  Learns mean μ, covariance Σ from NMR training data
│     Module       │  Outputs conditioning context θ
└──────┬───────────┘
       │
       ▼
┌────────────────────────────────────────────────────┐
│   Generative Model                                 │
│   flow_matching  ot_cfm  ddpm  ddim  vae  score    │
│   (configurable — only flow_matching uses SE(3))   │
└──────┬─────────────────────────────────────────────┘
       │
       ▼
Output: N conformer CA coordinate arrays  (N, L, 3)
```

---

## Stage 1 — Data Pipeline

### Fetching and Parsing

The pipeline begins by querying RCSB for NMR entries with at least 5 deposited conformers and their paired X-ray structures.

Each NMR entry is parsed into a tensor of shape `(M, L, 4, 3)`:
- `M` — number of conformer models (typically 10–50)
- `L` — sequence length
- `4` — backbone atoms: N, Cα, C, Cβ
- `3` — XYZ coordinates

Missing backbone atoms (common in NMR depositions) are imputed using ideal Engh-Huber bond lengths (N–Cα: 1.46 Å, Cα–C: 1.52 Å). These imputed atoms are used only for frame construction, never in the loss.

### Quality Filtering and Splits

After parsing, entries are filtered by:
- Minimum sequence length and atom completeness thresholds
- Ensemble spread (removes trivially rigid entries)
- 3-sigma outlier removal (conformers whose Cα RMSD from the ensemble mean exceeds 3 standard deviations are dropped as refinement artifacts)

Sequence redundancy is then removed at 30% identity using MMseqs2, and the dataset is split by deposition date: proteins deposited before 2020 go to training/validation, proteins from 2020 onward form the held-out test set. This temporal split prevents data leakage from homologous sequences.

---

## Stage 2 — Structure Representation

The first step is converting raw backbone coordinates into a per-residue feature representation. Four options are available, selected via `representation.structure` in the config.

### `se3_frames` (default)

Every residue is represented as a local reference frame in 3D space, constructed from the backbone geometry in `frames.py`.

For residue `i`:
1. Compute two vectors: Cα→C and Cα→N
2. Apply Gram-Schmidt orthogonalization to get three orthonormal axes
3. The result is a rotation matrix `R_i` (3×3) and a translation `t_i` (the Cα position)

This gives each residue an element of SE(3) — the group of rigid body motions in 3D space. Relative frames between residues `i` and `j` are:

```
R_rel = R_i @ R_j^T
t_rel = R_i @ (t_j - t_i)
```

Any global rotation applied to the whole protein cancels exactly. This is an exact guarantee, not an approximation. Chirality is preserved because the frame handedness encodes it.

### `distance_matrix`

Pairwise Cα distances are encoded with 64 radial basis functions (RBFs) spanning 0–40 Å. For each residue, the row-wise maximum of the RBF responses across all neighbours is used as the per-residue feature.

- **SE(3)-invariant**: distances are unchanged by rotation or translation
- **Loses chirality**: mirror-image structures produce identical distance matrices
- Faster to compute than full frames; useful as an ablation baseline

### `torsion_angles`

Backbone dihedral angles φ, ψ, and ω are computed from N, Cα, C atom positions and encoded as `(sin, cos)` pairs to avoid discontinuities at ±π. This gives 6 features per residue.

- **SE(3)-invariant**: torsion angles are unchanged by global rigid body motion
- Very compact input (6 values vs. a full distance matrix)
- Loses absolute 3D context — two residues with the same local geometry but different spatial arrangement are indistinguishable
- Boundary residues (first has no φ/ω, last has no ψ) are zero-padded

### `cartesian`

Raw Cα Cartesian coordinates (x, y, z), centred by subtracting the mean Cα position of each conformer before input.

- **Not SE(3)-invariant**: different orientations of the same structure produce different inputs
- Simplest possible representation; useful for debugging
- Requires all training structures to be pre-aligned if used seriously

### Representation Comparison

| Option | SE(3)-invariant | Preserves chirality | Relative cost |
|--------|----------------|---------------------|---------------|
| `se3_frames` | Yes (exact) | Yes | Medium |
| `distance_matrix` | Yes | No | Low |
| `torsion_angles` | Yes | Yes | Low |
| `cartesian` | No | Yes | Lowest |

---

## Stage 3 — Encoder

The encoder (`encoders.py`) projects the chosen structure representation plus sequence features into rich per-residue embeddings of shape `(B, M, L, d_model)`.

All four structure representations share the same transformer backbone (`_TransformerStack`), sinusoidal positional encoding, and sequence input. What differs is only the input projection layer.

**For `se3_frames`**, the encoder uses **Invariant Point Attention (IPA)** from `encoder.py` — the same mechanism as AlphaFold2. IPA combines two types of features per attention head:
1. **Scalar features** — standard dot-product attention over per-residue embeddings
2. **Point features** — learnable 3D query/key points transformed into local frames before computing distances

Both are SE(3)-invariant by construction. The output is `(B, M, L, d_model)` where `d_model = 256` by default.

**For the other three representations**, a standard multi-head self-attention transformer is used after projecting the respective features to `d_model`.

Sequence encoding supports:
- One-hot amino acid identity (default)
- ESM-2 pretrained language model embeddings (650M or 3B parameter variants)

---

## Stage 4 — Ensemble Statistics Module

During training, the model has access to M NMR conformers. The ensemble statistics module (`ensemble_stats.py`) aggregates these into a latent distribution that captures the protein's conformational flexibility.

Given M conformer embeddings `(B, M, L, d_model)`:
1. Compute per-residue mean `μ` and log-variance across conformers
2. Learn full cross-residue covariance `Σ` via cross-attention (captures correlated motions between distant residues — e.g., both ends of a helix moving together)
3. Output a conditioning context `θ` that encodes the flexibility signature

The latent space has dimension `d_latent = 16`, chosen to match the ~10–20 slow collective modes (principal components) typical of NMR ensembles. Small enough to force compression and prevent memorization; large enough to capture real physical flexibility.

---

## Stage 5 — Generative Model

Six generative model backends are available, selected via `generative_model.type` in the config. They all share the same interface (`training_step` / `generate`) and feed into the same loss functions.

**Key distinction**: only `flow_matching` operates on full SE(3) frames (rotations + translations). All other backends work on Cα coordinates directly and are not SE(3)-equivariant.

---

### `flow_matching` (default)

Learns a time-dependent vector field that continuously deforms random SE(3) noise frames into protein conformer frames (`flow_matching.py`).

**Training** — for each step:
1. Sample `t ∈ [0, 1]` uniformly
2. Sample random starting frames `x_0` (Gaussian noise on SE(3))
3. Sample a target conformer `x_1` from the NMR ensemble
4. Interpolate: `x_t = (1 - t) * x_0 + t * x_1`
5. Target velocity: `u_t = x_1 - (1 - σ_min) * x_0`
6. Train the network to predict `u_t` from `(x_t, t, θ)`

`σ_min = 0.01` keeps residual noise so the trajectory never collapses to a deterministic point, preserving output diversity.

**Inference** — integrate the ODE from `t = 0` to `t = 1`:
- **Heun (2nd-order)**: default, 20 steps ≈ 40 Euler steps in quality
- **Euler (1st-order)**: simpler, needs 100+ steps for equivalent quality

Repeat with independent noise samples to generate N conformers.

---

### `ot_cfm` — Optimal Transport CFM

Same flow matching training objective, but before computing the interpolation, noise–data pairs within each mini-batch are matched using the **linear assignment algorithm** (`scipy.optimize.linear_sum_assignment`) to minimise total squared Cα distance. This reduces gradient variance compared to random pairing.

- Coordinate-based (not SE(3)-equivariant)
- Falls back to random matching if `scipy` is unavailable
- Inference is identical to `flow_matching` (Euler or Heun ODE)
- Best when training is unstable due to high variance gradients

---

### `ddpm` — Denoising Diffusion Probabilistic Model

Classical DDPM on Cα coordinates with a **cosine noise schedule**.

**Training**: sample a random integer timestep `s ∈ [0, T]`, corrupt coordinates with `x_s = √ᾱ_s · x_1 + √(1 - ᾱ_s) · ε`, train the network to predict the noise `ε`.

**Inference**: stochastic reverse diffusion over `T` steps (sub-sampled to `n_steps` via striding). Each step re-adds a small amount of noise, so generation is stochastic — two runs with different seeds give different conformers.

- `T = 1000` steps by default; `n_steps` controls how many are actually run via striding
- Coordinates are clamped to `[-5, 5]` during reverse diffusion for stability
- Highest quality but slowest inference among the diffusion-based options

---

### `ddim` — Denoising Diffusion Implicit Models

Shares **identical training** with DDPM (same `_DenoisingTransformer`, same noise prediction objective), but uses **deterministic inference** (η = 0):

```
x_{s-1} = √ᾱ_{s-1} · x̂_0 + √(1 - ᾱ_{s-1}) · ε_θ(x_s)
```

No noise is added at each step, so the trajectory is fully deterministic given the starting noise `x_T`. Enables high-quality generation in far fewer steps than DDPM (typically 20–50 vs. 1000).

- Diversity still comes from sampling different `x_T ~ N(0, I)` for each conformer
- Drop-in replacement for DDPM at inference — uses the same checkpoint

---

### `vae` — Variational Auto-Encoder

Direct single-pass decoder: `(θ, z) → x_0`. No iterative denoising.

**Training**: `x_0_pred = decoder(θ, z)`, loss is reconstruction MSE against `x_1` (ground-truth Cα). `t_flow` is fixed to 1 so the schedule factor in `losses.py` is neutral.

**Inference**: one forward pass per conformer — fastest generator by a wide margin. `n_steps` and `method` are ignored.

- Diversity comes entirely from sampling different `z ~ N(0, I)` values
- Quality is generally lower than flow/diffusion models but generation is O(1) in steps

---

### `score_matching` — Score-Based Model

Implements the score matching framework (Song & Ermon, 2020) with a **geometric noise schedule**: `σ(t) = σ_min · (σ_max / σ_min)^t`, where `σ_min = 0.01` and `σ_max = 50.0`.

**Training**: corrupt `x_t = x_1 + σ(t) · ε`, train network to predict the noise `ε` (equivalent to learning the score `s_θ ≈ -ε/σ`).

**Inference**: Euler-Maruyama SDE stepping from high noise (`σ_max · N(0, I)`) down to clean coordinates. Each step:
```
dx = -σ² · score · dt
xt = xt + dx + √(σ² - σ_prev²) · noise
```

- Stochastic inference (adds noise at each step like DDPM)
- Geometric schedule means noise levels span several orders of magnitude, which can capture both large-scale and fine-grained flexibility

---

### Generative Model Comparison

| Type | SE(3)-equivariant | Inference style | Steps needed | Relative speed |
|------|-------------------|-----------------|--------------|----------------|
| `flow_matching` | Yes | ODE (deterministic) | 20 (Heun) | Fast |
| `ot_cfm` | No | ODE (deterministic) | 20 (Heun) | Fast |
| `ddpm` | No | SDE (stochastic) | 1000 (strided) | Slow |
| `ddim` | No | ODE (deterministic) | 20–50 | Fast |
| `vae` | No | Single pass | 1 | Fastest |
| `score_matching` | No | SDE (stochastic) | 100+ | Medium |

---

## Stage 6 — Loss Functions

Six loss terms are combined during training (`losses.py`):

| Loss | Weight | Purpose |
|------|--------|---------|
| Flow matching | 1.0 | SE(3) geodesic distance between predicted and target velocity |
| Ensemble reconstruction | 1.0 | Match mean structure RMSD, per-residue variance, pairwise distance distributions |
| KL divergence | 0.01 | Regularize latent space; free bits threshold of 0.5 nats prevents over-regularization |
| Diversity | auto | Hinge loss on pairwise RMSD — prevents mode collapse (all conformers identical) |
| Geometry | 0.1 | Backbone bond angle penalties for physical validity |
| Chirality | 0.1 | Penalizes negative triple products (D-amino acid configurations) |

The KL weight (0.01) is deliberately small: early in training, the encoder needs room to learn useful representations before regularization pressure is applied.

---

## Stage 7 — Training Infrastructure

Training (`trainer.py`) uses:
- **Distributed Data Parallel (DDP)** with NCCL backend for multi-GPU runs
- **Learning rate schedule**: 1,000-step linear warmup followed by cosine decay to a floor of 1e-6 over 100,000 steps. The warmup stabilizes the frame module before the main learning phase begins.
- **Exponential Moving Average (EMA)** of model weights with decay 0.9999 (averaging over ~10,000 steps). All validation and inference use EMA weights — they produce a smoother vector field and generalize better than the raw checkpoint weights.
- **Mixed precision (bf16)** for throughput on modern GPUs (A100, H100, B200)
- **Gradient clipping** to prevent instability during SE(3) learning

Only rank 0 writes checkpoints and logs. The best checkpoint is selected by validation ensemble RMSD.

---

## Stage 8 — Evaluation Metrics

Evaluation compares generated ensembles against held-out NMR ground truth (`metrics.py`):

**Level 1 — RMSD**
- *Coverage*: for each NMR conformer, find the nearest generated conformer (min RMSD). Mean across NMR set.
- *Precision*: for each generated conformer, find the nearest NMR conformer. Mean across generated set.
- *Mean pairwise RMSD*: average diversity within the generated ensemble.

**Level 2 — RMSF Correlation**
- Per-residue root-mean-square fluctuation computed for both generated and NMR ensembles.
- Pearson and Spearman correlation between the two RMSF profiles. Measures whether the model captures *where* the protein is flexible, not just *how much*.

**Level 3 — Ramachandran Favored Fraction**
- Fraction of residues with backbone dihedrals (φ, ψ) in favored regions of the Ramachandran plot. Measures stereochemical quality of generated conformers.

**Level 4 — Clash Score**
- Number of steric clashes (atom pairs closer than their van der Waals radii allow) per 100 residues. Lower is better; measures physical realism.

---

## Inference API

```python
from inference.predict import ConformerFlowPredictor

predictor = ConformerFlowPredictor("checkpoints/ckpt_best.pt")

result = predictor.predict(
    "input_protein.pdb",
    n_conformers=20,   # how many conformers to generate
    n_steps=20,        # ODE integration steps (Heun integrator)
    method="heun",     # "heun" or "euler"
)

# result.ca_coords: numpy array of shape (20, L, 3)
# L = number of residues in input structure
```

The predictor:
1. Parses the input PDB into backbone frames (X-ray or predicted structure)
2. Runs the encoder to get per-residue embeddings
3. Samples N independent noise initializations
4. Integrates each through the ODE in a batched forward pass
5. Returns Cα coordinates for all conformers

---

## Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| `d_latent` | 16 | Matches ~10–20 slow collective NMR modes; prevents memorization |
| `sigma_min` | 0.01 | Keeps noise in ODE target; ensures generated diversity |
| EMA decay | 0.9999 | Smooths over ~10,000 steps; standard for flow matching |
| `lambda_kl` | 0.01 | Very small — lets encoder learn before regularization dominates |
| `free_bits` | 0.5 nats | Only over-threshold dimensions get KL penalty; prevents dimension collapse |
| Temporal split cutoff | 2020 | Ensures test proteins were deposited after all training data |
| Sequence identity cutoff | 30% | Standard structural biology homology threshold (MMseqs2) |
| Chirality penalty | soft (0.1) | Model learns L-amino acid geometry from data; loss catches rare mirror images |

---

## File Map

```
files/
├── configs/base_config.yaml     # All hyperparameters
├── data/
│   ├── fetch_pdb.py             # Download NMR/X-ray entries from RCSB
│   ├── parse_nmr.py             # Extract (M, L, 4, 3) tensors from NMR PDBs
│   ├── parse_xray.py            # Parse single X-ray structures
│   ├── filter.py                # Quality filter, clustering, train/val/test split
│   └── dataset.py               # PyTorch Dataset with padding and masking
├── model/
│   ├── frames.py                # SE(3) backbone frame construction
│   ├── encoder.py               # Sequence encoder + IPA attention
│   ├── ensemble_stats.py        # Covariance statistics from NMR conformers
│   ├── flow_matching.py         # Flow matching ODE on SE(3)
│   ├── generative_models.py     # Alternative backends (DDPM, VAE, score matching)
│   └── model_factory.py         # Assembles full model from config
├── training/
│   ├── losses.py                # Six loss functions
│   └── trainer.py               # DDP training loop with EMA and LR schedule
├── evaluation/
│   ├── metrics.py               # 4-level metric suite
│   └── validate.py              # Systematic evaluation on held-out set
├── inference/
│   └── predict.py               # Public API: ConformerFlowPredictor
└── scripts/
    └── train.py                 # CLI entry point (60+ flags)
```

---

## Further Reading

- `SCIENCE.md` — detailed mathematical and biological rationale for every design decision
- `RUNBOOK.md` — step-by-step commands for running the full pipeline
- `PUBLICATION_CHECKLIST.md` — current gaps and publication blockers (P0–P4 priority tiers)
