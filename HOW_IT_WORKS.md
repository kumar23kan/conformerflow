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
┌────────────────────┐
│   Flow Matching    │  ODE on SE(3) frames: noise → conformer
│   (SE(3) frames)   │
└──────┬─────────────┘
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

## Stage 2 — SE(3) Frame Representation

Every residue is represented as a local reference frame in 3D space, constructed from the backbone geometry in `frames.py`.

For residue `i`:
1. Compute two vectors: Cα→C and Cα→N
2. Apply Gram-Schmidt orthogonalization to get three orthonormal axes
3. The result is a rotation matrix `R_i` (3×3) and a translation `t_i` (the Cα position)

This gives each residue an element of SE(3) — the group of rigid body motions in 3D space.

**Why SE(3) frames?** Because global rotation and translation of the entire protein must not change the model's output. Relative frames between residues `i` and `j` are defined as:

```
R_rel = R_i @ R_j^T
t_rel = R_i @ (t_j - t_i)
```

These quantities are invariant to global rigid body motions: any global rotation `G` applied to both frames cancels exactly in the relative computation. This guarantee is exact, not approximate.

---

## Stage 3 — Encoder

The encoder (`encoder.py`) transforms raw frames and sequence into rich per-residue embeddings using **Invariant Point Attention (IPA)**, the same mechanism used in AlphaFold2.

IPA combines two types of features per attention head:
1. **Scalar features** — standard dot-product attention over per-residue embeddings
2. **Point features** — learnable 3D query/key points that are transformed into local frames before computing distances

Both contributions are SE(3)-invariant by construction. The output is a tensor of shape `(B, L, d_model)` where `d_model = 256` by default.

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

## Stage 5 — Flow Matching on SE(3)

The generative model (`flow_matching.py`) learns a time-dependent vector field that continuously deforms random noise frames into protein conformer frames.

### Training

For each training step:
1. Sample a random time `t ∈ [0, 1]`
2. Sample random starting frames `x_0` (Gaussian noise)
3. Sample a target conformer `x_1` from the NMR ensemble
4. Interpolate: `x_t = (1 - t) * x_0 + t * x_1`
5. Define target velocity: `u_t = x_1 - (1 - σ_min) * x_0`
6. Train the network to predict `u_t` from `(x_t, t, θ)`

The small constant `σ_min = 0.01` keeps a residual noise contribution so the trajectory never collapses to a deterministic point, preserving diversity in generation.

Loss is the geodesic distance between predicted and target frames — Frobenius norm of the rotation error plus MSE of the translation.

### Inference (ODE Integration)

To generate a new conformer:
1. Sample random SE(3) frames `x_0 ~ N(0, I)` (one frame per residue)
2. Numerically integrate the learned ODE from `t = 0` to `t = 1` using the trained vector field `v_θ(x_t, t, θ)`
3. Extract Cα positions from the final frames

Two integrators are available:
- **Heun (2nd-order)** — default, 20 steps, equivalent accuracy to ~40 Euler steps
- **Euler (1st-order)** — simpler, requires 100+ steps for equivalent quality

Repeat from step 1 with independent noise samples to generate N conformers.

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
