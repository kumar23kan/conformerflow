# ConformerFlow Runbook

Step-by-step instructions to run the full pipeline from a clean environment.
All commands are run from the project root: `/home/kumar-perinbam/AI-work/conformerflow/files/`

---

## 0. Smoke Test

Run this first to verify the environment before starting any long step.

```bash
python - <<'EOF'
import torch, numpy, scipy, biopython, einops, requests, tqdm
print(f"PyTorch   : {torch.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        gb   = torch.cuda.get_device_properties(i).total_memory // (1024**3)
        print(f"  GPU {i}: {name} ({gb} GB)")
import numpy, scipy, einops
print("NumPy / SciPy / einops: OK")
from Bio import PDB
print("BioPython PDB: OK")
EOF
```

Expected: no import errors, CUDA available, at least one GPU listed.

---

## 1. Environment Setup

### 1a. Create conda environment

```bash
conda create -n conformerflow python=3.11 -y
conda activate conformerflow
```

### 1b. Install PyTorch (CUDA 12.8 / B200-optimised build)

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

> For older CUDA builds replace `cu128` with `cu118` or `cu121`.

### 1c. Install project dependencies

```bash
pip install \
    biopython \
    requests \
    tqdm \
    numpy \
    scipy \
    einops \
    pyyaml \
    pandas
```

### 1d. Optional: ESM-2 sequence embeddings

```bash
pip install fair-esm
```

Only needed when passing `--seq_encoder esm2_650m` to `train.py`.
If not installed, the model falls back to one-hot encoding automatically.

### 1e. Optional: Weights & Biases logging

```bash
pip install wandb
wandb login
```

### 1f. Optional: MMseqs2 for sequence redundancy removal

```bash
conda install -c conda-forge -c bioconda mmseqs2 -y
```

`filter.py` falls back to a slower Python pairwise implementation if
`mmseqs2` is not on `PATH`.

---

## 2. Data Pipeline

All three scripts write their output to `pdb_data/` by default.
Run them in order: fetch → parse → filter.

### 2a. Fetch PDB data

Downloads NMR ensemble PDB files and paired X-ray structures from RCSB.

```bash
python data/fetch_pdb.py \
    --output_dir     pdb_data \
    --min_conformers 5 \
    --max_nmr        15000 \
    --max_workers    8
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--output_dir` | `./pdb_data` | Root directory for all downloaded files |
| `--min_conformers` | `5` | Skip NMR entries with fewer than this many models |
| `--max_nmr` | `15000` | Maximum NMR entries to fetch from RCSB |
| `--max_workers` | `8` | Parallel download threads |
| `--no_download` | off | Fetch IDs only, skip PDB file downloads |

Output layout:
```
pdb_data/
  nmr/          *.pdb   (NMR training files)
  paired/       *.pdb   (held-out NMR+X-ray pairs)
  nmr_all.json
  nmr_train.json
  paired.json
  paired_nmr_ids.json
  paired_xray_ids.json
```

Expected time: 2–6 hours (RCSB is rate-limited; safe to run overnight).

### 2b. Parse NMR structures

Extracts backbone coordinates (N, CA, C, CB) from every NMR MODEL record,
imputes missing N/C atoms, removes outlier conformers (>3-sigma CA RMSD),
and saves compressed `.npz` tensor files.

```bash
python data/parse_nmr.py \
    --nmr_dir    pdb_data/nmr \
    --output_dir pdb_data/parsed_nmr \
    --min_conformers 5 \
    --max_residues   1000
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--nmr_dir` | required | Directory of NMR `.pdb` files |
| `--output_dir` | required | Where to write `.npz` and `.json` files |
| `--min_conformers` | `5` | Drop entries with fewer valid models after parsing |
| `--max_residues` | `1000` | Drop entries longer than this |

Output per entry: `PDBID.npz` (coords shape `(M, L, 4, 3)`) + `PDBID.json` (metadata).

Expected time: 30–90 minutes.

### 2c. Filter and split

Applies quality filters, enforces sequence redundancy between train and
val/test using MMseqs2 at 30% identity, and produces train/val/test splits.

```bash
python data/filter.py \
    --parsed_dir       pdb_data/parsed_nmr \
    --output_dir       pdb_data/splits \
    --paired_ids_json  pdb_data/paired_nmr_ids.json \
    --min_conformers   5 \
    --min_residues     20 \
    --max_residues     800 \
    --cluster_identity 0.30 \
    --temporal_split \
    --test_cutoff_year 2020
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--parsed_dir` | required | Directory of `.npz` files from parse step |
| `--output_dir` | required | Where to write split JSON files |
| `--paired_ids_json` | `None` | Exclude these NMR IDs from training (held-out set) |
| `--min_conformers` | `5` | Minimum conformers after parsing |
| `--min_residues` | `20` | Minimum sequence length |
| `--max_residues` | `800` | Maximum sequence length |
| `--cluster_identity` | `0.30` | MMseqs2 identity threshold; `0` skips clustering |
| `--temporal_split` | off | Split by deposition year instead of random |
| `--test_cutoff_year` | `2020` | Entries deposited >= this year go to test |

Output: `pdb_data/splits/train.json`, `val.json`, `test.json`, `filtered_manifest.json`.

---

## 3. Training

### 3a. Single GPU (with bf16)

```bash
python scripts/train.py \
    --config         configs/base_config.yaml \
    --train_manifest pdb_data/splits/train.json \
    --val_manifest   pdb_data/splits/val.json \
    --bf16 \
    --batch_size     4 \
    --max_epochs     100 \
    --lr             1e-4 \
    --seed           42
```

### 3b. Multi-GPU with torchrun (DDP)

```bash
torchrun \
    --nproc_per_node 4 \
    scripts/train.py \
    --config         configs/base_config.yaml \
    --train_manifest pdb_data/splits/train.json \
    --val_manifest   pdb_data/splits/val.json \
    --bf16 \
    --multi_gpu \
    --batch_size     16 \
    --max_epochs     100
```

Set `--nproc_per_node` to the number of GPUs on the node.

### 3c. Resume from checkpoint

```bash
python scripts/train.py \
    --config      configs/base_config.yaml \
    --resume_from checkpoints/ckpt_latest.pt \
    --bf16
```

### 3d. Full flag reference for train.py

**Data:**

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `configs/base_config.yaml` | YAML config file |
| `--train_manifest` | `pdb_data/splits/train.json` | Train split JSON |
| `--val_manifest` | `pdb_data/splits/val.json` | Validation split JSON |

**Architecture:**

| Flag | Choices | Default |
|---|---|---|
| `--struct_repr` | `se3_frames`, `cartesian`, `distance_matrix`, `torsion_angles` | `se3_frames` |
| `--generative_model` | `flow_matching`, `ot_cfm`, `ddpm`, `ddim`, `vae`, `score_matching` | `flow_matching` |
| `--seq_encoder` | `onehot`, `esm2_650m`, `esm2_3b`, `prot_t5`, `none` | `onehot` |
| `--loss_schedule` | `fixed`, `geometry_warmup`, `kl_annealing`, `curriculum` | `fixed` |

**Training hyperparameters:**

| Flag | Default | Meaning |
|---|---|---|
| `--batch_size` | `4` | Proteins per batch |
| `--max_epochs` | `100` | Total training epochs |
| `--lr` | `1e-4` | Peak learning rate |
| `--seed` | `42` | Global random seed |
| `--resume_from` | `None` | Path to checkpoint to resume from |
| `--bf16` | off | bfloat16 mixed precision |
| `--multi_gpu` | off | DistributedDataParallel (use with torchrun) |
| `--use_wandb` | off | Enable Weights & Biases logging |

**Loss weights:**

| Flag | Default | Meaning |
|---|---|---|
| `--lambda_flow` | `1.0` | SE(3) flow matching loss weight |
| `--lambda_ensemble` | `1.0` | Ensemble reconstruction loss weight |
| `--lambda_kl` | `0.01` | KL divergence regularisation weight |
| `--lambda_diversity` | `0.5` | Diversity hinge loss weight |
| `--lambda_geometry` | `0.1` | Backbone geometry loss weight |
| `--lambda_chirality` | `0.1` | L-amino acid chirality penalty weight |

Checkpoints are saved to `checkpoints/ckpt_best.pt`, `ckpt_latest.pt`, and
`ckpt_final.pt` by default. Override with `--output_dir`.

---

## 4. Validation

Evaluates the trained model on the held-out NMR/X-ray paired set. For each
pair, the X-ray structure is the model input and the deposited NMR ensemble
is ground truth.

```bash
python evaluation/validate.py \
    --checkpoint   checkpoints/ckpt_best.pt \
    --pairs_json   pdb_data/paired.json \
    --xray_dir     pdb_data/paired \
    --nmr_dir      pdb_data/paired \
    --output_dir   validation_results \
    --n_conformers 20 \
    --n_steps      20 \
    --method       heun
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | required | Path to trained model checkpoint |
| `--pairs_json` | required | `paired.json` from Phase 1 |
| `--xray_dir` | required | Directory with X-ray PDB files |
| `--nmr_dir` | required | Directory with NMR PDB files |
| `--output_dir` | `validation_results` | Where to write per-protein JSON results |
| `--n_conformers` | `20` | Number of conformers to generate per protein |
| `--n_steps` | `20` | ODE integration steps (20 = fast, 50 = more accurate) |
| `--method` | `heun` | ODE integrator: `heun` or `euler` |
| `--max_proteins` | `None` | Limit for quick tests |

Output: one JSON per protein pair + `validation_results/validation_summary.json`
with aggregated metrics across all four metric levels.

---

## 5. Inference

Generates a conformational ensemble from a single input PDB file.

```bash
python inference/predict.py \
    --checkpoint   checkpoints/ckpt_best.pt \
    --input        your_protein.pdb \
    --n_conformers 20 \
    --n_steps      20 \
    --method       heun \
    --output       predictions/your_protein
```

The predictor accepts both X-ray single-model and NMR multi-model PDB files.
Output: `PredictionResult.ca_coords` array of shape `(n_conformers, L, 3)`.

To call from Python:

```python
from inference.predict import ConformerFlowPredictor

predictor = ConformerFlowPredictor("checkpoints/ckpt_best.pt")
result    = predictor.predict("your_protein.pdb", n_conformers=20, n_steps=20, method="heun")
# result.ca_coords: numpy array (20, L, 3)
# result.sequence:  string
```

---

## 6. B200-Specific Notes

The B200 GPU provides 80 GB HBM3e and native bfloat16 throughput.

**Always use `--bf16`**: bfloat16 is the native compute format on B200/H100/A100.
It does not require `GradScaler` (unlike fp16) and avoids gradient underflow.
The trainer applies `GradScaler` only when fp16 is used.

**Batch size**: with 80 GB VRAM the config validator auto-scales `batch_size` to
64 when using the default `max_residues=800`. To override:

```bash
python scripts/train.py --bf16 --batch_size 64 ...
```

**Multi-GPU with torchrun** (4 x B200 example):

```bash
torchrun --nproc_per_node 4 scripts/train.py \
    --config         configs/base_config.yaml \
    --train_manifest pdb_data/splits/train.json \
    --val_manifest   pdb_data/splits/val.json \
    --bf16 \
    --multi_gpu \
    --batch_size 64 \
    --max_epochs 100 \
    --lr 1e-4 \
    --seed 42
```

DDP uses NCCL backend. Each rank gets a disjoint data shard via
`DistributedSampler`. Only rank 0 writes checkpoints and logs.

**EMA**: the trainer maintains EMA weights (`ema_decay=0.9999`). Validation
and checkpoint saving always use EMA weights, not live weights. This is the
standard practice for flow matching models.

---

## 7. Quick Reference: File Locations

```
conformerflow/files/
  configs/base_config.yaml       # all hyperparameters with defaults
  data/fetch_pdb.py              # Phase 1: download from RCSB
  data/parse_nmr.py              # Phase 1: extract conformer tensors
  data/filter.py                 # Phase 1: quality filter and split
  model/frames.py                # SE(3) backbone frame builder
  model/encoder.py               # IPA-based structure encoder
  model/ensemble_stats.py        # distribution parameter module
  model/flow_matching.py         # SE(3) flow matching + ODE sampler
  training/losses.py             # all 6 loss components
  training/trainer.py            # training loop with DDP and EMA
  scripts/train.py               # CLI entry point for training
  evaluation/validate.py         # held-out evaluation pipeline
  evaluation/metrics.py          # 4-level metric suite
  inference/predict.py           # inference API
  checkpoints/ckpt_best.pt       # best validation checkpoint
  checkpoints/ckpt_latest.pt     # most recent checkpoint
  pdb_data/splits/train.json     # training manifest
  pdb_data/splits/val.json       # validation manifest
  pdb_data/splits/test.json      # test manifest
  pdb_data/paired.json           # held-out NMR/X-ray pairs
```
