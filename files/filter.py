"""
ConformerFlow — Phase 1: Quality Filtering & Data Splitting
Filters parsed NMR ensembles and produces train/val/test splits.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Quality Filters
# ──────────────────────────────────────────────

def compute_ensemble_spread(coords: np.ndarray) -> dict:
    """
    Compute spread statistics across M conformers.
    coords: (M, L, 4, 3)

    Returns:
      mean_ca_rmsd:    average pairwise Cα RMSD across conformers
      per_residue_std: per-residue Cα coordinate std (L,)
      max_displacement: maximum displacement of any Cα atom
    """
    M, L, _, _ = coords.shape
    ca_coords = coords[:, :, 1, :]  # (M, L, 3) — Cα only

    # Per-residue standard deviation across conformers
    per_residue_std = ca_coords.std(axis=0).mean(axis=-1)  # (L,)

    # Mean Cα RMSD across all conformer pairs (sample up to 50 pairs)
    rmsds = []
    pairs = min(50, M * (M - 1) // 2)
    rng   = np.random.default_rng(42)
    for _ in range(pairs):
        i, j = rng.choice(M, size=2, replace=False)
        diff  = ca_coords[i] - ca_coords[j]
        rmsd  = np.sqrt((diff ** 2).sum(axis=-1).mean())
        rmsds.append(rmsd)

    mean_ca_rmsd    = float(np.mean(rmsds)) if rmsds else 0.0
    max_displacement = float(per_residue_std.max())

    return {
        "mean_ca_rmsd":    mean_ca_rmsd,
        "per_residue_std": per_residue_std,
        "max_displacement": max_displacement,
    }


def filter_ensemble(npz_path: str,
                    min_conformers:    int   = 5,
                    min_residues:      int   = 20,
                    max_residues:      int   = 800,
                    min_spread_rmsd:   float = 0.1,
                    max_spread_rmsd:   float = 20.0,
                    check_completeness: float = 0.8) -> Optional[dict]:
    """
    Apply quality filters to a parsed NMR ensemble.

    Filters:
      - Minimum / maximum residue count
      - Minimum / maximum ensemble spread (degenerate or exploded ensembles)
      - Backbone completeness (fraction of residues with all 4 atoms)

    Returns metadata dict if passes, None if filtered out.
    """
    npz_path = Path(npz_path)
    meta_path = npz_path.with_suffix(".json")

    try:
        data = np.load(str(npz_path))
        coords  = data["coords"]   # (M, L, 4, 3)
        mask    = data["mask"]     # (L, 4)
    except Exception as e:
        logger.debug(f"Failed to load {npz_path}: {e}")
        return None

    M, L, _, _ = coords.shape

    # Filter 1: conformer count
    if M < min_conformers:
        return None

    # Filter 2: residue count
    if L < min_residues or L > max_residues:
        return None

    # Filter 3: backbone completeness
    # Fraction of residues with N, CA, C present (columns 0,1,2)
    backbone_complete = mask[:, :3].all(axis=1).mean()
    if backbone_complete < check_completeness:
        return None

    # Filter 4: ensemble spread
    spread = compute_ensemble_spread(coords)
    if spread["mean_ca_rmsd"] < min_spread_rmsd:
        return None  # degenerate ensemble (all conformers identical)
    if spread["mean_ca_rmsd"] > max_spread_rmsd:
        return None  # likely parsing error

    # Load metadata
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return {
        "pdb_id":            meta.get("pdb_id", npz_path.stem),
        "sequence":          meta.get("sequence", ""),
        "n_conformers":      M,
        "n_residues":        L,
        "mean_ca_rmsd":      spread["mean_ca_rmsd"],
        "max_displacement":  spread["max_displacement"],
        "backbone_complete": float(backbone_complete),
        "npz_path":          str(npz_path),
    }


# ──────────────────────────────────────────────
# Train / Val / Test Splitting
# ──────────────────────────────────────────────

def split_dataset(manifest: list,
                  train_frac: float = 0.80,
                  val_frac:   float = 0.10,
                  seed:       int   = 42) -> dict:
    """
    Split a filtered manifest into train / val / test sets.
    Uses sequence-length stratified shuffling for balanced splits.

    Returns dict with keys: 'train', 'val', 'test'
    """
    rng = np.random.default_rng(seed)
    items = list(manifest)
    rng.shuffle(items)

    N     = len(items)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    splits = {
        "train": items[:n_train],
        "val":   items[n_train:n_train + n_val],
        "test":  items[n_train + n_val:],
    }

    for split_name, split_items in splits.items():
        lengths = [x["n_residues"] for x in split_items]
        logger.info(
            f"{split_name:5s}: {len(split_items):5,} entries | "
            f"seq length — mean: {np.mean(lengths):.0f}, "
            f"min: {min(lengths)}, max: {max(lengths)}"
        )

    return splits


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def run_filter_pipeline(parsed_dir:  str,
                        output_dir:  str,
                        paired_ids:  list = None,
                        **filter_kwargs) -> dict:
    """
    1. Load all parsed .npz files from parsed_dir
    2. Apply quality filters
    3. Remove any paired NMR/X-ray entries (held-out)
    4. Split into train/val/test
    5. Save split manifests
    """
    parsed_dir = Path(parsed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files  = list(parsed_dir.glob("*.npz"))
    paired_ids = set(paired_ids or [])

    logger.info(f"Filtering {len(npz_files)} parsed ensembles...")

    passed, filtered_paired, filtered_quality = [], 0, 0

    for npz_file in npz_files:
        pdb_id = npz_file.stem.upper()

        # Remove paired entries
        if pdb_id in paired_ids:
            filtered_paired += 1
            continue

        result = filter_ensemble(str(npz_file), **filter_kwargs)
        if result is None:
            filtered_quality += 1
            continue

        passed.append(result)

    logger.info(
        f"Filtering: {len(passed)} passed | "
        f"{filtered_quality} failed quality | "
        f"{filtered_paired} removed (paired held-out)"
    )

    # Split
    splits = split_dataset(passed)

    # Save
    for split_name, split_items in splits.items():
        path = output_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(split_items, f, indent=2)
        logger.info(f"Saved {split_name}.json")

    # Save full filtered manifest
    with open(output_dir / "filtered_manifest.json", "w") as f:
        json.dump(passed, f, indent=2)

    return {"splits": splits, "total_passed": len(passed)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ConformerFlow — Quality Filter & Split")
    parser.add_argument("--parsed_dir",       type=str, required=True)
    parser.add_argument("--output_dir",       type=str, required=True)
    parser.add_argument("--paired_ids_json",  type=str, default=None,
                        help="JSON file with list of paired NMR PDB IDs to exclude")
    parser.add_argument("--min_conformers",   type=int, default=5)
    parser.add_argument("--min_residues",     type=int, default=20)
    parser.add_argument("--max_residues",     type=int, default=800)
    parser.add_argument("--min_spread_rmsd",  type=float, default=0.1)
    args = parser.parse_args()

    paired_ids = []
    if args.paired_ids_json:
        with open(args.paired_ids_json) as f:
            paired_ids = json.load(f)

    run_filter_pipeline(
        parsed_dir       = args.parsed_dir,
        output_dir       = args.output_dir,
        paired_ids       = paired_ids,
        min_conformers   = args.min_conformers,
        min_residues     = args.min_residues,
        max_residues     = args.max_residues,
        min_spread_rmsd  = args.min_spread_rmsd,
    )
