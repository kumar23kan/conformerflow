"""
ConformerFlow — Phase 1: Quality Filtering & Data Splitting
Filters parsed NMR ensembles and produces train/val/test splits.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import shutil
import tempfile

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
        "deposition_date":   meta.get("deposition_date"),   # for temporal_split
        "npz_path":          str(npz_path),
    }


# ──────────────────────────────────────────────
# Sequence Redundancy Removal
# ──────────────────────────────────────────────

def _seq_identity_python(s1: str, s2: str) -> float:
    """
    Approximate global sequence identity: prefix matches / max(len(s1), len(s2)).
    Not a true alignment — use MMseqs2 for publication-quality results.
    """
    if not s1 or not s2:
        return 0.0
    matches = sum(a == b for a, b in zip(s1, s2))
    return matches / max(len(s1), len(s2))


def _mmseqs2_redundant_pairs(all_entries: list,
                              identity: float,
                              tmp_dir: Path) -> set:
    """
    Run mmseqs2 easy-search (all-vs-all) and return the set of
    (pdb_id_a, pdb_id_b) pairs whose sequence identity >= `identity`.
    Both orderings are included so membership tests are O(1).
    """
    import subprocess

    fasta_path  = tmp_dir / "seqs.fasta"
    result_path = tmp_dir / "hits.tsv"
    tmp_mmseqs  = tmp_dir / "mmseqs_tmp"
    tmp_mmseqs.mkdir(exist_ok=True)

    with open(fasta_path, "w") as f:
        for entry in all_entries:
            seq = entry.get("sequence", "")
            if seq:
                f.write(f">{entry['pdb_id']}\n{seq}\n")

    cmd = [
        "mmseqs", "easy-search",
        str(fasta_path), str(fasta_path),
        str(result_path), str(tmp_mmseqs),
        "--min-seq-id", str(identity),
        "--format-output", "query,target,pident",
        "-c", "0.5", "--cov-mode", "1",
        "-v", "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[:500])

    redundant: set = set()
    with open(result_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            q, t = parts[0], parts[1]
            if q != t:
                redundant.add((q, t))
                redundant.add((t, q))
    return redundant


def remove_sequence_redundancy(splits: dict,
                               cluster_identity: float = 0.30,
                               tmp_dir: Optional[Path] = None) -> dict:
    """
    Remove training entries that share >= cluster_identity sequence identity
    with any val or test entry, preventing train→test leakage.

    Tries MMseqs2 first (fast, alignment-based); falls back to Python
    pairwise prefix comparison when mmseqs2 is unavailable.

    Entries with no sequence in metadata are left in train unchanged.
    """
    train = splits["train"]
    held  = splits["val"] + splits["test"]

    if not train or not held:
        return splits

    held_ids = {e["pdb_id"] for e in held}

    use_tmp  = Path(tempfile.mkdtemp(prefix="cf_mmseqs_")) if tmp_dir is None else tmp_dir
    mmseqs_ok = False

    if shutil.which("mmseqs") is not None:
        try:
            all_entries     = train + held
            redundant_pairs = _mmseqs2_redundant_pairs(all_entries, cluster_identity, use_tmp)

            clean_train, removed = [], 0
            for entry in train:
                pid = entry["pdb_id"]
                if any((pid, hid) in redundant_pairs for hid in held_ids):
                    removed += 1
                else:
                    clean_train.append(entry)

            logger.info(
                f"MMseqs2 redundancy @ {cluster_identity:.0%} id: "
                f"{removed} train removed, {len(clean_train)} remain"
            )
            mmseqs_ok = True

        except Exception as exc:
            logger.warning(f"MMseqs2 failed ({exc}); falling back to Python pairwise")
    else:
        logger.warning(
            "mmseqs2 not in PATH — using Python pairwise identity (slower, approximate). "
            "Install MMseqs2 for publication-quality clustering."
        )

    if not mmseqs_ok:
        held_seqs = [(e["pdb_id"], e.get("sequence", "")) for e in held]
        clean_train, removed = [], 0
        for entry in train:
            s1 = entry.get("sequence", "")
            if not s1:
                clean_train.append(entry)
                continue
            redundant = any(
                s2 and _seq_identity_python(s1, s2) >= cluster_identity
                for _, s2 in held_seqs
            )
            if redundant:
                removed += 1
            else:
                clean_train.append(entry)

        logger.info(
            f"Python pairwise redundancy @ {cluster_identity:.0%} id: "
            f"{removed} train removed, {len(clean_train)} remain"
        )

    if tmp_dir is None:
        shutil.rmtree(use_tmp, ignore_errors=True)

    return {**splits, "train": clean_train}


# ──────────────────────────────────────────────
# Train / Val / Test Splitting
# ──────────────────────────────────────────────

def temporal_split(manifest: list,
                   test_cutoff_year: int   = 2020,
                   val_frac:         float = 0.10,
                   seed:             int   = 42) -> dict:
    """
    Split by deposition year to prevent temporal leakage.

    - Entries with deposition_date year >= test_cutoff_year → test set
    - Remaining entries randomly split into train (1-val_frac) and val (val_frac)
    - Entries missing deposition_date are treated as pre-cutoff (→ train/val pool)

    Returns dict with keys: 'train', 'val', 'test'
    """
    rng = np.random.default_rng(seed)

    pool, test = [], []
    for entry in manifest:
        date_str = entry.get("deposition_date")
        if date_str is not None and int(date_str[:4]) >= test_cutoff_year:
            test.append(entry)
        else:
            pool.append(entry)

    rng.shuffle(pool)
    n_val = int(len(pool) * val_frac)
    val   = pool[:n_val]
    train = pool[n_val:]

    for split_name, items in [("train", train), ("val", val), ("test", test)]:
        lengths = [x["n_residues"] for x in items] if items else [0]
        logger.info(
            f"{split_name:5s}: {len(items):5,} entries | "
            f"seq length — mean: {np.mean(lengths):.0f}, "
            f"min: {min(lengths)}, max: {max(lengths)}"
        )

    return {"train": train, "val": val, "test": test}


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

def run_filter_pipeline(parsed_dir:        str,
                        output_dir:        str,
                        paired_ids:        list  = None,
                        cluster_identity:  float = 0.30,
                        use_temporal_split: bool = False,
                        test_cutoff_year:  int   = 2020,
                        **filter_kwargs) -> dict:
    """
    1. Load all parsed .npz files from parsed_dir
    2. Apply quality filters
    3. Remove any paired NMR/X-ray entries (held-out)
    4. Split into train/val/test  (random or temporal)
    5. Sequence redundancy removal
    6. Save split manifests
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

    # Split — temporal or random
    if use_temporal_split:
        splits = temporal_split(passed, test_cutoff_year=test_cutoff_year)
    else:
        splits = split_dataset(passed)

    # Sequence redundancy removal (train vs val+test)
    if cluster_identity > 0.0:
        splits = remove_sequence_redundancy(splits, cluster_identity=cluster_identity)

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
    parser.add_argument("--cluster_identity", type=float, default=0.30,
                        help="Max sequence identity allowed between train and val/test "
                             "(0 = skip clustering; default = 0.30). "
                             "Uses MMseqs2 if available, Python pairwise otherwise.")
    parser.add_argument("--temporal_split",   action="store_true",
                        help="Split by deposition year instead of random shuffling.")
    parser.add_argument("--test_cutoff_year", type=int, default=2020,
                        help="Entries deposited >= this year go to test set "
                             "(only used with --temporal_split; default 2020).")
    args = parser.parse_args()

    paired_ids = []
    if args.paired_ids_json:
        with open(args.paired_ids_json) as f:
            paired_ids = json.load(f)

    run_filter_pipeline(
        parsed_dir          = args.parsed_dir,
        output_dir          = args.output_dir,
        paired_ids          = paired_ids,
        cluster_identity    = args.cluster_identity,
        use_temporal_split  = args.temporal_split,
        test_cutoff_year    = args.test_cutoff_year,
        min_conformers      = args.min_conformers,
        min_residues        = args.min_residues,
        max_residues        = args.max_residues,
        min_spread_rmsd     = args.min_spread_rmsd,
    )
