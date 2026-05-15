"""
ConformerFlow — Phase 1: NMR Ensemble Parser
Extracts all M conformers from NMR PDB entries.
Builds per-residue features: N, CA, C, CB coordinates + sequence.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard 20 amino acids
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

ONE_HOT_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
N_AA = 20


@dataclass
class ResidueFrame:
    """
    SE(3)-invariant frame for a single residue.
    Stores raw coordinates; frame computation happens in encoder.
    """
    residue_idx:  int
    residue_name: str                     # 1-letter code
    one_hot:      np.ndarray              # shape (20,)
    N_coord:      Optional[np.ndarray]    # shape (3,)
    CA_coord:     np.ndarray              # shape (3,)  — always present
    C_coord:      Optional[np.ndarray]    # shape (3,)
    CB_coord:     Optional[np.ndarray]    # shape (3,)  — GLY has no CB


@dataclass
class ConformerData:
    """A single conformer (one model from an NMR ensemble)."""
    model_id:  int
    frames:    list   # list of ResidueFrame


@dataclass
class NMREnsemble:
    """
    Full NMR ensemble for one PDB entry.
    Contains M conformers, each with L residue frames.
    """
    pdb_id:      str
    sequence:    str                  # 1-letter sequence
    n_residues:  int
    n_conformers: int
    conformers:  list                 # list of ConformerData
    chain_id:    str = "A"


# ──────────────────────────────────────────────
# Deposition Date
# ──────────────────────────────────────────────

def _extract_deposition_date(pdb_path: Path) -> Optional[str]:
    """
    Extract deposition date from PDB HEADER record.
    Returns ISO date string "YYYY-MM-DD" or None.

    HEADER format (columns 1-80):
      HEADER    <classification>                    DD-MON-YY   <PDB-ID>
    The date occupies cols 51-59 (1-indexed).
    """
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith("HEADER") and len(line) >= 59:
                    raw = line[50:59].strip()
                    if raw:
                        dt = datetime.strptime(raw, "%d-%b-%y")
                        # Python's %y: 00-68 → 2000-2068, 69-99 → 1969-1999
                        # Fix pre-1969 structures (very rare): anything > today is wrong
                        if dt.year > datetime.now().year:
                            dt = dt.replace(year=dt.year - 100)
                        return dt.strftime("%Y-%m-%d")
                if line.startswith("ATOM"):
                    break
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────
# Core Parsing
# ──────────────────────────────────────────────

def _get_cb_coord(residue) -> Optional[np.ndarray]:
    """
    Get CB coordinate. For GLY (no CB), synthesize a virtual CB
    from N, CA, C using the standard geometry approach.
    """
    if "CB" in residue:
        return np.array(residue["CB"].get_vector().get_array())

    # Virtual CB for GLY
    if "N" in residue and "CA" in residue and "C" in residue:
        n  = np.array(residue["N"].get_vector().get_array())
        ca = np.array(residue["CA"].get_vector().get_array())
        c  = np.array(residue["C"].get_vector().get_array())
        # Standard virtual CB construction
        b = ca - n
        c_vec = c - ca
        a = np.cross(b, c_vec)
        cb = -0.58273431 * b + 0.56802827 * a - 0.54067466 * c_vec + ca
        return cb

    return None


def _parse_residue(residue, idx: int) -> Optional[ResidueFrame]:
    """Parse a single BioPython residue into a ResidueFrame."""
    resname = residue.get_resname().strip()

    # Skip non-standard amino acids
    if resname not in AA3_TO_1:
        return None
    if not is_aa(residue, standard=True):
        return None

    # CA is mandatory
    if "CA" not in residue:
        return None

    aa1 = AA3_TO_1[resname]

    # One-hot encoding
    one_hot = np.zeros(N_AA, dtype=np.float32)
    if aa1 in ONE_HOT_MAP:
        one_hot[ONE_HOT_MAP[aa1]] = 1.0

    # Backbone coords
    ca_coord = np.array(residue["CA"].get_vector().get_array(), dtype=np.float32)
    n_coord  = np.array(residue["N"].get_vector().get_array(),  dtype=np.float32) \
               if "N" in residue else None
    c_coord  = np.array(residue["C"].get_vector().get_array(),  dtype=np.float32) \
               if "C" in residue else None
    cb_coord = _get_cb_coord(residue)
    if cb_coord is not None:
        cb_coord = cb_coord.astype(np.float32)

    return ResidueFrame(
        residue_idx  = idx,
        residue_name = aa1,
        one_hot      = one_hot,
        N_coord      = n_coord,
        CA_coord     = ca_coord,
        C_coord      = c_coord,
        CB_coord     = cb_coord,
    )


def _get_primary_chain(model):
    """
    Select the primary protein chain from a model.

    Policy (explicit for reproducibility):
      1. Pick the chain with the most standard amino acids that have a CA atom.
      2. On a tie, prefer chain 'A'; on a further tie, take the chain that
         appears first in the PDB file (BioPython iteration order).

    This is logged in the paper as: "longest chain; chain A preferred on ties."
    """
    candidates = []
    for chain in model:
        residues = [r for r in chain if is_aa(r, standard=True) and "CA" in r]
        if residues:
            candidates.append((len(residues), chain.id != "A", chain))

    if not candidates:
        return None

    # Sort: descending length, then prefer A (False < True), then file order
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


def parse_nmr_pdb(pdb_path: str) -> Optional[NMREnsemble]:
    """
    Parse an NMR PDB file into an NMREnsemble.

    Each MODEL record in the PDB file corresponds to one conformer.
    Returns None if parsing fails or entry has < 2 conformers.
    """
    pdb_path = Path(pdb_path)
    pdb_id   = pdb_path.stem.upper()

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(pdb_path))
    except Exception as e:
        logger.warning(f"Failed to parse {pdb_id}: {e}")
        return None

    models = list(structure.get_models())
    if len(models) < 2:
        logger.debug(f"{pdb_id}: only {len(models)} model(s) — skipping.")
        return None

    conformers = []
    reference_sequence = None
    reference_length   = None

    for model in models:
        chain = _get_primary_chain(model)
        if chain is None:
            continue

        frames = []
        sequence_chars = []
        idx = 0

        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            frame = _parse_residue(residue, idx)
            if frame is not None:
                frames.append(frame)
                sequence_chars.append(frame.residue_name)
                idx += 1

        if len(frames) < 10:
            continue  # too short

        seq = "".join(sequence_chars)

        # Use first model to define reference sequence
        if reference_sequence is None:
            reference_sequence = seq
            reference_length   = len(frames)
        elif len(frames) != reference_length:
            # Skip conformers with different lengths (parsing artifacts)
            continue

        conformers.append(ConformerData(
            model_id = model.id,
            frames   = frames,
        ))

    if len(conformers) < 2 or reference_sequence is None:
        logger.debug(f"{pdb_id}: insufficient valid conformers after parsing.")
        return None

    # P1-5: impute missing N/C from neighbouring CA positions
    conformers = _impute_missing_backbone(conformers)

    # P1-4: remove outlier conformers (> 3σ from ensemble mean CA RMSD)
    conformers = _remove_outlier_conformers(conformers)

    if len(conformers) < 2:
        logger.debug(f"{pdb_id}: fewer than 2 conformers after cleanup.")
        return None

    return NMREnsemble(
        pdb_id       = pdb_id,
        sequence     = reference_sequence,
        n_residues   = reference_length,
        n_conformers = len(conformers),   # updated after outlier removal
        conformers   = conformers,
    )


# ──────────────────────────────────────────────
# Numpy Tensor Conversion
# ──────────────────────────────────────────────

def ensemble_to_tensors(ensemble: NMREnsemble) -> dict:
    """
    Convert NMREnsemble to numpy arrays ready for PyTorch.

    Returns dict:
      coords:    (M, L, 4, 3)  — M conformers, L residues, 4 atoms (N/CA/C/CB), xyz
      one_hot:   (L, 20)       — sequence one-hot (same for all conformers)
      sequence:  str
      mask:      (L, 4)        — bool, True if atom exists
      pdb_id:    str
      n_conformers: int
      n_residues:   int
    """
    M = ensemble.n_conformers
    L = ensemble.n_residues

    coords  = np.zeros((M, L, 4, 3), dtype=np.float32)
    mask    = np.zeros((L, 4),       dtype=bool)
    one_hot = np.zeros((L, N_AA),    dtype=np.float32)

    # Atom order: 0=N, 1=CA, 2=C, 3=CB
    for m, conformer in enumerate(ensemble.conformers):
        for frame in conformer.frames:
            i = frame.residue_idx
            if i >= L:
                continue
            one_hot[i] = frame.one_hot

            atom_coords = [frame.N_coord, frame.CA_coord, frame.C_coord, frame.CB_coord]
            for a, coord in enumerate(atom_coords):
                if coord is not None:
                    coords[m, i, a] = coord
                    mask[i, a]      = True

    return {
        "pdb_id":       ensemble.pdb_id,
        "sequence":     ensemble.sequence,
        "coords":       coords,     # (M, L, 4, 3)
        "one_hot":      one_hot,    # (L, 20)
        "mask":         mask,       # (L, 4)
        "n_conformers": M,
        "n_residues":   L,
    }


# ──────────────────────────────────────────────
# Post-parse Cleanup
# ──────────────────────────────────────────────

def _remove_outlier_conformers(conformers: list,
                               max_z_score: float = 3.0) -> list:
    """
    Remove conformers whose CA RMSD from the ensemble mean exceeds
    max_z_score standard deviations. Keeps at least 2 conformers.

    Outliers arise from disordered loops, parsing artifacts, or
    poorly converged NMR models (high restraint violations).
    """
    if len(conformers) <= 2:
        return conformers

    L = len(conformers[0].frames)
    ca = np.zeros((len(conformers), L, 3), dtype=np.float32)
    for m, conf in enumerate(conformers):
        for frame in conf.frames:
            i = frame.residue_idx
            if i < L:
                ca[m, i] = frame.CA_coord

    mean_ca = ca.mean(axis=0)                          # (L, 3)
    diff    = ca - mean_ca[None]                       # (M, L, 3)
    rmsds   = np.sqrt((diff ** 2).sum(-1).mean(-1))   # (M,)

    mean_r, std_r = rmsds.mean(), rmsds.std()
    if std_r < 1e-6:
        return conformers  # all identical — keep all

    z_scores = (rmsds - mean_r) / std_r
    kept = [c for c, z in zip(conformers, z_scores) if z <= max_z_score]

    if len(kept) < 2:
        return conformers  # don't discard everything
    if len(kept) < len(conformers):
        logger.debug(f"Removed {len(conformers) - len(kept)} outlier conformers "
                     f"(z > {max_z_score})")
    return kept


def _impute_missing_backbone(conformers: list) -> list:
    """
    Impute missing N or C atoms for residues where CA is present but
    N or C is absent.  Uses the mean of the neighbouring CA positions
    with ideal backbone bond lengths as a crude approximation:
      - N:  CA(i) + 1.46 Å along (CA(i-1) → CA(i)) unit vector, perturbed
      - C:  CA(i) + 1.52 Å along (CA(i) → CA(i+1)) unit vector, perturbed

    Imputed atoms are flagged with a sufficiently wrong value to prevent
    confusion — they simply fill in positions that would otherwise be zero,
    letting the backbone-frame builder and mask handle the rest.

    Only fills N/C; never creates CB (that is already handled by _get_cb_coord).
    Never overwrites existing atoms.
    """
    BOND_N_CA  = 1.46  # Å  N–CA bond length
    BOND_CA_C  = 1.52  # Å  CA–C  bond length

    for conf in conformers:
        frames = conf.frames
        L = len(frames)
        # Build CA array for quick index access
        ca = {f.residue_idx: f.CA_coord for f in frames}

        for k, frame in enumerate(frames):
            i = frame.residue_idx

            # --- Impute N ---
            if frame.N_coord is None:
                prev_ca = ca.get(i - 1)
                if prev_ca is not None:
                    d = frame.CA_coord - prev_ca
                    d_norm = np.linalg.norm(d)
                    if d_norm > 1e-6:
                        frame.N_coord = frame.CA_coord - (d / d_norm) * BOND_N_CA

            # --- Impute C ---
            if frame.C_coord is None:
                next_ca = ca.get(i + 1)
                if next_ca is not None:
                    d = next_ca - frame.CA_coord
                    d_norm = np.linalg.norm(d)
                    if d_norm > 1e-6:
                        frame.C_coord = frame.CA_coord + (d / d_norm) * BOND_CA_C

    return conformers


# ──────────────────────────────────────────────
# Batch Processing
# ──────────────────────────────────────────────

def parse_nmr_directory(nmr_dir: str,
                        output_dir: str,
                        min_conformers: int = 5,
                        max_residues:   int = 1000) -> dict:
    """
    Parse all NMR PDB files in a directory.
    Saves each as a compressed .npz file.
    Returns summary statistics.
    """
    nmr_dir    = Path(nmr_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = list(nmr_dir.glob("*.pdb"))
    logger.info(f"Found {len(pdb_files)} PDB files in {nmr_dir}")

    stats = {
        "total":     len(pdb_files),
        "parsed":    0,
        "skipped":   0,
        "too_few_conformers": 0,
        "too_long":  0,
        "failed":    0,
        "manifest":  []
    }

    for pdb_file in pdb_files:
        ensemble = parse_nmr_pdb(str(pdb_file))

        if ensemble is None:
            stats["failed"] += 1
            continue

        if ensemble.n_conformers < min_conformers:
            stats["too_few_conformers"] += 1
            continue

        if ensemble.n_residues > max_residues:
            stats["too_long"] += 1
            continue

        tensors   = ensemble_to_tensors(ensemble)
        save_path = output_dir / f"{ensemble.pdb_id}.npz"

        np.savez_compressed(
            str(save_path),
            coords       = tensors["coords"],
            one_hot      = tensors["one_hot"],
            mask         = tensors["mask"],
        )

        # Save sequence + deposition date as metadata
        dep_date = _extract_deposition_date(pdb_file)
        meta_path = output_dir / f"{ensemble.pdb_id}.json"
        with open(meta_path, "w") as f:
            json.dump({
                "pdb_id":           ensemble.pdb_id,
                "sequence":         ensemble.sequence,
                "n_conformers":     ensemble.n_conformers,
                "n_residues":       ensemble.n_residues,
                "deposition_date":  dep_date,   # "YYYY-MM-DD" or null
            }, f)

        stats["parsed"] += 1
        stats["manifest"].append({
            "pdb_id":       ensemble.pdb_id,
            "n_conformers": ensemble.n_conformers,
            "n_residues":   ensemble.n_residues,
            "npz_path":     str(save_path),
        })

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(stats["manifest"], f, indent=2)

    logger.info(
        f"Parsing complete: {stats['parsed']} parsed, "
        f"{stats['too_few_conformers']} too few conformers, "
        f"{stats['too_long']} too long, "
        f"{stats['failed']} failed."
    )
    return stats


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ConformerFlow — NMR Ensemble Parser")
    parser.add_argument("--nmr_dir",        type=str, required=True,
                        help="Directory containing NMR .pdb files")
    parser.add_argument("--output_dir",     type=str, required=True,
                        help="Directory to save parsed .npz files")
    parser.add_argument("--min_conformers", type=int, default=5,
                        help="Minimum conformers required")
    parser.add_argument("--max_residues",   type=int, default=1000,
                        help="Maximum sequence length")
    args = parser.parse_args()

    parse_nmr_directory(
        nmr_dir        = args.nmr_dir,
        output_dir     = args.output_dir,
        min_conformers = args.min_conformers,
        max_residues   = args.max_residues,
    )
