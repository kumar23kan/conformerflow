"""
ConformerFlow — Phase 1: X-ray Structure Parser
Parses single-conformation X-ray PDB entries.
Produces the same tensor format as parse_nmr.py for unified inference.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_HOT_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
N_AA = 20


def _get_cb_coord(residue) -> Optional[np.ndarray]:
    """Get CB coordinate; synthesize virtual CB for GLY."""
    if "CB" in residue:
        return np.array(residue["CB"].get_vector().get_array(), dtype=np.float32)
    if "N" in residue and "CA" in residue and "C" in residue:
        n  = np.array(residue["N"].get_vector().get_array())
        ca = np.array(residue["CA"].get_vector().get_array())
        c  = np.array(residue["C"].get_vector().get_array())
        b  = ca - n
        c_vec = c - ca
        a  = np.cross(b, c_vec)
        cb = -0.58273431 * b + 0.56802827 * a - 0.54067466 * c_vec + ca
        return cb.astype(np.float32)
    return None


def _get_primary_chain(model):
    """Select the longest standard protein chain."""
    best_chain, best_len = None, 0
    for chain in model:
        residues = [r for r in chain if is_aa(r, standard=True) and "CA" in r]
        if len(residues) > best_len:
            best_len, best_chain = len(residues), chain
    return best_chain


def parse_xray_pdb(pdb_path: str) -> Optional[dict]:
    """
    Parse a single X-ray PDB file.
    Returns the same tensor format as NMREnsemble tensors,
    with M=1 (single conformer) for unified model interface.
    """
    pdb_path = Path(pdb_path)
    pdb_id   = pdb_path.stem.upper()

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(pdb_path))
    except Exception as e:
        logger.warning(f"Failed to parse {pdb_id}: {e}")
        return None

    # X-ray: take first model only
    try:
        model = next(structure.get_models())
    except StopIteration:
        return None

    chain = _get_primary_chain(model)
    if chain is None:
        return None

    coords_list  = []
    one_hot_list = []
    mask_list    = []
    sequence     = []

    for residue in chain:
        if not is_aa(residue, standard=True):
            continue
        resname = residue.get_resname().strip()
        if resname not in AA3_TO_1 or "CA" not in residue:
            continue

        aa1 = AA3_TO_1[resname]
        sequence.append(aa1)

        # One-hot
        oh = np.zeros(N_AA, dtype=np.float32)
        if aa1 in ONE_HOT_MAP:
            oh[ONE_HOT_MAP[aa1]] = 1.0
        one_hot_list.append(oh)

        # Atom coordinates — order: N, CA, C, CB
        ca_coord = np.array(residue["CA"].get_vector().get_array(), dtype=np.float32)
        n_coord  = np.array(residue["N"].get_vector().get_array(),  dtype=np.float32) \
                   if "N"  in residue else np.zeros(3, dtype=np.float32)
        c_coord  = np.array(residue["C"].get_vector().get_array(),  dtype=np.float32) \
                   if "C"  in residue else np.zeros(3, dtype=np.float32)
        cb_coord = _get_cb_coord(residue)
        if cb_coord is None:
            cb_coord = np.zeros(3, dtype=np.float32)

        atom_coords = np.stack([n_coord, ca_coord, c_coord, cb_coord], axis=0)  # (4, 3)
        coords_list.append(atom_coords)

        atom_mask = np.array([
            "N"  in residue,
            True,            # CA always present
            "C"  in residue,
            "CB" in residue or resname == "GLY",  # virtual CB for GLY
        ], dtype=bool)
        mask_list.append(atom_mask)

    if len(sequence) < 10:
        return None

    L = len(sequence)

    # Shape: (1, L, 4, 3) — M=1 for X-ray single conformation
    coords  = np.stack(coords_list,  axis=0)[np.newaxis]   # (1, L, 4, 3)
    one_hot = np.stack(one_hot_list, axis=0)                # (L, 20)
    mask    = np.stack(mask_list,    axis=0)                # (L, 4)

    return {
        "pdb_id":       pdb_id,
        "sequence":     "".join(sequence),
        "coords":       coords,
        "one_hot":      one_hot,
        "mask":         mask,
        "n_conformers": 1,
        "n_residues":   L,
        "source":       "xray",
    }


def parse_xray_directory(xray_dir: str,
                         output_dir: str,
                         max_residues: int = 1000) -> dict:
    """
    Parse all X-ray PDB files in a directory.
    Saves each as a compressed .npz file.
    """
    xray_dir   = Path(xray_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = list(xray_dir.glob("*.pdb"))
    logger.info(f"Found {len(pdb_files)} X-ray PDB files")

    stats = {"total": len(pdb_files), "parsed": 0, "failed": 0, "manifest": []}

    for pdb_file in pdb_files:
        result = parse_xray_pdb(str(pdb_file))
        if result is None or result["n_residues"] > max_residues:
            stats["failed"] += 1
            continue

        save_path = output_dir / f"{result['pdb_id']}.npz"
        np.savez_compressed(
            str(save_path),
            coords  = result["coords"],
            one_hot = result["one_hot"],
            mask    = result["mask"],
        )

        meta_path = output_dir / f"{result['pdb_id']}.json"
        with open(meta_path, "w") as f:
            json.dump({
                "pdb_id":     result["pdb_id"],
                "sequence":   result["sequence"],
                "n_residues": result["n_residues"],
                "source":     "xray",
            }, f)

        stats["parsed"] += 1
        stats["manifest"].append({
            "pdb_id":     result["pdb_id"],
            "n_residues": result["n_residues"],
            "npz_path":   str(save_path),
        })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(stats["manifest"], f, indent=2)

    logger.info(f"X-ray parsing: {stats['parsed']} parsed, {stats['failed']} failed.")
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ConformerFlow — X-ray Structure Parser")
    parser.add_argument("--xray_dir",    type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--max_residues",type=int, default=1000)
    args = parser.parse_args()
    parse_xray_directory(args.xray_dir, args.output_dir, args.max_residues)
