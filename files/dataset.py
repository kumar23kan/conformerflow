"""
ConformerFlow — Phase 1: PyTorch Dataset Classes
Unified dataset for NMR ensembles (training) and X-ray structures (inference).
Handles variable-length proteins via dynamic padding and masking.
"""

import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# NMR Dataset (Training)
# ──────────────────────────────────────────────

class NMREnsembleDataset(Dataset):
    """
    Dataset for NMR ensemble entries.

    Each item returns:
      coords:       (M, L, 4, 3)  — all M conformers, L residues, N/CA/C/CB
      one_hot:      (L, 20)       — sequence one-hot encoding
      mask:         (L, 4)        — atom existence mask
      seq_mask:     (L,)          — residue existence mask (for padding)
      n_conformers: int
      n_residues:   int
      pdb_id:       str
    """

    def __init__(self,
                 manifest_path: str,
                 max_residues: int = 800,
                 max_conformers: int = 50,
                 augment: bool = True):
        """
        Args:
            manifest_path:  Path to split JSON manifest (train/val/test)
            max_residues:   Pad/truncate sequences to this length
            max_conformers: Cap conformers per entry (memory management)
            augment:        If True, randomly subsample conformers during training
        """
        self.max_residues   = max_residues
        self.max_conformers = max_conformers
        self.augment        = augment

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        logger.info(f"NMREnsembleDataset: {len(self.manifest)} entries from {manifest_path}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        entry    = self.manifest[idx]
        npz_path = entry["npz_path"]

        try:
            data = np.load(npz_path)
        except Exception as e:
            logger.warning(f"Failed to load {npz_path}: {e} — returning zeros")
            return self._empty_item()

        coords  = data["coords"]   # (M, L, 4, 3)
        one_hot = data["one_hot"]  # (L, 20)
        mask    = data["mask"]     # (L, 4)

        M, L = coords.shape[:2]

        # Cap conformers
        if M > self.max_conformers:
            if self.augment:
                # Random subsample during training for augmentation
                idx_m = np.random.choice(M, self.max_conformers, replace=False)
                idx_m = np.sort(idx_m)
            else:
                idx_m = np.arange(self.max_conformers)
            coords = coords[idx_m]
            M = self.max_conformers

        # Truncate long sequences
        if L > self.max_residues:
            if self.augment:
                start = np.random.randint(0, L - self.max_residues)
            else:
                start = 0
            coords  = coords[:, start:start + self.max_residues]
            one_hot = one_hot[start:start + self.max_residues]
            mask    = mask[start:start + self.max_residues]
            L = self.max_residues

        # Pad to max_residues
        L_pad  = self.max_residues
        seq_mask = np.zeros(L_pad, dtype=bool)
        seq_mask[:L] = True

        coords_pad  = np.zeros((M, L_pad, 4, 3), dtype=np.float32)
        one_hot_pad = np.zeros((L_pad, 20),       dtype=np.float32)
        mask_pad    = np.zeros((L_pad, 4),         dtype=bool)

        coords_pad[:, :L]  = coords
        one_hot_pad[:L]    = one_hot
        mask_pad[:L]       = mask

        return {
            "coords":       torch.from_numpy(coords_pad),      # (M, L, 4, 3)
            "one_hot":      torch.from_numpy(one_hot_pad),     # (L, 20)
            "mask":         torch.from_numpy(mask_pad),        # (L, 4)
            "seq_mask":     torch.from_numpy(seq_mask),        # (L,)
            "n_conformers": M,
            "n_residues":   L,
            "pdb_id":       entry["pdb_id"],
        }

    def _empty_item(self) -> dict:
        L = self.max_residues
        M = 1
        return {
            "coords":       torch.zeros(M, L, 4, 3),
            "one_hot":      torch.zeros(L, 20),
            "mask":         torch.zeros(L, 4, dtype=torch.bool),
            "seq_mask":     torch.zeros(L, dtype=torch.bool),
            "n_conformers": M,
            "n_residues":   0,
            "pdb_id":       "UNKNOWN",
        }


# ──────────────────────────────────────────────
# X-ray Dataset (Inference / Validation)
# ──────────────────────────────────────────────

class XRayStructureDataset(Dataset):
    """
    Dataset for X-ray crystal structures.
    Single-conformation input; same format as NMREnsembleDataset
    with M=1 for unified model interface.
    """

    def __init__(self,
                 manifest_path: str,
                 max_residues: int = 800):
        self.max_residues = max_residues

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        logger.info(f"XRayStructureDataset: {len(self.manifest)} entries")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        entry    = self.manifest[idx]
        npz_path = entry["npz_path"]

        try:
            data = np.load(npz_path)
        except Exception as e:
            logger.warning(f"Failed to load {npz_path}: {e}")
            return self._empty_item()

        coords  = data["coords"]   # (1, L, 4, 3)
        one_hot = data["one_hot"]  # (L, 20)
        mask    = data["mask"]     # (L, 4)

        L = coords.shape[1]

        # Truncate if needed
        if L > self.max_residues:
            coords  = coords[:, :self.max_residues]
            one_hot = one_hot[:self.max_residues]
            mask    = mask[:self.max_residues]
            L = self.max_residues

        L_pad    = self.max_residues
        seq_mask = np.zeros(L_pad, dtype=bool)
        seq_mask[:L] = True

        coords_pad  = np.zeros((1, L_pad, 4, 3), dtype=np.float32)
        one_hot_pad = np.zeros((L_pad, 20),       dtype=np.float32)
        mask_pad    = np.zeros((L_pad, 4),         dtype=bool)

        coords_pad[0, :L] = coords[0]
        one_hot_pad[:L]   = one_hot
        mask_pad[:L]      = mask

        return {
            "coords":       torch.from_numpy(coords_pad),
            "one_hot":      torch.from_numpy(one_hot_pad),
            "mask":         torch.from_numpy(mask_pad),
            "seq_mask":     torch.from_numpy(seq_mask),
            "n_conformers": 1,
            "n_residues":   L,
            "pdb_id":       entry["pdb_id"],
        }

    def _empty_item(self) -> dict:
        L = self.max_residues
        return {
            "coords":       torch.zeros(1, L, 4, 3),
            "one_hot":      torch.zeros(L, 20),
            "mask":         torch.zeros(L, 4, dtype=torch.bool),
            "seq_mask":     torch.zeros(L, dtype=torch.bool),
            "n_conformers": 1,
            "n_residues":   0,
            "pdb_id":       "UNKNOWN",
        }


# ──────────────────────────────────────────────
# Collate Function (variable-M batching)
# ──────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """
    Custom collate for variable M (conformer count) across batch items.
    Pads conformer dimension to max M in the batch.
    """
    max_M = max(item["n_conformers"] for item in batch)
    L     = batch[0]["coords"].shape[1]  # already padded to max_residues

    coords_batch   = []
    one_hot_batch  = []
    mask_batch     = []
    seq_mask_batch = []
    conformer_mask = []  # (B, M) — which conformers are real vs padded
    n_residues     = []
    pdb_ids        = []

    for item in batch:
        M_item = item["n_conformers"]
        M_pad  = max_M - M_item

        # Pad conformer dimension with zeros
        coords = item["coords"]  # (M_item, L, 4, 3)
        if M_pad > 0:
            pad    = torch.zeros(M_pad, L, 4, 3)
            coords = torch.cat([coords, pad], dim=0)

        # Conformer validity mask
        c_mask = torch.zeros(max_M, dtype=torch.bool)
        c_mask[:M_item] = True

        coords_batch.append(coords)
        one_hot_batch.append(item["one_hot"])
        mask_batch.append(item["mask"])
        seq_mask_batch.append(item["seq_mask"])
        conformer_mask.append(c_mask)
        n_residues.append(item["n_residues"])
        pdb_ids.append(item["pdb_id"])

    return {
        "coords":        torch.stack(coords_batch),    # (B, M, L, 4, 3)
        "one_hot":       torch.stack(one_hot_batch),   # (B, L, 20)
        "mask":          torch.stack(mask_batch),      # (B, L, 4)
        "seq_mask":      torch.stack(seq_mask_batch),  # (B, L)
        "conformer_mask":torch.stack(conformer_mask),  # (B, M)
        "n_residues":    torch.tensor(n_residues),     # (B,)
        "pdb_ids":       pdb_ids,
    }


# ──────────────────────────────────────────────
# DataLoader Factory
# ──────────────────────────────────────────────

def build_dataloaders(train_manifest: str,
                      val_manifest:   str,
                      test_manifest:  str,
                      batch_size:     int = 4,
                      max_residues:   int = 800,
                      max_conformers: int = 50,
                      num_workers:    int = 4) -> dict:
    """
    Build train / val / test DataLoaders.
    Returns dict with keys 'train', 'val', 'test'.
    """
    train_ds = NMREnsembleDataset(train_manifest, max_residues, max_conformers, augment=True)
    val_ds   = NMREnsembleDataset(val_manifest,   max_residues, max_conformers, augment=False)
    test_ds  = NMREnsembleDataset(test_manifest,  max_residues, max_conformers, augment=False)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=True),
    }
    return loaders
