"""
ConformerFlow — Inference Predictor
Generates a conformational ensemble from a single PDB file.

After setup (step 1), this file lives at inference/predict.py.
Imported by evaluation/validate.py as:
    from inference.predict import ConformerFlowPredictor
"""

import sys
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

logger = logging.getLogger(__name__)

AA1_ORDER = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class PredictionResult:
    """Return value of ConformerFlowPredictor.predict()"""
    ca_coords:    np.ndarray   # (N_conformers, L, 3)
    sequence:     str
    pdb_id:       str
    n_conformers: int
    n_residues:   int


class ConformerFlowPredictor:
    """
    Loads a trained ConformerFlow checkpoint and generates
    conformational ensembles from input PDB files.

    Usage:
        predictor = ConformerFlowPredictor("checkpoints/ckpt_best.pt")
        result    = predictor.predict("protein.pdb", n_conformers=20)
        # result.ca_coords: numpy array (20, L, 3)
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        from model.model_factory import load_model
        self.model, self.cfg = load_model(checkpoint_path, device=str(self.device))
        self.model.eval()

        logger.info(f"ConformerFlowPredictor ready on {self.device}")

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_pdb(self, pdb_path: str) -> dict:
        """
        Parse a PDB file into model-ready numpy arrays.
        Tries the X-ray parser first, falls back to the NMR parser for
        multi-model files. Returns a dict with coords, one_hot, mask, sequence.
        """
        try:
            from data.parse_xray import parse_xray_pdb
            result = parse_xray_pdb(pdb_path)
            if result is not None:
                return result
        except Exception as e:
            logger.debug(f"X-ray parser failed ({e}), trying NMR parser")

        try:
            from data.parse_nmr import parse_nmr_pdb, ensemble_to_tensors
            ensemble = parse_nmr_pdb(pdb_path)
            if ensemble is not None:
                tensors = ensemble_to_tensors(ensemble)
                return {
                    "coords":   tensors["coords"][:1],   # use first model only
                    "one_hot":  tensors["one_hot"],
                    "mask":     tensors["mask"],
                    "sequence": ensemble.sequence,
                }
        except Exception as e:
            logger.debug(f"NMR parser also failed: {e}")

        raise ValueError(f"Could not parse PDB file: {pdb_path}")

    def _build_batch(self, parsed: dict) -> tuple:
        """
        Convert parsed arrays into a B=1 batch dict on the target device.
        Returns (batch, n_residues).
        """
        max_res = self.cfg.get("data", {}).get("max_residues", 800)

        coords  = parsed["coords"]    # (M, L, 4, 3)
        one_hot = parsed["one_hot"]   # (L, 20)
        mask    = parsed["mask"]      # (L, 4)

        L = min(coords.shape[1], max_res)
        coords  = coords[:1, :L]     # single model, truncate to max_res
        one_hot = one_hot[:L]
        mask    = mask[:L]

        coords_t  = torch.from_numpy(coords.astype(np.float32))    # (1, L, 4, 3)
        one_hot_t = torch.from_numpy(one_hot.astype(np.float32))   # (L, 20)
        mask_t    = torch.from_numpy(mask.astype(bool))             # (L, 4)
        seq_mask  = mask_t[:, 1]                                    # CA exists → residue valid

        batch = {
            "coords":         coords_t.unsqueeze(0).to(self.device),      # (1, 1, L, 4, 3)
            "one_hot":        one_hot_t.unsqueeze(0).to(self.device),     # (1, L, 20)
            "mask":           mask_t.unsqueeze(0).to(self.device),        # (1, L, 4)
            "seq_mask":       seq_mask.unsqueeze(0).to(self.device),      # (1, L)
            "conformer_mask": torch.ones(1, 1, dtype=torch.bool,
                                         device=self.device),              # (1, 1)
        }
        return batch, L

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self,
                pdb_path:     str,
                n_conformers: int = 20,
                n_steps:      int = 20,
                method:       str = "heun") -> PredictionResult:
        """
        Generate a conformational ensemble from a single PDB structure.

        Args:
            pdb_path:     path to input PDB file (X-ray or NMR)
            n_conformers: number of conformers to generate
            n_steps:      ODE integration steps (20=fast, 50=accurate)
            method:       ODE integrator ('heun' or 'euler')

        Returns:
            PredictionResult with .ca_coords (n_conformers, L, 3) numpy array
        """
        pdb_path = str(pdb_path)
        pdb_id   = Path(pdb_path).stem

        logger.info(f"Generating ensemble for {pdb_id}  "
                    f"(n_conformers={n_conformers}, steps={n_steps}, method={method})")

        parsed        = self._parse_pdb(pdb_path)
        batch, L      = self._build_batch(parsed)
        sequence      = parsed.get("sequence", "")

        ca_ensemble = self.model.predict_ensemble(
            batch,
            n_conformers = n_conformers,
            n_steps      = n_steps,
            method       = method,
        )   # (1, N, L, 3)

        ca_np = ca_ensemble[0].cpu().numpy()   # (N, L, 3)

        return PredictionResult(
            ca_coords    = ca_np,
            sequence     = sequence,
            pdb_id       = pdb_id,
            n_conformers = n_conformers,
            n_residues   = L,
        )
