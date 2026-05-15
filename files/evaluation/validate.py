"""
ConformerFlow — Phase 6: Full Validation Pipeline

Runs systematic evaluation on the held-out set:
proteins with BOTH X-ray AND NMR structures in PDB.

For each held-out protein:
  1. Feed X-ray structure → ConformerFlow → predicted ensemble
  2. Load deposited NMR ensemble (ground truth)
  3. Run all 4 levels of metrics
  4. Aggregate results and produce summary report
"""

import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

import torch

from inference.predict    import ConformerFlowPredictor
from data.parse_nmr       import parse_nmr_pdb, ensemble_to_tensors
from data.parse_xray      import parse_xray_pdb
from evaluation.metrics   import evaluate_ensemble

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# VALIDATION RUNNER
# ─────────────────────────────────────────────────────────

class ConformerFlowValidator:
    """
    Systematic validation of ConformerFlow on the held-out set.

    Held-out proteins: those with BOTH X-ray and NMR entries
    in the PDB (identified during Phase 1 data preparation).

    For each pair:
      - X-ray structure → input to ConformerFlow
      - NMR ensemble    → ground truth for comparison
    """

    def __init__(self,
                 checkpoint_path: str,
                 n_conformers:    int   = 20,
                 n_steps:         int   = 20,
                 method:          str   = "heun",
                 device:          str   = "auto"):

        self.predictor    = ConformerFlowPredictor(
            checkpoint_path, device=device
        )
        self.n_conformers = n_conformers
        self.n_steps      = n_steps
        self.method       = method

    def _load_nmr_coords(self, nmr_pdb_path: str) -> tuple:
        """Load NMR ensemble CA coordinates and sequence."""
        ensemble = parse_nmr_pdb(nmr_pdb_path)
        if ensemble is None:
            return None, None
        tensors  = ensemble_to_tensors(ensemble)
        # CA is atom index 1: (M, L, 3)
        ca_coords = tensors["coords"][:, :, 1, :]
        return ca_coords, ensemble.sequence

    def _load_xray_coords(self, xray_pdb_path: str) -> tuple:
        """Load X-ray structure CA coordinates and sequence."""
        result = parse_xray_pdb(xray_pdb_path)
        if result is None:
            return None, None
        # CA is atom index 1: (1, L, 3)
        ca_coords = result["coords"][:, :, 1, :]
        return ca_coords, result["sequence"]

    def validate_one(self,
                     xray_pdb:  str,
                     nmr_pdb:   str,
                     pdb_id:    str = "") -> dict:
        """
        Validate predictions for one protein pair.

        Args:
            xray_pdb: path to X-ray PDB file (model input)
            nmr_pdb:  path to NMR PDB file (ground truth)
            pdb_id:   protein identifier

        Returns:
            dict with all evaluation metrics
        """
        logger.info(f"Validating: {pdb_id}")

        # ── Generate predicted ensemble ──
        try:
            pred_result = self.predictor.predict(
                pdb_path     = xray_pdb,
                n_conformers = self.n_conformers,
                n_steps      = self.n_steps,
                method       = self.method,
            )
            pred_coords = pred_result.ca_coords   # (N_pred, L, 3)
            pred_seq    = pred_result.sequence
        except Exception as e:
            logger.warning(f"  Prediction failed: {e}")
            return {"pdb_id": pdb_id, "status": "prediction_failed",
                    "error": str(e)}

        # ── Load NMR ground truth ──
        try:
            true_coords, true_seq = self._load_nmr_coords(nmr_pdb)
            if true_coords is None:
                raise ValueError("NMR parsing returned None")
        except Exception as e:
            logger.warning(f"  NMR loading failed: {e}")
            return {"pdb_id": pdb_id, "status": "nmr_load_failed",
                    "error": str(e)}

        # ── Align sequence lengths ──
        L_pred = pred_coords.shape[1]
        L_true = true_coords.shape[1]
        L      = min(L_pred, L_true)

        if L < 10:
            return {"pdb_id": pdb_id, "status": "too_short"}

        pred_coords = pred_coords[:, :L, :]
        true_coords = true_coords[:, :L, :]

        # ── Run all metrics ──
        try:
            metrics = evaluate_ensemble(
                pred_coords = pred_coords,
                true_coords = true_coords,
                pdb_id      = pdb_id,
            )
            metrics["status"]          = "success"
            metrics["n_pred"]          = pred_coords.shape[0]
            metrics["n_true"]          = true_coords.shape[0]
            metrics["n_residues"]      = L
            metrics["pred_seq_len"]    = L_pred
            metrics["true_seq_len"]    = L_true

            logger.info(
                f"  coverage_rmsd={metrics['coverage_rmsd']:.3f}  "
                f"rmsf_r={metrics['rmsf_pearson_r']:.3f}  "
                f"tm={metrics['tm_score']:.3f}  "
                f"cov_sim={metrics['covariance_frobenius_sim']:.3f}"
            )

        except Exception as e:
            logger.warning(f"  Metrics computation failed: {e}")
            return {"pdb_id": pdb_id, "status": "metrics_failed",
                    "error": str(e)}

        return metrics

    def validate_dataset(self,
                          pairs_json:  str,
                          xray_dir:    str,
                          nmr_dir:     str,
                          output_dir:  str,
                          max_proteins: int = None) -> dict:
        """
        Validate on the full held-out paired dataset.

        Args:
            pairs_json:   path to paired_ids JSON from Phase 1
            xray_dir:     directory with X-ray PDB files
            nmr_dir:      directory with NMR PDB files
            output_dir:   where to save results
            max_proteins: limit for testing (None = all)

        Returns:
            summary dict with aggregated metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(pairs_json) as f:
            pairs = json.load(f)

        if max_proteins:
            pairs = pairs[:max_proteins]

        logger.info(f"Validating on {len(pairs)} protein pairs...")

        all_results = []
        for i, pair in enumerate(pairs):
            nmr_id  = pair["nmr_id"]
            xray_id = pair["xray_id"]

            xray_path = Path(xray_dir) / f"{xray_id.upper()}.pdb"
            nmr_path  = Path(nmr_dir)  / f"{nmr_id.upper()}.pdb"

            if not xray_path.exists() or not nmr_path.exists():
                logger.warning(
                    f"Missing files for {nmr_id}/{xray_id} — skipping"
                )
                continue

            result = self.validate_one(
                xray_pdb = str(xray_path),
                nmr_pdb  = str(nmr_path),
                pdb_id   = f"{xray_id}_vs_{nmr_id}",
            )
            all_results.append(result)

            # Save individual result
            out_path = output_dir / f"{xray_id}_vs_{nmr_id}.json"
            # Exclude numpy arrays from JSON
            serializable = {
                k: v for k, v in result.items()
                if not isinstance(v, np.ndarray)
            }
            with open(out_path, "w") as f:
                json.dump(serializable, f, indent=2)

        # ── Aggregate metrics ──
        summary = self._aggregate_results(all_results)
        summary["n_total"]  = len(pairs)
        summary["n_success"]= sum(1 for r in all_results
                                   if r.get("status") == "success")

        # Save summary
        summary_path = output_dir / "validation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self._print_report(summary)
        return summary

    def _aggregate_results(self, results: list) -> dict:
        """Aggregate metrics across all successful predictions."""
        success = [r for r in results if r.get("status") == "success"]
        if not success:
            return {"error": "No successful predictions"}

        scalar_metrics = [
            "coverage_rmsd", "precision_rmsd", "mean_pairwise_rmsd",
            "mean_struct_rmsd", "tm_score", "gdt_ts",
            # P2-2: threshold coverage metrics
            "nmr_recall_1A", "nmr_recall_2A", "nmr_recall_5A",
            "gen_precision_1A", "gen_precision_2A", "gen_precision_5A",
            "rmsf_pearson_r", "rmsf_spearman_r", "rmsf_mae",
            "torsion_overlap", "torsion_js_div",
            "covariance_frobenius_sim", "covariance_pearson_r",
            "top_mode_overlap",
            "bond_len_mean", "bond_len_mae", "bond_len_outliers",
            "clash_score_per100",   # P2-4: MolProbity convention
            "clash_fraction",
        ]

        summary = {}
        for metric in scalar_metrics:
            vals = [r[metric] for r in success if metric in r
                    and isinstance(r[metric], (int, float))
                    and not np.isnan(r[metric])]
            if vals:
                summary[f"{metric}_mean"] = float(np.mean(vals))
                summary[f"{metric}_std"]  = float(np.std(vals))
                summary[f"{metric}_median"]= float(np.median(vals))

        return summary

    def _print_report(self, summary: dict):
        """Print a human-readable validation report."""
        print()
        print("=" * 65)
        print("  ConformerFlow Validation Report")
        print("=" * 65)
        print(f"  Proteins evaluated: {summary.get('n_success', 0)} / "
              f"{summary.get('n_total', 0)}")
        print()

        sections = [
            ("Level 1 — Structural",
             ["coverage_rmsd", "precision_rmsd",
              "mean_struct_rmsd", "tm_score", "gdt_ts"]),
            ("Level 1b — Coverage at RMSD thresholds",
             ["nmr_recall_1A", "nmr_recall_2A", "nmr_recall_5A",
              "gen_precision_1A", "gen_precision_2A", "gen_precision_5A"]),
            ("Level 2 — Conformational Distribution",
             ["rmsf_pearson_r", "rmsf_spearman_r",
              "torsion_overlap", "torsion_js_div"]),
            ("Level 3 — Correlated Motions",
             ["covariance_frobenius_sim", "covariance_pearson_r",
              "top_mode_overlap"]),
            ("Level 4 — Physical Validity",
             ["bond_len_mean", "bond_len_mae",
              "bond_len_outliers", "clash_score_per100"]),
        ]

        for section_name, metrics in sections:
            print(f"  {section_name}")
            print(f"  {'-' * 55}")
            for m in metrics:
                mean_key = f"{m}_mean"
                std_key  = f"{m}_std"
                if mean_key in summary:
                    mean = summary[mean_key]
                    std  = summary.get(std_key, 0)
                    print(f"    {m:35s}: {mean:7.4f} ± {std:.4f}")
            print()

        print("=" * 65)


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(
        description="ConformerFlow — Validation Pipeline"
    )
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--pairs_json",    required=True,
                        help="paired.json from Phase 1 data pipeline")
    parser.add_argument("--xray_dir",      required=True)
    parser.add_argument("--nmr_dir",       required=True)
    parser.add_argument("--output_dir",    default="validation_results")
    parser.add_argument("--n_conformers",  type=int,   default=20)
    parser.add_argument("--n_steps",       type=int,   default=20)
    parser.add_argument("--method",        default="heun")
    parser.add_argument("--max_proteins",  type=int,   default=None)
    parser.add_argument("--device",        default="auto")
    args = parser.parse_args()

    validator = ConformerFlowValidator(
        checkpoint_path = args.checkpoint,
        n_conformers    = args.n_conformers,
        n_steps         = args.n_steps,
        method          = args.method,
        device          = args.device,
    )
    validator.validate_dataset(
        pairs_json   = args.pairs_json,
        xray_dir     = args.xray_dir,
        nmr_dir      = args.nmr_dir,
        output_dir   = args.output_dir,
        max_proteins = args.max_proteins,
    )
