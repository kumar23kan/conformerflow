"""
ConformerFlow — Ablation Study Runner

Iterates over a hardcoded table of ablation configurations, runs
ConformerFlowValidator on each available checkpoint, aggregates the
per-run summary JSONs, and prints/saves a formatted ASCII (and
optionally LaTeX) comparison table.

Usage:
    python scripts/run_ablations.py \\
        --pairs_json data/paired.json \\
        --xray_dir   data/xray_pdbs \\
        --nmr_dir    data/nmr_pdbs \\
        --checkpoints_dir checkpoints \\
        --output_dir  results/ablations \\
        --n_conformers 20 \\
        --latex
"""

import sys
import json
import logging
import argparse
from pathlib import Path

# Allow running as  python scripts/run_ablations.py  (parent = project root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.validate import ConformerFlowValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION TABLE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_TABLE = [
    # Reference / full model
    {"name": "full_model",    "run_name": "full_model"},
    # Encoder ablations
    {"name": "enc_cartesian", "run_name": "enc_cartesian"},
    {"name": "enc_distance",  "run_name": "enc_distance"},
    {"name": "enc_torsion",   "run_name": "enc_torsion"},
    # Generative model ablations
    {"name": "gen_ot_cfm",    "run_name": "gen_ot_cfm"},
    {"name": "gen_ddpm",      "run_name": "gen_ddpm"},
    {"name": "gen_ddim",      "run_name": "gen_ddim"},
    {"name": "gen_vae",       "run_name": "gen_vae"},
    {"name": "gen_score",     "run_name": "gen_score"},
    # Latent ablation
    {"name": "latent_global", "run_name": "latent_global"},
    # Multi-head training
    {"name": "multi_head",    "run_name": "multi_head"},
]

# Metrics shown in the comparison table (key → display header)
TABLE_METRICS = [
    ("coverage_rmsd",           "coverage_rmsd"),
    ("precision_rmsd",          "precision_rmsd"),
    ("rmsf_pearson_r",          "rmsf_pearson_r"),
    ("torsion_overlap",         "torsion_overlap"),
    ("covariance_frobenius_sim","cov_frobenius_sim"),
    ("clash_score_per100",      "clash_per100"),
    ("rama_pred_favored",       "rama_favored"),
]

# For each metric: True means *lower* is better, False means *higher* is better
LOWER_IS_BETTER = {
    "coverage_rmsd":           True,
    "precision_rmsd":          True,
    "rmsf_pearson_r":          False,
    "torsion_overlap":         False,
    "covariance_frobenius_sim":False,
    "clash_score_per100":      True,
    "rama_pred_favored":       False,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_summary(summary_path: Path) -> dict:
    """Load a validation_summary.json and return it (or empty dict on error)."""
    try:
        with open(summary_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load summary at {summary_path}: {e}")
        return {}


def _extract_metric(summary: dict, metric: str):
    """
    Return (mean, std) floats for *metric* from a summary dict, or (None, None).

    The validator stores aggregated metrics under the keys
    ``{metric}_mean`` and ``{metric}_std``.
    """
    mean_key = f"{metric}_mean"
    std_key  = f"{metric}_std"
    mean = summary.get(mean_key)
    std  = summary.get(std_key)
    if mean is None:
        return None, None
    return float(mean), float(std) if std is not None else 0.0


def _fmt_cell(mean, std):
    """Format a mean ± std cell, or '—' when data are absent."""
    if mean is None:
        return "—"
    return f"{mean:.4f} ± {std:.4f}"


def _best_index(values, lower_is_better: bool):
    """
    Return the index of the best non-None value in *values*.
    Returns None if all values are absent.
    """
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    if lower_is_better:
        return min(valid, key=lambda x: x[1])[0]
    return max(valid, key=lambda x: x[1])[0]


# ─────────────────────────────────────────────────────────────────────────────
# TABLE RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def _build_table_data(rows: list[dict]) -> tuple[list[str], list[list[str]], list[str]]:
    """
    Build (headers, cells, best_markers) for the comparison table.

    rows: list of dicts with keys:
        name, summary (dict), n_success (int)
    Returns:
        headers    — list of column header strings
        cells      — list of row lists (strings)
        col_widths — list of column widths
    """
    headers = ["Config"] + [hdr for _, hdr in TABLE_METRICS] + ["n_success"]

    # Collect mean values for best-value detection
    metric_means: dict[str, list] = {m: [] for m, _ in TABLE_METRICS}
    for row in rows:
        for metric, _ in TABLE_METRICS:
            mean, _ = _extract_metric(row["summary"], metric)
            metric_means[metric].append(mean)

    # Best-value index per metric column
    best_idx = {}
    for metric, _ in TABLE_METRICS:
        best_idx[metric] = _best_index(
            metric_means[metric], LOWER_IS_BETTER[metric]
        )

    # Build cell matrix
    cells = []
    for row_i, row in enumerate(rows):
        cell_row = [row["name"]]
        for metric, _ in TABLE_METRICS:
            mean, std = _extract_metric(row["summary"], metric)
            text = _fmt_cell(mean, std)
            if best_idx[metric] == row_i and mean is not None:
                text = f"*{text}"
            cell_row.append(text)
        cell_row.append(str(row.get("n_success", "—")))
        cells.append(cell_row)

    # Column widths
    col_widths = [len(h) for h in headers]
    for cell_row in cells:
        for ci, cell in enumerate(cell_row):
            col_widths[ci] = max(col_widths[ci], len(cell))

    return headers, cells, col_widths


def _render_ascii(headers, cells, col_widths) -> str:
    """Render an ASCII table and return it as a string."""
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    def fmt_row(row):
        return "| " + " | ".join(
            cell.ljust(col_widths[ci]) for ci, cell in enumerate(row)
        ) + " |"

    lines = [sep, fmt_row(headers), sep]
    for row in cells:
        lines.append(fmt_row(row))
    lines.append(sep)
    lines.append("")
    lines.append("  * = best value in column")
    return "\n".join(lines)


def _render_latex(headers, cells) -> str:
    """Render a LaTeX booktabs table and return it as a string."""
    n_cols = len(headers)
    col_spec = "l" + "r" * (n_cols - 1)

    def escape(s: str) -> str:
        return s.replace("_", r"\_").replace("±", r"$\pm$").replace("*", r"\textbf{*}")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{ConformerFlow Ablation Study}",
        r"\label{tab:ablations}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{escape(h)}}}" for h in headers) + r" \\",
        r"\midrule",
    ]
    for ci, row in enumerate(cells):
        # Separate full_model from ablations with a midrule
        if ci == 1:
            lines.append(r"\midrule")
        lines.append(" & ".join(escape(c) for c in row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "ConformerFlow Ablation Runner — evaluates every configuration "
            "in ABLATION_TABLE and produces a comparison table."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs_json",
        required=True,
        help="Path to paired.json containing X-ray/NMR pairs for evaluation.",
    )
    parser.add_argument(
        "--xray_dir",
        required=True,
        help="Directory containing X-ray PDB files (named {PDB_ID}.pdb).",
    )
    parser.add_argument(
        "--nmr_dir",
        required=True,
        help="Directory containing NMR PDB files (named {PDB_ID}.pdb).",
    )
    parser.add_argument(
        "--checkpoints_dir",
        required=True,
        help=(
            "Root directory for checkpoints. Each ablation checkpoint is "
            "expected at {checkpoints_dir}/{run_name}/ckpt_best.pt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Root directory for output. Per-run results are saved under "
            "{output_dir}/{run_name}/; summary files at {output_dir}/."
        ),
    )
    parser.add_argument(
        "--n_conformers",
        type=int,
        default=20,
        help="Number of conformers to generate per protein per ablation run.",
    )
    parser.add_argument(
        "--max_proteins",
        type=int,
        default=None,
        help="Maximum number of proteins to evaluate (None = all pairs).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Device for inference: 'auto' selects GPU if available, "
            "otherwise CPU."
        ),
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Energy-minimize generated conformers with OpenMM before scoring.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help=(
            "Additionally write a LaTeX booktabs table to "
            "{output_dir}/ablation_table.tex."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir      = Path(args.output_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []  # will hold {name, run_name, summary, n_success} per entry

    # ── Run validation for each ablation configuration ────────────────────────
    for entry in ABLATION_TABLE:
        name     = entry["name"]
        run_name = entry["run_name"]
        ckpt     = checkpoints_dir / run_name / "ckpt_best.pt"

        if not ckpt.exists():
            logger.warning(
                f"[SKIP] Checkpoint not found for '{name}': {ckpt}"
            )
            rows.append({
                "name":      name,
                "run_name":  run_name,
                "summary":   {},
                "n_success": None,
                "skipped":   True,
            })
            continue

        logger.info(f"[RUN] Starting ablation: {name}  (checkpoint: {ckpt})")

        run_output_dir = output_dir / run_name

        try:
            validator = ConformerFlowValidator(
                checkpoint_path=str(ckpt),
                n_conformers=args.n_conformers,
                device=args.device,
                minimize=getattr(args, "minimize", False),
            )
            summary = validator.validate_dataset(
                pairs_json=args.pairs_json,
                xray_dir=args.xray_dir,
                nmr_dir=args.nmr_dir,
                output_dir=str(run_output_dir),
                max_proteins=args.max_proteins,
            )
        except Exception as exc:
            logger.error(f"[ERROR] Ablation '{name}' failed with: {exc}")
            summary = {"error": str(exc)}

        # validate_dataset saves validation_summary.json; reload for consistency
        summary_path = run_output_dir / "validation_summary.json"
        if summary_path.exists():
            summary = _load_summary(summary_path)

        rows.append({
            "name":      name,
            "run_name":  run_name,
            "summary":   summary,
            "n_success": summary.get("n_success"),
            "skipped":   False,
        })

        logger.info(f"[DONE] Ablation: {name}  n_success={summary.get('n_success')}")

    # ── Build and render comparison table ─────────────────────────────────────
    headers, cells, col_widths = _build_table_data(rows)
    ascii_table = _render_ascii(headers, cells, col_widths)

    print()
    print("=" * 70)
    print("  ConformerFlow Ablation Comparison Table")
    print("=" * 70)
    print(ascii_table)

    # Save ASCII table
    ascii_path = output_dir / "ablation_table.txt"
    ascii_path.write_text(ascii_table)
    logger.info(f"ASCII table saved to: {ascii_path}")

    # Save aggregated results JSON
    results_json_path = output_dir / "ablation_results.json"
    serializable_rows = []
    for row in rows:
        serializable_rows.append({
            "name":      row["name"],
            "run_name":  row["run_name"],
            "skipped":   row.get("skipped", False),
            "n_success": row.get("n_success"),
            "summary":   row["summary"],
        })
    with open(results_json_path, "w") as f:
        json.dump(serializable_rows, f, indent=2)
    logger.info(f"Aggregated results JSON saved to: {results_json_path}")

    # Optionally save LaTeX table
    if args.latex:
        latex_table = _render_latex(headers, cells)
        latex_path  = output_dir / "ablation_table.tex"
        latex_path.write_text(latex_table)
        logger.info(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
