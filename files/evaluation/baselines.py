"""
ConformerFlow — Baseline Methods for Conformational Ensemble Generation

This module provides three simple baseline methods that serve as reference
points when evaluating ConformerFlow's generated ensembles against NMR data.
They span the spectrum from "no diversity" to "data-informed Gaussian model",
establishing lower and approximate upper bounds for what a generative model
should achieve without learning protein-specific dynamics.

All baselines share the same interface:
    generate(xray_ca, n_conformers, nmr_ca=None) -> np.ndarray (N, L, 3)

Usage example::

    from evaluation.baselines import evaluate_baselines
    results = evaluate_baselines(xray_ca, nmr_ca, n_conformers=20)
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np

# Relative import — works when this module is part of the evaluation package.
# Falls back to an absolute import so the file can also be run standalone.
try:
    from .metrics import evaluate_ensemble
except ImportError:
    try:
        from evaluation.metrics import evaluate_ensemble
    except ImportError:
        evaluate_ensemble = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _check_xray_ca(xray_ca: np.ndarray) -> np.ndarray:
    """Validate and return a (L, 3) float64 CA coordinate array."""
    xray_ca = np.asarray(xray_ca, dtype=np.float64)
    if xray_ca.ndim != 2 or xray_ca.shape[1] != 3:
        raise ValueError(
            f"xray_ca must be (L, 3), got shape {xray_ca.shape}"
        )
    return xray_ca


def _check_nmr_ca(nmr_ca: np.ndarray) -> np.ndarray:
    """Validate and return a (N_nmr, L, 3) float64 CA coordinate array."""
    nmr_ca = np.asarray(nmr_ca, dtype=np.float64)
    if nmr_ca.ndim == 2:
        # Single NMR conformer supplied as (L, 3) — wrap into (1, L, 3)
        nmr_ca = nmr_ca[np.newaxis]
    if nmr_ca.ndim != 3 or nmr_ca.shape[2] != 3:
        raise ValueError(
            f"nmr_ca must be (N_nmr, L, 3), got shape {nmr_ca.shape}"
        )
    return nmr_ca


def _rmsf_from_ensemble(coords: np.ndarray) -> np.ndarray:
    """
    Compute per-residue RMSF from an ensemble.

    Parameters
    ----------
    coords : np.ndarray, shape (N, L, 3)

    Returns
    -------
    rmsf : np.ndarray, shape (L,)
    """
    mean = coords.mean(axis=0)          # (L, 3)
    diff = coords - mean[np.newaxis]    # (N, L, 3)
    return np.sqrt((diff ** 2).sum(axis=-1).mean(axis=0))  # (L,)


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 1 — STATIC (TRIVIAL LOWER BOUND)
# ─────────────────────────────────────────────────────────────────────────────

class StaticBaseline:
    """
    Trivial "no-diversity" baseline: repeat the X-ray CA structure N times.

    Scientific interpretation
    -------------------------
    This baseline represents the null hypothesis that the crystal structure
    alone fully characterises the conformational landscape — i.e., there is
    zero dynamics.  Its ensemble metrics (RMSF, coverage RMSD, covariance
    similarity, …) therefore define the performance floor: any method that
    cannot outperform a static crystal structure is not modelling protein
    flexibility at all.

    Because all N conformers are identical:
      * per-residue RMSF = 0 everywhere
      * coverage RMSD = nearest-neighbour RMSD to each NMR conformer from
        the X-ray structure (measures how close the crystal is to the
        solution ensemble on average)
      * torsion overlap and covariance similarity will be minimal

    No ``nmr_ca`` is required; if supplied it is ignored.
    """

    def generate(
        self,
        xray_ca: np.ndarray,
        n_conformers: int,
        nmr_ca: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate an ensemble by tiling the X-ray structure.

        Parameters
        ----------
        xray_ca : np.ndarray, shape (L, 3)
            CA coordinates of the X-ray crystal structure.
        n_conformers : int
            Number of conformers to generate (N).
        nmr_ca : np.ndarray or None
            Ignored for this baseline; accepted for API compatibility.

        Returns
        -------
        ensemble : np.ndarray, shape (N, L, 3)
            N identical copies of the input X-ray structure.
        """
        xray_ca = _check_xray_ca(xray_ca)
        if n_conformers < 1:
            raise ValueError(f"n_conformers must be >= 1, got {n_conformers}")
        # Tile: (1, L, 3) -> (N, L, 3) — no data is shared, use explicit copy
        return np.tile(xray_ca[np.newaxis], (n_conformers, 1, 1))


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 2 — ISOTROPIC GAUSSIAN NOISE
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNoiseBaseline:
    """
    Random-diversity baseline: add independent isotropic Gaussian noise to
    each CA position.

    Scientific interpretation
    -------------------------
    Each conformer is generated by drawing independent N(0, σ²) noise for
    every CA atom in every spatial dimension and adding it to the crystal
    structure.  The noise is identically distributed across all residues —
    there is no knowledge of which regions are genuinely flexible.

    When ``nmr_ca`` is supplied, σ is automatically calibrated so that the
    expected per-residue RMSF of the generated ensemble equals the mean
    per-residue RMSF of the NMR ensemble::

        σ_calibrated = mean_residue_RMSF(NMR) / sqrt(3)

    The 1/√3 factor converts from 3-D RMSF to the 1-D standard deviation
    of each Cartesian component (since RMSF² = 3σ² for isotropic noise).

    This baseline isolates the contribution of the *magnitude* of diversity
    from the *spatial pattern* of diversity.  A model is only genuinely
    useful if it outperforms uniform noise at the correctly calibrated σ.

    Generated coordinates are clamped to ±10 Å of the input to prevent
    rare large outlier conformers.

    Parameters
    ----------
    sigma : float
        Default isotropic noise standard deviation in Ångström.  Overridden
        by NMR calibration when ``nmr_ca`` is provided to ``generate``.
    seed : int or None
        Random seed for reproducibility.
    """

    # Hard limit: generated CA cannot drift more than this far from input.
    _CLAMP_RADIUS: float = 10.0

    def __init__(self, sigma: float = 1.0, seed: Optional[int] = None) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def _calibrate_sigma(self, nmr_ca: np.ndarray) -> float:
        """
        Derive σ from the NMR ensemble's mean per-residue RMSF.

        RMSF per residue (3-D) = sqrt(3) * σ_1D  =>  σ_1D = RMSF / sqrt(3)

        Parameters
        ----------
        nmr_ca : np.ndarray, shape (N_nmr, L, 3)

        Returns
        -------
        sigma : float
        """
        rmsf = _rmsf_from_ensemble(nmr_ca)      # (L,)
        mean_rmsf = float(rmsf.mean())
        if mean_rmsf < 1e-8:
            # Degenerate NMR ensemble (all conformers identical) — use default
            return self.sigma
        return mean_rmsf / np.sqrt(3.0)

    def generate(
        self,
        xray_ca: np.ndarray,
        n_conformers: int,
        nmr_ca: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate an ensemble by adding isotropic Gaussian noise.

        Parameters
        ----------
        xray_ca : np.ndarray, shape (L, 3)
            CA coordinates of the X-ray crystal structure.
        n_conformers : int
            Number of conformers to generate (N).
        nmr_ca : np.ndarray or None, shape (N_nmr, L, 3)
            If provided, σ is calibrated to match the NMR RMSF.

        Returns
        -------
        ensemble : np.ndarray, shape (N, L, 3)
            Noisy conformers, clamped to ±10 Å from the input.
        """
        xray_ca = _check_xray_ca(xray_ca)
        if n_conformers < 1:
            raise ValueError(f"n_conformers must be >= 1, got {n_conformers}")

        if nmr_ca is not None:
            nmr_ca = _check_nmr_ca(nmr_ca)
            if nmr_ca.shape[1] != xray_ca.shape[0]:
                raise ValueError(
                    f"nmr_ca sequence length ({nmr_ca.shape[1]}) does not "
                    f"match xray_ca ({xray_ca.shape[0]})"
                )
            sigma = self._calibrate_sigma(nmr_ca)
        else:
            sigma = self.sigma

        L = xray_ca.shape[0]
        noise = self.rng.normal(loc=0.0, scale=sigma, size=(n_conformers, L, 3))
        ensemble = xray_ca[np.newaxis] + noise  # (N, L, 3) broadcast

        # Clamp each CA to stay within _CLAMP_RADIUS of the input position
        delta = ensemble - xray_ca[np.newaxis]
        delta = np.clip(delta, -self._CLAMP_RADIUS, self._CLAMP_RADIUS)
        ensemble = xray_ca[np.newaxis] + delta

        return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 3 — PER-RESIDUE GAUSSIAN FITTED TO NMR ENSEMBLE (ORACLE BASELINE)
# ─────────────────────────────────────────────────────────────────────────────

class NMRMeanBaseline:
    """
    Data-informed oracle baseline: sample from a diagonal Gaussian fitted to
    the NMR ensemble on a per-residue basis.

    Scientific interpretation
    -------------------------
    This baseline fits a multivariate Gaussian to the NMR ground truth with:
      * mean μ_i  = mean CA position of residue i across NMR conformers
      * variance  = per-residue, per-axis sample variance (diagonal covariance)

    New conformers are drawn from N(μ, diag(σ²)).  Unlike
    ``GaussianNoiseBaseline``, the noise magnitude varies per residue,
    correctly reflecting which parts of the structure are flexible.  However,
    the model ignores all inter-residue correlations and uses the NMR data
    directly — it is an oracle that no blind method can exceed in terms of
    capturing the *marginal* per-residue distributions.

    Use-case: establishes a soft upper bound.  If ConformerFlow cannot beat
    or approach this baseline, the model is not capturing even the marginal
    flexibility correctly.

    Requires ``nmr_ca`` in both ``fit`` (or automatically in ``generate``).
    Works on proteins with L >= 1.  For single-conformer NMR ensembles the
    variance is zero and all generated conformers equal the NMR mean (same
    behaviour as ``StaticBaseline`` centred on the NMR mean).

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._mu: Optional[np.ndarray] = None    # (L, 3)
        self._std: Optional[np.ndarray] = None   # (L, 3)

    def fit(self, nmr_ca: np.ndarray) -> "NMRMeanBaseline":
        """
        Fit the diagonal Gaussian to an NMR ensemble.

        Parameters
        ----------
        nmr_ca : np.ndarray, shape (N_nmr, L, 3)

        Returns
        -------
        self
        """
        nmr_ca = _check_nmr_ca(nmr_ca)
        self._mu = nmr_ca.mean(axis=0)        # (L, 3)
        # ddof=1 for unbiased estimate; clamp to 0 for robustness when N=1
        if nmr_ca.shape[0] > 1:
            self._std = nmr_ca.std(axis=0, ddof=1)   # (L, 3)
        else:
            self._std = np.zeros_like(self._mu)
        return self

    def generate(
        self,
        xray_ca: np.ndarray,
        n_conformers: int,
        nmr_ca: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Sample conformers from the per-residue Gaussian fitted to NMR data.

        ``nmr_ca`` must be supplied either here or via a prior call to
        ``fit()``.  If both are supplied, the array passed here takes
        precedence and ``fit()`` is called internally.

        Parameters
        ----------
        xray_ca : np.ndarray, shape (L, 3)
            CA coordinates of the X-ray crystal structure.  Used only to
            validate sequence length consistency; the generated ensemble is
            centred on the NMR mean, not the crystal structure.
        n_conformers : int
            Number of conformers to generate (N).
        nmr_ca : np.ndarray or None, shape (N_nmr, L, 3)
            NMR ground-truth ensemble.  Required if ``fit()`` has not been
            called previously.

        Returns
        -------
        ensemble : np.ndarray, shape (N, L, 3)
        """
        xray_ca = _check_xray_ca(xray_ca)
        if n_conformers < 1:
            raise ValueError(f"n_conformers must be >= 1, got {n_conformers}")

        if nmr_ca is not None:
            nmr_ca = _check_nmr_ca(nmr_ca)
            self.fit(nmr_ca)

        if self._mu is None or self._std is None:
            raise RuntimeError(
                "NMRMeanBaseline has not been fitted. "
                "Pass nmr_ca to generate() or call fit() first."
            )

        L = xray_ca.shape[0]
        if self._mu.shape[0] != L:
            raise ValueError(
                f"Fitted model has L={self._mu.shape[0]} residues but "
                f"xray_ca has L={L}."
            )

        # Sample: (N, L, 3) = mu + std * z
        z = self.rng.standard_normal((n_conformers, L, 3))
        ensemble = self._mu[np.newaxis] + self._std[np.newaxis] * z
        return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baselines(
    xray_ca: np.ndarray,
    nmr_ca: np.ndarray,
    n_conformers: int = 20,
    mask: Optional[np.ndarray] = None,
    pdb_id: str = "",
) -> dict:
    """
    Run all three baselines and evaluate each against the NMR ground truth.

    Parameters
    ----------
    xray_ca : np.ndarray, shape (L, 3)
        CA coordinates of the X-ray crystal structure used as input.
    nmr_ca : np.ndarray, shape (N_nmr, L, 3)
        NMR CA coordinates used as the evaluation target.
    n_conformers : int
        Number of conformers each baseline should generate.
    mask : np.ndarray or None, shape (L,) bool
        True for real (non-padded) residues.  Passed through to
        ``evaluate_ensemble``.
    pdb_id : str
        PDB identifier string for logging/reporting.

    Returns
    -------
    results : dict
        Outer keys are baseline names ("static", "gaussian_noise",
        "nmr_mean"); inner values are the metric dicts returned by
        ``evaluate_ensemble``.

    Raises
    ------
    RuntimeError
        If ``evaluate_ensemble`` could not be imported.
    """
    if evaluate_ensemble is None:
        raise RuntimeError(
            "evaluate_ensemble could not be imported from evaluation.metrics. "
            "Ensure the package is installed or on sys.path."
        )

    xray_ca = _check_xray_ca(xray_ca)
    nmr_ca = _check_nmr_ca(nmr_ca)

    baselines = {
        "static": StaticBaseline(),
        "gaussian_noise": GaussianNoiseBaseline(seed=0),
        "nmr_mean": NMRMeanBaseline(seed=0),
    }

    results: dict = {}
    for name, baseline in baselines.items():
        ensemble = baseline.generate(
            xray_ca=xray_ca,
            n_conformers=n_conformers,
            nmr_ca=nmr_ca,
        )
        metrics = evaluate_ensemble(
            pred_coords=ensemble,
            true_coords=nmr_ca,
            mask=mask,
            pdb_id=f"{pdb_id}_{name}" if pdb_id else name,
        )
        results[name] = metrics

    return results


# ─────────────────────────────────────────────────────────────────────────────
# __main__ CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ca_from_pdb(pdb_path: str) -> np.ndarray:
    """
    Minimal PDB parser: extract CA coordinates from the first MODEL (or
    ATOM records) of a PDB file, returning (N_models, L, 3) for multi-model
    files and (1, L, 3) for single-model files.

    Only ATOM records with atom name " CA " are processed.
    """
    models: list[list[list[float]]] = []
    current: list[list[float]] = []
    in_model = False
    found_model_record = False

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                found_model_record = True
                current = []
                in_model = True
            elif rec == "ENDMDL":
                if current:
                    models.append(current)
                current = []
                in_model = False
            elif rec == "ATOM":
                atom_name = line[12:16]
                if atom_name == " CA ":
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        current.append([x, y, z])
                    except ValueError:
                        pass

    # Single-model PDB (no MODEL records)
    if not found_model_record and current:
        models.append(current)
    elif not found_model_record and not models and current:
        models.append(current)

    if not models:
        raise ValueError(f"No CA atoms found in {pdb_path!r}")

    # Pad to equal length (take the minimum across models to be safe)
    min_len = min(len(m) for m in models)
    arr = np.array([m[:min_len] for m in models], dtype=np.float64)
    return arr  # (N_models, L, 3)


def _print_table(results: dict) -> None:
    """Print a human-readable comparison table of baseline metrics."""
    scalar_keys = [
        "coverage_rmsd",
        "precision_rmsd",
        "mean_pairwise_rmsd",
        "mean_struct_rmsd",
        "rmsf_pearson_r",
        "rmsf_spearman_r",
        "rmsf_mae",
        "torsion_overlap",
        "torsion_js_div",
        "covariance_frobenius_sim",
        "covariance_pearson_r",
        "top_mode_overlap",
        "bond_len_mae",
        "bond_len_outliers",
        "clash_score_per100",
        "tm_score",
        "gdt_ts",
    ]

    baseline_names = list(results.keys())
    col_w = max(22, *(len(n) + 2 for n in baseline_names))
    metric_w = 30

    # Header
    header = f"{'Metric':<{metric_w}}" + "".join(
        f"{n:>{col_w}}" for n in baseline_names
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for key in scalar_keys:
        row = f"{key:<{metric_w}}"
        for name in baseline_names:
            val = results[name].get(key, float("nan"))
            if isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)

    print(sep)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ConformerFlow baseline methods against an NMR ensemble."
        )
    )
    parser.add_argument(
        "--xray_pdb",
        required=True,
        metavar="PATH",
        help="Path to single-model PDB file containing the X-ray structure.",
    )
    parser.add_argument(
        "--nmr_pdb",
        required=True,
        metavar="PATH",
        help="Path to multi-model PDB file containing the NMR ensemble.",
    )
    parser.add_argument(
        "--n_conformers",
        type=int,
        default=20,
        metavar="N",
        help="Number of conformers each baseline should generate (default: 20).",
    )
    parser.add_argument(
        "--pdb_id",
        default="",
        metavar="ID",
        help="PDB identifier string used in reporting (optional).",
    )
    args = parser.parse_args(argv)

    print(f"Loading X-ray structure from: {args.xray_pdb}", file=sys.stderr)
    xray_models = _parse_ca_from_pdb(args.xray_pdb)
    if xray_models.shape[0] > 1:
        print(
            f"  WARNING: {xray_models.shape[0]} models found in xray PDB; "
            "using first model only.",
            file=sys.stderr,
        )
    xray_ca = xray_models[0]  # (L, 3)

    print(f"Loading NMR ensemble from:    {args.nmr_pdb}", file=sys.stderr)
    nmr_ca = _parse_ca_from_pdb(args.nmr_pdb)  # (N_nmr, L, 3)
    print(
        f"  X-ray: {xray_ca.shape[0]} residues | "
        f"NMR: {nmr_ca.shape[0]} models × {nmr_ca.shape[1]} residues",
        file=sys.stderr,
    )

    # Align sequence lengths
    L = min(xray_ca.shape[0], nmr_ca.shape[1])
    xray_ca = xray_ca[:L]
    nmr_ca = nmr_ca[:, :L, :]

    print(
        f"\nGenerating {args.n_conformers} conformers per baseline …\n",
        file=sys.stderr,
    )

    results = evaluate_baselines(
        xray_ca=xray_ca,
        nmr_ca=nmr_ca,
        n_conformers=args.n_conformers,
        pdb_id=args.pdb_id,
    )

    print(f"\n{'='*60}")
    print(f"  Baseline comparison  —  {args.pdb_id or args.xray_pdb}")
    print(f"  n_conformers={args.n_conformers}, L={L}")
    print(f"{'='*60}\n")
    _print_table(results)

    # Quick narrative summary
    print("\nSummary (coverage_rmsd — lower is better):")
    for name, m in results.items():
        val = m.get("coverage_rmsd", float("nan"))
        print(f"  {name:<20} coverage_rmsd = {val:.3f} Å")


if __name__ == "__main__":
    main()
