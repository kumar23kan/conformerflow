"""
ConformerFlow — Optional CA-level energy minimization of generated conformers.

Hierarchy:
  1. OpenMM (amber14 CA-only model, harmonic position restraints)
  2. No-op fallback — returns input unchanged

Usage:
    from evaluation.minimize import minimize_ensemble
    result = minimize_ensemble(ca_coords, sequence)  # (N, L, 3) float32
    clean_coords = result["coords"]
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _minimize_one_openmm(ca_coords: np.ndarray,
                          sequence: str,
                          n_steps: int,
                          tolerance: float,
                          restraint_k: float) -> tuple:
    """
    CA-only OpenMM minimisation for a single conformer.

    Force field:
      - Harmonic CA-CA bonds (3.8 Å ideal, stiff spring to preserve topology)
      - Harmonic position restraints (pull CA back toward input positions)

    restraint_k: kJ/mol/nm² — controls how far atoms move from their input positions.
    Returns (coords_angstrom, energy_kJ_mol) or raises on failure.
    """
    import openmm as mm
    from openmm import app, unit

    L    = len(sequence)
    assert ca_coords.shape == (L, 3), \
        f"ca_coords shape {ca_coords.shape} does not match sequence length {L}"

    # Build topology
    topology = app.Topology()
    chain    = topology.addChain()
    for i, aa1 in enumerate(sequence):
        res_name = _AA3.get(aa1, "ALA")
        residue  = topology.addResidue(res_name, chain, str(i + 1))
        topology.addAtom("CA", app.Element.getBySymbol("C"), residue)

    positions_nm = [(ca_coords[i] * 0.1).tolist()   # Å → nm
                    for i in range(L)]

    # Minimal system: carbon masses
    system = mm.System()
    for _ in range(L):
        system.addParticle(12.011)

    # Harmonic CA-CA virtual bonds
    bond_force = mm.HarmonicBondForce()
    for i in range(L - 1):
        bond_force.addBond(i, i + 1, 0.38, 50000.0)   # 3.8 Å, stiff
    system.addForce(bond_force)

    # Harmonic position restraints
    restraint = mm.CustomExternalForce(
        "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint.addGlobalParameter("k", restraint_k)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    for i, pos in enumerate(positions_nm):
        restraint.addParticle(i, pos)
    system.addForce(restraint)

    integrator = mm.LangevinMiddleIntegrator(300, 1.0, 0.004)
    platform   = mm.Platform.getPlatformByName("Reference")   # portable CPU
    context    = mm.Context(system, integrator, platform)
    context.setPositions(
        [mm.Vec3(*p) * unit.nanometer for p in positions_nm]
    )

    mm.LocalEnergyMinimizer.minimize(
        context, tolerance=tolerance, maxIterations=n_steps
    )

    state   = context.getState(getPositions=True, getEnergy=True)
    pos_out = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    energy  = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    return np.array(pos_out, dtype=np.float32), float(energy)


def minimize_ensemble(ca_coords: np.ndarray,
                       sequence: str,
                       method: str = "openmm",
                       n_steps: int = 1000,
                       tolerance: float = 10.0,
                       restraint_k: float = 100.0) -> dict:
    """
    Energy-minimize a predicted CA conformer ensemble.

    Args:
        ca_coords:   (N, L, 3)  CA coordinates in Angstroms
        sequence:    str         one-letter amino acid sequence (length L)
        method:      "openmm" | "none"
        n_steps:     max L-BFGS iterations per conformer
        tolerance:   convergence tolerance (kJ/mol)
        restraint_k: position-restraint stiffness (kJ/mol/nm²);
                     higher = stays closer to input structure

    Returns:
        {
            "coords":   np.ndarray (N, L, 3)  minimized CA coords (Å)
            "energies": np.ndarray (N,)        final energy kJ/mol (NaN if unavailable)
            "method":   str                    method actually used
            "n_failed": int                    conformers that fell back to input
        }
    """
    N, L, _ = ca_coords.shape
    out_coords = ca_coords.copy().astype(np.float32)
    energies   = np.full(N, np.nan, dtype=np.float32)
    n_failed   = 0

    if method == "none" or not sequence:
        return {
            "coords":   out_coords,
            "energies": energies,
            "method":   "none",
            "n_failed": N,
        }

    if method == "openmm":
        try:
            import openmm  # noqa: F401
        except ImportError:
            logger.warning(
                "OpenMM not installed — minimization skipped. "
                "Install: conda install -c conda-forge openmm"
            )
            return {
                "coords":   out_coords,
                "energies": energies,
                "method":   "none",
                "n_failed": N,
            }

        for i in range(N):
            try:
                coords_min, energy = _minimize_one_openmm(
                    ca_coords[i], sequence, n_steps, tolerance, restraint_k
                )
                out_coords[i] = coords_min
                energies[i]   = energy
            except Exception as e:
                logger.debug(f"OpenMM failed for conformer {i}: {e}")
                n_failed += 1

        n_ok = N - n_failed
        if n_ok > 0:
            logger.info(
                f"OpenMM minimization: {n_ok}/{N} conformers succeeded, "
                f"mean energy = {float(np.nanmean(energies)):.1f} kJ/mol"
            )
        if n_failed > 0:
            logger.warning(
                f"OpenMM minimization: {n_failed}/{N} conformers fell back to input"
            )

        return {
            "coords":   out_coords,
            "energies": energies,
            "method":   "openmm",
            "n_failed": n_failed,
        }

    raise ValueError(f"Unknown minimization method: {method!r}. "
                     f"Choose 'openmm' or 'none'.")
