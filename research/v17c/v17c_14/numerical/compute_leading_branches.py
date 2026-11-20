"""Recompute local15 leading-only circular roots and reduced Jacobian spectra."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from make_figures import P, fixed_points


def jsonable_root(root: dict) -> dict:
    return {
        "rho": float(root["rho"]),
        "omega_over_Omega1": float(root["omega"]),
        "stable_reduced_model": bool(root["stable"]),
        "eigenvalues_real_imag": [[float(v.real), float(v.imag)]
                                   for v in root["eigenvalues"]],
    }


def main() -> None:
    result = {
        "method": "leading p=2,s=4 RF average; independent species; Coulomb omitted",
        "source_commit": "5438b1cdd79c47abab45c6cbaf99c6daec9fbb60",
        "scan": {"beta_min": 0.005, "beta_max": 0.030, "samples": 301,
                 "A": P["A"], "B": P["B"], "nu": P["nu"],
                 "lambda1": P["lambda1"], "lambda2": P["lambda2"],
                 "alpha1": 1.0, "alpha2": P["alpha2"]},
        "classification": "algebraic roots 0<rho<1; local stability by eigenvalues of reduced 3D polar Jacobian",
        "trajectory_initial_conditions": None,
        "trajectory_horizon": None,
        "species_at_local15_beta": {},
        "scan_root_counts": {},
    }
    beta = P["beta"]
    for species, alpha, lam in [(1, 1.0, P["lambda1"]),
                                (2, P["alpha2"], P["lambda2"])]:
        roots = fixed_points(alpha, lam, beta)
        result["species_at_local15_beta"][str(species)] = [jsonable_root(r) for r in roots]
        counts = {"stable": 0, "unstable": 0, "total": 0}
        for b in np.linspace(result["scan"]["beta_min"], result["scan"]["beta_max"],
                             result["scan"]["samples"]):
            for r in fixed_points(alpha, lam, float(b)):
                counts["total"] += 1
                counts["stable" if r["stable"] else "unstable"] += 1
        result["scan_root_counts"][str(species)] = counts
    out = Path(__file__).resolve().parents[1] / "results/leading_branch.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
