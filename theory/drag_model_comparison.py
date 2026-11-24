"""Paired M3 re-solves at one common grid with and without SN nonlinearity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/v17c/v17c_21/code"))
sys.path.insert(0, str(ROOT / "research/v17c/v17c_21a/code"))
from analytical_model import load_inputs  # noqa: E402
from m3_harmonic_balance import HBConfig  # noqa: E402
from m3_system import initial_root, solve_m3_root  # noqa: E402


def evaluate() -> dict:
    data = load_inputs()
    archive = json.loads((ROOT / "research/v17c/v17c_21a/results/full_grid_campaign.json").read_text())
    root0 = archive["stages"][-1]["root"]
    seed = initial_root(root0)
    base = dict(ntheta=96, nphase=8, cutoff=0.05, oversample=2,
                method="newton_krylov", tolerance=1e-7,
                equation_tolerance=5e-5, max_iter=40,
                relaxation=0.5, history=6)
    outputs = []
    current = seed
    for scale in (0.0, 1.0):
        config = HBConfig(**base, sn_scale=scale)
        result = solve_m3_root(data, current, config, max_nfev=80,
                               scaled_balance_tolerance=2e-6,
                               require_convergence=True)
        current = np.array([result["root"]["radii_R"][0],
                            result["root"]["omega_over_Omega1"][0],
                            result["root"]["radii_R"][1],
                            result["root"]["omega_over_Omega1"][1],
                            result["root"]["mean_z_R"][0],
                            result["root"]["mean_z_R"][1]])
        outputs.append({"sn_scale": scale, "status": result["status"],
                        "root": result["root"],
                        "balance_residual_inf": result["scaled_balance_max_norm"],
                        "inner_residuals": [p["relative_original_fast_equation_residual"]
                                            for p in result["inner"]],
                        "outer_evaluations": result["balance_evaluations"],
                        "wall_time_s": result["wall_time_s"], "grid": [96, 8, 8],
                        "oversample": 2})
    nonlinear, linear = outputs[1], outputs[0]
    freq_ratio_changes = [100 * (f1 - f0) / abs(f1)
                          for f1, f0 in zip(nonlinear["root"]["frequency_Hz"],
                                            linear["root"]["frequency_Hz"])]
    radius_changes = [100 * (r1 - r0) / r1
                      for r1, r0 in zip(nonlinear["root"]["radii_R"],
                                        linear["root"]["radii_R"])]
    return {"source": "V21A M3 code and V20B refine12 registry",
            "paired_runs": outputs,
            "SN_vs_linear_change_percent_relative_to_SN": {
                "frequency": freq_ratio_changes, "radius": radius_changes},
            "interpretation": "Same parameter vector and common collocation settings; the SN factor is removed only for the linear control. This isolates the model effect within M3, not hydrodynamic model validity."}


if __name__ == "__main__":
    out = ROOT / "data/m3_schiller_naumann_vs_linear.json"
    out.write_text(json.dumps(evaluate(), indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
