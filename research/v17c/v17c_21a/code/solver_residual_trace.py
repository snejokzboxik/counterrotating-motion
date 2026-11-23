"""Save nonlinear fixed-point residual histories for the p1 solver comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "v17c" / "v17c_21" / "code"))
import analytical_model as legacy  # noqa: E402
from m3_harmonic_balance import HBConfig, solve_harmonic_balance  # noqa: E402


def main():
    data = legacy.load_inputs()
    campaign = json.loads((BASE / "results" / "full_grid_campaign.json").read_text(encoding="utf-8"))
    root = next(x for x in campaign["stages"] if x["grid"] == [192, 16, 16])["root"]
    output = {"study": "V17C.21.1 p1 inner solver residual traces",
              "grid": [192, 16, 16], "particle": 1,
              "frozen_root": {"r": root["radii_R"][0], "omega": root["omega_over_Omega1"][0]},
              "rows": []}
    for method, tol, max_iter in (
        ("fixed_point", 2e-9, 160), ("adaptive", 1e-5, 320),
        ("anderson", 1e-5, 240), ("newton_krylov", 1e-7, 30)
    ):
        result = solve_harmonic_balance(
            root["radii_R"][0], root["omega_over_Omega1"][0], data["alpha"][0],
            data["lambda"][0], data["Re_per_speed"][0], data["beta"], data["modes"],
            HBConfig(192, 16, cutoff=0.05, oversample=1, method=method,
                     tolerance=tol, max_iter=max_iter, relaxation=0.5, history=6),
            require_convergence=False,
        )
        row = {k: result[k] for k in (
            "status", "converged", "iterations", "wall_time_s", "relative_nonlinear_residual",
            "fixed_point_change_rms", "minimum_fast_mode_denominator", "rms_unresolved_fast_force",
            "relative_original_fast_equation_residual", "equation_tolerance",
            "rms_unresolved_slow_force", "residual_trace"
        )}
        row.update({"method": method, "tolerance": tol, "max_iter": max_iter})
        output["rows"].append(row)
    (BASE / "results" / "solver_residual_trace.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("method", "status", "iterations", "relative_nonlinear_residual")}
                      for r in output["rows"]], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
