"""Reproduce the V21 large-grid failure and compare inner M3 solvers."""
from __future__ import annotations

import json
import sys
import time
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
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    data = legacy.load_inputs()
    pred = json.loads((ROOT / "research" / "v17c" / "v17c_21" / "results" /
                       "analytical_predictions.json").read_text(encoding="utf-8"))
    m3 = pred["models"]["M3_full_MFS_SN_torus_Galerkin"]
    configs = [
        HBConfig(192, 16, method="fixed_point", tolerance=2e-9, max_iter=160, relaxation=0.5),
        HBConfig(192, 16, method="adaptive", tolerance=1e-5, max_iter=320, relaxation=0.5),
        HBConfig(192, 16, method="anderson", tolerance=1e-5, max_iter=320, relaxation=0.5, history=6),
        HBConfig(192, 16, method="newton_krylov", tolerance=1e-7, max_iter=30, relaxation=0.5),
    ]
    rows = []
    for i in range(2):
        for cfg in configs:
            t0 = time.perf_counter()
            result = solve_harmonic_balance(
                m3["radii_R"][i], m3["omega_over_Omega1"][i], data["alpha"][i],
                data["lambda"][i], data["Re_per_speed"][i], data["beta"], data["modes"],
                cfg, require_convergence=False,
            )
            old = legacy.nonlinear_micromotion_full(
                m3["radii_R"][i], m3["omega_over_Omega1"][i], data["alpha"][i],
                data["lambda"][i], data["Re_per_speed"][i], data["beta"], data["modes"],
                192, 16, tolerance=2e-9, max_iter=160, slow_mode_cutoff=0.05,
            )
            row = {k: result[k] for k in (
                "converged", "status", "iterations", "wall_time_s", "relative_nonlinear_residual",
                "relative_original_fast_equation_residual", "equation_tolerance",
                "fixed_point_change_rms", "minimum_fast_mode_denominator", "rms_rf_force",
                "rms_unresolved_fast_force", "rms_unresolved_slow_force",
                "rms_nonlinear_force_tail", "relative_nonlinear_force_tail", "rms_micromotion",
                "mean_total_RF_torque", "mean_total_drag_torque", "phase_grid", "evaluation_grid",
            )}
            row["legacy"] = {k: old[k] for k in (
                "converged", "iterations", "relative_fixed_point_residual", "rms_rf_force",
                "rms_micromotion", "mean_total_RF_torque", "mean_total_drag_torque",
                "rms_fast_mode_equation_residual", "rms_slow_mode_residual",
            )}
            row["new_minus_legacy"] = {
                "rms_rf_force": row["rms_rf_force"] - old["rms_rf_force"],
                "rms_micromotion": row["rms_micromotion"] - old["rms_micromotion"],
                "mean_total_RF_torque": row["mean_total_RF_torque"] - old["mean_total_RF_torque"],
                "mean_total_drag_torque": row["mean_total_drag_torque"] - old["mean_total_drag_torque"],
            }
            row.update({"particle": i + 1, "method": cfg.method,
                        "oversample": cfg.oversample, "tolerance": cfg.tolerance,
                        "max_iter": cfg.max_iter})
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("particle", "method", "converged",
                  "iterations", "relative_nonlinear_residual", "rms_rf_force", "rms_micromotion",
                  "mean_total_RF_torque", "new_minus_legacy")}, ensure_ascii=False), flush=True)
    out = {
        "study": "V17C.21.1 inner M3 solver benchmark",
        "source_commit": "05a8d8ce811c9858ab1e74b67b15139c557018b1",
        "frozen_reference_root_only_for_diagnostic": True,
        "grid": [192, 16, 16],
        "rows": rows,
        "qualification": "This compares the inner solve at the archived V21 M3 roots; it is not the required full-root grid convergence campaign.",
    }
    (BASE / "results" / "solver_benchmark.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
