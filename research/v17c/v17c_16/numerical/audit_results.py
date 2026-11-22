"""Aggregate numerical comparisons without changing raw run outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    results = ROOT/"results"
    analytic = read(results/"analytic_screen.json")
    d16 = read(results/"trajectory_validation_1000p_s16"/"summary.json")
    d32 = read(results/"trajectory_validation_ideal1000_s32"/"summary.json")
    mfs = read(results/"trajectory_validation_mfs1000_s16"/"summary.json")
    e16 = {r["case"]: r for r in d16["results"]}
    emfs = {r["case"]: r for r in mfs["results"]}
    e32 = d32["results"][0]
    ideal_nominal = e16["oseen_coulomb_nominal_start01"]
    mfs_nominal = emfs["oseen_coulomb_nominal_start01"]
    exact_models = {"ideal_leading_harmonic": ideal_nominal,
                    "full_2d_MFS": mfs_nominal}
    model_errors = {}
    for i in (1, 2):
        branch = analytic[f"particle_{i}"]["oseen"]
        model_errors[f"particle_{i}"] = {}
        for name, row in exact_models.items():
            rho = row["mean_radius_dimensionless_last_80pct"][i-1]
            omega = row["mean_angular_velocity_dimensionless_last_80pct"][i-1]
            model_errors[f"particle_{i}"][name] = {
                "analytical_radius_R": branch["rho"], "numerical_radius_R": rho,
                "radius_relative_error_percent": abs(branch["rho"]-rho)/abs(rho)*100,
                "analytical_omega_Omega1": branch["omega"],
                "numerical_omega_Omega1": omega,
                "omega_relative_error_percent": abs(branch["omega"]-omega)/abs(omega)*100,
                "omega_absolute_error_Omega1": abs(branch["omega"]-omega),
            }
    step_refinement = {}
    for key in ("mean_radius_dimensionless_last_80pct",
                "mean_angular_velocity_dimensionless_last_80pct"):
        values16 = ideal_nominal[key]
        values32 = e32[key]
        step_refinement[key] = [
            {"particle": i+1, "steps16": values16[i], "steps32": values32[i],
             "relative_change_percent": abs(values32[i]-values16[i])
                /max(abs(values32[i]), 1e-30)*100}
            for i in range(2)]
    mfs_vs_ideal = {}
    for key in ("mean_radius_dimensionless_last_80pct",
                "mean_angular_velocity_dimensionless_last_80pct"):
        mfs_vs_ideal[key] = [
            {"particle": i+1,
             "ideal": ideal_nominal[key][i],
             "MFS": mfs_nominal[key][i],
             "relative_change_percent": abs(mfs_nominal[key][i]-ideal_nominal[key][i])
                /max(abs(ideal_nominal[key][i]), 1e-30)*100}
            for i in range(2)]
    starts = [r for r in d16["results"] if r["oseen_quasisteady"]]
    initial_condition_spread = {}
    for key in ("mean_radius_dimensionless_last_80pct",
                "mean_angular_velocity_dimensionless_last_80pct"):
        values = np.asarray([r[key] for r in starts])
        initial_condition_spread[key] = {
            "cases": [r["case"] for r in starts],
            "min_by_particle": values.min(axis=0).tolist(),
            "max_by_particle": values.max(axis=0).tolist(),
            "max_abs_spread_by_particle": (values.max(axis=0)-values.min(axis=0)).tolist(),
        }
    lyapunov = {}
    for tag in ("qrf100", "qrf200", "qrf400"):
        path = results/f"lyapunov_{tag}.json"
        if path.exists():
            item = read(path)
            lyapunov[tag] = {"periods": item["primary_RF_periods"],
                             "largest_per_tau": item["largest_exponent_per_tau"],
                             "spectrum_per_tau": item["finite_time_exponents_per_tau"]}
    payload = {
        "analytical_vs_exact_errors": model_errors,
        "DOP853_max_step_refinement_1000_periods": step_refinement,
        "leading_harmonic_vs_full_MFS_1000_periods": mfs_vs_ideal,
        "finite_start_spread_ideal_field": initial_condition_spread,
        "finite_time_Lyapunov_spectra": lyapunov,
        "MFS_two_phase_transfer_checks": mfs_nominal["mfs_leading_transfer_check"],
        "interpretation": [
            "Numerical step refinement is distinct from physical drag/model validity.",
            "Initial-condition spread is a fixed deterministic panel, not a basin probability.",
            "Finite-time QR exponents for the quasiperiodic drive are not Floquet multipliers.",
            "The MFS field is 2D infinite rods with ideal DC added separately, not a 3D FEM validation.",
        ],
    }
    out = results/"comparison_audit.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n",
                   encoding="utf-8")
    print(out)
    print(json.dumps({"errors": model_errors, "step_refinement": step_refinement,
                      "mfs_vs_ideal": mfs_vs_ideal}, indent=2))


if __name__ == "__main__":
    main()
