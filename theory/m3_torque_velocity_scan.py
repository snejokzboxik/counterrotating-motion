"""Fixed-radius M3 torque as a function of slow angular velocity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/v17c/v17c_21/code"))
sys.path.insert(0, str(ROOT / "research/v17c/v17c_21a/code"))
from analytical_model import load_inputs  # noqa: E402
from m3_harmonic_balance import HBConfig, solve_harmonic_balance  # noqa: E402


def run() -> dict:
    data = load_inputs()
    archive = json.loads((ROOT / "research/v17c/v17c_21a/results/full_grid_campaign.json").read_text())
    stage = archive["stages"][2]  # independently verified 128x12x12 root
    assert stage["grid"] == [128, 12, 12] and stage["status"] == "VERIFIED_ROOT"
    config = HBConfig(**stage["config"])
    rows = []
    for i in range(2):
        r = stage["root"]["radii_R"][i]
        nominal_omega = stage["root"]["omega_over_Omega1"][i]
        sequence = [nominal_omega, 0.0, -0.001, 0.001, -0.002, 0.002]
        xi_seed = None
        for omega in sequence:
            try:
                hb = solve_harmonic_balance(
                    r, omega, data["alpha"][i], data["lambda"][i],
                    data["Re_per_speed"][i], data["beta"], data["modes"],
                    config, xi0=xi_seed, require_convergence=True,
                )
                xi_seed = hb["xi_grid"]
                rf = hb["mean_total_RF_torque"]
                drag = hb["mean_total_drag_torque"]
                rows.append({
                    "particle": i+1, "r_over_R_fixed": r, "omega_over_Omega1": omega,
                    "RF_torque": rf, "drag_torque": drag,
                    "net_torque_without_ring_averaged_Coulomb": rf+drag,
                    "HB_converged": hb["converged"],
                    "HB_original_fast_residual": hb["relative_original_fast_equation_residual"],
                    "HB_iterations": hb["iterations"], "HB_wall_s": hb["wall_time_s"],
                    "nominal_M3_root": omega == nominal_omega,
                })
            except Exception as exc:
                rows.append({"particle": i+1, "r_over_R_fixed": r,
                             "omega_over_Omega1": omega,
                             "HB_converged": False, "error": str(exc)})
    return {
        "model": "V21A M3 full 8-mode RF and nonlinear drag; frozen published 128x12x12 radii",
        "source_commit": "94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8",
        "config": stage["config"], "rows": rows,
        "caveat": "A fixed-radius torque section is not the coupled radial-angular stationary branch or a stability proof. Coulomb tangential mean is zero only under the independent-phase ring average.",
    }


if __name__ == "__main__":
    result = run()
    out = ROOT / "data/m3_fixed_radius_torque_vs_omega.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for row in result["rows"]:
        print(row["particle"], row["omega_over_Omega1"], row.get("net_torque_without_ring_averaged_Coulomb"), row["HB_converged"], flush=True)
