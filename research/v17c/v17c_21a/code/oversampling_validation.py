"""Repeat the high-grid full M3 root with padded nonlinear quadrature."""
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
    outdir = BASE / "results"
    campaign = json.loads((outdir / "full_grid_campaign.json").read_text(encoding="utf-8"))
    base = next(x for x in campaign["stages"] if x["grid"] == [192, 16, 16])
    root = base["root"]
    with np.load(outdir / "xi_seed_192_16.npz") as f:
        xi_seed = [f["xi1"], f["xi2"]]
    data = legacy.load_inputs()
    root_vec = [root["radii_R"][0], root["omega_over_Omega1"][0],
                root["radii_R"][1], root["omega_over_Omega1"][1],
                root["mean_z_R"][0], root["mean_z_R"][1]]
    print("START frozen-root HB comparison with 2x nonlinear-force oversampling", flush=True)
    rows = []
    xi2 = []
    for i in range(2):
        metrics = []
        for oversample in (1, 2):
            cfg = HBConfig(192, 16, cutoff=0.05, oversample=oversample,
                           method="newton_krylov", tolerance=1e-7, max_iter=30)
            h = solve_harmonic_balance(
                root["radii_R"][i], root["omega_over_Omega1"][i],
                data["alpha"][i], data["lambda"][i], data["Re_per_speed"][i],
                data["beta"], data["modes"], cfg, xi0=xi_seed[i],
                require_convergence=True,
            )
            xi2.append(h["xi_grid"]) if oversample == 2 else None
            metrics.append({k: v for k, v in h.items() if k not in
                            {"xi_grid", "xi_fourier_coefficients", "nonlinear_force_fourier_coefficients"}})
        old, new = metrics
        rows.append({
            "particle": i + 1, "oversample1": old, "oversample2": new,
            "delta": {
                "rms_rf_force": new["rms_rf_force"] - old["rms_rf_force"],
                "rms_micromotion": new["rms_micromotion"] - old["rms_micromotion"],
                "mean_total_RF_torque": new["mean_total_RF_torque"] - old["mean_total_RF_torque"],
                "mean_total_drag_torque": new["mean_total_drag_torque"] - old["mean_total_drag_torque"],
            },
        })
    np.savez_compressed(outdir / "xi_oversample2_192_16.npz", xi1=xi2[0], xi2=xi2[1])
    metrics = {"study": "V17C.21.1 nonlinear-force aliasing and padding check",
               "base_grid": [192, 16, 16], "oversample_factor": 2,
               "root_fixed_at_192_16_16": root, "outer_root_resolved_again": False,
               "scope": "Frozen slow root: compares inner Fourier solutions/quadrature only; does not re-solve six slow balances.",
               "comparison": rows}
    (outdir / "oversampling_validation.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE_FROZEN_ROOT_ALIAS_CHECK",
                      "tail_comparison": [{"particle": q["particle"],
                          "tail_rel": q["oversample2"]["relative_nonlinear_force_tail"],
                          "dRFtorque": q["delta"]["mean_total_RF_torque"]} for q in rows]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
