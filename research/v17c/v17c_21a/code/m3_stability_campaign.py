"""Finite-difference local stability of the declared adiabatic M3 slow flow."""
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
from m3_adiabatic_slow_flow import slow_flow_jacobian  # noqa: E402
from m3_harmonic_balance import HBConfig  # noqa: E402


def main():
    outdir = BASE / "results"
    campaign = json.loads((outdir / "full_grid_campaign.json").read_text(encoding="utf-8"))
    base = next(x for x in campaign["stages"] if x["grid"] == [192, 16, 16])
    r = base["root"]["radii_R"]
    w = base["root"]["omega_over_Omega1"]
    z = base["root"]["mean_z_R"]
    state = np.array([r[0], 0.0, w[0], z[0], 0.0,
                      r[1], 0.0, w[1], z[1], 0.0], dtype=float)
    data = legacy.load_inputs()
    cfg = HBConfig(96, 12, cutoff=0.05, oversample=1, method="newton_krylov",
                   tolerance=1e-7, max_iter=30, relaxation=0.5, history=6)
    central = np.array([1e-4, 1e-4, 2e-6, 1e-4, 1e-4] * 2)
    output = {
        "study": "V17C.21.1 local stability of adiabatic M3 slow flow",
        "config": cfg.__dict__,
        "source_root_grid": [192, 16, 16],
        "jacobian_grid": [96, 12, 12],
        "finite_difference_steps_central": central.tolist(),
        "variants": [],
        "status": "IN_PROGRESS",
    }
    for factor in (1.0, 0.5, 2.0):
        steps = central * factor
        deriv_steps = np.array([1e-4, 1e-4, 2e-6, 1e-4, 1e-4])
        deriv_steps *= factor
        path = outdir / f"m3_slowflow_jacobian_progress_h{factor:g}.json"
        print(f"START slow-flow Jacobian finite-difference factor={factor:g}", flush=True)
        row = slow_flow_jacobian(data, state, cfg, steps, deriv_steps, output_path=path)
        row["step_factor"] = factor
        output["variants"].append(row)
        output["status"] = "PARTIAL"
        (outdir / "m3_slowflow_stability.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"factor": factor, "max_real": row["max_real_eigenvalue"],
                          "eigenvalues": row["eigenvalues_real_imag"],
                          "Lomega": row["dL_domega"]}, ensure_ascii=False), flush=True)
    eig0 = np.asarray([complex(*q) for q in output["variants"][0]["eigenvalues_real_imag"]])
    for row in output["variants"][1:]:
        eig = np.asarray([complex(*q) for q in row["eigenvalues_real_imag"]])
        remaining = list(eig)
        matched = []
        for value in eig0:
            idx = int(np.argmin([abs(value - q) for q in remaining]))
            matched.append(remaining.pop(idx))
        row["eigenvalue_max_nearest_difference_vs_central"] = float(
            max(abs(a - b) for a, b in zip(eig0, matched)))
    output["status"] = "COMPLETE"
    (outdir / "m3_slowflow_stability.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("COMPLETE adiabatic M3 local stability", flush=True)


if __name__ == "__main__":
    main()
