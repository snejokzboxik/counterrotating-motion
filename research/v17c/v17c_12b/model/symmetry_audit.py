"""Permutation-equivariance test for identical-particle V17C dynamics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import basin_scan as model  # noqa: E402


def swap(y: np.ndarray) -> np.ndarray:
    return y.reshape(2, 6)[::-1].reshape(12).copy()


def main() -> None:
    seed = json.loads((ROOT/"results"/"physical_seed.json").read_text(encoding="utf-8"))
    chosen = None
    src = ROOT/"results"/"identical_alumina_screen_corrected"/"trajectories.jsonl"
    for line in src.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["category"] == "CO_PLUS_PLUS":
            chosen = row
            break
    if chosen is None:
        raise RuntimeError("no retained symmetric-grid trajectory available")
    p = np.asarray(seed["p"], dtype=np.float64)
    _, family, k, coeff = model.load_model()
    q = seed["particle"]["charge_C"]
    m = seed["particle"]["mass_kg"]
    radius = seed["particle"]["radius_m"]
    lam = p[6]
    omega = seed["trap"]["omega1_rad_s"]
    rho_g = seed["medium"]["density_kg_m3"]
    eta = seed["medium"]["dynamic_viscosity_Pa_s"]
    recoef = 2*rho_g*radius*seed["trap"]["free_aperture_radius_R_m"]*omega/eta
    cc = seed["dimensionless_checks"]["Cunningham_slip_factor_approx"]
    gravity = -9.80665/(seed["trap"]["free_aperture_radius_R_m"]*omega*omega)

    results = []
    for steps in (64, 96, 128):
        kwargs = dict(qratio=1.0, oseen_Re_per_speed=recoef, oseen_Re2_per_speed=recoef,
            hydro_mode=2, a1_over_R=radius/seed["trap"]["free_aperture_radius_R_m"],
            a2_over_R=radius/seed["trap"]["free_aperture_radius_R_m"],
            gravity_z=gravity, drag_mode=2, slip_correction=cc)
        forward = model.integrate_one(np.asarray(chosen["initial"]), np.asarray(chosen["phase"]),
            p, family, k, coeff, 15000, steps, 200, **kwargs)
        reverse = model.integrate_one(swap(np.asarray(chosen["initial"])), np.asarray(chosen["phase"]),
            p, family, k, coeff, 15000, steps, 200, **kwargs)
        fstate, fturns, fw, fmin, fmax, fz, _, fd, fperiods, fretained, *_ = forward
        rstate, rturns, rw, rmin, rmax, rz, _, rd, rperiods, rretained, *_ = reverse
        results.append({"steps_per_rf_period": steps, "forward_category": str(forward[12]),
            "swapped_category": str(reverse[12]), "periods_completed": int(fperiods),
            "swapped_periods_completed": int(rperiods),
            "retained": bool(fretained), "swapped_retained": bool(rretained),
            "max_abs_final_state_swap_error": float(np.max(np.abs(fstate-swap(rstate)))),
            "turns_forward": fturns.tolist(), "turns_swapped": rturns.tolist(),
            "max_abs_turn_swap_error": float(np.max(np.abs(fturns-rturns[::-1]))),
            "min_distance_forward": float(fd), "min_distance_swapped": float(rd),
            "max_abs_z_forward": fz.tolist(), "max_abs_z_swapped": rz.tolist()})
    out = {"study": "identical-particle label-exchange audit",
        "candidate_source": "first retained CO++ row in the corrected strict 512-start physical baseline",
        "start_index": int(chosen["start_index"]), "initial": chosen["initial"], "phase": chosen["phase"],
        "parameters": p.tolist(), "physical_drag": {"drag_mode": "Schiller-Naumann",
            "hydrodynamic_mode": "pairwise Rotne-Prager-like cross mobility mode 2",
            "Cunningham_factor_each": cc, "Re_per_dimensionless_speed_each": recoef,
            "gravity_z": gravity}, "resolutions": results,
        "equivariance_pass": all(r["forward_category"] == r["swapped_category"] and
            r["max_abs_final_state_swap_error"] < 1e-9 and r["max_abs_turn_swap_error"] < 1e-9
            for r in results),
        "interpretation": "This verifies label-swap symmetry of the equations and numerical trajectory for one representative state. It does not establish basin sizes or attractor stability."}
    path = ROOT/"results"/"symmetry_audit.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"equivariance_pass": out["equivariance_pass"], "resolutions": results},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
