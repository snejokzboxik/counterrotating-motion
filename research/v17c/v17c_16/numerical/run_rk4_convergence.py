"""Short-horizon RK4 64/96/128-per-fastest-period refinement audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from physical_screen import G
from run_trajectory_validation import _branch_rows, _make_initial
from two_particle_oseen import local15_trial_model, rhs, rk4_integrate


def main() -> None:
    model = local15_trial_model(coulomb_scale=1.0, gravity=True)
    model = model.__class__(**{
        **model.__dict__,
        "gravity_hat": (0.0, 0.0,
                        -G/(model.radial_scale_m*model.omega1_rad_s**2)),
    })
    y0 = _make_initial(model, _branch_rows(),
                       {"angles": [0.0, np.pi/2]}, "vertical")
    tau_end = 2*np.pi*10.0
    nu = max(m.frequency for m in model.rf_modes)
    reference = solve_ivp(lambda t, y: rhs(t, y, model, oseen=True,
                                           coulomb=True),
                         (0.0, tau_end), y0, method="DOP853", rtol=2e-13,
                         atol=2e-15, max_step=2*np.pi/(512*nu))
    if not reference.success:
        raise RuntimeError(reference.message)
    yref = reference.y[:, -1]
    results = []
    for steps in (64, 96, 128):
        h = 2*np.pi/(steps*nu)
        ts, ys = rk4_integrate(model, y0, tau_end, h,
                               oseen=True, coulomb=True)
        delta = ys[-1]-yref
        results.append({
            "steps_per_fast_RF_period": steps,
            "rk4_step_tau": float(ts[1]-ts[0]),
            "final_state_max_abs_error": float(np.max(np.abs(delta))),
            "final_state_rms_abs_error": float(np.sqrt(np.mean(delta**2))),
            "final_state_scaled_rms_error": float(np.sqrt(np.mean(
                (delta/np.maximum(np.abs(yref), 1e-3))**2))),
            "final_state": ys[-1].tolist(),
        })
    out = Path(__file__).resolve().parents[1]/"results"/"rk4_convergence.json"
    out.write_text(json.dumps({
        "model": "ideal leading-harmonic RF + ideal DC; full two-particle Coulomb; Oseen drag; vertical gravity/buoyancy",
        "initial_condition": "nominal Oseen branch, angles 0 and pi/2, vertical gravity-set z, RF phases zero",
        "horizon_primary_RF_periods": 10,
        "reference": {"integrator": "DOP853", "rtol": 2e-13,
                      "atol": 2e-15, "max_step_fast_period_fraction": 1/512,
                      "nfev": int(reference.nfev), "final_state": yref.tolist()},
        "results": results,
        "interpretation": "short-horizon local integrator convergence only; not a long-time branch or physical-model validation",
    }, indent=2)+"\n", encoding="utf-8")
    print(out)
    for row in results:
        print(row["steps_per_fast_RF_period"],
              row["final_state_max_abs_error"],
              row["final_state_rms_abs_error"])


if __name__ == "__main__":
    main()
