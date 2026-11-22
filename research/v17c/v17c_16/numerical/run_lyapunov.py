"""Finite-time QR/Lyapunov check for the ideal-field Oseen two-body model.

This is a nonperiodic/quasiperiodic finite-time stability diagnostic. It does
not report Floquet multipliers or claim an asymptotic attractor.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from physical_screen import G
from two_particle_oseen import local15_trial_model, rhs, tangent_jacobian
from run_trajectory_validation import _branch_rows, _make_initial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=200)
    parser.add_argument("--steps-per-fast-period", type=int, default=16)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--tag", default="qr200")
    args = parser.parse_args()

    model = local15_trial_model(coulomb_scale=1.0, gravity=True)
    model = replace(model, gravity_hat=(
        0.0, 0.0, -G/(model.radial_scale_m*model.omega1_rad_s**2)))
    initial = _make_initial(model, _branch_rows(),
                            {"angles": [0.0, np.pi/2]}, "vertical")
    q = np.eye(12)
    y = np.concatenate([initial, q.ravel()])
    nu = max(m.frequency for m in model.rf_modes)
    max_step = 2*np.pi/(args.steps_per_fast_period*nu)
    period = 2*np.pi
    logs = np.zeros(12)
    nfev_total = 0
    for n in range(args.periods):
        def augmented(tau, state):
            x = state[:12]
            tangent = state[12:].reshape(12, 12)
            dx = rhs(tau, x, model, oseen=True, coulomb=True)
            dT = tangent_jacobian(tau, x, model, oseen=True,
                                  coulomb=True) @ tangent
            return np.concatenate([dx, dT.ravel()])

        result = solve_ivp(augmented, (n*period, (n+1)*period), y,
                           method="DOP853", rtol=args.rtol, atol=args.atol,
                           max_step=max_step)
        if not result.success:
            raise RuntimeError(result.message)
        nfev_total += int(result.nfev)
        y = result.y[:, -1]
        q, r = np.linalg.qr(y[12:].reshape(12, 12))
        diagonal = np.abs(np.diag(r))
        if np.any(diagonal <= 0.0):
            raise FloatingPointError("singular tangent QR factor")
        logs += np.log(diagonal)
        y[12:] = q.ravel()
    exponents = logs/(args.periods*period)
    out = Path(__file__).resolve().parents[1]/"results"/f"lyapunov_{args.tag}.json"
    payload = {
        "model": "ideal leading-harmonic RF plus ideal DC, two-particle Coulomb, quasi-steady Oseen, vertical gravity/buoyancy",
        "method": "DOP853 state plus variational matrix; QR once per primary RF period",
        "classification": "finite-time Lyapunov spectrum; quasiperiodic drive; no Floquet claim",
        "primary_RF_periods": args.periods,
        "steps_per_fast_period_bound": args.steps_per_fast_period,
        "rtol": args.rtol,
        "atol": args.atol,
        "finite_time_exponents_per_tau": exponents.tolist(),
        "finite_time_exponents_per_primary_period": (exponents*period).tolist(),
        "largest_exponent_per_tau": float(np.max(exponents)),
        "largest_exponent_per_primary_period": float(np.max(exponents)*period),
        "final_state": y[:12].tolist(),
        "nfev_total": nfev_total,
        "limitations": [
            "finite horizon only; exponent convergence versus horizon not established",
            "ideal 2D infinite-rod RF field and ideal DC quadrupole",
            "quasi-steady Oseen closure remains a hydrodynamic screening model",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(out)
    print(json.dumps({"largest_per_tau": payload["largest_exponent_per_tau"],
                      "spectrum_per_tau": exponents.tolist()}, indent=2))


if __name__ == "__main__":
    main()
