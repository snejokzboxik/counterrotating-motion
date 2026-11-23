"""Fail-fast M3 averaged balances and outer six-variable root solve."""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from analytical_model import coulomb_ring_average
from m3_harmonic_balance import HBConfig, HarmonicBalanceFailure, solve_harmonic_balance


class M3OuterFailure(RuntimeError):
    pass


def initial_root(seed: dict) -> np.ndarray:
    return np.array([seed["radii_R"][0], seed["omega_over_Omega1"][0],
                     seed["radii_R"][1], seed["omega_over_Omega1"][1],
                     seed["mean_z_R"][0], seed["mean_z_R"][1]], dtype=float)


class M3PairSystem:
    """M3 slow force/moment balances with explicit HB status propagation."""

    def __init__(self, data: dict, config: HBConfig, xi_seed: list[np.ndarray] | None = None):
        self.data = data
        self.config = config
        self._xi_cache: list[np.ndarray | None] = (
            [None, None] if xi_seed is None else [np.asarray(x).copy() for x in xi_seed]
        )
        self.evaluations = 0

    def balance(self, x: np.ndarray, require_inner: bool = True):
        r1, w1, r2, w2, z1, z2 = map(float, x)
        d = self.data
        fc1, fc2 = coulomb_ring_average(r1, r2, z1, z2, d["C"], d["mass_ratio_m1_m2"])
        rows = []
        particles = []
        for i, (r, omega, z, fc) in enumerate(((r1, w1, z1, fc1), (r2, w2, z2, fc2))):
            hb = solve_harmonic_balance(
                r, omega, d["alpha"][i], d["lambda"][i], d["Re_per_speed"][i],
                d["beta"], d["modes"], self.config, xi0=self._xi_cache[i],
                require_convergence=require_inner,
            )
            if hb["converged"]:
                self._xi_cache[i] = hb["xi_grid"].copy()
            rf = hb["mean_rf_force_rotating_xy"]
            drag = hb["mean_drag_force_rotating_xyz"]
            rows.extend([
                (r * omega**2 + d["alpha"][i] * d["beta"] * r + fc[0] + rf[0] + drag[0]) * 100.0,
                (hb["mean_total_RF_torque"] + hb["mean_total_drag_torque"] + r * fc[1]) * 1e4,
                (-2.0 * d["alpha"][i] * d["beta"] * z + d["gravity_z"] + fc[2] + drag[2]) * 100.0,
            ])
            particles.append(hb)
        self.evaluations += 1
        return np.asarray(rows, dtype=float), {"particles": particles}


def solve_m3_root(
    data: dict,
    seed: np.ndarray,
    config: HBConfig,
    max_nfev: int = 180,
    scaled_balance_tolerance: float = 2e-6,
    require_convergence: bool = True,
    xi_seed: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Solve all six averaged balances on this grid, never accepting bad HB."""
    started = perf_counter()
    system = M3PairSystem(data, config, xi_seed=xi_seed)
    lower = np.array([0.1, -0.05, 0.1, -0.05, -2.0, -2.0])
    upper = np.array([0.99, 0.05, 0.99, 0.05, 2.0, 2.0])

    def objective(x):
        value = system.balance(x, require_inner=True)[0]
        if system.evaluations % 10 == 0:
            print(f"M3 eval={system.evaluations} max_scaled_residual={np.max(np.abs(value)):.3e}", flush=True)
        return value

    sol = least_squares(objective, np.asarray(seed, dtype=float), bounds=(lower, upper),
                        xtol=2e-10, ftol=2e-10, gtol=2e-10, max_nfev=max_nfev)
    residual, details = system.balance(sol.x, require_inner=True)
    x = np.asarray(sol.x, dtype=float)
    balance_norm = float(np.max(np.abs(residual)))
    inner_ok = all(p["converged"] for p in details["particles"])
    verified = bool(sol.success and inner_ok and balance_norm <= scaled_balance_tolerance)
    result = {
        "status": "VERIFIED_ROOT" if verified else "OUTER_ROOT_NOT_VERIFIED",
        "outer_solver_success": bool(sol.success),
        "outer_message": str(sol.message),
        "outer_cost": float(sol.cost),
        "outer_nfev": int(sol.nfev),
        "balance_evaluations": system.evaluations,
        "wall_time_s": perf_counter() - started,
        "scaled_balance_residual": residual.tolist(),
        "scaled_balance_max_norm": balance_norm,
        "scaled_balance_tolerance": scaled_balance_tolerance,
        "inner_all_converged": inner_ok,
        "root": {
            "radii_R": [float(x[0]), float(x[2])],
            "omega_over_Omega1": [float(x[1]), float(x[3])],
            "frequency_Hz": [float(x[1] * data["f1_Hz"]), float(x[3] * data["f1_Hz"])],
            "mean_z_R": [float(x[4]), float(x[5])],
        },
        "inner": [{k: v for k, v in p.items() if k not in
                    {"xi_grid", "xi_fourier_coefficients", "nonlinear_force_fourier_coefficients"}}
                  for p in details["particles"]],
        "config": asdict(config),
    }
    result["_xi_grids"] = [p["xi_grid"] for p in details["particles"]]
    result["_xi_fourier"] = [p["xi_fourier_coefficients"] for p in details["particles"]]
    if require_convergence and not verified:
        raise M3OuterFailure(
            f"M3 outer root not verified: solver={sol.success}, inner={inner_ok}, "
            f"scaled residual={balance_norm:.3e}, tolerance={scaled_balance_tolerance:.3e}"
        )
    return result
