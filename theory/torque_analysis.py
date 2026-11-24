"""Analytical leading-mode torques and converged nominal M3 mode decomposition."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/v17c/v17c_21/code"))
sys.path.insert(0, str(ROOT / "research/v17c/v17c_21a/code"))
from analytical_model import _phase_grid, _rf_field_rotating, load_inputs  # noqa: E402
from m3_harmonic_balance import HBConfig, solve_harmonic_balance  # noqa: E402


def leading_mode_torques(r: float, alpha: float, lam: float, beta: float,
                         amplitude: float, coefficient: complex, frequency: float,
                         k: int) -> dict[str, float]:
    """Exact zero-orbit-frequency M2 response for one circular traveling mode.

    RF torque contains torque on both the mean orbit and the RF micromotion;
    drag on micromotion must be subtracted to obtain the slow-orbit torque.
    """
    n = abs(k)
    chirality = 1 if k > 0 else -1
    denom = (frequency**2 + alpha * beta)**2 + (lam * frequency)**2
    base = alpha**2 * n**2 * abs(amplitude * coefficient)**2 * r**(2*n-2)
    micro_l = chirality * base * frequency / denom
    micro_rf = lam * micro_l
    orbit_rf = (n - 1) * micro_rf
    return {
        "k": k, "radial_power": 2*n-2, "denominator": denom,
        "mean_orbit_RF_torque": orbit_rf,
        "micromotion_RF_torque": micro_rf,
        "total_RF_torque": n * micro_rf,
        "micromotion_angular_momentum": micro_l,
        "micromotion_drag_torque": -lam * micro_l,
        "net_slow_torque_at_zero_omega": orbit_rf,
    }


def critical_radii(A: float, B: float, nu: float, alpha: float,
                   lam: float, beta: float,
                   c2: complex = 1.0, cm4: complex = 1.0) -> dict[str, float]:
    """Leading-only zero RF and zero slow torque radii at omega=0."""
    d1 = (1 + alpha * beta)**2 + lam**2
    d2 = (nu**2 + alpha * beta)**2 + (lam * nu)**2
    common = A*A*abs(c2)**2*d2/(B*B*abs(cm4)**2*nu*d1)
    return {
        "zero_total_RF_torque_r_over_R": (common/8)**0.25,
        "zero_net_slow_torque_r_over_R": (common/12)**0.25,
        "simple_zero_RF_beta_lambda_0": (A*A*nu**3/(8*B*B))**0.25,
        "simple_zero_slow_beta_lambda_0": (A*A*nu**3/(12*B*B))**0.25,
    }


def nominal_m3_decomposition() -> dict:
    data = load_inputs()
    campaign = json.loads((ROOT / "research/v17c/v17c_21a/results/full_grid_campaign.json").read_text())
    stage = campaign["stages"][-1]
    assert stage["status"] == "VERIFIED_ROOT" and stage["grid"] == [192, 16, 16]
    cfg = HBConfig(**stage["config"])
    root = stage["root"]
    theta, phases = _phase_grid(cfg.ntheta, cfg.nphase)
    particles = []
    for i in range(2):
        r = root["radii_R"][i]
        omega = root["omega_over_Omega1"][i]
        hb = solve_harmonic_balance(
            r, omega, data["alpha"][i], data["lambda"][i],
            data["Re_per_speed"][i], data["beta"], data["modes"], cfg,
            require_convergence=True,
        )
        xi = hb["xi_grid"]
        w = (r + xi[..., 0]) + 1j*xi[..., 1]
        per_mode = []
        for mode in data["modes"]:
            field = _rf_field_rotating(w, theta, phases, data["alpha"][i], [mode])
            torque = (r + xi[..., 0])*field.imag - xi[..., 1]*field.real
            per_mode.append({
                "family": int(mode.family) + 1, "k": int(mode.k),
                "torque": float(np.mean(torque)),
                "mean_radial_force": float(np.mean(field.real)),
                "mean_tangential_force": float(np.mean(field.imag)),
            })
        rf_sum = sum(x["torque"] for x in per_mode)
        assert abs(rf_sum - hb["mean_total_RF_torque"]) < 2e-13
        family = [sum(x["torque"] for x in per_mode if x["family"] == j) for j in (1, 2)]
        lead = [leading_mode_torques(r, data["alpha"][i], data["lambda"][i], data["beta"],
                                     mode.amplitude, mode.coefficient, mode.frequency, mode.k)
                for mode in data["modes"] if mode.k in (2, -4)]
        particles.append({
            "index": i+1, "r_over_R": r, "omega_over_Omega1": omega,
            "frequency_Hz": root["frequency_Hz"][i], "alpha": data["alpha"][i],
            "lambda": data["lambda"][i], "Re_star": data["Re_per_speed"][i],
            "m3_modes": per_mode, "m3_RF_family_torques": family,
            "m3_RF_total_torque": rf_sum,
            "m3_drag_total_torque": hb["mean_total_drag_torque"],
            "m3_RF_force_rotating_xy": hb["mean_rf_force_rotating_xy"],
            "m3_drag_force_rotating_xyz": hb["mean_drag_force_rotating_xyz"],
            "m3_inner_relative_residual": hb["relative_original_fast_equation_residual"],
            "m3_inner_converged": hb["converged"],
            "leading_linear_at_fixed_m3_radius_and_zero_omega": lead,
            "leading_critical_radii": critical_radii(
                data["exact_p"][0], data["exact_p"][1], data["exact_p"][2],
                data["alpha"][i], data["lambda"][i], data["beta"]),
        })
    return {
        "model": "V21A M3, full 8-mode MFS, nonlinear SN drag, fixed published M3 root",
        "source_sha": "94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8",
        "root_source": "research/v17c/v17c_21a/results/full_grid_campaign.json",
        "HB_grid": stage["grid"], "HB_config": stage["config"],
        "parameters": data["exact_p"], "particles": particles,
        "caveat": "Per-mode RF torque decomposition is exact for the recomputed M3 collocation field; changing modes would require re-solving micromotion. Leading formulas are zero-omega linear-response approximations, not M3 boundary conditions.",
    }


if __name__ == "__main__":
    result = nominal_m3_decomposition()
    dest = ROOT / "data/m3_nominal_torque_decomposition.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for p in result["particles"]:
        print(p["index"], p["r_over_R"], p["m3_RF_family_torques"],
              p["m3_RF_total_torque"], p["m3_drag_total_torque"], flush=True)
