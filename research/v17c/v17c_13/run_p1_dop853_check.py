"""Independent Python/Numpy DOP853 check of a late N=6 #505 crystal segment.

The RHS below is separately written and does not call the Numba RK4 force
routine.  It integrates a post-load segment from a saved state so the RF
phases remain at the exact absolute simulation time.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import nbody_dynamics as nb

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "p1_long_candidates" / "P2_505_angle0_phase0_long_trajectory.npz"
OUT = HERE / "p1_dop853"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
TAU = 2.0 * math.pi


def independent_rhs(t: float, y_flat: np.ndarray, p: np.ndarray,
                    phases: np.ndarray, family: np.ndarray, modes: np.ndarray,
                    coeff: np.ndarray, qratio: float) -> np.ndarray:
    """Direct 3D acceleration from modal electric gradients and pair forces."""
    y = y_flat.reshape((6, 6))
    dydt = np.zeros_like(y)
    amp_a, amp_b, nu, beta, inv_mass_2, coupling, lam1, lam2 = p
    carrier_a = np.exp(-1j * (t + phases[0]))
    carrier_b = np.exp(-1j * (nu * t + phases[1]))
    for i in range(6):
        x, yy, z, vx, vy, vz = y[i]
        u = complex(x, yy)
        field_x = 0.0
        field_y = 0.0
        for h, mode in enumerate(modes):
            degree = abs(int(mode))
            base = u if mode > 0 else u.conjugate()
            carrier = amp_a * carrier_a if family[h] == 0 else amp_b * carrier_b
            grad = degree * carrier * coeff[h] * base ** (degree - 1)
            field_x -= grad.real
            field_y += grad.imag if mode > 0 else -grad.imag
        if GROUPS[i] == 0:
            qi, inv_mass, lam = 1.0, 1.0, lam1
        else:
            qi, inv_mass, lam = qratio, inv_mass_2, lam2
        q_over_m = qi * inv_mass
        ax = q_over_m * (field_x + beta * x) - lam * vx
        ay = q_over_m * (field_y + beta * yy) - lam * vy
        az = q_over_m * (-2.0 * beta * z) - lam * vz
        dydt[i, :3] = (vx, vy, vz)
        dydt[i, 3:] = (ax, ay, az)

    # Independent equal-and-opposite central pair force assembly.
    positions = y[:, :3]
    for i in range(5):
        qi = 1.0 if GROUPS[i] == 0 else qratio
        mi_inv = 1.0 if GROUPS[i] == 0 else inv_mass_2
        for j in range(i + 1, 6):
            qj = 1.0 if GROUPS[j] == 0 else qratio
            mj_inv = 1.0 if GROUPS[j] == 0 else inv_mass_2
            dr = positions[i] - positions[j]
            d2 = float(np.dot(dr, dr))
            if d2 > 1.0e-28:
                pair_force_unit = coupling * (qi * qj / qratio) * dr / (d2 * math.sqrt(d2))
                dydt[i, 3:] += mi_inv * pair_force_unit
                dydt[j, 3:] -= mj_inv * pair_force_unit
    return dydt.ravel()


def angles_and_metrics(t: np.ndarray, states: np.ndarray) -> dict:
    pos = states.reshape((len(t), 6, 6)).transpose(1, 2, 0)
    theta = np.unwrap(np.arctan2(pos[:, 1, :], pos[:, 0, :]), axis=1)
    radii = np.hypot(pos[:, 0, :], pos[:, 1, :])
    turns = (theta[:, -1] - theta[:, 0]) / TAU
    duration_cycles = (t[-1] - t[0]) / TAU
    frequency = turns * nb.F1_HZ / duration_cycles
    psi3 = [np.abs(np.mean(np.exp(3j * theta[g * 3:(g + 1) * 3]), axis=0)) for g in (0, 1)]
    return {
        "turns_each": turns.tolist(),
        "frequency_Hz_each": frequency.tolist(),
        "mean_frequency_Hz_by_group": [float(np.mean(frequency[:3])), float(np.mean(frequency[3:]))],
        "frequency_spread_by_group": [float(np.std(frequency[:3]) / abs(np.mean(frequency[:3]))),
                                      float(np.std(frequency[3:]) / abs(np.mean(frequency[3:])))],
        "mean_radius_R_by_particle": np.mean(radii, axis=1).tolist(),
        "mean_Psi3_by_group": [float(np.mean(v)) for v in psi3],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p, family, modes, coeff, qratio = nb.load_local15()
    with np.load(SOURCE) as data:
        logs = data["trajectory"]
        skip = int(data["post_load_skip"])
        phases = data["phases"].astype(np.float64)
    start_idx = skip + 10_000
    segment_cycles = 3_000
    stop_idx = start_idx + segment_cycles
    t_ref = logs[start_idx:stop_idx + 1, 0, 0]
    y0 = logs[start_idx, :, 1:7].ravel().copy()
    reference = logs[start_idx:stop_idx + 1, :, 1:7].reshape((len(t_ref), 36))
    tic = time.perf_counter()
    solution = solve_ivp(
        independent_rhs, (float(t_ref[0]), float(t_ref[-1])), y0,
        args=(p, phases, family, modes, coeff, qratio),
        method="DOP853", t_eval=t_ref, rtol=1e-11, atol=1e-12,
        max_step=TAU / 96.0,
    )
    elapsed = time.perf_counter() - tic
    result_states = solution.y.T
    errors = np.abs(result_states - reference)
    rk_metrics = angles_and_metrics(t_ref, reference)
    dop_metrics = angles_and_metrics(solution.t, result_states)
    summary = {
        "status": "passed" if solution.success and len(solution.t) == len(t_ref) else "failed",
        "independent_integrator": "scipy.solve_ivp(method='DOP853')",
        "rhs": "separately implemented Python/Numpy modal gradient, DC, drag, and all 15 Coulomb pairs",
        "source_trajectory": SOURCE.name,
        "start_primary_period_after_load": 10_000,
        "segment_primary_periods": segment_cycles,
        "rtol": 1e-11, "atol": 1e-12, "max_step_tau": TAU / 96.0,
        "nfev": int(solution.nfev), "elapsed_seconds": elapsed,
        "solver_message": solution.message,
        "max_abs_state_difference_at_end": float(np.max(errors[-1])),
        "max_abs_state_difference_over_sampled_segment": float(np.max(errors)),
        "max_abs_frequency_difference_Hz": float(np.max(np.abs(
            np.asarray(rk_metrics["frequency_Hz_each"]) - np.asarray(dop_metrics["frequency_Hz_each"])))),
        "rk4_reference_metrics": rk_metrics,
        "dop853_metrics": dop_metrics,
    }
    np.savez_compressed(OUT / "dop853_late_segment.npz", tau=solution.t,
                        state=result_states.reshape((-1, 6, 6)),
                        reference_state=reference.reshape((-1, 6, 6)), phases=phases)
    (OUT / "dop853_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rk4_reference_metrics", "dop853_metrics")}, indent=2))
    print("RK4 frequencies", rk_metrics["frequency_Hz_each"])
    print("DOP853 frequencies", dop_metrics["frequency_Hz_each"])
    if summary["status"] != "passed":
        raise SystemExit("DOP853 segment did not complete")


if __name__ == "__main__":
    main()
