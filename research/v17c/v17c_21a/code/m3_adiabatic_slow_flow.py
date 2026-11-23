"""Adiabatic M3 slow-flow closure derived from the converged torus HB.

This is a declared approximation: the fast M3 micromotion is assumed to track
the instantaneous slow state.  Coulomb terms use the same independent-phase
ring averages as V21.  It is not an exact nonstationary averaging theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from analytical_model import coulomb_ring_average
from m3_harmonic_balance import HBConfig, solve_harmonic_balance


@dataclass
class M3AveragedObservables:
    radial_acceleration: np.ndarray
    axial_acceleration: np.ndarray
    torque: np.ndarray
    angular_momentum: np.ndarray
    hb: list[dict[str, Any]]


def evaluate_observables(data: dict, state: np.ndarray, config: HBConfig,
                         xi_seed: list[np.ndarray] | None = None) -> M3AveragedObservables:
    """Evaluate averaged accelerations/torques at a frozen slow state.

    State ordering is (r1,u1,omega1,z1,vz1,r2,u2,omega2,z2,vz2), all in
    V21 dimensionless units and derivatives with respect to tau=Omega_1*t.
    """
    x = np.asarray(state, dtype=float).reshape(2, 5)
    r, u, omega, z, vz = x.T
    fc = coulomb_ring_average(r[0], r[1], z[0], z[1],
                              data["C"], data["mass_ratio_m1_m2"])
    hb_rows = []
    radial = np.zeros(2)
    axial = np.zeros(2)
    torque = np.zeros(2)
    angular_momentum = np.zeros(2)
    for i in range(2):
        h = solve_harmonic_balance(
            r[i], omega[i], data["alpha"][i], data["lambda"][i],
            data["Re_per_speed"][i], data["beta"], data["modes"], config,
            u_r=u[i], v_z=vz[i],
            xi0=None if xi_seed is None else xi_seed[i],
            require_convergence=True,
        )
        hb_rows.append(h)
        rf = h["mean_rf_force_rotating_xy"]
        drag = h["mean_drag_force_rotating_xyz"]
        radial[i] = (r[i] * omega[i] ** 2 + data["alpha"][i] * data["beta"] * r[i]
                     + fc[i][0] + rf[0] + drag[0])
        axial[i] = (-2.0 * data["alpha"][i] * data["beta"] * z[i]
                    + data["gravity_z"] + fc[i][2] + drag[2])
        torque[i] = h["mean_total_RF_torque"] + h["mean_total_drag_torque"]
        angular_momentum[i] = h["mean_angular_momentum"]
    return M3AveragedObservables(radial, axial, torque, angular_momentum, hb_rows)


def angular_momentum_gradient(data: dict, state: np.ndarray, config: HBConfig,
                              base_hb: list[dict], steps: np.ndarray) -> np.ndarray:
    """Finite-difference dL_i/d(r_i,u_i,omega_i,vz_i), with repeated HB solves.

    The ideal infinite-electrode RF field and DC quadrupole are z independent,
    hence dL_i/dz_i=0 in this model.  Coulomb's ring-averaged torque is zero.
    """
    x = np.asarray(state, dtype=float).reshape(2, 5)
    grad = np.zeros((2, 5))
    local_indices = (0, 1, 2, 4)
    for i in range(2):
        for j in local_indices:
            h = float(steps[j])
            xp, xm = x.copy(), x.copy()
            xp[i, j] += h
            xm[i, j] -= h
            lp = evaluate_observables(data, xp.ravel(), config,
                                      [q["xi_grid"] for q in base_hb]).angular_momentum[i]
            lm = evaluate_observables(data, xm.ravel(), config,
                                      [q["xi_grid"] for q in base_hb]).angular_momentum[i]
            grad[i, j] = (lp - lm) / (2.0 * h)
    return grad


def slow_flow_jacobian(data: dict, state: np.ndarray, config: HBConfig,
                       state_steps: np.ndarray, derivative_steps: np.ndarray,
                       output_path=None) -> dict[str, Any]:
    """Build J of the adiabatic M3 flow using finite-difference HB sensitivities.

    The equilibrium derivative is assembled from partial derivatives of the
    averaged force/torque balances and dL/dX. This avoids differentiating the
    full ratio away from equilibrium and does not omit the implicit HB
    dependence: every perturbation re-solves the inner Fourier system.
    """
    x = np.asarray(state, dtype=float)
    x2 = x.reshape(2, 5)
    hb = evaluate_observables(data, x, config)
    xigrid = [row["xi_grid"] for row in hb.hb]
    Lgrad = angular_momentum_gradient(data, x, config, hb.hb, derivative_steps)
    if np.any(np.abs(Lgrad[:, 2]) < 1e-10):
        raise ArithmeticError(f"small dL/domega prevents solving angular flow: {Lgrad[:, 2]}")

    # g(X) contains the averaged radial acceleration, torque, and axial
    # acceleration for both particles.  Its Jacobian includes the implicit
    # response derivative because each perturbed point solves the HB closure.
    def g_of(y: np.ndarray) -> np.ndarray:
        obs = evaluate_observables(data, y, config, xigrid)
        return np.array([obs.radial_acceleration[0], obs.torque[0], obs.axial_acceleration[0],
                         obs.radial_acceleration[1], obs.torque[1], obs.axial_acceleration[1]])

    g0 = g_of(x)
    G = np.zeros((6, 10))
    column_records = []
    for j, h in enumerate(state_steps):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        G[:, j] = (g_of(xp) - g_of(xm)) / (2.0 * h)
        row = {"column": j, "step": float(h), "derivative_column": G[:, j].tolist()}
        column_records.append(row)
        if output_path is not None:
            import json
            output_path.write_text(json.dumps({
                "status": "JACOBIAN_IN_PROGRESS", "completed_columns": column_records,
                "root_state": x.tolist(), "L_gradient": Lgrad.tolist(),
                "g_equilibrium": g0.tolist(),
            }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    J = np.zeros((10, 10))
    for i in range(2):
        base = 5 * i
        J[base, base + 1] = 1.0  # rdot = u
        J[base + 1, :] = G[3 * i, :]
        # Angular momentum chain rule:
        # L_r*u + L_u*udot + L_omega*omegadot + L_z*vz + L_vz*vzdot = torque.
        J[base + 2, :] = (G[3 * i + 1, :] - Lgrad[i, 1] * J[base + 1, :]
                          - Lgrad[i, 0] * np.eye(10)[base + 1, :]
                          - Lgrad[i, 4] * G[3 * i + 2, :]) / Lgrad[i, 2]
        J[base + 3, base + 4] = 1.0  # zdot = vz
        J[base + 4, :] = G[3 * i + 2, :]
    eig = np.linalg.eigvals(J)
    return {
        "status": "ADIABATIC_M3_SLOW_FLOW_JACOBIAN",
        "state_order": ["r1", "u1", "omega1", "z1", "vz1",
                        "r2", "u2", "omega2", "z2", "vz2"],
        "time": "tau=Omega1*t",
        "root_state": x.tolist(),
        "equilibrium_acceleration_torque_vector": g0.tolist(),
        "angular_momentum_gradient_local_coordinates": Lgrad.tolist(),
        "dL_domega": Lgrad[:, 2].tolist(),
        "observables_inner_solve_status": [q["status"] for q in hb.hb],
        "observables_inner_residual": [q["relative_nonlinear_residual"] for q in hb.hb],
        "g_jacobian": G.tolist(),
        "slow_flow_jacobian": J.tolist(),
        "eigenvalues_real_imag": [[float(v.real), float(v.imag)] for v in eig],
        "max_real_eigenvalue": float(np.max(eig.real)),
        "all_modes_stable_in_reduced_flow": bool(np.max(eig.real) < 0),
        "state_difference_steps": state_steps.tolist(),
        "angular_momentum_difference_steps": derivative_steps.tolist(),
        "approximation": (
            "Quasi-steady frozen-torus M3 closure: micromotion is recomputed at each slow state; "
            "slow accelerations are omitted from the fast HB operator; Coulomb uses independent-phase "
            "ring averages. This is a local adiabatic averaged-flow model, not a full-system Floquet or Lyapunov spectrum."
        ),
    }
