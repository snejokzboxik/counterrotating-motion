"""Single-particle moving-orbit averaging with torus-averaged Oseen drag.

This is a perturbative predictor: it solves the Stokes linear micromotion
response, evaluates the nonlinear Oseen average on the resulting two-frequency
velocity torus, and balances mean radial/tangential forces. It is not the exact
two-particle solution and does not include Coulomb or gravity.
"""
from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import root

from physical_screen import LOCAL15, PAIR
from two_particle_oseen import RFMode, local15_trial_model


def unsteady_stokes_drag_impedance(lam: float, radius_m: float,
                                   omega_dimless: float,
                                   omega1_rad_s: float,
                                   air_kinematic_viscosity_m2_s: float) -> complex:
    """Linear Basset/added-mass impedance for exp(-i omega t) convention."""
    omega_dimensional = abs(omega_dimless) * omega1_rad_s
    chi = radius_m * math.sqrt(omega_dimensional/air_kinematic_viscosity_m2_s)
    return lam * (1.0 + chi*(1.0-1.0j)/math.sqrt(2.0)
                  - 1.0j*chi*chi/9.0)


def rf_mode_response(radius: float, omega: float, alpha: float, lam: float,
                     mode: RFMode, beta: float,
                     unsteady: bool = False, particle_index: int = 0) -> dict:
    """First-order mode response in the co-rotating frame."""
    k, n = mode.k, abs(mode.k)
    chirality = 1.0 if k > 0 else -1.0
    sigma = mode.frequency - k * omega
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    I = np.eye(2)
    slow_jacobian = alpha * beta * I
    mode_lam = lam
    if unsteady:
        model = local15_trial_model(coulomb_scale=0.0, gravity=False)
        nu_f = model.air_dynamic_viscosity_Pa_s/model.air_density_kg_m3
        mode_lam = unsteady_stokes_drag_impedance(
            lam, model.radii_m[particle_index], mode.frequency,
            model.omega1_rad_s, nu_f)
    D = (-sigma**2*I - 1j*sigma*(mode_lam*I + 2.0*omega*J)
         -omega**2*I + mode_lam*omega*J - slow_jacobian)
    fhat = (-alpha*mode.amplitude*n*radius**(n-1)
            * np.array([1.0, 1j*chirality], dtype=complex))
    H = (-alpha*mode.amplitude*n*(n-1)*radius**(n-2)
         * np.array([[1.0, 1j*chirality],
                     [1j*chirality, -1.0]], dtype=complex))
    xi = np.linalg.solve(D, fhat)
    mean_rf = 0.5*np.real(H @ np.conjugate(xi))
    return {"sigma": float(sigma), "xi": xi, "mean_rf": mean_rf,
            "condition_number": float(np.linalg.cond(D)),
            "micromotion_amplitude": float(np.linalg.norm(xi))}


def two_mode_response(radius: float, omega: float, alpha: float,
                      lam: float, modes: Sequence[RFMode], beta: float,
                      unsteady: bool = False, particle_index: int = 0) -> list[dict]:
    return [rf_mode_response(radius, omega, alpha, lam, mode, beta,
                             unsteady, particle_index)
            for mode in modes]


def torus_oseen_average(radius: float, omega: float, alpha: float,
                        lam: float, modes: Sequence[RFMode], beta: float,
                        particle_index: int, samples: int = 128) -> dict:
    """Quadrature for <|V+w|(V+w)> in the co-rotating plane.

    The laboratory-frame velocity expressed in the rotating basis includes
    both d(xi_rot)/dt and omega J xi_rot. RF phases are independently sampled,
    i.e. this evaluates the two-torus mean, not a short-time random maximum.
    """
    if samples < 8:
        raise ValueError("at least 8 phase samples are required")
    model = local15_trial_model(coulomb_scale=0.0, gravity=False)
    i = int(particle_index)
    if i not in (0, 1):
        raise ValueError("particle_index must be 0 or 1")
    a = model.radii_m[i]
    rho_f = model.air_density_kg_m3
    eta = model.air_dynamic_viscosity_Pa_s
    scale = model.radial_scale_m*model.omega1_rad_s
    K = 2.0*rho_f*a*scale/eta
    cO = 3.0*K/16.0
    responses = two_mode_response(radius, omega, alpha, lam, modes, beta)
    phases = 2.0*math.pi*np.arange(samples)/samples
    p1, p2 = np.meshgrid(phases, phases, indexing="ij")
    phases_2d = (p1, p2)
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    total = np.zeros((samples, samples, 2), dtype=float)
    for phase, mode, response in zip(phases_2d, modes, responses):
        xi = response["xi"]
        sigma = response["sigma"]
        carrier = np.exp(-1j*phase)
        xi_rot = np.real(carrier[..., None]*xi[None, None, :])
        dxi_rot = np.real((-1j*sigma)*carrier[..., None]*xi[None, None, :])
        total += dxi_rot + omega*np.einsum("ab,...b->...a", J, xi_rot)
    slow_velocity = np.array([0.0, omega*radius])
    total += slow_velocity
    speed = np.linalg.norm(total, axis=-1)
    G = np.mean(speed[..., None]*total, axis=(0, 1))
    G_abs = float(np.mean(speed))
    stokes_drag = -lam*slow_velocity
    oseen_drag = -lam*cO*G
    return {
        "samples_per_RF_phase": samples,
        "mean_speed_dimensionless": G_abs,
        "mean_speed_m_s": G_abs*scale,
        "G_vector_dimensionless": G.tolist(),
        "G_r_dimensionless": float(G[0]),
        "G_theta_dimensionless": float(G[1]),
        "stokes_drag_r_dimensionless": float(stokes_drag[0]),
        "stokes_drag_theta_dimensionless": float(stokes_drag[1]),
        "oseen_drag_r_dimensionless": float(oseen_drag[0]),
        "oseen_drag_theta_dimensionless": float(oseen_drag[1]),
        "diameter_Re_phase_rms": float(K*np.sqrt(np.mean(speed**2))),
        "diameter_Re_mean_speed": K*G_abs,
        "oseen_linear_factor_at_mean_speed": cO*G_abs,
        "response": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in row.items() if k != "xi"} for row in responses],
    }


def mean_force_balance(state: Sequence[float], particle_index: int,
                       oseen: bool = True, samples: int = 128,
                       unsteady: bool = False) -> np.ndarray:
    """Mean force balance for an isolated rotating branch.

    The RF response uses the Stokes moving-orbit susceptibility. Model B adds
    the exactly quadrature-averaged nonlinear drag evaluated on that response.
    This is first order in the Oseen coefficient; micromotion feedback from
    higher Fourier components of the nonlinear drag remains a residual.
    """
    i = int(particle_index)
    model = local15_trial_model(coulomb_scale=0.0, gravity=False)
    radius, omega = map(float, state)
    if radius <= 0.0:
        return np.array([1e3 + radius, omega])
    alpha = model.charges_relative[i]/model.masses_relative[i]
    lam = model.lambdas_stokes[i]
    modes = model.rf_modes
    rf = two_mode_response(radius, omega, alpha, lam, modes, model.beta2,
                           unsteady, i)
    FRF = np.sum([mode["mean_rf"] for mode in rf], axis=0)
    Fdrag = -lam*np.array([0.0, omega*radius])
    if oseen:
        os = torus_oseen_average(radius, omega, alpha, lam, modes,
                                 model.beta2, i, samples=samples)
        Fdrag += np.array([os["oseen_drag_r_dimensionless"],
                            os["oseen_drag_theta_dimensionless"]])
    radial = omega**2*radius + alpha*model.beta2*radius + FRF[0] + Fdrag[0]
    tangential = FRF[1] + Fdrag[1]
    return np.array([radial, tangential])


def solve_branch(particle_index: int, initial: tuple[float, float],
                 oseen: bool = True, samples: int = 128,
                 unsteady: bool = False) -> dict:
    solution = root(mean_force_balance, np.asarray(initial, dtype=float),
                    args=(particle_index, oseen, samples, unsteady), method="hybr",
                    options={"xtol": 1e-10})
    radius, omega = map(float, solution.x)
    residual = mean_force_balance(solution.x, particle_index, oseen, samples,
                                  unsteady)
    return {"success": bool(solution.success), "message": str(solution.message),
            "rho": radius, "omega": omega,
            "residual": residual.tolist(),
            "oseen": oseen, "unsteady_stokes_control": unsteady,
            "phase_samples": samples}


def frozen_local15_starts() -> tuple[tuple[float, float], tuple[float, float]]:
    # Archived V17C.15 Stokes moving-orbit fixed points used as independent seeds.
    return (0.551745266004148, 0.0012771817038559458), \
           (0.7662833082551475, -0.0010694770940450292)


def branch_diagnostics(branch: dict, particle_index: int,
                       unsteady: bool = False) -> dict:
    model = local15_trial_model(coulomb_scale=0.0, gravity=False)
    i = int(particle_index)
    alpha = model.charges_relative[i]/model.masses_relative[i]
    lam = model.lambdas_stokes[i]
    response = two_mode_response(branch["rho"], branch["omega"], alpha,
                                 lam, model.rf_modes, model.beta2,
                                 unsteady, i)
    torques = [branch["rho"]*float(mode["mean_rf"][1])
               for mode in response]
    denom = sum(abs(value) for value in torques)
    return {
        "RF_mode_mean_torques_dimensionless": torques,
        "RF_net_mean_torque_dimensionless": sum(torques),
        "absolute_net_over_sum_absolute_modes":
            abs(sum(torques))/denom if denom else 0.0,
        "mode_response": [{
            "sigma": mode["sigma"],
            "mean_RF_force_xy_dimensionless": mode["mean_rf"].tolist(),
            "micromotion_amplitude_dimensionless": mode["micromotion_amplitude"],
        } for mode in response],
    }


if __name__ == "__main__":
    starts = frozen_local15_starts()
    results = {}
    for i, start in enumerate(starts):
        stokes = solve_branch(i, start, oseen=False)
        oseen = solve_branch(i, start, oseen=True, samples=192)
        unsteady = solve_branch(i, start, oseen=False, unsteady=True)
        results[f"particle_{i+1}"] = {
            "stokes": {**stokes, "diagnostics": branch_diagnostics(stokes, i)},
            "oseen": {**oseen, "diagnostics": branch_diagnostics(oseen, i)},
            "unsteady_stokes_control": {
                **unsteady, "diagnostics": branch_diagnostics(unsteady, i, True)},
            "oseen_torus_quadrature_128": torus_oseen_average(
                *start, 1.0 if i == 0 else 0.34126029835390936,
                local15_trial_model().lambdas_stokes[i],
                local15_trial_model().rf_modes, LOCAL15["beta"], i, 128),
            "oseen_torus_quadrature_256": torus_oseen_average(
                *start, 1.0 if i == 0 else 0.34126029835390936,
                local15_trial_model().lambdas_stokes[i],
                local15_trial_model().rf_modes, LOCAL15["beta"], i, 256),
        }
    output = Path(__file__).resolve().parents[1] / "results" / "analytic_screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for name, rows in results.items():
        serializable[name] = {}
        for key, value in rows.items():
            serializable[name][key] = value
    output.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(output)
