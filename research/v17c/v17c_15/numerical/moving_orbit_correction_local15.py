"""Evaluate the local moving-orbit RF response for the frozen Local15 field.

This is a reduced-model prediction, compared without refitting to the archived
15,000-period exact-RF panel.  All frequencies, times, positions, and forces
are dimensionless.  It is not a new trajectory campaign.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import root


ROOT = Path(__file__).resolve().parents[1]
PARAMS = {
    "A": 0.03716194907442204,
    "B": 0.1626938199343468,
    "nu": 3.1118029452860356,
    "beta": 0.017273303124515367,
    "species1": {"alpha": 1.0, "lambda": 0.20975541409005094,
                 "leading_rho": 0.5503691530067869,
                 "leading_omega": 0.001440054163088106},
    "species2": {"alpha": 0.34126014359705537,
                 "lambda": 0.12617311138613094,
                 "leading_rho": 0.7663818561742767,
                 "leading_omega": -0.0010579232360122394},
}


def planar_rf_force_jet(radius: float, alpha: float, k: int,
                        amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Return complex force amplitude and its planar Jacobian at (radius,0)."""
    n = abs(k)
    chirality = 1.0 if k > 0 else -1.0
    z = complex(radius, 0.0)
    scale = complex(amplitude)
    # Acceleration force is -alpha times the gradient of the potential phasor.
    grad = scale*n*z**(n-1)*np.array([1.0, 1j*chirality], dtype=complex)
    h = scale*n*(n-1)*z**(n-2)
    hess = h*np.array([[1.0, 1j*chirality],
                       [1j*chirality, -1.0]], dtype=complex)
    return -alpha*grad, -alpha*hess


def planar_rf_force_at(position: np.ndarray, alpha: float, k: int,
                       amplitude: float) -> np.ndarray:
    """Complex force phasor at a general planar position."""
    n = abs(k)
    chirality = 1.0 if k > 0 else -1.0
    z = complex(float(position[0]), chirality*float(position[1]))
    grad = amplitude*n*z**(n-1)*np.array([1.0, 1j*chirality], dtype=complex)
    return -alpha*grad


def phase_quadrature_check(radius: float, alpha: float, k: int,
                           amplitude: float, response: dict,
                           samples: int = 65536) -> dict:
    """Compare 1st-order force with exact field on the linear-response cycle."""
    phase = 2*np.pi*np.arange(samples)/samples
    xihat = np.asarray([complex(*pair) for pair in response["micromotion_phasor"]],
                       dtype=complex)
    carrier = np.exp(-1j*phase)
    displacement = np.real(carrier[:, None]*xihat[None, :])
    position = np.array([radius, 0.0])+displacement
    phasor = np.array([planar_rf_force_at(y, alpha, k, amplitude)
                       for y in position])
    exact_mean = np.mean(np.real(phasor*carrier[:, None]), axis=0)
    first_order = np.asarray(response["mean_force_xy"], dtype=float)
    return {"samples_per_cycle": samples,
            "exact_field_mean_on_linear_response_cycle": exact_mean.tolist(),
            "first_order_mean_force": first_order.tolist(),
            "nonlinear_displacement_residual": (exact_mean-first_order).tolist(),
            "residual_norm": float(np.linalg.norm(exact_mean-first_order))}


def rotating_response(radius: float, omega: float, alpha: float,
                      lam: float, k: int, carrier: float,
                      amplitude: float,
                      include_slow_jacobian: bool = True) -> dict:
    """Compute first-order micromotion and averaged force for one RF mode."""
    sigma = carrier-k*omega
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    P = np.eye(2)
    # The static slow-force Jacobian is the planar DC acceleration Jacobian.
    A_s = alpha*PARAMS["beta"]*P if include_slow_jacobian else np.zeros((2, 2))
    D = (-sigma**2*np.eye(2)
         -1j*sigma*(lam*np.eye(2)+2.0*omega*J)
         -omega**2*P+lam*omega*J-A_s)
    fhat, Df = planar_rf_force_jet(radius, alpha, k, amplitude)
    xihat = np.linalg.solve(D, fhat)
    mean_force = 0.5*np.real(Df @ np.conjugate(xihat))
    return {
        "k": k,
        "includes_slow_jacobian": include_slow_jacobian,
        "carrier": carrier,
        "sigma": float(sigma),
        "D_condition_number": float(np.linalg.cond(D)),
        "micromotion_amplitude": float(np.linalg.norm(xihat)),
        "micromotion_over_radius": float(np.linalg.norm(xihat)/radius),
        "micromotion_phasor": [[float(v.real), float(v.imag)] for v in xihat],
        "mean_force_xy": mean_force.tolist(),
    }


def corrected_balance(state: np.ndarray, species: dict,
                      include_slow_jacobian: bool = True) -> np.ndarray:
    radius, omega = map(float, state)
    alpha, lam = species["alpha"], species["lambda"]
    if radius <= 0:
        return np.array([1e3+radius, omega])
    response2 = rotating_response(radius, omega, alpha, lam, +2, 1.0,
                                  PARAMS["A"], include_slow_jacobian)
    response4 = rotating_response(radius, omega, alpha, lam, -4,
                                  PARAMS["nu"], PARAMS["B"],
                                  include_slow_jacobian)
    average = np.array(response2["mean_force_xy"])+np.array(response4["mean_force_xy"])
    # In the rotating frame: (-omega^2 I + lambda*omega*J)y = F_DC + F_RF,avg.
    radial = -omega**2*radius-alpha*PARAMS["beta"]*radius-average[0]
    tangential = lam*omega*radius-average[1]
    return np.array([radial, tangential])


def solve_variant(species: dict, initial: np.ndarray,
                  include_slow_jacobian: bool
                  ) -> tuple[float, float, list[dict], bool, str]:
    solved = root(corrected_balance, initial,
                  args=(species, include_slow_jacobian), method="hybr",
                  options={"xtol": 1e-11})
    radius, omega = map(float, solved.x)
    alpha, lam = species["alpha"], species["lambda"]
    modes = [rotating_response(radius, omega, alpha, lam, +2, 1.0,
                               PARAMS["A"], include_slow_jacobian),
             rotating_response(radius, omega, alpha, lam, -4, PARAMS["nu"],
                               PARAMS["B"], include_slow_jacobian)]
    return radius, omega, modes, bool(solved.success), str(solved.message)


def summarize_species(name: str, species: dict, exact: dict) -> dict:
    initial = np.array([species["leading_rho"], species["leading_omega"]])
    rho, omega, modes, success, message = solve_variant(species, initial, True)
    rho_kin, omega_kin, modes_kin, kinematic_success, kinematic_message = solve_variant(
        species, initial, False)
    lead_rho, lead_omega = initial
    exact_rho = float(exact["mean_late_rho"])
    exact_omega = float(exact["mean_late_omega"])
    alpha, lam = species["alpha"], species["lambda"]
    zero_speed_modes = [
        rotating_response(lead_rho, 0.0, alpha, lam, +2, 1.0,
                          PARAMS["A"], False),
        rotating_response(lead_rho, 0.0, alpha, lam, -4, PARAMS["nu"],
                          PARAMS["B"], False),
    ]
    k1 = alpha**2*4*PARAMS["A"]**2/(1+lam**2)
    k2 = alpha**2*48*PARAMS["B"]**2/(PARAMS["nu"]**2+lam**2)
    analytic_force = np.array([
        -k1*lead_rho-k2*lead_rho**5,
        lam*(k1*lead_rho-(k2/PARAMS["nu"])*lead_rho**5),
    ])
    recovered_force = sum((np.asarray(m["mean_force_xy"]) for m in zero_speed_modes),
                          start=np.zeros(2))
    phase_checks = {
        "k2": phase_quadrature_check(rho, species["alpha"], +2,
                                     PARAMS["A"], modes[0]),
        "k_minus4": phase_quadrature_check(rho, species["alpha"], -4,
                                           PARAMS["B"], modes[1]),
    }
    return {
        "solver_success": success,
        "solver_message": message,
        "leading": {"rho": float(lead_rho), "omega": float(lead_omega)},
        "moving_orbit_corrected": {"rho": rho, "omega": omega,
                                   "balance_residual": corrected_balance(
                                       np.array([rho, omega]), species).tolist()},
        "moving_kinematic_only": {"solver_success": kinematic_success,
                                   "solver_message": kinematic_message,
                                   "rho": rho_kin, "omega": omega_kin,
                                   "balance_residual": corrected_balance(
                                       np.array([rho_kin, omega_kin]), species, False).tolist(),
                                   "mode_responses": modes_kin},
        "exact_rf_panel_15k": {"rho": exact_rho, "omega": exact_omega,
                                "retained": int(exact["retained"])},
        "errors": {
            "same_convention_as_report_percent": {
                "leading_radius": 100*(exact_rho/lead_rho-1),
                "kinematic_only_radius": 100*(exact_rho/rho_kin-1),
                "full_moving_radius": 100*(exact_rho/rho-1),
                "leading_abs_frequency": 100*(abs(exact_omega)/abs(lead_omega)-1),
                "kinematic_only_abs_frequency": 100*(abs(exact_omega)/abs(omega_kin)-1),
                "full_moving_abs_frequency": 100*(abs(exact_omega)/abs(omega)-1),
            },
            "prediction_minus_exact_over_exact_percent": {
                "leading_radius": 100*(lead_rho/exact_rho-1),
                "kinematic_only_radius": 100*(rho_kin/exact_rho-1),
                "full_moving_radius": 100*(rho/exact_rho-1),
                "leading_abs_frequency": 100*(abs(lead_omega)/abs(exact_omega)-1),
                "kinematic_only_abs_frequency": 100*(abs(omega_kin)/abs(exact_omega)-1),
                "full_moving_abs_frequency": 100*(abs(omega)/abs(exact_omega)-1),
            },
            "leading_frequency_absolute": float(lead_omega-exact_omega),
            "corrected_frequency_absolute": float(omega-exact_omega),
        },
        "mode_responses": modes,
        "zero_orbit_speed_recovery_of_leading_formula": {
            "analytic_mean_force_xy": analytic_force.tolist(),
            "complex_response_mean_force_xy": recovered_force.tolist(),
            "absolute_vector_error": float(np.linalg.norm(analytic_force-recovered_force)),
        },
        "phase_quadrature_on_linear_response_cycle": phase_checks,
    }


def main() -> None:
    panel = json.loads((ROOT/"results/local15_exact_rf/independent_audit.json")
                       .read_text(encoding="utf-8"))
    output = {
        "status": "computed reduced moving-orbit response; compared to frozen exact-RF panel",
        "method": "co-rotating 2x2 susceptibility including Coriolis, centrifugal, drag, and DC Jacobian; mean RF force 1/2 Re[D f xi*]",
        "parameters": PARAMS,
        "panel_sha256": panel["panel_sha256"],
        "horizon_primary_periods": panel["protocol"]["periods"],
        "warning": "The corrected equilibria are asymptotic reduced-model predictions. Existing RF data are a fixed panel, not a probability estimate; this comparison does not establish full RF Floquet stability or validate omitted harmonics.",
        "species": {
            key: summarize_species(key, PARAMS[key], panel["species"][key])
            for key in ("species1", "species2")
        },
    }
    out = ROOT/"results/moving_orbit_correction_local15.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False)+"\n",
                   encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
