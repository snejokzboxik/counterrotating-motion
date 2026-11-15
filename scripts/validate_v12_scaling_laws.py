"""Lightweight trajectory and phase-volume check of V12 scaling laws.

The two physical parameter sets below share the same dimensionless multi-mode
field, Lambda, normalized aperture, initial state, and RF phase.  The script
integrates each set once and checks that the normalized trajectories coincide.
It also checks the conversion A4 = R**4 * Omega**2 * a4 using the exact volume
of a small shared dimensionless test domain.  This is a scaling validation,
not an acceptance estimate or an independent FEM validation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap.dimensionless_dynamics_v12 import DimensionlessRFModel  # noqa: E402

OUT = ROOT / "outputs/octupole_comparison_home/dimensionless_acceptance_v12/validation"
E_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_KG = 1.66053906660e-27
HORIZON_RF = 10
PHASE_RAD = 0.37
RTOL = 1e-11
ATOL = 1e-13
MAX_STEP_RF = 1 / 128
TRACE_SAMPLES = 1001


def build_common_field() -> tuple[np.ndarray, np.ndarray]:
    """Return one identical dimensionless FEM-like multi-harmonic profile."""
    cosine = np.zeros(9, dtype=np.complex128)
    sine = np.zeros(9, dtype=np.complex128)
    # Leading rotating m=4 term plus a smaller, oppositely handed m=8 term.
    # Coefficients are per applied volt and are shared by similar geometries.
    cosine[4] = 0.5413596632303368
    sine[4] = -1j * cosine[4]
    cosine[8] = 0.043
    sine[8] = 1j * cosine[8]
    return cosine, sine


def integrate_trace(model: DimensionlessRFModel, omega_rad_s: float, horizon_rf: int) -> dict:
    """Integrate one normalized trajectory and return its trace and outcome."""
    state0 = np.array([0.2, 0.1, -0.002, 0.003], dtype=float)
    duration_tau = 2.0 * math.pi * horizon_rf

    def first_exit(tau: float, state: np.ndarray) -> float:
        return float(np.linalg.norm(state[:2]) - model.aperture_rho)

    first_exit.terminal = True
    first_exit.direction = 1

    tau_eval = np.linspace(0.0, duration_tau, TRACE_SAMPLES)
    solution = solve_ivp(
        model.rhs,
        (0.0, duration_tau),
        state0,
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
        max_step=2.0 * math.pi * MAX_STEP_RF,
        t_eval=tau_eval,
        events=first_exit,
    )
    if not solution.success:
        raise RuntimeError(f"DOP853 failed: {solution.message}")
    exit_tau = float(solution.t_events[0][0]) if len(solution.t_events[0]) else None
    status = "escaped" if exit_tau is not None else "retained"
    reached_tau = exit_tau if exit_tau is not None else duration_tau
    trace = solution.y.T
    # solve_ivp with t_eval can omit a final event state; append the localized
    # first-exit point so the two event times and terminal states are comparable.
    if exit_tau is not None and (not len(solution.t) or solution.t[-1] < exit_tau):
        event_state = solution.y_events[0][0]
        trace = np.vstack((trace, event_state))
    sample_tau = solution.t.copy()
    if exit_tau is not None and (not len(sample_tau) or sample_tau[-1] < exit_tau):
        sample_tau = np.r_[sample_tau, exit_tau]
    return {
        "status": status,
        "escape_RF": None if exit_tau is None else exit_tau / (2.0 * math.pi),
        "reached_RF": reached_tau / (2.0 * math.pi),
        "state_trace": trace,
        "tau_trace": sample_tau,
        "initial_state": state0,
        "nfev": int(solution.nfev),
        "max_radius_sampled_rho": float(np.linalg.norm(trace[:, :2], axis=1).max()),
        "physical_time_s": reached_tau / omega_rad_s,
    }


def main() -> None:
    # Reference system: same physical scale as the V12 baseline.
    m1 = 100_000 * ATOMIC_MASS_KG
    q1 = E_CHARGE_C
    omega1 = 2.0 * math.pi * 100_000.0
    radius1 = 8.0e-3
    voltage1 = 250.0

    # A distinct mass, RF frequency, and size. Choose voltage from equal Lambda.
    m2 = 2.0 * m1
    q2 = q1
    omega2 = 1.5 * omega1
    radius2 = 1.25 * radius1
    voltage2 = voltage1 * (m2 / m1) * (omega2 / omega1) ** 2 * (radius2 / radius1) ** 2
    aperture1, aperture2 = radius1, radius2
    cosine, sine = build_common_field()

    models = {
        "set_1": DimensionlessRFModel.from_physical(
            cosine, sine, charge_c=q1, mass_kg=m1, voltage_v=voltage1,
            omega_rad_s=omega1, length_m=radius1, reference_radius_m=radius1,
            aperture_m=aperture1, phase_rad=PHASE_RAD, name="scale_check_1",
        ),
        "set_2": DimensionlessRFModel.from_physical(
            cosine, sine, charge_c=q2, mass_kg=m2, voltage_v=voltage2,
            omega_rad_s=omega2, length_m=radius2, reference_radius_m=radius2,
            aperture_m=aperture2, phase_rad=PHASE_RAD, name="scale_check_2",
        ),
    }
    results = {
        "set_1": integrate_trace(models["set_1"], omega1, HORIZON_RF),
        "set_2": integrate_trace(models["set_2"], omega2, HORIZON_RF),
    }
    trace1, trace2 = results["set_1"]["state_trace"], results["set_2"]["state_trace"]
    tau1, tau2 = results["set_1"]["tau_trace"], results["set_2"]["tau_trace"]
    n_common = min(len(trace1), len(trace2))
    if len(trace1) != len(trace2) or not np.array_equal(tau1, tau2):
        raise RuntimeError("Normalized trace grids differ between equivalent systems")
    delta = trace1[:n_common] - trace2[:n_common]
    max_abs_state_difference = float(np.max(np.abs(delta)))
    max_position_difference = float(np.max(np.linalg.norm(delta[:, :2], axis=1)))

    lambda1 = models["set_1"].coupling
    lambda2 = models["set_2"].coupling
    field_difference = max(float(np.max(np.abs(models["set_1"].cosine - models["set_2"].cosine))),
                           float(np.max(np.abs(models["set_1"].sine - models["set_2"].sine))))
    factor1 = radius1**4 * omega1**2
    factor2 = radius2**4 * omega2**2
    expected_factor_ratio = (radius2 / radius1) ** 4 * (omega2 / omega1) ** 2

    # Exact volume of one common normalized test domain, not an acceptance result.
    rho_injection = 0.75
    u_max = 0.02
    a4_domain = math.pi**2 * rho_injection**2 * u_max**2
    domain_volume1 = factor1 * a4_domain
    domain_volume2 = factor2 * a4_domain
    measured_conversion_ratio = domain_volume2 / domain_volume1
    results_match = (
        math.isclose(lambda1, lambda2, rel_tol=2e-15, abs_tol=0.0)
        and field_difference == 0.0
        and results["set_1"]["status"] == results["set_2"]["status"]
        and results["set_1"]["escape_RF"] == results["set_2"]["escape_RF"]
        and max_abs_state_difference <= 5e-12
        and math.isclose(measured_conversion_ratio, expected_factor_ratio, rel_tol=2e-15)
    )

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "study": "V12 dimensionless scaling-law trajectory and phase-volume check",
        "classification": "lightweight verification only; not an acceptance estimate or independent FEM validation",
        "horizon_RF": HORIZON_RF,
        "phase_rad": PHASE_RAD,
        "solver": {"method": "DOP853", "rtol": RTOL, "atol": ATOL,
                   "max_step_RF_fraction": MAX_STEP_RF, "trace_samples": TRACE_SAMPLES},
        "normalized_field": {
            "description": "same two-quadrature complex polynomial for both geometrically similar systems; m=4 and m=8 terms",
            "cosine_coefficients_real_imag": [[float(z.real), float(z.imag)] for z in cosine],
            "sine_coefficients_real_imag": [[float(z.real), float(z.imag)] for z in sine],
            "max_coefficient_difference": field_difference,
        },
        "normalized_initial_state": results["set_1"]["initial_state"].tolist(),
        "normalized_aperture": {name: float(model.aperture_rho) for name, model in models.items()},
        "systems": {
            "set_1": {"mass_kg": m1, "charge_C": q1, "voltage_V": voltage1,
                      "omega_rad_s": omega1, "frequency_Hz": omega1 / (2*math.pi),
                      "length_m": radius1, "aperture_m": aperture1,
                      "lambda": lambda1, "physical_time_at_horizon_s": results["set_1"]["physical_time_s"],
                      "status": results["set_1"]["status"], "escape_RF": results["set_1"]["escape_RF"],
                      "reached_RF": results["set_1"]["reached_RF"], "nfev": results["set_1"]["nfev"],
                      "max_radius_sampled_rho": results["set_1"]["max_radius_sampled_rho"]},
            "set_2": {"mass_kg": m2, "charge_C": q2, "voltage_V": voltage2,
                      "omega_rad_s": omega2, "frequency_Hz": omega2 / (2*math.pi),
                      "length_m": radius2, "aperture_m": aperture2,
                      "lambda": lambda2, "physical_time_at_horizon_s": results["set_2"]["physical_time_s"],
                      "status": results["set_2"]["status"], "escape_RF": results["set_2"]["escape_RF"],
                      "reached_RF": results["set_2"]["reached_RF"], "nfev": results["set_2"]["nfev"],
                      "max_radius_sampled_rho": results["set_2"]["max_radius_sampled_rho"]},
        },
        "trajectory_comparison": {
            "trace_rows": int(n_common),
            "max_abs_state_difference": max_abs_state_difference,
            "max_position_difference_rho": max_position_difference,
            "status_match": results["set_1"]["status"] == results["set_2"]["status"],
            "escape_RF_match": results["set_1"]["escape_RF"] == results["set_2"]["escape_RF"],
        },
        "phase_volume_conversion": {
            "relation": "A4 = R^4 * Omega^2 * a4",
            "a4_probe": a4_domain,
            "normalized_test_domain": {"rho_injection": rho_injection, "u_max": u_max,
                                       "measure": "uniform position disk × uniform velocity disk"},
            "R4_Omega2_set_1_m4_s2": factor1,
            "R4_Omega2_set_2_m4_s2": factor2,
            "physical_domain_volume_set_1_m4_s2": domain_volume1,
            "physical_domain_volume_set_2_m4_s2": domain_volume2,
            "expected_set2_over_set1": expected_factor_ratio,
            "measured_set2_over_set1": measured_conversion_ratio,
            "note": "The test-domain volume is exact; a4_probe is not a measured retained fraction.",
        },
        "pass": results_match,
        "trace_csv": "scaling_law_trajectory_v12.csv",
        "summary_csv": "scaling_law_summary_v12.csv",
    }
    json_path = OUT / "scaling_law_v12.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    trace_path = OUT / "scaling_law_trajectory_v12.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "period_RF", "tau", "rho_x_set1", "rho_y_set1", "u_x_set1", "u_y_set1",
            "rho_x_set2", "rho_y_set2", "u_x_set2", "u_y_set2",
            "delta_rho_x", "delta_rho_y", "delta_u_x", "delta_u_y",
        ])
        writer.writeheader()
        for tau, a, b in zip(tau1, trace1, trace2):
            writer.writerow({
                "period_RF": tau / (2.0 * math.pi), "tau": tau,
                "rho_x_set1": a[0], "rho_y_set1": a[1], "u_x_set1": a[2], "u_y_set1": a[3],
                "rho_x_set2": b[0], "rho_y_set2": b[1], "u_x_set2": b[2], "u_y_set2": b[3],
                "delta_rho_x": a[0]-b[0], "delta_rho_y": a[1]-b[1],
                "delta_u_x": a[2]-b[2], "delta_u_y": a[3]-b[3],
            })

    summary_path = OUT / "scaling_law_summary_v12.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["set", "mass_kg", "charge_C", "voltage_V", "frequency_Hz", "omega_rad_s",
                  "length_m", "aperture_m", "lambda", "status", "escape_RF", "reached_RF",
                  "physical_time_at_horizon_s", "nfev", "max_radius_sampled_rho",
                  "R4_Omega2_m4_s2", "normalized_test_domain_volume_a4", "physical_domain_volume_m4_s2"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, values in summary["systems"].items():
            writer.writerow({"set": name, **values,
                             "R4_Omega2_m4_s2": factor1 if name == "set_1" else factor2,
                             "normalized_test_domain_volume_a4": a4_domain,
                             "physical_domain_volume_m4_s2": domain_volume1 if name == "set_1" else domain_volume2})

    print(json.dumps({"pass": results_match, "json": str(json_path),
                      "trajectory_csv": str(trace_path), "summary_csv": str(summary_path),
                      "max_abs_state_difference": max_abs_state_difference,
                      "status": [results[name]["status"] for name in ("set_1", "set_2")],
                      "phase_volume_ratio": measured_conversion_ratio}, indent=2))
    if not results_match:
        raise SystemExit("Scaling-law check failed; see the validation JSON for diagnostics")


if __name__ == "__main__":
    main()
