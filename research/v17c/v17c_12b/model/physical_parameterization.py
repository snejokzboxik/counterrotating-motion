"""Map V17C dimensionless parameters to dimensioned particle/field quantities.

This script is deliberately an analytical screen. It does not label the
candidate as breakdown-safe; detailed field maxima are computed separately.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23
T_AIR = 300.0
ETA_AIR = 1.81e-5
RHO_AIR = 1.177
LAMBDA_AIR = 65.1e-9
RHO_ALUMINA = 3990.0
H2 = 0.870316891091623
H4 = 0.7982172624273378
MFS_RESIDUAL_P32 = {"rf1": 1.5064367853959374e-7, "rf2": 1.486684510277422e-7}

CANDIDATES = {
    "candidate52": {
        "p": [0.03956095746691348, 0.15032976498886072, 2.8996277751401065,
              0.015917322789905167, 0.11307613545330587, 9.281679945656495e-6,
              0.1868668374144136, 0.10359181855828002],
        "qratio": 3.5274259874597194,
    },
    "local15": {
        "p": [0.03716194907442204, 0.1626938199343468, 3.1118029452860356,
              0.017273303124515367, 0.10944955727644175, 9.463121589050818e-6,
              0.20975541409005094, 0.12617311138613094],
        "qratio": 3.1179673274978996,
    },
    "refine12": {
        "p": [0.036536559669839035, 0.17246401028140085, 3.2239347513765098,
              0.014742040731828608, 0.12340655648341282, 1.2076805896660412e-5,
              0.20865112927753002, 0.13669668777275765],
        "qratio": 2.852663503959775,
    },
    "refine12_sobol17": {
        "p": [0.03492413282400757, 0.15631667332941065, 3.2196773638017477,
              0.015388559140477825, 0.10963763906505944, 9.747265456858355e-6,
              0.24575362874758605, 0.17750147818833362],
        "qratio": 3.4490536225028334,
    },
}


def candidate_compatibility() -> list[dict[str, float | str]]:
    # V17C.10 used the same finite radii in all strict candidate tests.
    a1_over_R, a2_over_R = 0.0002, 0.00189
    radius_ratio = a2_over_R / a1_over_R
    rows = []
    for name, d in CANDIDATES.items():
        p = d["p"]
        s, lam1, lam2 = p[4], p[6], p[7]
        same_fluid_lambda_ratio = radius_ratio * s
        same_density_mass_ratio = radius_ratio ** -3
        density_ratio_needed = s * radius_ratio**3
        radius_ratio_for_stokes = lam2 / (s * lam1)
        density_ratio_at_stokes_radius = s * radius_ratio_for_stokes**3
        rows.append({
            "candidate": name,
            "radius_ratio_a2_a1_from_V17C10": radius_ratio,
            "mass_ratio_m1_m2": s,
            "mass_ratio_if_same_density_spheres": same_density_mass_ratio,
            "required_density_ratio_rho1_rho2_at_fixed_radii": density_ratio_needed,
            "lambda2_lambda1_model": lam2 / lam1,
            "lambda2_lambda1_Stokes_same_fluid_fixed_radii": same_fluid_lambda_ratio,
            "model_over_Stokes_drag_ratio": (lam2 / lam1) / same_fluid_lambda_ratio,
            "radius_ratio_for_Stokes_same_fluid_fixed_masses": radius_ratio_for_stokes,
            "density_ratio_at_that_radius_ratio": density_ratio_at_stokes_radius,
            "qratio": d["qratio"],
        })
    return rows


def physical_seed() -> dict:
    # Preserve the V17C.10 leader's RF/DC/coupling shape but impose exact
    # particle identity: same material, radius, mass, charge, and Stokes drag.
    src = CANDIDATES["refine12_sobol17"]["p"]
    A, B, nu, beta, _, C, lam, _ = src
    radius = 17e-6  # 34 µm diameter; measured alumina charging data cover this scale.
    R = 0.010
    m = 4.0 * math.pi * RHO_ALUMINA * radius**3 / 3.0
    omega1 = 4.5 * ETA_AIR / (RHO_ALUMINA * radius**2 * lam)
    f1 = omega1 / (2.0 * math.pi)
    q = math.sqrt(C * 4.0 * math.pi * EPS0 * m * R**3 * omega1**2)
    voltage_scale = m * R**2 * omega1**2 / q
    u1 = A * voltage_scale / H2
    u2 = B * voltage_scale / H4
    v2 = beta * voltage_scale
    dimensionless_gravity = 9.80665 / (R * omega1**2)
    gravity_z_equilibrium = -dimensionless_gravity / (2.0 * beta)
    D = R / 0.95
    rod_radius = 0.05 * D
    return {
        "label": "alumina_identical_symmetric_ref17_shape",
        "status": "dimensionally_consistent_screen_not_a_validated_orbit_or_breakdown_pass",
        "source_candidate": "refine12_sobol17 with s=1, q2/q1=1, lambda2=lambda1",
        "p": [A, B, nu, beta, 1.0, C, lam, lam],
        "qratio": 1.0,
        "particle": {
            "material": "Al2O3",
            "shape_assumption": "solid sphere",
            "radius_m": radius,
            "diameter_m": 2.0 * radius,
            "density_kg_m3": RHO_ALUMINA,
            "mass_kg": m,
            "charge_C": q,
            "charge_e": q / E_CHARGE,
            "specific_charge_C_kg": q / m,
            "surface_field_V_m_isolated_particle": q / (4.0 * math.pi * EPS0 * radius**2),
        },
        "medium": {
            "gas": "dry air, approximate 300 K, 1 atm",
            "dynamic_viscosity_Pa_s": ETA_AIR,
            "density_kg_m3": RHO_AIR,
            "mean_free_path_m": LAMBDA_AIR,
            "temperature_K": T_AIR,
        },
        "trap": {
            "free_aperture_radius_R_m": R,
            "electrode_center_radius_D_m": D,
            "rod_radius_m": rod_radius,
            "electrode_count": 24,
            "geometry_a_over_D": 0.05,
            "geometry_shield_radius_over_D": 1.10,
            "nearest_rod_clear_gap_m": (2.0 * math.sin(math.pi / 24.0) - 0.10) * D,
            "rod_to_shield_clear_gap_m": 0.05 * D,
            "f1_Hz": f1,
            "f2_Hz": nu * f1,
            "omega1_rad_s": omega1,
            "omega2_rad_s": nu * omega1,
        },
        "electrode_drive": {
            "U1_peak_V_electrode_not_RMS_or_peak_to_peak": u1,
            "U2_peak_V_electrode_not_RMS_or_peak_to_peak": u2,
            "U2_over_U1": u2 / u1,
            "leading_transfer_h2": H2,
            "leading_transfer_h_minus4": H4,
            "leading_RF2_coefficient_V_at_R": B * voltage_scale,
            "leading_RF1_coefficient_V_at_R": A * voltage_scale,
            "ideal_DC_quadrupole_coefficient_V_not_an_endcap_voltage": v2,
            "MFS_boundary_residual_p32": MFS_RESIDUAL_P32,
        },
        "dimensionless_checks": {
            "lambda_reconstructed": 6.0 * math.pi * ETA_AIR * radius / (m * omega1),
            "C_reconstructed": q**2 / (4.0 * math.pi * EPS0 * m * R**3 * omega1**2),
            "gravity_acceleration_over_Romega1_squared": dimensionless_gravity,
            "gravity_shift_z_over_R_in_ideal_DC_quadrupole": gravity_z_equilibrium,
            "Kn_mean_free_path_over_radius": LAMBDA_AIR / radius,
            "Cunningham_slip_factor_approx": 1.0 + (LAMBDA_AIR / radius) * (1.257 + 0.4 * math.exp(-1.1 * radius / LAMBDA_AIR)),
            "Brownian_diffusivity_m2_s": K_B * T_AIR / (6.0 * math.pi * ETA_AIR * radius),
            "Re_per_dimensionless_speed": 2.0 * RHO_AIR * radius * R * omega1 / ETA_AIR,
            "unsteady_Stokes_parameter_Omega_a_squared_over_nu_air": omega1 * radius**2 / (ETA_AIR / RHO_AIR),
        },
    }


def main() -> None:
    rows = candidate_compatibility()
    seed = physical_seed()
    (OUT / "candidate_drag_compatibility.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (OUT / "physical_seed.json").write_text(json.dumps(seed, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_drag_compatibility": rows, "physical_seed": seed}, indent=2))


if __name__ == "__main__":
    main()
