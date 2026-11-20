"""Dimensionally map legacy V17C.10 candidates to one-air/sphere case."""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.optimize import brentq

from physical_parameterization import CANDIDATES, H2, H4

HERE = Path(__file__).resolve().parent
OUT = HERE.parent/"results"/"legacy_candidate_physical_map_10mm.json"

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
ETA = 1.81e-5
RHO_GAS = 1.177
MFP = 65.1e-9
RHO1 = 3990.0
T = 300.0
KB = 1.380649e-23
R = 0.010
A2 = 50e-6
PRESSURE_TORR = 760.0


def cunningham(a: float) -> float:
    kn = MFP/a
    return 1.0+kn*(1.257+0.4*math.exp(-1.1/kn))


def paschen(pd_torr_cm: float, A: float = 15.0, B: float = 365.0,
            gamma: float = 0.01) -> float:
    return B*pd_torr_cm/(math.log(A*pd_torr_cm)-math.log(math.log(1+1/gamma)))


def main() -> None:
    rows = []
    for name, cand in CANDIDATES.items():
        p = cand["p"]
        A, B, nu, beta, s, C, lam1, lam2 = p
        qratio = cand["qratio"]
        target_ratio = lam2/lam1
        # Match the code-defined Stokes lambda ratio in one gas. Cunningham
        # is applied once by the drag force after this mapping.
        def residual(ratio: float) -> float:
            return ratio*s-target_ratio
        radius_ratio = brentq(residual, 0.1, 20.0, xtol=1e-14)
        a1 = A2/radius_ratio
        rho2 = RHO1/(s*radius_ratio**3)
        m1 = 4*math.pi*RHO1*a1**3/3
        m2 = m1/s
        cc1, cc2 = cunningham(a1), cunningham(A2)
        omega1 = 4.5*ETA/(RHO1*a1*a1*lam1)
        q1 = math.sqrt(C*4*math.pi*EPS0*m1*R**3*omega1**2/qratio)
        q2 = qratio*q1
        voltage_scale = m1*R**2*omega1**2/q1
        u1, u2, v2 = A*voltage_scale/H2, B*voltage_scale/H4, beta*voltage_scale
        D = R/0.95
        gap_screen = 0.05*D
        gap_rods = (2*math.sin(math.pi/24)-0.1)*D
        cp = PRESSURE_TORR*100*gap_screen
        pp = PRESSURE_TORR*100*gap_rods
        # If f_i = 0.03 Re is used as a small-Re Oseen estimate, this reports
        # the coefficient at one dimensionless speed; actual Re scales with it.
        re1 = 2*RHO_GAS*a1*R*omega1/ETA
        re2 = 2*RHO_GAS*A2*R*omega1/ETA
        rows.append({"candidate": name, "source_dimensionless": {
                "A": A, "B": B, "nu": nu, "beta": beta, "m1_over_m2": s,
                "C": C, "lambda1_stokes_target": lam1,
                "lambda2_stokes_target": lam2, "q2_over_q1": qratio},
            "assumptions": {"particle1": "solid spherical Al2O3", "rho1_kg_m3": RHO1,
                "particle2": "spherical low-density glass/hollow particle; density is solved, not an identified grade",
                "radius2_m": A2, "medium": "dry air, 300 K, 1 atm",
                "eta_Pa_s": ETA, "mean_free_path_m": MFP,
                "trap_aperture_radius_R_m": R},
            "dimensioned": {"radius1_m": a1, "radius2_m": A2,
                "diameter1_um": 2*a1*1e6, "diameter2_um": 2*A2*1e6,
                "rho2_kg_m3": rho2, "mass1_kg": m1, "mass2_kg": m2,
                "mass_ratio_reconstructed": m1/m2,
                "lambda1_reconstructed": 4.5*ETA/(RHO1*a1*a1*omega1),
                "lambda2_reconstructed": 4.5*ETA/(rho2*A2*A2*omega1),
                "lambda1_effective_with_Cunningham": lam1*cc1,
                "lambda2_effective_with_Cunningham": lam2*cc2,
                "radius_ratio_a2_a1": radius_ratio,
                "f1_Hz": omega1/(2*math.pi), "f2_Hz": nu*omega1/(2*math.pi),
                "q1_C": q1, "q2_C": q2, "q1_e": q1/E_CHARGE, "q2_e": q2/E_CHARGE,
                "E1_isolated_surface_V_m": q1/(4*math.pi*EPS0*a1*a1),
                "E2_isolated_surface_V_m": q2/(4*math.pi*EPS0*A2*A2),
                "U1_peak_electrode_V": u1, "U2_peak_electrode_V": u2,
                "V2_ideal_DC_coefficient_V_not_endcap": v2,
                "U1_plus_U2_peak_bound_V": u1+u2,
                "leading_RF_amplitudes_at_R_V": {"RF1": A*voltage_scale, "RF2": B*voltage_scale},
                "Kn1": MFP/a1, "Kn2": MFP/A2, "Cunningham1": cc1, "Cunningham2": cc2,
                "Re_per_dimensionless_speed1": re1, "Re_per_dimensionless_speed2": re2,
                "Omega_a2_over_nu1": omega1*a1*a1/(ETA/RHO_GAS),
                "Omega_a2_over_nu2": omega1*A2*A2/(ETA/RHO_GAS),
                "Brownian_D1_m2_s": KB*T/(6*math.pi*ETA*a1*cc1),
                "z_gravity_equilibrium_over_R_if_axis_vertical": -9.80665/(2*beta*R*omega1**2),
                "trap_geometry": {"rod_radius_m": 0.05*D,
                    "rod_to_shield_gap_m": gap_screen, "adjacent_rod_gap_m": gap_rods,
                    "pd_rod_shield_Torr_cm": cp, "pd_adjacent_rods_Torr_cm": pp,
                    "flat_air_Paschen_rod_shield_V": paschen(cp),
                    "flat_air_Paschen_adjacent_rods_V": paschen(pp),
                    "RF_bound_rod_shield_V": u1+u2,
                    "RF_bound_adjacent_rods_V": 2*u1*math.sin(math.pi/24)+u2}}})
    out = {"study": "candidate mapping with same-air Stokes lambda; Cunningham applied once in the physical drag",
        "rows": rows,
        "limits": [
            "The solved rho2 is a target density, not proof of a commercially available particle with the required radius, charge, and surface state.",
            "Paschen values use the plane-parallel Townsend formula and are screening estimates only; no corona/RF breakdown pass is claimed.",
            "Isolated particle surface fields omit the applied trap field and neighboring particles.",
            "Finite rod ends, feeds, surface roughness, and real DC electrodes are absent."]}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
