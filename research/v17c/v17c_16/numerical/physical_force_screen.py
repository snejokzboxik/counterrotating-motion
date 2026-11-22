"""Order-of-magnitude screen for Brownian, dielectrophoretic and wall effects."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from physical_screen import (EPS0, K_B, PAIR, build_screen)
from mfs_two_particle_dynamics import MFSRFField
from two_particle_oseen import local15_trial_model


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = build_screen()
    mfs_run = json.loads((root/"results"/"trajectory_validation_mfs1000_s16"/
                          "oseen_coulomb_nominal_start01.json").read_text(
                              encoding="utf-8"))
    model = local15_trial_model(coulomb_scale=1.0, gravity=False)
    field = MFSRFField(model, order_per_rod=24)
    Escale = field.vscale/field.R
    eps_r = 2.6
    contrast = (eps_r-1.0)/(eps_r+2.0)
    t_horizon = 1000.0/PAIR["f1_Hz"]
    rows = []
    for i, particle in enumerate(base["particles"]):
        a = particle["radius_m"]
        rho = mfs_run["mean_radius_dimensionless_last_80pct"][i]
        pos = np.array([rho, 0.0, 0.0])
        dipole_alpha = 4.0*math.pi*EPS0*a**3*contrast

        def mean_E2(at):
            g1, g2 = field.rf_gradient_phasors_dimensionless(at)
            return (0.5*Escale**2*(float(np.vdot(g1[:2], g1[:2]).real)
                                   +float(np.vdot(g2[:2], g2[:2]).real)))

        h = 1.0e-4
        grad_E2_dimless = np.array([
            (mean_E2(pos+np.eye(3)[j]*h)-mean_E2(pos-np.eye(3)[j]*h))/(2*h)
            for j in range(3)])
        grad_E2_SI = grad_E2_dimless/PAIR["radial_scale_R_m"]
        F_dep = 0.5*dipole_alpha*grad_E2_SI
        F_dep_mag = float(np.linalg.norm(F_dep))
        weight = particle["mass_kg"]*9.80665
        coulomb_R = base["gravity_and_coulomb"]["Coulomb_force_at_separation_R_N"]
        D = K_B*PAIR["temperature_K"]/(6.0*math.pi*
            PAIR["air_dynamic_viscosity_Pa_s"]*a)
        brownian_rms_displacement = math.sqrt(2.0*D*t_horizon)
        v_th_rms = math.sqrt(3.0*K_B*PAIR["temperature_K"]/particle["mass_kg"])
        orbit_speed = abs(mfs_run["mean_angular_velocity_rad_s_last_80pct"][i])
        gap_R = mfs_run["minimum_rod_clearance_over_R_by_particle"][i]
        gap_m = gap_R*PAIR["radial_scale_R_m"]
        center_wall = a+gap_m
        planar_parallel_drag_increment = (9.0/16.0)*a/center_wall
        rod_radius = 0.05/(0.95)*PAIR["radial_scale_R_m"]
        rows.append({
            "particle": i+1,
            "dielectric_assumption": "epsilon_r=2.6, real Clausius-Mossotti contrast, zero loss; estimate only",
            "polarizability_C_m2_V": dipole_alpha,
            "mean_RF_E2_at_mean_orbit_V2_m2": mean_E2(pos),
            "mean_RF_field_rms_V_m": math.sqrt(mean_E2(pos)),
            "dielectrophoretic_force_estimate_N": F_dep.tolist(),
            "dielectrophoretic_force_magnitude_N": F_dep_mag,
            "DEP_to_weight_ratio": F_dep_mag/weight,
            "DEP_to_coulomb_force_at_R_ratio": F_dep_mag/coulomb_R,
            "thermal_diffusivity_m2_s": D,
            "Brownian_rms_displacement_over_1000_primary_periods_m": brownian_rms_displacement,
            "Brownian_rms_displacement_over_R": brownian_rms_displacement/
                PAIR["radial_scale_R_m"],
            "thermal_rms_speed_m_s": v_th_rms,
            "orbit_mean_speed_m_s": orbit_speed,
            "thermal_to_orbit_speed_ratio": v_th_rms/orbit_speed,
            "minimum_rod_surface_gap_m": gap_m,
            "minimum_rod_gap_over_particle_radius": gap_m/a,
            "leading_planar_wall_parallel_drag_increment_estimate": planar_parallel_drag_increment,
            "particle_radius_over_rod_radius": a/rod_radius,
            "wall_model_status": "not applied to Oseen RHS: flat-wall Stokes asymptotic, curved rod, RF-unsteady and finite-Re corrections are not controlled here",
        })
    out = root/"results"/"secondary_force_screen.json"
    payload = {
        "status": "order-of-magnitude controls at the full-MFS mean orbit; not a coupled force simulation",
        "method": "RF-torus mean square field from independent complex MFS phasors; central finite differences in dimensionless coordinates; real-dielectric dipole estimate",
        "horizon_s": t_horizon,
        "particles": rows,
        "limits": [
            "Polarizability uses a real low-frequency bulk-polystyrene epsilon and neglects dielectric loss/charge redistribution.",
            "The DEP estimate samples the mean orbit at one azimuth; its magnitude is an order estimate, not a resolved orbit-averaged force.",
            "Brownian values are free-air diffusion screens without electric-field response or wall modification.",
            "The planar-wall correction is only a leading Stokes estimate; rod curvature a/Rrod is not asymptotically negligible and the wall term is not added to the Oseen integrator.",
        ],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n",
                   encoding="utf-8")
    print(out)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
