"""Fast-torus Coulomb-force screen on prescribed slow Local15 orbit branches.

The phase difference between the two slow orbits is retained as a coordinate;
the calculation does not impose a fixed relative phase or feed the averaged
force back into a reduced trajectory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from physical_screen import G
from two_particle_oseen import local15_trial_model
from averaged_theory import two_mode_response


ROOT = Path(__file__).resolve().parents[1]


def _rot(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def force_vs_relative_phase(samples_rf: int = 96,
                            samples_slow: int = 64) -> dict:
    model = local15_trial_model(coulomb_scale=1.0, gravity=True)
    branch_file = ROOT/"results"/"analytic_screen.json"
    data = json.loads(branch_file.read_text(encoding="utf-8"))
    b1, b2 = data["particle_1"]["oseen"], data["particle_2"]["oseen"]
    branch = [b1, b2]
    rho = np.array([b1["rho"], b2["rho"]])
    omega = np.array([b1["omega"], b2["omega"]])
    z = []
    ghat = G/(model.radial_scale_m*model.omega1_rad_s**2)
    for i in range(2):
        alpha = model.charges_relative[i]/model.masses_relative[i]
        buoy = 1.0-model.air_density_kg_m3/model.densities_kg_m3[i]
        z.append(-ghat*buoy/(2.0*alpha*model.beta2))
    angles = np.linspace(0.0, 2.0*math.pi, samples_slow, endpoint=False)
    phases = 2.0*math.pi*np.arange(samples_rf)/samples_rf
    p1, p2 = np.meshgrid(phases, phases, indexing="ij")
    phase_mesh = (p1, p2)
    responses = []
    for i in range(2):
        alpha = model.charges_relative[i]/model.masses_relative[i]
        responses.append(two_mode_response(rho[i], omega[i], alpha,
                                           model.lambdas_stokes[i],
                                           model.rf_modes, model.beta2))
    torque_rows = []
    force_rows = []
    zforce_rows = []
    min_gap = []
    for delta in angles:
        theta = (0.0, float(delta))
        pos = []
        for i in range(2):
            xi = np.zeros((samples_rf, samples_rf, 2), dtype=float)
            for s, mode in enumerate(model.rf_modes):
                ph = phase_mesh[s]
                carrier = np.exp(-1j*(ph-mode.k*theta[i]))
                xi += np.real(carrier[..., None]
                              * responses[i][s]["xi"][None, None, :])
            local = np.empty_like(xi)
            local[..., 0] = rho[i]+xi[..., 0]
            local[..., 1] = xi[..., 1]
            xy = local @ _rot(theta[i]).T
            pos.append(np.column_stack((xy.reshape(-1, 2),
                                        np.full(samples_rf*samples_rf, z[i]))))
        d = pos[0]-pos[1]
        d2 = np.sum(d*d, axis=1)
        coefficient = (model.kappa_base*model.charges_relative[0]
                       * model.charges_relative[1])
        f0 = coefficient*d/(d2[:, None]*np.sqrt(d2[:, None]))
        f0bar = np.mean(f0, axis=0)
        f1bar = -f0bar
        e_theta0 = np.array([0.0, 1.0, 0.0])
        e_theta1 = np.array([-math.sin(theta[1]), math.cos(theta[1]), 0.0])
        torque0 = rho[0]*float(f0bar[:2] @ e_theta0[:2])
        torque1 = rho[1]*float(f1bar[:2] @ e_theta1[:2])
        torque_rows.append([torque0, torque1])
        force_rows.append([f0bar.tolist(), f1bar.tolist()])
        zforce_rows.append([float(f0bar[2]), float(f1bar[2])])
        min_gap.append(float(np.min(np.sqrt(d2))
                             - sum(model.radii_m)/model.radial_scale_m))
    return {
        "status": "leading fast-torus Coulomb average on prescribed single-particle Oseen branches; not a self-consistent pair solution",
        "slow_branch": [{"rho": float(rho[i]), "omega": float(omega[i]),
                         "z_over_R_gravity_equilibrium": float(z[i])}
                        for i in range(2)],
        "relative_phase_definition": "theta2-theta1; theta1 fixed at zero",
        "RF_torus_samples_per_phase": samples_rf,
        "relative_phase_samples": samples_slow,
        "relative_phase_rad": angles.tolist(),
        "mean_Coulomb_force_by_phase_dimensionless": force_rows,
        "mean_Coulomb_torque_by_phase_dimensionless": torque_rows,
        "mean_axial_Coulomb_force_by_phase_dimensionless": zforce_rows,
        "minimum_surface_gap_over_fast_phases_by_slow_phase_over_R": min_gap,
        "relative_slow_phase_rate": float(omega[1]-omega[0]),
        "relative_phase_cycle_in_primary_RF_periods": float(1.0/abs(omega[1]-omega[0])),
        "limitations": [
            "RF micromotion is linear Stokes-response and prescribed; nonlinear Oseen modifies it.",
            "Coulomb force is averaged over the shared RF phase torus while slow phase is held fixed.",
            "The averaged force is not fed back into radius/phase equations and no phase-locking claim is made.",
            "The vertical offsets use the ideal local DC quadrupole and infinite-rod RF geometry.",
        ],
    }


def main() -> None:
    out = force_vs_relative_phase(samples_rf=128, samples_slow=96)
    path = ROOT/"results"/"averaged_coulomb_screen.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n",
                    encoding="utf-8")
    torques = np.asarray(out["mean_Coulomb_torque_by_phase_dimensionless"])
    gaps = np.asarray(out["minimum_surface_gap_over_fast_phases_by_slow_phase_over_R"])
    print(path)
    print({"torque1_range": [float(torques[:, 0].min()), float(torques[:, 0].max())],
           "torque2_range": [float(torques[:, 1].min()), float(torques[:, 1].max())],
           "min_surface_gap_over_R": float(gaps.min()),
           "relative_phase_cycle_RF_periods": out["relative_phase_cycle_in_primary_RF_periods"]})


if __name__ == "__main__":
    main()
