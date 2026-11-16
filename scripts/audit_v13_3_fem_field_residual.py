"""Compare the cached full h50 P4 FEM force with V13.3 leading harmonics."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
SAMPLE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1/fem/superposition_points.npz"
POINTS = BASE / "fem_control_points.csv"
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
M0 = 3200.04652261404 * AMU
R = .008
OMEGA = 2 * math.pi * 100_000.
C2 = D2 = .7718255558274212
C4 = .5413395699439091


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from three_phase_trap.mass_filter_v13_3 import acceleration

    with POINTS.open(newline="", encoding="utf-8") as stream:
        points = list(csv.DictReader(stream))
    sample = np.load(SAMPLE)
    xy = sample["points_m"]
    rho = np.hypot(xy[:, 0], xy[:, 1]) / R
    g4c, g4s, g2 = (sample[k] for k in ("G4c_grad_V_per_m", "G4s_grad_V_per_m", "G2_grad_V_per_m"))
    scale = E_CHARGE / (M0 * OMEGA**2 * R)
    radial_cuts = (.25, .40, .60, .90)
    times = (0., .37, math.pi/2, 2.4)
    phases = (0., math.pi/2)
    rows = []
    for point in points:
        alpha, beta, kappa = map(float, (point["alpha"], point["beta"], point["kappa"]))
        u2 = beta * M0 * OMEGA**2 * R**2 / E_CHARGE / C2
        vdc = alpha * M0 * OMEGA**2 * R**2 / E_CHARGE / D2
        u4 = kappa * M0 * OMEGA**2 * R**2 / E_CHARGE / C4
        for cutoff in radial_cuts:
            mask = (rho <= cutoff) & (rho > 1e-4)
            error_sq = reference_sq = 0.0
            n_vectors = 0
            for tau0 in phases:
                for tau in times:
                    phase = tau + tau0
                    fem_grad = u4 * (g4c[mask] * math.cos(phase) + g4s[mask] * math.sin(phase))
                    fem_grad += (u2 * math.cos(phase) + vdc) * g2[mask]
                    fem_acc = -scale * fem_grad
                    ideal_acc = np.asarray([acceleration(x, y, tau, kappa, beta, alpha,
                                                          delta=0., tau0=tau0)
                                            for x, y in xy[mask] / R])
                    error_sq += float(np.sum((fem_acc - ideal_acc)**2))
                    reference_sq += float(np.sum(ideal_acc**2))
                    n_vectors += len(ideal_acc)
            rows.append({"candidate_id": point["candidate_id"], "alpha": alpha,
                         "beta": beta, "kappa": kappa, "eta": kappa/beta,
                         "rho_cut": cutoff, "point_time_phase_samples": n_vectors,
                         "relative_RMS_acceleration_residual": math.sqrt(error_sq/reference_sq),
                         "diagnostic": "600 cached full-mesh h50 P4 points; not a trajectory error bound"})
    out = BASE / "fem_controls"
    with (out / "ideal_vs_fem_force_residuals.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {"sample_npz": str(SAMPLE.relative_to(ROOT)),
                "sample_sha256": hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),
                "points": int(len(xy)), "radius_normalization_m": R,
                "time_samples_rad": list(times), "injection_phases_rad": list(phases),
                "relative_phase_delta_rad": 0.0, "coefficient_normalization": {"c2": C2, "d2": D2, "c4": C4},
                "metric": "relative RMS of vector acceleration difference over sample points and listed phases/times",
                "limitation": "The points are a fixed V13.1 h50 FEM sample, not a uniform area sample or an error bound along all later trajectories."}
    (out / "ideal_vs_fem_force_residuals.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
