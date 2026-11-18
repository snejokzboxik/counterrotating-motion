"""Screen individual complex electrode drives around ADCFEB.

This is a 2-D electrostatic MFS harmonic/field screen, not a particle-orbit
calculation. It uses the archived circular-cylinder geometry and full signed
K=72 spectrum. RF phasors use exp(-i Omega t), so a rod voltage with phase
phi is represented by exp(+i phi).
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import differential_evolution

HERE = Path(__file__).resolve().parent
V17C4 = HERE.parents[1]
ARCH_SCAN = V17C4 / "input" / "archive_contents" / "v17c_six_phase" / "scan.py"
FEM_JSON = V17C4 / "results" / "fem_sixphase" / "G1_G2_12electrode_6phase_P1_FEM.json"
OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)

N = 12
A_D = .05
B_D = 1.10
KMAX = 72
PHASE_INDEX = np.asarray([0, 3, 2, 5, 4, 1] * 2)
V_BASE = np.exp(1j * PHASE_INDEX * np.pi / 3)


def load_scan():
    spec = importlib.util.spec_from_file_location("mfs_scan_adaptive_phases", ARCH_SCAN)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def surface_field_basis(samples_per_rod: int = 128, p: int = 64):
    """Complex normal/tangential field phasor at electrode surfaces per drive."""
    theta = np.arange(N) * 2 * np.pi / N
    centers = np.exp(1j * theta)
    mfs_angles = theta[:, None] + np.arange(p) * 2 * np.pi / p
    targets_mfs = (centers[:, None] + A_D * np.exp(1j * mfs_angles)).ravel()
    sources = (centers[:, None] + .65 * A_D * np.exp(1j * mfs_angles)).ravel()
    source_conj = sources.conj()[None, :]

    def green(z):
        return np.log(np.abs(B_D * B_D - z[:, None] * source_conj) /
                      (B_D * np.abs(z[:, None] - sources[None, :])))

    rhs = np.repeat(np.eye(N, dtype=np.complex128), p, axis=0)
    source_coeff = np.linalg.solve(green(targets_mfs), rhs)

    sample = np.arange(samples_per_rod) * 2 * np.pi / samples_per_rod
    surface_targets = (centers[:, None] + A_D * np.exp(1j * sample[None, :])).ravel()
    den1 = B_D * B_D - surface_targets[:, None] * source_conj
    diff = surface_targets[:, None] - sources[None, :]
    gx = np.real(-source_conj / den1 - 1.0 / diff)
    gy = np.real(-1j * source_conj / den1 - 1j / diff)
    ex = -(gx @ source_coeff)
    ey = -(gy @ source_coeff)
    return ex, ey


def spectrum_field_matrices(k, transfer, r: float, ntheta: int = 512):
    """Return Ex/Ey angular samples as linear maps from rod phasors to field."""
    theta = np.arange(ntheta) * 2 * np.pi / ntheta
    z = r * np.exp(1j * theta)
    # One output column per rod phasor.
    ex = np.zeros((ntheta, N), np.complex128)
    ey = np.zeros_like(ex)
    for row, signed_k in enumerate(k):
        kk = int(signed_k)
        n = abs(kk)
        base = z if kk > 0 else z.conjugate()
        term = -n * base[:, None] ** (n - 1) * transfer[row][None, :]
        ex += term
        ey += (-1j * term) if kk > 0 else (1j * term)
    return ex, ey


def metrics(v, k, transfer, surface_ex, surface_ey, field_mats):
    v = np.asarray(v, np.complex128)
    v = v / max(float(np.max(np.abs(v))), 1e-30)  # equal peak electrode voltage
    coeff = transfer @ v
    n = np.abs(k).astype(float)
    torque_weights = np.sign(k) * n * n * (n - 1)
    abs_weights = np.abs(torque_weights)
    radii = np.linspace(.20, .75, 111)
    powers = radii[:, None] ** (2 * n[None, :] - 2)
    torque = powers @ (torque_weights * np.abs(coeff) ** 2)
    torque_abs = powers @ (abs_weights * np.abs(coeff) ** 2)
    chi = torque / np.maximum(torque_abs, 1e-30)

    neg_band = (radii >= .45) & (radii <= .75) & (chi < -.10)
    neg_width = float(np.sum(neg_band) * (radii[1] - radii[0]))
    outer = (radii >= .47) & (radii <= .65)
    inner = (radii >= .20) & (radii <= .40)
    mean_chi_outer = float(np.mean(chi[outer]))
    mean_chi_inner = float(np.mean(chi[inner]))
    outer_neg_frac = float(np.mean(chi[outer] < -.10))
    inner_pos_frac = float(np.mean(chi[inner] > .10))

    profiles = {}
    for rr, (exmat, eymat) in field_mats.items():
        ef = np.abs(exmat @ v) ** 2 + np.abs(eymat @ v) ** 2
        profiles[str(rr)] = {
            "mean_E2": float(np.mean(ef)),
            "corrugation": float((np.max(ef) - np.min(ef)) / max(np.mean(ef), 1e-30)),
            "barrier_per_kappa2": float((np.max(ef) - np.min(ef)) / 4.0),
        }
    surface = np.abs(surface_ex @ v) ** 2 + np.abs(surface_ey @ v) ** 2
    surface_emax = float(np.sqrt(np.max(surface)))
    adjacent = float(np.max(np.abs(v - np.roll(v, -1))))
    mean_inner_e2 = float(np.mean([profiles[x]["mean_E2"] for x in ("0.3", "0.35", "0.4")]))
    mean_outer_e2 = float(np.mean([profiles[x]["mean_E2"] for x in ("0.5", "0.55", "0.6", "0.65")]))
    return {
        "negative_band_width_r45_r75_chi_lt_m0p1": neg_width,
        "outer_mean_chirality_r47_r65": mean_chi_outer,
        "outer_negative_fraction": outer_neg_frac,
        "inner_mean_chirality_r20_r40": mean_chi_inner,
        "inner_positive_fraction": inner_pos_frac,
        "outer_mean_corrugation": float(np.mean([profiles[x]["corrugation"] for x in ("0.47", "0.5", "0.55", "0.6", "0.65")])),
        "inner_mean_corrugation": float(np.mean([profiles[x]["corrugation"] for x in ("0.3", "0.35", "0.4")])),
        "outer_mean_E2": mean_outer_e2,
        "inner_mean_E2": mean_inner_e2,
        "surface_Emax_U0_over_D": surface_emax,
        "max_neighbor_voltage_difference_U0": adjacent,
        "min_radial_chirality": float(np.min(chi)),
        "max_radial_chirality": float(np.max(chi)),
        "metrics_by_radius": profiles,
        "torque_roots_approx": [float((radii[i] + radii[i + 1]) / 2)
                                for i in range(len(radii) - 1) if chi[i] * chi[i + 1] < 0],
        "coefficients": [[int(kk), float(c.real), float(c.imag)] for kk, c in zip(k, coeff)],
        "voltage_phasors": [[float(x.real), float(x.imag)] for x in v],
    }


def score(m):
    """Balanced screening score; hard violations are heavily penalized."""
    intensity = min(m["inner_mean_E2"], m["outer_mean_E2"])
    emax = m["surface_Emax_U0_over_D"]
    return float(
        5.0 * m["negative_band_width_r45_r75_chi_lt_m0p1"]
        + 2.0 * max(0.0, -m["outer_mean_chirality_r47_r65"])
        + 1.5 * max(0.0, m["inner_mean_chirality_r20_r40"])
        + 0.65 * m["outer_negative_fraction"]
        + 0.30 * m["inner_positive_fraction"]
        - 0.85 * m["outer_mean_corrugation"]
        - 0.25 * m["inner_mean_corrugation"]
        + 0.18 * np.log(max(intensity, 1e-12))
        - 8.0 * max(0.0, .55 - intensity) ** 2
        - 1.5 * max(0.0, emax / 26.50818 - 1.12) ** 2
        - 1.0 * max(0.0, emax / 26.50818 - 1.0)
        - 1.0 * max(0.0, m["max_neighbor_voltage_difference_U0"] - 2.05) ** 2
    )


def main():
    scan = load_scan()
    k, transfer, boundary_error = scan.transfer(N, A_D, B_D, p=64, K=KMAX)
    exs, eys = surface_field_basis(samples_per_rod=128, p=64)
    radii = (.30, .35, .40, .47, .50, .55, .60, .65)
    fm = {str(rr): spectrum_field_matrices(k, transfer, rr) for rr in radii}
    base = metrics(V_BASE, k, transfer, exs, eys, fm)
    fem = json.loads(FEM_JSON.read_text(encoding="utf-8"))
    stored = fem["geometries"]["ADCFEBADCFEB"]["mesh_levels"][-1]["mfs_coefficients"]
    stored_d = {int(kk): complex(float(re), float(im)) for kk, re, im in stored}
    computed_d = {int(kk): complex(re, im) for kk, re, im in base["coefficients"]}
    coeff_delta = max(abs(stored_d[kk] - computed_d[kk]) for kk in stored_d)
    baseline_summary = {key: value for key, value in base.items()
                        if key not in ("coefficients", "voltage_phasors", "metrics_by_radius")}
    baseline_summary["MFS_boundary_residual"] = float(np.max(np.abs(boundary_error @ V_BASE)))
    baseline_summary["max_MFS_coefficient_delta_vs_saved"] = float(coeff_delta)
    baseline_summary["metrics_by_radius"] = base["metrics_by_radius"]

    # Variables are normalized rod amplitudes in [0.9,1.1] and eleven phase
    # offsets (rod 0 fixes the irrelevant global RF-time phase) in +/-20 deg.
    # Each trial is rescaled to a maximum peak voltage of U0.
    bounds = [(0.9, 1.1)] * N + [(-np.deg2rad(20), np.deg2rad(20))] * (N - 1)
    evaluations = []

    def decode(x):
        amp = np.asarray(x[:N], float)
        dphase = np.r_[0.0, np.asarray(x[N:], float)]
        v = amp * np.exp(1j * (PHASE_INDEX * np.pi / 3 + dphase))
        return v / np.max(np.abs(v))

    def objective(x):
        mm = metrics(decode(x), k, transfer, exs, eys, fm)
        val = score(mm)
        # Preserve every successful evaluation so interrupted optimization can
        # be resumed/re-ranked without repeating MFS/FEM setup.
        evaluations.append((val, np.asarray(x, float).copy(), mm))
        return -val

    # Seed with the exact baseline, then a compact differential-evolution
    # screen around it. Full RF/DC particle dynamics is still required before
    # any candidate can be considered an orbit-supporting design.
    x0 = np.r_[np.ones(N), np.zeros(N - 1)]
    base_score = score(base)
    evaluations.append((base_score, x0.copy(), base))
    result = differential_evolution(
        objective, bounds, strategy="best1bin", maxiter=18, popsize=6,
        tol=.004, mutation=(.45, 1.0), recombination=.72,
        init="latinhypercube", polish=False, seed=20261118,
        updating="immediate", workers=1, disp=True,
    )
    # Retain a reproducible best-first shortlist; remove essentially duplicate
    # phasor patterns using their normalized complex-drive distance.
    evaluations.sort(key=lambda row: row[0], reverse=True)
    shortlist = []
    seen = []
    for val, x, mm in evaluations:
        v = decode(x)
        if any(np.linalg.norm(v - old) < 0.03 for old in seen):
            continue
        seen.append(v)
        shortlist.append((val, x, mm))
        if len(shortlist) >= 40:
            break

    rows = []
    for rank, (val, x, mm) in enumerate(shortlist, 1):
        row = {"rank": rank, "score": float(val),
               **{key: item for key, item in mm.items()
                  if key not in ("coefficients", "voltage_phasors", "metrics_by_radius")},
               "amplitudes_normalized": np.abs(decode(x)).tolist(),
               "phase_offsets_deg_from_ADCFEB": np.r_[0.0, np.asarray(x[N:]) * 180 / np.pi].tolist(),
               "electrode_phase_deg_absolute": (np.angle(decode(x)) * 180 / np.pi % 360).tolist(),
               "coefficients": mm["coefficients"],
               "voltage_phasors": mm["voltage_phasors"],
               "metrics_by_radius": mm["metrics_by_radius"]}
        rows.append(row)
    summary = {
        "scope": "2D infinite-cylinder MFS spectrum screen; not full 3D FEM or particle dynamics",
        "geometry": {"electrodes": 12, "a_over_D": A_D, "B_over_D": B_D,
                     "R_over_D": 1 - A_D, "Kmax": KMAX, "MFS_p": 64},
        "phase_convention": "V_j(t)=Re[v_j exp(-i Omega t)]; v_j=amplitude*exp(+i phase)",
        "voltage_constraints": {"normalized_max_peak": 1.0,
                                "pre_normalization_amplitude_bounds": [.9, 1.1],
                                "phase_offsets_from_ADCFEB_deg": [-20, 20],
                                "electrode_0_phase_fixed": True},
        "MFS_boundary_residual_baseline": baseline_summary["MFS_boundary_residual"],
        "max_coefficient_delta_vs_saved_MFS": coeff_delta,
        "baseline_ADCFEB": baseline_summary,
        "optimizer": {"method": "scipy differential_evolution best1bin",
                      "maxiter": 18, "popsize": 6, "seed": 20261118,
                      "evaluations": len(evaluations), "success": bool(result.success),
                      "message": str(result.message), "best_score": float(-result.fun)},
        "score_definition": "negative-band width and inner/outer chirality; penalize angular corrugation, weak field, surface E above baseline, and adjacent voltage differences above 2U0",
        "top_candidates": rows,
        "interpretation": "These candidates rank perturbative field metrics only. Every candidate needs full RF-DC 3D two-particle integration, Coulomb coupling, friction, independent solver, long-horizon and perturbation checks.",
    }
    (OUT / "continuous_phase_amplitude_screen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "continuous_phase_amplitude_top40.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "score", "negative_band_width_r45_r75_chi_lt_m0p1",
            "outer_mean_chirality_r47_r65", "outer_negative_fraction", "inner_mean_chirality_r20_r40",
            "inner_positive_fraction", "outer_mean_corrugation", "inner_mean_corrugation",
            "outer_mean_E2", "inner_mean_E2", "surface_Emax_U0_over_D",
            "max_neighbor_voltage_difference_U0", "amplitudes_normalized",
            "phase_offsets_deg_from_ADCFEB", "electrode_phase_deg_absolute"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    print(json.dumps({"baseline": baseline_summary,
                      "best": {key: rows[0][key] for key in
                               ("rank", "score", "negative_band_width_r45_r75_chi_lt_m0p1",
                                "outer_mean_chirality_r47_r65", "outer_mean_corrugation",
                                "surface_Emax_U0_over_D")},
                      "evaluations": len(evaluations)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
