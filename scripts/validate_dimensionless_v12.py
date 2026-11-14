"""Cross-check the V12 dimensionless integrator on saved h35/h50 FEM fits.

This compares the new dimensionless equations with the established physical-SI
N48 integrator for identical FEM coefficient arrays and physical initial
states.  The saved P4 trajectories are reported separately as independent
spatial-field checks; this script does not rebuild or modify any FEM solution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "outputs/octupole_comparison_home/confinement_maps_v4"))
from three_phase_trap.dimensionless_dynamics_v12 import DimensionlessRFModel, integrate_to_horizon
from three_phase_trap.pilot_dynamics import PolynomialRF, integrate_rf
from three_phase_trap.rf_reconstruction import HarmonicQuadratures, RFReconstruction
import run_v4


OUT = ROOT / "outputs/octupole_comparison_home/dimensionless_acceptance_v12/validation"
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
MASS = 100_000 * AMU
FREQUENCY = 100_000.0
OMEGA = 2.0 * math.pi * FREQUENCY
LENGTH = 8.0e-3
APERTURE = 8.0e-3


def load_field(mesh_um: int, model: str) -> HarmonicQuadratures:
    if model in ("A", "B"):
        path = ROOT / f"outputs/abc_validation/direct_fem_validation/v2/h{mesh_um}_P4/diagnostic_N48.json"
        q = RFReconstruction.load(path).unit_quadratures
        if model == "B":
            return HarmonicQuadratures(np.zeros_like(q.cosine), q.sine.copy(), q.reference_radius)
        return q
    path = ROOT / f"outputs/octupole_comparison_home/confinement_maps_v11/fields/h{mesh_um}_P4_C/reconstruction_N48_C.json"
    return RFReconstruction.load(path).unit_quadratures


def initial_states():
    rows = []
    # The central zero-speed state tests the flat center of m>2 fields.
    rows.append(("center_zero", np.zeros(2), np.zeros(2), 0.0))
    # The same physical phase-space launch used in the V6/V7 early-boundary checks.
    for azimuth_deg in (0.0, 9.6240234375, 10.26123046875, 22.5):
        theta = math.radians(azimuth_deg)
        radius = 1.0e-3
        speed = math.sqrt(2.0 * 2.0 * E_CHARGE / MASS)
        position = radius * np.array([math.cos(theta), math.sin(theta)])
        velocity = speed * np.array([-math.sin(theta), math.cos(theta)])
        rows.append((f"r1mm_E2eV_tangent_a{azimuth_deg:g}", position, velocity, 0.0))
    # Direct C-P4 early escape probe reported in V11.
    rows.append(("C_native_P4_early_xy", np.array([3.75e-3, -6e-3]), np.zeros(2), 0.0))
    # Moderate radial launch tests the velocity scaling independently.
    r, theta = 2.0e-3, math.radians(37.0)
    speed = math.sqrt(2.0 * 1.0 * E_CHARGE / MASS)
    rows.append(("r2mm_E1eV_radial_out", r*np.array([math.cos(theta), math.sin(theta)]),
                 speed*np.array([math.cos(theta), math.sin(theta)]), math.pi/4))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--segment", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    starts = initial_states()
    if args.limit:
        starts = starts[:args.limit]
    rows = []
    started = time.perf_counter()
    for mesh in (35, 50):
        for config in "ABC":
            q = load_field(mesh, config)
            voltage = 250.0 * math.sqrt(2.0) if config == "B" else 250.0
            phys_field = PolynomialRF(q)
            dimless = DimensionlessRFModel.from_physical(
                q.cosine, q.sine, charge_c=E_CHARGE, mass_kg=MASS,
                voltage_v=voltage, omega_rad_s=OMEGA, length_m=LENGTH,
                reference_radius_m=q.reference_radius, aperture_m=APERTURE,
                name=f"{config}_h{mesh}")
            for label, position, velocity, phase in starts:
                state_dimless = np.r_[position / LENGTH, velocity / (LENGTH * OMEGA)]
                t0 = time.perf_counter()
                dim = integrate_to_horizon(
                    dimless, state_dimless, args.horizon,
                    segment_periods=args.segment, rtol=2e-10, atol=2e-12,
                    max_step_rf=1/64, samples_per_rf=4,
                )
                dim_elapsed = time.perf_counter() - t0
                t0 = time.perf_counter()
                phys = integrate_rf(
                    phys_field, np.r_[position, velocity], E_CHARGE * voltage,
                    MASS, FREQUENCY, phase, 0.0, args.horizon / FREQUENCY,
                    output_samples_per_period=1, max_steps_per_period=64,
                    rtol=2e-10, atol=(2e-13, 2e-13, 2e-11, 2e-11), radius=APERTURE,
                )
                phys_elapsed = time.perf_counter() - t0
                phys_status = "escaped" if phys.exit_time is not None else "retained"
                phys_exit_rf = None if phys.exit_time is None else phys.exit_time * FREQUENCY
                state_from_physical = np.r_[phys.states[-1, :2] / LENGTH,
                                           phys.states[-1, 2:] / (LENGTH * OMEGA)]
                mismatch = float(np.linalg.norm(dim.final_state - state_from_physical))
                rows.append({
                    "mesh_um": mesh, "configuration": config, "launch": label,
                    "phase_deg": math.degrees(phase), "voltage_V": voltage,
                    "horizon_RF": args.horizon,
                    "dimensionless_status": dim.status,
                    "physical_SI_status": phys_status,
                    "status_match": dim.status == phys_status,
                    "dimensionless_exit_RF": dim.escape_period,
                    "physical_exit_RF": phys_exit_rf,
                    "exit_time_difference_RF": (None if dim.escape_period is None or phys_exit_rf is None
                                                 else dim.escape_period - phys_exit_rf),
                    "final_state_mismatch_norm_dimensionless": mismatch,
                    "dimensionless_max_rho_sampled": dim.max_radius_sampled_rho,
                    "dimensionless_nfev": dim.nfev,
                    "dimensionless_wall_s": dim_elapsed,
                    "physical_nfev": phys.nfev,
                    "physical_wall_s": phys_elapsed,
                    "field_reference_radius_m": q.reference_radius,
                    "field_max_order": len(q.cosine)-1,
                })
                print(f"h{mesh} {config} {label}: {dim.status}/{phys_status}; "
                      f"dt_exit={rows[-1]['exit_time_difference_RF']}; dstate={mismatch:.3e}", flush=True)
    out_csv = OUT / f"dimensionless_vs_physical_horizon{args.horizon}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "study": "V12 dimensionless-to-SI equation verification",
        "horizon_RF": args.horizon,
        "meshes_um": [35, 50], "configurations": ["A", "B", "C"],
        "new_FEM_solves": 0,
        "fields": "Saved P4-derived N48 Fourier reconstructions, h35 and h50; C uses the independent alternating-electrode P4 solve.",
        "solver": "DOP853 both formulations; dimensionless tau=Omega*t, same SI and phase inputs; 64 steps/RF maximum step.",
        "trajectories": len(rows),
        "all_statuses_match": all(r["status_match"] for r in rows),
        "max_exit_time_difference_RF": max((abs(r["exit_time_difference_RF"]) for r in rows
                                             if r["exit_time_difference_RF"] is not None), default=None),
        "max_final_state_mismatch_dimensionless": max(r["final_state_mismatch_norm_dimensionless"] for r in rows),
        "wall_seconds": time.perf_counter() - started,
        "csv": out_csv.name,
    }
    (OUT / f"dimensionless_vs_physical_horizon{args.horizon}.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
