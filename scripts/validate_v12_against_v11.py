"""Replay selected, already-computed V11 FEM-map starts with V12 equations.

Only the new dimensionless trajectories are integrated here. Reference
classifications and first-exit times are read from V11 checkpoints/results, so
the physical SI trajectories are not repeated.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap.dimensionless_dynamics_v12 import DimensionlessRFModel, integrate_to_horizon
from three_phase_trap.rf_reconstruction import HarmonicQuadratures, RFReconstruction

OUT = ROOT / "outputs/octupole_comparison_home/dimensionless_acceptance_v12/validation"
MAPS = ROOT / "outputs/octupole_comparison_home/confinement_maps_v11/xy"
CHECKS = ROOT / "outputs/octupole_comparison_home/confinement_maps_v11/numerical_checks"
FIELDS = ROOT / "outputs/octupole_comparison_home/confinement_maps_v11/fields"
E_CHARGE = 1.602176634e-19
MASS = 100_000 * 1.66053906660e-27
OMEGA = 2 * math.pi * 100_000.0
LENGTH = 8e-3


def q_for(mesh: int, model: str):
    if model in "AB":
        path = ROOT / f"outputs/abc_validation/direct_fem_validation/v2/h{mesh}_P4/diagnostic_N48.json"
        q = RFReconstruction.load(path).unit_quadratures
        if model == "B":
            q = HarmonicQuadratures(np.zeros_like(q.cosine), q.sine.copy(), q.reference_radius)
        return q, path
    path = FIELDS / f"h{mesh}_P4_C/reconstruction_N48_C.json"
    return RFReconstruction.load(path).unit_quadratures, path


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_model_at(path: Path, model: str, x: float, y: float):
    matches = [r for r in read_rows(path) if r["model"] == model
               and abs(float(r["x0_m"]) - x) < 1e-15
               and abs(float(r["y0_m"]) - y) < 1e-15]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {model} at {x},{y} in {path}; got {len(matches)}")
    return matches[0]


def load_checks():
    cases = []
    # A common escape-sensitive zero-velocity XY start checked on h35 and h50.
    h35_path = CHECKS / "h35_edge_equal_phase0/results.csv"
    h50_path = MAPS / "4ceec3a67c8fdfcd/results.csv"
    for mesh, path in ((35, h35_path), (50, h50_path)):
        for model in "ABC":
            r = select_model_at(path, model, -0.003, -0.00675)
            cases.append((f"xy_edge_phase0_h{mesh}", mesh, model, r))
    # Same physical coordinates and phase 45, from the completed V11 map.
    phase45_path = MAPS / "7c854b0d1f8f059f/results.csv"
    for model in "ABC":
        r = select_model_at(phase45_path, model, -0.003, -0.00675)
        cases.append(("xy_edge_phase45_h50", 50, model, r))
    return cases


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for label, mesh, model_name, ref in load_checks():
        q, source = q_for(mesh, model_name)
        voltage = float(ref["voltage_V"])
        phase = float(ref["phase_rad"])
        state_si = np.array([float(ref[k]) for k in ("x0_m", "y0_m", "vx0_m_s", "vy0_m_s")])
        state_d = np.r_[state_si[:2] / LENGTH, state_si[2:] / (LENGTH * OMEGA)]
        model = DimensionlessRFModel.from_physical(
            q.cosine, q.sine, charge_c=E_CHARGE, mass_kg=MASS,
            voltage_v=voltage, omega_rad_s=OMEGA, length_m=LENGTH,
            reference_radius_m=q.reference_radius, aperture_m=LENGTH,
            phase_rad=phase, name=f"{model_name}_h{mesh}")
        dim = integrate_to_horizon(
            model, state_d, 1000, segment_periods=100,
            rtol=2e-9,
            atol=np.array([2.5e-11, 2.5e-11,
                           2e-11/(LENGTH*OMEGA), 2e-11/(LENGTH*OMEGA)]),
            max_step_rf=1/50, samples_per_rf=4,
        )
        ref_status = ref["status"]
        ref_exit = None if not ref["escape_RF"] else float(ref["escape_RF"])
        ref_final_si = np.asarray(json.loads(ref["final_state_SI"]), dtype=float)
        ref_final_d = np.r_[ref_final_si[:2] / LENGTH, ref_final_si[2:] / (LENGTH * OMEGA)]
        rows.append({
            "case": label, "mesh_um": mesh, "model": model_name,
            "field_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
            "x0_m": state_si[0], "y0_m": state_si[1], "vx0_m_s": state_si[2], "vy0_m_s": state_si[3],
            "phase_rad": phase, "voltage_V": voltage,
            "reference_status_V11": ref_status, "dimensionless_status": dim.status,
            "status_match": ref_status == dim.status,
            "reference_exit_RF": ref_exit, "dimensionless_exit_RF": dim.escape_period,
            "exit_time_difference_RF": None if ref_exit is None or dim.escape_period is None else dim.escape_period-ref_exit,
            "final_state_difference_norm_dimensionless": float(np.linalg.norm(dim.final_state-ref_final_d)),
            "max_radius_sampled_rho": dim.max_radius_sampled_rho,
            "nfev": dim.nfev,
        })
        print(f"{label} {model_name} h{mesh}: {dim.status}, exit={dim.escape_period}", flush=True)
    out_csv = OUT / "dimensionless_replay_of_v11_xy.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "study": "V12 dimensionless replay against completed V11 XY outcomes",
        "new_dimensionless_trajectories": len(rows), "new_physical_SI_trajectories": 0,
        "new_FEM_solves": 0,
        "reference": "Existing h35/h50 V11 results at phase 0 and h50 V11 results at phase 45; same physical states, voltage and N48 field files.",
        "solver": "DOP853, 100 RF segment restart, rtol=2e-9, max step T/50, phase preserved.",
        "all_statuses_match": all(r["status_match"] for r in rows),
        "escape_comparisons": sum(r["exit_time_difference_RF"] is not None for r in rows),
        "max_abs_exit_time_difference_RF": max((abs(r["exit_time_difference_RF"]) for r in rows
                                                  if r["exit_time_difference_RF"] is not None), default=None),
        "wall_seconds": time.perf_counter()-started, "csv": out_csv.name,
        "results": rows,
    }
    (OUT / "dimensionless_replay_of_v11_xy.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
