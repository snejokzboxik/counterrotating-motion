"""Compare new dimensionless trajectories to saved direct-P4 controls.

The old direct-P4 SI trajectories are read from V10/V11 artifacts.  This
script only integrates the V12 dimensionless N48 reductions; it does not
repeat any native P4 or SI trajectory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

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
        source = ROOT / f"outputs/abc_validation/direct_fem_validation/v2/h{mesh}_P4/diagnostic_N48.json"
        q = RFReconstruction.load(source).unit_quadratures
        if model == "B":
            q = HarmonicQuadratures(np.zeros_like(q.cosine), q.sine.copy(), q.reference_radius)
    else:
        source = FIELDS / f"h{mesh}_P4_C/reconstruction_N48_C.json"
        q = RFReconstruction.load(source).unit_quadratures
    return q, source


def select_csv(path: Path, model: str, x: float, y: float):
    with path.open(newline="", encoding="utf-8") as f:
        found = [r for r in csv.DictReader(f) if r["model"] == model
                 and abs(float(r["x0_m"])-x) < 1e-15 and abs(float(r["y0_m"])-y) < 1e-15]
    if len(found) != 1:
        raise ValueError(f"expected one result row, got {len(found)} in {path}")
    return found[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    x, y = 3.75e-3, -6e-3
    for mesh, ref_path in ((35, CHECKS / "h35_edge_equal_phase0/results.csv"),
                           (50, MAPS / "4ceec3a67c8fdfcd/results.csv")):
        for kind in "ABC":
            ref = select_csv(ref_path, kind, x, y)
            q, field_path = q_for(mesh, kind)
            phase = float(ref["phase_rad"])
            voltage = float(ref["voltage_V"])
            state_si = np.array([x, y, 0.0, 0.0])
            state_d = np.r_[state_si[:2]/LENGTH, state_si[2:]/(LENGTH*OMEGA)]
            model = DimensionlessRFModel.from_physical(
                q.cosine, q.sine, charge_c=E_CHARGE, mass_kg=MASS,
                voltage_v=voltage, omega_rad_s=OMEGA, length_m=LENGTH,
                reference_radius_m=q.reference_radius, aperture_m=LENGTH,
                phase_rad=phase, name=f"{kind}_h{mesh}")
            dim = integrate_to_horizon(model, state_d, 1000, segment_periods=100,
                                       rtol=2e-9,
                                       atol=np.array([2.5e-11, 2.5e-11,
                                                      2e-11/(LENGTH*OMEGA), 2e-11/(LENGTH*OMEGA)]),
                                       max_step_rf=1/50, samples_per_rf=4)
            p4_ref = None
            if kind == "C":
                p4_path = CHECKS / "native_C_early_exit" / f"h{mesh}.json"
                p4_ref = json.loads(p4_path.read_text(encoding="utf-8"))
            rows.append({
                "mesh_um": mesh, "model": kind, "x0_m": x, "y0_m": y,
                "phase_rad": phase, "voltage_V": voltage,
                "field_sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
                "V11_N48_status": ref["status"], "dimensionless_status": dim.status,
                "V11_N48_exit_RF": ref["escape_RF"] or None,
                "dimensionless_exit_RF": dim.escape_period,
                "dimensionless_minus_V11_exit_RF": None if not ref["escape_RF"] or dim.escape_period is None else dim.escape_period-float(ref["escape_RF"]),
                "native_P4_status": (None if p4_ref is None else "escaped"),
                "native_P4_exit_RF": (None if p4_ref is None else p4_ref["escape_RF"]),
                "dimensionless_minus_native_P4_exit_RF": (None if p4_ref is None or dim.escape_period is None else dim.escape_period-p4_ref["escape_RF"]),
                "nfev": dim.nfev,
            })
            print(f"h{mesh} {kind}: {dim.status}, tau_exit={dim.escape_period}", flush=True)
    out_csv = OUT / "dimensionless_vs_native_p4_c_early_exit.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "study": "V12 dimensionless reduction versus saved native P4 FEM controls",
        "new_dimensionless_trajectories": len(rows), "new_native_P4_or_SI_trajectories": 0,
        "new_FEM_solves": 0, "launch": {"x_m": x, "y_m": y, "v_m_s": [0, 0], "phase_deg": 0},
        "models_and_meshes": "A/B/C at h35 and h50, from independent P4-derived N48 coefficient sets; direct native P4 C controls are saved h35/h50 trajectories.",
        "status_matches_V11_N48": all(r["V11_N48_status"] == r["dimensionless_status"] for r in rows),
        "C_native_P4_exit_differences_RF": {f"h{r['mesh_um']}": r["dimensionless_minus_native_P4_exit_RF"] for r in rows if r["model"] == "C"},
        "csv": out_csv.name,
    }
    (OUT / "dimensionless_vs_native_p4_c_early_exit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
