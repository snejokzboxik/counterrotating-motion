"""Integrity and summary audit for the N=6 Coulomb continuation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "p1_coulomb_scan"
OUT = DATA / "coupling_audit.json"
EXPECTED = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


def main() -> None:
    rows = [json.loads(s) for s in (DATA / "coulomb_runs.jsonl").read_text().splitlines() if s.strip()]
    errors = []
    factors = [float(r["C_multiplier"]) for r in rows]
    for r in rows:
        path = DATA / f"{r['task_id']}_trajectory.npz"
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue
        with np.load(path) as arr:
            if not np.isfinite(arr["trajectory"]).all() or not np.isfinite(arr["energy_balance"]).all():
                errors.append(f"non-finite trajectory/work: {path.name}")
            if arr["trajectory"].shape[1:] != (6, 8):
                errors.append(f"unexpected trajectory shape: {path.name}")
    summaries = []
    for r in sorted(rows, key=lambda x: x["C_multiplier"]):
        groups = r["structure_after_load"]["groups"]
        fmean = [g["mean_frequency_Hz"] for g in groups]
        summary = {
            "C_multiplier": r["C_multiplier"], "C_absolute": r["C_absolute"],
            "completed_periods": r["completed_periods"], "retained": r["retained"],
            "integrator_category": r["category"],
            "mean_frequency_Hz_by_group": fmean,
            "mean_turns_by_group": [float(np.mean(r["turns_each_after_load"][:3])),
                                    float(np.mean(r["turns_each_after_load"][3:]))],
            "frequency_spread_percent_by_group": [100.0 * g["relative_frequency_spread"] for g in groups],
            "mean_Psi3_by_group": [g["mean_Psi3"] for g in groups],
            "p05_Psi3_by_group": [g["p05_Psi3"] for g in groups],
            "late_gap_max_error_deg_by_group": [g["late_max_abs_gap_error_deg"] for g in groups],
            "mean_radius_R_by_group": [float(np.mean(g["mean_radius_R_each"])) for g in groups],
            "max_abs_late_phase_drift_deg_per_1000_periods": [
                max(abs(x["linear_drift_deg_per_1000_periods"])
                    for x in g["relative_phase_late_diagnostics"][1:]) for g in groups],
            "minimum_pair_distance_R": min(r["minimum_pair_distances_R_i_lt_j"]),
            "wall_clearance_min_R": min(1.0 - radius - particle_radius for radius, particle_radius in
                                        zip(r["r_max_R_each"], [0.0002] * 3 + [0.00189] * 3)),
            "energy_relative_max_residual": r["energy_balance"]["relative_max_residual_vs_RF_work"],
        }
        summary["opposite_group_directions"] = fmean[0] * fmean[1] < 0.0
        summaries.append(summary)
    completed = len(rows) == len(EXPECTED) and sorted(factors) == list(EXPECTED) and not errors
    output = {
        "status": "passed" if completed else "failed",
        "runs": len(rows), "unique_task_ids": len({r["task_id"] for r in rows}),
        "trajectory_count": len(list(DATA.glob("*_trajectory.npz"))),
        "data_errors": errors,
        "interpretation": "For this selected ordered start, counterrotation persists through C/C0=2 and both rings corotate by C/C0=5; the transition is bracketed only within that interval and this loading history.",
        "results": summaries,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "passed":
        raise SystemExit("Coulomb continuation integrity audit failed")


if __name__ == "__main__":
    main()
