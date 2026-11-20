"""Audit completed long P1 candidates and assign explicit structural criteria."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "p1_long_candidates"
OUT = DATA / "long_audit.json"


def classify(row: dict) -> tuple[str, dict]:
    s = row["structure_after_load"]
    groups = s["groups"]
    phase_slopes = [abs(x["linear_drift_deg_per_1000_periods"])
                    for g in groups for x in g["relative_phase_late_diagnostics"][1:]]
    phase_ranges = [x["detrended_range_deg"]
                    for g in groups for x in g["relative_phase_late_diagnostics"][1:]]
    predicates = {
        "all_six_retained": bool(row["retained"] and row["completed_periods"] == row["total_periods_requested"]),
        "all_six_at_least_10_turns": bool(s.get("all_particles_at_least_10_turns")),
        "opposite_group_frequencies": bool(s.get("opposite_group_frequency_signs")),
        "directions_persist_in_three_windows": bool(s.get("directions_constant_in_three_windows")),
        "each_group_frequency_spread_below_1_percent": all(g["relative_frequency_spread"] < 0.01 for g in groups),
        "each_group_mean_psi3_at_least_0_9": all(g["mean_Psi3"] >= 0.9 for g in groups),
        "each_group_p05_psi3_at_least_0_9": all(g["p05_Psi3"] >= 0.9 for g in groups),
        "late_gap_deviation_at_most_10_deg": all(g["late_max_abs_gap_error_deg"] <= 10.0 for g in groups),
        "relative_phase_drift_at_most_0_25_deg_per_1000_periods": max(phase_slopes, default=float("inf")) <= 0.25,
        "detrended_relative_phase_range_at_most_20_deg": max(phase_ranges, default=float("inf")) <= 20.0,
        "no_particle_or_wall_contact": row["failure_status"] == 0,
    }
    cls = "COUNTERROTATING_CRYSTALS_3_PLUS_3" if all(predicates.values()) else (
        "COUNTERROTATING_UNORDERED" if predicates["opposite_group_frequencies"] else "UNRESOLVED")
    return cls, predicates


def main() -> None:
    rows = [json.loads(x) for x in (DATA / "long_runs.jsonl").read_text().splitlines() if x.strip()]
    ids = [r["task_id"] for r in rows]
    errors = []
    for r in rows:
        f = DATA / f"{r['task_id']}_trajectory.npz"
        if not f.exists():
            errors.append(f"missing {f.name}")
            continue
        with np.load(f) as z:
            if not np.isfinite(z["trajectory"]).all() or not np.isfinite(z["energy_balance"]).all():
                errors.append(f"non-finite values in {f.name}")
            if z["trajectory"].shape[1:] != (6, 8):
                errors.append(f"unexpected state shape in {f.name}")
    annotated = []
    for row in rows:
        cls, predicates = classify(row)
        annotated.append({
            "task_id": row["task_id"], "protocol": row["protocol"],
            "scientific_classification": cls, "criteria": predicates,
            "turns_each": row["turns_each_after_load"],
            "frequencies_Hz_each": row["frequencies_Hz_each"],
            "mean_radius_R_each_group": [g["mean_radius_R_each"] for g in row["structure_after_load"]["groups"]],
            "mean_Psi3_each_group": [g["mean_Psi3"] for g in row["structure_after_load"]["groups"]],
            "p05_Psi3_each_group": [g["p05_Psi3"] for g in row["structure_after_load"]["groups"]],
            "frequency_spread_each_group": [g["relative_frequency_spread"] for g in row["structure_after_load"]["groups"]],
            "late_gap_error_each_group_deg": [g["late_max_abs_gap_error_deg"] for g in row["structure_after_load"]["groups"]],
            "center_z_R_each_group": [g["center_z_mean_R"] for g in row["structure_after_load"]["groups"]],
            "energy_relative_max_residual": row["energy_balance"]["relative_max_residual_vs_RF_work"],
            "minimum_pair_distance_R": min(row["minimum_pair_distances_R_i_lt_j"]),
            "wall_clearance_min_R": min(1.0 - r - a for r, a in zip(row["r_max_R_each"], [0.0002] * 3 + [0.00189] * 3)),
        })
    output = {
        "status": "passed" if len(rows) == 7 and len(set(ids)) == 7 and not errors else "failed",
        "runs": len(rows), "unique_task_ids": len(set(ids)), "trajectory_npz_count": len(list(DATA.glob("*_trajectory.npz"))),
        "data_errors": errors, "criteria_note": "Finite-horizon structural classification; not a claim of asymptotic attractor or random-basin success.",
        "candidates": annotated,
        "crystal_candidates": [r["task_id"] for r in annotated if r["scientific_classification"] == "COUNTERROTATING_CRYSTALS_3_PLUS_3"],
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "passed":
        raise SystemExit("long-candidate data audit failed")


if __name__ == "__main__":
    main()
