"""Audit both pre-load and post-formation perturbation comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_verified(path: Path, jsonl: str, expected: int) -> list[dict]:
    rows = [json.loads(s) for s in (path / jsonl).read_text().splitlines() if s.strip()]
    errors = []
    if len(rows) != expected or len({r["task_id"] for r in rows}) != expected:
        errors.append("incorrect row or unique-task count")
    for row in rows:
        f = path / f"{row['task_id']}_trajectory.npz"
        if not f.exists():
            errors.append(f"missing {f.name}")
            continue
        with np.load(f) as a:
            if not np.isfinite(a["trajectory"]).all() or not np.isfinite(a["energy_balance"]).all():
                errors.append(f"non-finite arrays in {f.name}")
            if a["trajectory"].shape[1:] != (6, 8):
                errors.append(f"wrong trajectory shape in {f.name}")
    if errors:
        raise ValueError(f"{path.name} integrity errors: {errors}")
    return rows


def summarize(row: dict, structure_key: str, turns_key: str) -> dict:
    s = row[structure_key]
    groups = s["groups"]
    return {
        "task_id": row["task_id"], "angle_deg": row.get("initial_angle_perturbation_deg", row.get("position_perturbation_deg")),
        "C_multiplier": row["C_multiplier"], "retained": row["retained"],
        "failure_status": row["failure_status"], "category": row["category"],
        "min_abs_turns": min(abs(x) for x in row[turns_key]),
        "opposite_group_frequencies": s.get("opposite_group_frequency_signs"),
        "directions_stable_in_windows": s.get("directions_constant_in_three_windows"),
        "mean_frequency_Hz_by_group": [g["mean_frequency_Hz"] for g in groups],
        "frequency_spread_percent_by_group": [100 * g["relative_frequency_spread"] for g in groups],
        "mean_Psi3_by_group": [g["mean_Psi3"] for g in groups],
        "p05_Psi3_by_group": [g["p05_Psi3"] for g in groups],
        "late_gap_max_error_deg_by_group": [g["late_max_abs_gap_error_deg"] for g in groups],
        "max_abs_relative_phase_drift_deg_per_1000_periods": [
            max(abs(x["linear_drift_deg_per_1000_periods"])
                for x in g["relative_phase_late_diagnostics"][1:]) for g in groups],
        "min_particle_distance_R": min(row["minimum_pair_distances_R_i_lt_j"]),
    }


def main() -> None:
    initial_dir = HERE / "p1_coulomb_perturbation"
    formed_dir = HERE / "p1_postformation_perturbations"
    initial_rows = read_verified(initial_dir, "perturbation_runs.jsonl", 4)
    formed_rows = read_verified(formed_dir, "postformation_runs.jsonl", 4)
    result = {
        "status": "passed",
        "initial_state_perturbations": {
            "meaning": "one particle in each ring rotated before loading; zero velocities retained",
            "results": [summarize(r, "structure_after_load", "turns_each_after_load") for r in initial_rows],
        },
        "postformation_perturbations": {
            "meaning": "formed candidate perturbed at 10,000 post-load cycles; RF phases propagated continuously; 15,000-cycle recovery window",
            "results": [summarize(r, "structure_after_perturbation", "turns_each_after_perturbation") for r in formed_rows],
        },
        "interpretation": "C=C0 more strongly restores the triangular geometry after the specified coordinate perturbations than C=0; the 20-degree perturbation still leaves residual inner-ring distortion and is not a strict full-recovery pass.",
    }
    out = HERE / "p1_perturbation_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
