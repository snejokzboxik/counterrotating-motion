"""Integrity and outcome summary for the five frozen N=6 formation ensembles."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_p1_crystal_ensembles import COUNT, ENSEMBLES, OUT

POST_LOAD = 15_000


def main() -> None:
    result_path = OUT / "formation_runs.jsonl"
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate task IDs")
    if len(rows) != len(ENSEMBLES) * COUNT:
        errors.append(f"expected {len(ENSEMBLES)*COUNT} result rows, found {len(rows)}")
    meta = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        errors.append(f"run status is {meta.get('status')!r}, not complete")

    trajectory_files = list(OUT.glob("*_trajectory.npz"))
    if len(trajectory_files) != len(rows):
        errors.append(f"JSONL rows {len(rows)} != trajectory files {len(trajectory_files)}")
    by_ensemble = {}
    for ensemble in ENSEMBLES:
        group_rows = [r for r in rows if r["ensemble"] == ensemble]
        if len(group_rows) != COUNT:
            errors.append(f"{ensemble}: expected {COUNT} rows, found {len(group_rows)}")
        start_meta = json.loads((OUT / f"{ensemble}_metadata.json").read_text(encoding="utf-8"))
        if any(r["start_file_sha256"] != start_meta["start_and_phase_sha256"] for r in group_rows):
            errors.append(f"{ensemble}: result row start hash mismatch")
        for row in group_rows:
            path = OUT / f"{row['task_id']}_trajectory.npz"
            if not path.exists():
                errors.append(f"missing {path.name}")
                continue
            with np.load(path) as data:
                trajectory = data["trajectory"]
                work = data["energy_balance"]
                if trajectory.shape[1:] != (6, 8):
                    errors.append(f"{path.name}: trajectory shape {trajectory.shape}")
                if not np.isfinite(trajectory).all() or not np.isfinite(work).all():
                    errors.append(f"{path.name}: non-finite data")
                if len(trajectory) != row["completed_periods"] + 1:
                    errors.append(f"{path.name}: row/log completed cycle mismatch")
                if row["retained"] and row["post_load_periods_observed"] != POST_LOAD:
                    errors.append(f"{path.name}: retained run did not reach full post-load horizon")

        category_counts = Counter(r["category"] for r in group_rows)
        crystal = [r for r in group_rows if r["category"] == "COUNTERROTATING_CRYSTALS_3_PLUS_3"]
        counterrotation = [r for r in group_rows if r.get("structure_after_load", {}).get("opposite_group_frequency_signs") is True]
        stable_particle_directions = []
        for r in group_rows:
            if not r["retained"]:
                stable_particle_directions.append(False)
                continue
            w = np.asarray(r["late_window_turns_each"], dtype=np.float64)
            stable_particle_directions.append(bool(
                np.all(w[:, :3] > 0.0) and np.all(w[:, 3:] < 0.0) or
                np.all(w[:, :3] < 0.0) and np.all(w[:, 3:] > 0.0)))
        retained_with_structure = [r for r in group_rows if r["retained"] and len(r.get("structure_after_load", {}).get("groups", [])) == 2]
        energy_rows = [r["energy_balance"] for r in group_rows if r.get("energy_balance") is not None]
        by_ensemble[ensemble] = {
            "n_total": len(group_rows),
            "outcome_counts": dict(sorted(category_counts.items())),
            "retained_all_six_full_horizon": sum(bool(r["retained"]) for r in group_rows),
            "retention_probability_all_starts": sum(bool(r["retained"]) for r in group_rows) / max(1, len(group_rows)),
            "strict_counterrotating_crystal_successes": len(crystal),
            "P_crystal_all_starts": len(crystal) / max(1, len(group_rows)),
            "opposite_group_mean_rotation_cases": len(counterrotation),
            "P_opposite_group_rotation_all_starts": len(counterrotation) / max(1, len(group_rows)),
            "all_six_directions_opposite_in_three_late_windows": sum(stable_particle_directions),
            "minimum_turns_each_over_retained_cases": min(
                (min(abs(x) for x in r["turns_each_total"]) for r in group_rows if r["retained"]),
                default=None),
            "mean_group_frequency_Hz_over_strict_crystals": (
                [float(np.mean([r["structure_after_load"]["groups"][g]["mean_frequency_Hz"] for r in crystal]))
                 for g in range(2)] if crystal else None),
            "mean_Psi3_over_strict_crystals": (
                [float(np.mean([r["structure_after_load"]["groups"][g]["mean_Psi3"] for r in crystal]))
                 for g in range(2)] if crystal else None),
            "mean_group_center_z_R_over_retained_structures": (
                [float(np.mean([r["structure_after_load"]["groups"][g]["center_z_mean_R"]
                                for r in retained_with_structure])) for g in range(2)]
                if retained_with_structure else None),
            "energy_balance_coverage": len(energy_rows),
            "maximum_relative_energy_residual_vs_RF_work": (
                max(e["relative_max_residual_vs_RF_work"] for e in energy_rows) if energy_rows else None),
            "start_sha256": start_meta["start_and_phase_sha256"],
        }

    all_success = sum(v["strict_counterrotating_crystal_successes"] for v in by_ensemble.values())
    all_retained = sum(v["retained_all_six_full_horizon"] for v in by_ensemble.values())
    early_boundary_failures = []
    for row in rows:
        if row["category"] == "WALL_CONTACT":
            initial = np.asarray(row["initial_state"], dtype=np.float64)
            early_boundary_failures.append({
                "task_id": row["task_id"], "completed_primary_periods": row["completed_periods"],
                "initial_max_r_R": float(np.max(np.hypot(initial[:, 0], initial[:, 1]))),
                "initial_max_abs_z_R": float(np.max(np.abs(initial[:, 2]))),
                "initial_max_speed": float(np.max(np.linalg.norm(initial[:, 3:6], axis=1))),
            })
    output = {
        "status": "passed" if not errors else "failed",
        "protocol": "P2_505", "steps_per_period": 96,
        "post_load_periods": POST_LOAD, "relax_cycles": meta["relax_after_load_cycles"],
        "n_total_starts": len(rows), "n_strict_crystals": all_success,
        "P_crystal_all_starts": all_success / max(1, len(rows)),
        "n_full_horizon_retained": all_retained,
        "P_retention_all_starts": all_retained / max(1, len(rows)),
        "n_opposite_group_rotation": sum(v["opposite_group_mean_rotation_cases"] for v in by_ensemble.values()),
        "P_opposite_group_rotation_all_starts": sum(v["opposite_group_mean_rotation_cases"] for v in by_ensemble.values()) / max(1, len(rows)),
        "denominator_note": "All generated, initially non-overlapping starts are included; early wall/contact losses are failures, never dropped.",
        "sampling_note": "Exploratory scrambled-Sobol N=16 per ensemble; point probabilities are not final IID statistical validation.",
        "integrity_errors": errors,
        "ensembles": by_ensemble,
        "category_counts_overall": dict(sorted(Counter(r["category"] for r in rows).items())),
        "early_boundary_failures": early_boundary_failures,
        "run_ids": ids,
    }
    (OUT / "formation_ensemble_audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": output["status"], "n_total": output["n_total_starts"],
        "n_crystal": output["n_strict_crystals"], "P_crystal": output["P_crystal_all_starts"],
        "n_retained": output["n_full_horizon_retained"], "P_retention": output["P_retention_all_starts"],
        "n_counterrotation": output["n_opposite_group_rotation"],
        "outcomes": output["category_counts_overall"],
        "per_ensemble": {k: {key: v[key] for key in ["retained_all_six_full_horizon", "strict_counterrotating_crystal_successes", "P_crystal_all_starts", "outcome_counts"]}
                         for k, v in by_ensemble.items()},
        "errors": errors,
    }, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit("formation ensemble audit failed")


if __name__ == "__main__":
    main()
