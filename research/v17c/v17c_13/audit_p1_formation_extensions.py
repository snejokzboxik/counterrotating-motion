"""Audit continuity, phase handling, energy balance, and structure in 120k traces."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from run_p1_long_candidates import structural_metrics

HERE = Path(__file__).resolve().parent
BASE = HERE / "p1_formation_ensembles"
OUT = HERE / "p1_formation_extensions"
TASKS = [
    "ordered_003",
    "random_azimuth_010",
    "random_position_rest_004",
    "random_position_rest_015",
    "random_position_moving_006",
]
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
TAU = 2.0 * math.pi


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wrap_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def check_task(task: str, nu: float) -> dict:
    task_dir = OUT / task
    progress = json.loads((task_dir / "progress.json").read_text(encoding="utf-8"))
    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
    base_path = BASE / f"{task}_trajectory.npz"
    with np.load(base_path) as data:
        base = data["trajectory"].copy()
        base_work = data["energy_balance"].copy()
        skip = int(data["post_load_skip"])
    errors: list[str] = []
    chunk_metrics = []
    prev_state = base[-1, :, 1:7].reshape(-1)
    prev_theta = base[-1, :, 7].copy()
    base_tau = float(base[-1, 0, 0])
    prev_energy = float(base_work[-1, 1])
    max_state_boundary_error = 0.0
    max_theta_boundary_error = 0.0
    max_energy_boundary_error = 0.0
    max_phase_error = 0.0
    chunk_count = 0
    for k in range(1, 8):
        path = task_dir / f"chunk_{k:02d}.npz"
        if not path.exists():
            errors.append(f"missing chunk {k}")
            continue
        with np.load(path) as data:
            logs = data["trajectory"].copy()
            work = data["energy_balance"].copy()
            state_end = data["state_end"].copy()
            phase_end = data["phase_end"].copy()
            tau_end = float(data["total_tau_end"])
            n_periods = int(data["total_extra_periods_end"])
            status = json.loads(str(data["result_json"]))
        chunk_count += 1
        state_start = logs[0, :, 1:7].reshape(-1)
        state_err = float(np.max(np.abs(state_start - prev_state)))
        theta_err = float(np.max(np.abs(logs[0, :, 7] - prev_theta)))
        energy_err = abs(float(work[0, 1]) - prev_energy)
        max_state_boundary_error = max(max_state_boundary_error, state_err)
        max_theta_boundary_error = max(max_theta_boundary_error, theta_err)
        max_energy_boundary_error = max(max_energy_boundary_error, energy_err)
        if state_err > 2e-12:
            errors.append(f"state discontinuity chunk {k}: {state_err:.3g}")
        if theta_err > 2e-10:
            errors.append(f"unwrapped-angle discontinuity chunk {k}: {theta_err:.3g}")
        if energy_err > 2e-8:
            errors.append(f"energy discontinuity chunk {k}: {energy_err:.3g}")
        expected_phase = np.asarray([0.0, 0.0])
        # Read the source RF phases and advance both carriers over the base plus
        # all completed primary periods. No phase reset occurs at a chunk edge.
        with np.load(base_path) as data:
            initial_phase = data["phases"].copy()
        expected_phase = np.mod(initial_phase + np.asarray([1.0, nu]) * tau_end, TAU)
        phase_err = float(np.max(np.abs(wrap_error(phase_end, expected_phase))))
        max_phase_error = max(max_phase_error, phase_err)
        if phase_err > 2e-9:
            errors.append(f"carrier phase mismatch chunk {k}: {phase_err:.3g} rad")
        if len(logs) != 15_001 or not status["retained"] or status["completed_periods"] != 15_000:
            errors.append(f"incomplete/failed integration chunk {k}")
        if len(work) != 15_001:
            errors.append(f"incomplete energy trace chunk {k}")
        chunk_metrics.append(status)
        prev_state = state_end
        prev_theta = logs[-1, :, 7].copy()
        prev_energy = float(work[-1, 1])

    full_path = task_dir / "full_120k_trajectory.npz"
    with np.load(full_path) as data:
        logs = data["trajectory"].copy()
        energy = data["energy_balance"].copy()
        full_skip = int(data["post_load_skip"])
    expected_cycles = 120_000 + full_skip
    if logs.shape[0] != expected_cycles + 1:
        errors.append(f"full trace length {logs.shape[0]} != {expected_cycles+1}")
    if progress["status"] != "complete" or progress["completed_extension_periods"] != 105_000:
        errors.append("extension checkpoint does not show full completion")
    if chunk_count != 7:
        errors.append(f"expected 7 chunks, found {chunk_count}")

    # Verify all six individual orbit directions in each of three contiguous
    # post-loading windows, not only the group-average sign.
    post_theta = logs[full_skip:, :, 7]
    particle_turns_by_window = []
    for w in range(3):
        lo = w * (len(post_theta) - 1) // 3
        hi = (w + 1) * (len(post_theta) - 1) // 3
        turns = (post_theta[hi] - post_theta[lo]) / TAU
        particle_turns_by_window.append(turns.tolist())
    signs = np.sign(np.asarray(particle_turns_by_window))
    individual_direction_pass = bool(
        np.all(signs[:, :3] == signs[0, 0])
        and np.all(signs[:, 3:] == signs[0, 3])
        and signs[0, 0] * signs[0, 3] < 0
    )
    if not individual_direction_pass:
        errors.append("individual particle direction does not remain counterrotating in all three windows")

    # Late-half structure tests distinguish eventual phase locking from the
    # early transient included in the full-horizon percentile.
    late_start = full_skip + int(0.5 * (len(logs) - full_skip))
    late_metrics = structural_metrics(logs[late_start:], 0)
    full_metrics = summary["metrics_full_trajectory"]
    groups_ok = []
    for g in late_metrics["groups"]:
        group_ok = bool(
            g["relative_frequency_spread"] < 0.01
            and g["mean_Psi3"] >= 0.9
            and g["p05_Psi3"] >= 0.9
            and g["late_max_abs_gap_error_deg"] <= 10.0
            and max(abs(x["linear_drift_deg_per_1000_periods"]) for x in g["relative_phase_late_diagnostics"]) <= 0.25
            and max(x["detrended_range_deg"] for x in g["relative_phase_late_diagnostics"]) <= 20.0
        )
        groups_ok.append(group_ok)
    if not all(groups_ok):
        errors.append("late-half crystal structure fails one or more stated thresholds")
    max_relative_energy = float(np.max(np.abs(energy[:, 4])) / max(abs(energy[-1, 2]), 1e-30))
    if max_relative_energy > 5e-5:
        errors.append(f"energy residual exceeds audit threshold: {max_relative_energy:.3g}")
    if not summary["retained_through_extension"]:
        errors.append("summary says the full extension was not retained")

    return {
        "task_id": task,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "full_trace_sha256": sha(full_path),
        "full_trace_shape": list(logs.shape),
        "chunk_count": chunk_count,
        "max_state_boundary_error": max_state_boundary_error,
        "max_unwrapped_angle_boundary_error_rad": max_theta_boundary_error,
        "max_total_energy_boundary_error": max_energy_boundary_error,
        "max_continuous_rf_phase_error_rad": max_phase_error,
        "three_window_turns_each": particle_turns_by_window,
        "individual_particle_directions_pass": individual_direction_pass,
        "late_half_structure_pass_by_group": groups_ok,
        "full_horizon_metrics": full_metrics,
        "late_half_metrics": late_metrics,
        "max_relative_energy_residual_vs_rf_work": max_relative_energy,
        "minimum_pair_distances_R_i_lt_j": summary["minimum_pair_distances_R_i_lt_j_full_horizon"],
        "minimum_radius_R_each": summary["minimum_radius_R_each_full_horizon"],
        "maximum_radius_R_each": summary["maximum_radius_R_each_full_horizon"],
        "maximum_abs_z_R_each": summary["maximum_abs_z_R_each_full_horizon"],
        "mean_group_center_z_R": [g["center_z_mean_R"] for g in full_metrics["groups"]],
    }


def main() -> None:
    params = json.loads((HERE / "local15_n6_parameters.json").read_text(encoding="utf-8"))
    nu = float(params["field"]["nu"])
    results = [check_task(task, nu) for task in TASKS]
    result = {
        "status": "passed" if all(row["status"] == "passed" for row in results) else "failed",
        "n_representative_extensions": len(results),
        "target_total_postload_periods": 120_000,
        "steps_per_period": 96,
        "phase_continuity_checked": True,
        "all_six_individual_directions_checked": True,
        "late_half_crystal_thresholds": {
            "relative_frequency_spread_max": 0.01,
            "mean_and_p05_Psi3_min": 0.9,
            "late_cyclic_gap_error_max_deg": 10.0,
            "relative_phase_drift_max_deg_per_1000_periods": 0.25,
            "detrended_phase_range_max_deg": 20.0,
        },
        "results": results,
    }
    (OUT / "extension_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "results": [{"task": r["task_id"], "status": r["status"], "errors": r["errors"], "p05_late_Psi3": [round(g["p05_Psi3"], 4) for g in r["late_half_metrics"]["groups"]], "energy": r["max_relative_energy_residual_vs_rf_work"]} for r in results]}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
