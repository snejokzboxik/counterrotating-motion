"""Resumable extension of representative N=6 formation runs to 120k cycles.

Each extension chunk starts from the exact prior endpoint. RF carrier phases are
advanced continuously, while the loading envelopes are replaced by the constant
local15 protocol after the already completed loading schedule.
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import nbody_dynamics as nb
from run_p1_long_candidates import structural_metrics

HERE = Path(__file__).resolve().parent
BASE = HERE / "p1_formation_ensembles"
OUT = HERE / "p1_formation_extensions"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
STEPS = 96
CHUNK_PERIODS = 15_000
EXTENSION_PERIODS = 105_000
WORKERS = 4
ALL_SELECTED = [
    "ordered_003",                 # positive control: strict crystal
    "random_azimuth_010",          # near-ordered random-angle example, phase slip
    "random_position_rest_004",    # unstructured zero-velocity start, phase slip
    "random_position_rest_015",    # unstructured start, ring exchange
    "random_position_moving_006",  # moving random start, retained phase slip
]
SELECTED = [x for x in os.environ.get("V17C_EXTENSION_TASKS", ",".join(ALL_SELECTED)).split(",") if x]


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_npz_atomic(path: Path, **arrays) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_rows() -> dict[str, dict]:
    rows = [json.loads(line) for line in (BASE / "formation_runs.jsonl").read_text().splitlines() if line.strip()]
    return {row["task_id"]: row for row in rows}


def task_paths(task_id: str) -> tuple[Path, Path, Path]:
    d = OUT / task_id
    return d, d / "progress.json", d / "state.npz"


def initialize(task_id: str, row: dict) -> dict:
    task_dir, progress_path, state_path = task_paths(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    base_path = BASE / f"{task_id}_trajectory.npz"
    with np.load(base_path) as source:
        trajectory = source["trajectory"]
        base_work = source["energy_balance"]
        skip = int(source["post_load_skip"])
    total_tau = float(trajectory[-1, 0, 0])
    nu = float(nb.load_local15()[0][2])
    phase = np.asarray(row["rf_phases_rad"], dtype=np.float64)
    phase = np.mod(phase + np.asarray([total_tau, nu * total_tau]), 2.0 * math.pi)
    save_npz_atomic(state_path,
                    state=trajectory[-1, :, 1:7].reshape(-1),
                    phase=phase,
                    last_theta=trajectory[-1, :, 7],
                    total_tau=np.asarray(total_tau),
                    total_extra_periods=np.asarray(0, dtype=np.int64),
                    rf_work_offset=np.asarray(float(base_work[-1, 2])),
                    drag_work_offset=np.asarray(float(base_work[-1, 3])))
    progress = {
        "task_id": task_id,
        "status": "running_resumable",
        "base_category": row["category"],
        "base_completed_periods": row["completed_periods"],
        "base_post_load_periods": row["post_load_periods_observed"],
        "base_post_load_skip": skip,
        "steps_per_period": STEPS,
        "chunk_periods": CHUNK_PERIODS,
        "extension_periods_target": EXTENSION_PERIODS,
        "completed_extension_periods": 0,
        "continuous_rf_phases_at_checkpoint_rad": phase.tolist(),
        "base_endpoint_tau": total_tau,
        "chunk_indices_complete": [],
    }
    atomic_json(progress_path, progress)
    return progress


def run_one(task_id: str, row: dict, p: np.ndarray, family: np.ndarray,
            modes: np.ndarray, coeff: np.ndarray, qratio: float,
            particle_radii: np.ndarray, protocol: np.ndarray) -> dict:
    task_dir, progress_path, state_path = task_paths(task_id)
    if not progress_path.exists() or not state_path.exists():
        progress = initialize(task_id, row)
    else:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    with np.load(state_path) as saved:
        state = saved["state"].copy()
        phase = saved["phase"].copy()
        last_theta = saved["last_theta"].copy()
        total_tau = float(saved["total_tau"])
        completed = int(saved["total_extra_periods"])
        rf_offset = float(saved["rf_work_offset"])
        drag_offset = float(saved["drag_work_offset"])
    chunk_index = completed // CHUNK_PERIODS
    target_chunks = EXTENSION_PERIODS // CHUNK_PERIODS
    segment_results = []
    start_clock = time.perf_counter()

    while chunk_index < target_chunks:
        chunk_path = task_dir / f"chunk_{chunk_index + 1:02d}.npz"
        if chunk_path.exists():
            with np.load(chunk_path) as saved:
                chunk_logs = saved["trajectory"].copy()
                chunk_work = saved["energy_balance"].copy()
                state = saved["state_end"].copy()
                phase = saved["phase_end"].copy()
                last_theta = saved["last_theta_end"].copy()
                total_tau = float(saved["total_tau_end"])
                completed = int(saved["total_extra_periods_end"])
                rf_offset = float(saved["rf_work_offset_end"])
                drag_offset = float(saved["drag_work_offset_end"])
                segment_results.append(json.loads(str(saved["result_json"])))
            chunk_index += 1
            continue

        tic = time.perf_counter()
        result = nb.integrate(state, phase, p, protocol, GROUPS, family, modes, coeff,
                              qratio, particle_radii, CHUNK_PERIODS, STEPS, 0, True, True)
        elapsed = time.perf_counter() - tic
        logs = result[13]
        work = result[14]
        # Preserve continuous unwrapped angles across chunk boundaries.
        logs[:, :, 7] += last_theta[None, :] - logs[0, :, 7][None, :]
        state_end = result[0].copy()
        last_theta_end = logs[-1, :, 7].copy()
        duration_tau = float(logs[-1, 0, 0])
        phase_end = np.mod(phase + np.asarray([duration_tau, p[2] * duration_tau]), 2.0 * math.pi)
        total_tau_end = total_tau + duration_tau
        completed_end = completed + int(result[9])
        rf_offset_end = rf_offset + (float(work[-1, 2]) if len(work) else 0.0)
        drag_offset_end = drag_offset + (float(work[-1, 3]) if len(work) else 0.0)
        status = {
            "chunk_index": chunk_index + 1,
            "requested_periods": CHUNK_PERIODS,
            "completed_periods": int(result[9]),
            "retained": bool(result[10]),
            "category": str(result[11]),
            "failure_status": int(result[12]),
            "minimum_pair_distances_R": np.asarray(result[8]).tolist(),
            "min_radius_R": np.asarray(result[4]).tolist(),
            "max_radius_R": np.asarray(result[5]).tolist(),
            "max_abs_z_R": np.asarray(result[6]).tolist(),
            "turns_in_chunk": np.asarray(result[1]).tolist(),
            "late_window_turns_in_chunk": np.asarray(result[2]).tolist(),
            "frequencies_Hz_in_chunk": np.asarray(result[3]).tolist(),
            "energy_samples": int(len(work)),
            "energy_final": float(work[-1, 1]) if len(work) else None,
            "rf_work_this_chunk": float(work[-1, 2]) if len(work) else None,
            "drag_work_this_chunk": float(work[-1, 3]) if len(work) else None,
            "max_abs_energy_residual_this_chunk": float(np.max(np.abs(work[:, 4]))) if len(work) else None,
            "relative_energy_residual_vs_rf_work_this_chunk": float(np.max(np.abs(work[:, 4])) / max(abs(work[-1, 2]), 1e-30)) if len(work) else None,
            "elapsed_seconds": elapsed,
        }
        save_npz_atomic(chunk_path,
                        trajectory=logs,
                        energy_balance=work,
                        state_end=state_end,
                        phase_end=phase_end,
                        last_theta_end=last_theta_end,
                        total_tau_end=np.asarray(total_tau_end),
                        total_extra_periods_end=np.asarray(completed_end, dtype=np.int64),
                        rf_work_offset_end=np.asarray(rf_offset_end),
                        drag_work_offset_end=np.asarray(drag_offset_end),
                        result_json=np.asarray(json.dumps(status, separators=(",", ":"))))
        segment_results.append(status)
        state, phase, last_theta = state_end, phase_end, last_theta_end
        total_tau, completed = total_tau_end, completed_end
        rf_offset, drag_offset = rf_offset_end, drag_offset_end
        chunk_index += 1
        progress.update({
            "status": "running_resumable" if status["retained"] else "terminated_on_failure",
            "completed_extension_periods": completed,
            "chunk_indices_complete": list(range(1, chunk_index + 1)),
            "continuous_rf_phases_at_checkpoint_rad": phase.tolist(),
            "current_category": status["category"],
            "last_chunk_seconds": elapsed,
            "elapsed_wall_seconds": time.perf_counter() - start_clock,
        })
        atomic_json(progress_path, progress)
        print(f"{task_id}: extension {completed}/{EXTENSION_PERIODS}; {status['category']}; {elapsed:.1f}s", flush=True)
        if not status["retained"] or status["completed_periods"] < CHUNK_PERIODS:
            break

    progress["status"] = "complete" if completed >= EXTENSION_PERIODS else progress.get("status", "partial_resumable")
    progress["completed_extension_periods"] = completed
    progress["elapsed_wall_seconds"] = time.perf_counter() - start_clock
    progress["segment_results"] = segment_results
    atomic_json(progress_path, progress)
    return progress


def assemble(task_id: str, row: dict, base_work_start: float) -> dict:
    task_dir, progress_path, _ = task_paths(task_id)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    base_path = BASE / f"{task_id}_trajectory.npz"
    with np.load(base_path) as source:
        base_logs = source["trajectory"].copy()
        base_work = source["energy_balance"].copy()
        skip = int(source["post_load_skip"])
        min_pair = np.asarray(row["minimum_pair_distances_R_i_lt_j"], dtype=np.float64)
        min_radius = np.asarray(row["r_min_R_each"], dtype=np.float64)
        max_radius = np.asarray(row["r_max_R_each"], dtype=np.float64)
        max_abs_z = np.asarray(row["max_abs_z_R_each"], dtype=np.float64)
    full_logs = [base_logs]
    full_work = [base_work]
    tau_offset = float(base_logs[-1, 0, 0])
    rf_offset = float(base_work[-1, 2])
    drag_offset = float(base_work[-1, 3])
    segment_rows = []
    for idx in range(1, progress["completed_extension_periods"] // CHUNK_PERIODS + 1):
        with np.load(task_dir / f"chunk_{idx:02d}.npz") as saved:
            logs = saved["trajectory"].copy()
            work = saved["energy_balance"].copy()
            status = json.loads(str(saved["result_json"]))
        min_pair = np.minimum(min_pair, np.asarray(status["minimum_pair_distances_R"], dtype=np.float64))
        min_radius = np.minimum(min_radius, np.asarray(status["min_radius_R"], dtype=np.float64))
        max_radius = np.maximum(max_radius, np.asarray(status["max_radius_R"], dtype=np.float64))
        max_abs_z = np.maximum(max_abs_z, np.asarray(status["max_abs_z_R"], dtype=np.float64))
        if len(logs):
            logs[:, :, 0] += tau_offset
            full_logs.append(logs[1:])
        if len(work):
            work[:, 0] += tau_offset
            work[:, 2] += rf_offset
            work[:, 3] += drag_offset
            work[:, 4] = work[:, 1] - base_work[0, 1] - work[:, 2] - work[:, 3]
            full_work.append(work[1:])
            rf_offset = float(work[-1, 2])
            drag_offset = float(work[-1, 3])
        tau_offset += 2.0 * math.pi * status["completed_periods"]
        segment_rows.append(status)
    logs_all = np.concatenate(full_logs, axis=0)
    work_all = np.concatenate(full_work, axis=0)
    trajectory_path = task_dir / "full_120k_trajectory.npz"
    save_npz_atomic(trajectory_path, trajectory=logs_all, energy_balance=work_all,
                    groups=GROUPS, post_load_skip=np.asarray(skip, dtype=np.int64),
                    start=np.asarray(row["initial_state"], dtype=np.float64),
                    phases=np.asarray(row["rf_phases_rad"], dtype=np.float64))
    metrics = structural_metrics(logs_all, skip)
    summary = {
        "task_id": task_id,
        "ensemble": row["ensemble"],
        "base_outcome": row["category"],
        "retained_through_extension": progress["completed_extension_periods"] >= EXTENSION_PERIODS and all(x["retained"] for x in segment_rows),
        "post_load_periods_total": int(logs_all.shape[0] - 1 - skip),
        "extension_periods_completed": int(progress["completed_extension_periods"]),
        "steps_per_period": STEPS,
        "metrics_full_trajectory": metrics,
        "minimum_pair_distances_R_i_lt_j_full_horizon": min_pair.tolist(),
        "minimum_radius_R_each_full_horizon": min_radius.tolist(),
        "maximum_radius_R_each_full_horizon": max_radius.tolist(),
        "maximum_abs_z_R_each_full_horizon": max_abs_z.tolist(),
        "energy_balance": {
            "samples": int(len(work_all)),
            "initial_postload_total_energy": float(work_all[0, 1]),
            "final_total_energy": float(work_all[-1, 1]),
            "integrated_RF_work": float(work_all[-1, 2]),
            "integrated_drag_work": float(work_all[-1, 3]),
            "final_balance_residual": float(work_all[-1, 4]),
            "max_abs_balance_residual": float(np.max(np.abs(work_all[:, 4]))),
            "relative_max_residual_vs_RF_work": float(np.max(np.abs(work_all[:, 4])) / max(abs(work_all[-1, 2]), 1e-30)),
        },
        "segment_results": segment_rows,
        "full_trajectory_file": str(trajectory_path.name),
        "full_trajectory_bytes": trajectory_path.stat().st_size,
    }
    atomic_json(task_dir / "summary.json", summary)
    progress["status"] = "complete" if summary["retained_through_extension"] else progress["status"]
    progress["final_summary"] = summary
    atomic_json(progress_path, progress)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    p, family, modes, coeff, qratio = nb.load_local15()
    protocols = nb.load_protocols()
    particle_radii = np.asarray([0.0002, 0.00189], dtype=np.float64)[GROUPS]
    meta_path = OUT / "extension_metadata.json"
    metadata = {
        "status": "running_resumable",
        "purpose": "extend frozen formation-ensemble representatives to 120k total post-load periods under constant local15",
        "base_protocol": "P2_505; its 15k post-load endpoint is used as exact state/phase checkpoint",
        "continuation_protocol": "constant local15; no amplitude modulation after pre-existing loading ends",
        "rf_phase_continuity": "phase_j(t0)=phase_j(0)+Omega_j*t0 mod 2pi; chunk endpoint phases advanced using exact integrated chunk duration",
        "total_post_load_periods_target": 120_000,
        "base_post_load_periods": 15_000,
        "extension_periods": EXTENSION_PERIODS,
        "chunk_periods": CHUNK_PERIODS,
        "steps_per_primary_period": STEPS,
        "workers": WORKERS,
        "selected_task_ids": ALL_SELECTED,
        "run_task_ids": SELECTED,
        "stability_note": "representative extensions address persistence; they do not estimate ensemble probabilities",
    }
    atomic_json(meta_path, metadata)
    futures = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for task_id in SELECTED:
            if task_id not in rows:
                raise KeyError(f"Missing source task {task_id}")
            futures[pool.submit(run_one, task_id, rows[task_id], p, family, modes, coeff,
                                qratio, particle_radii, protocols["constant"])] = task_id
        for future in as_completed(futures):
            task_id = futures[future]
            progress = future.result()
            if progress.get("completed_extension_periods", 0) > 0:
                assemble(task_id, rows[task_id], float(p[2]))
            print(f"finished {task_id}: {progress['status']} {progress.get('completed_extension_periods',0)} cycles", flush=True)
    all_complete = all(task_paths(t)[1].exists() and json.loads(task_paths(t)[1].read_text()).get("status") == "complete" for t in ALL_SELECTED)
    metadata["status"] = "complete" if all_complete else "partial_resumable"
    atomic_json(meta_path, metadata)
    print("extension stage", metadata["status"], flush=True)


if __name__ == "__main__":
    main()
