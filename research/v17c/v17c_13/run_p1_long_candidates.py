"""Resumable 15k-post-load-cycle verification of the strongest N=6 starts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import nbody_dynamics as nb

HERE = Path(__file__).resolve().parent
PILOT = HERE / "p1_ordered_pilot" / "pilot_runs.jsonl"
OUT = HERE / "p1_long_candidates"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
STEPS = 96
RELAX_AFTER_LOAD = 200
POST_LOAD_PERIODS = 15_000
WORKERS = 4


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def structural_metrics(logs: np.ndarray, skip: int) -> dict:
    # Main-RF-period stroboscopic data: [tau,x,y,z,vx,vy,vz,theta_unwrapped].
    if logs.shape[0] <= skip + 1:
        return {"groups": [], "opposite_group_frequency_signs": None}
    part = logs[skip:, :, :]
    theta = part[:, :, 7]
    radius = np.hypot(part[:, :, 1], part[:, :, 2])
    z = part[:, :, 3]
    result = {"groups": []}
    for g in (0, 1):
        idx = np.flatnonzero(GROUPS == g)
        th = theta[:, idx]
        rr = radius[:, idx]
        zz = z[:, idx]
        nperiod = max(1, len(part) - 1)
        turn = (th[-1] - th[0]) / (2.0 * math.pi)
        freq = turn * nb.F1_HZ / nperiod
        mean_freq = float(np.mean(freq))
        freq_spread = float(np.std(freq) / max(abs(mean_freq), 1.0e-30))

        # Relative phases are differences of continuous unwrapped angles.
        rel = th - th[:, :1]
        late_start = int(0.5 * len(rel))
        x = np.arange(len(rel) - late_start, dtype=np.float64)
        phase_rows = []
        for j in range(3):
            values = rel[late_start:, j]
            if j == 0:
                slope, intercept = 0.0, float(values[0])
                residual = values - intercept
            else:
                slope, intercept = np.polyfit(x, values, 1)
                residual = values - (slope * x + intercept)
            phase_rows.append({
                "linear_drift_deg_per_1000_periods": float(slope * 1000.0 * 180.0 / math.pi),
                "detrended_range_deg": float(np.ptp(residual) * 180.0 / math.pi),
                "raw_range_deg": float(np.ptp(values) * 180.0 / math.pi),
            })

        psi = np.abs(np.mean(np.exp(3j * th), axis=1))
        gaps = []
        for row in th:
            angles = np.sort(np.mod(row, 2.0 * math.pi))
            gaps.append(np.diff(np.r_[angles, angles[0] + 2.0 * math.pi]))
        gaps = np.asarray(gaps)
        late_gaps = gaps[len(gaps) // 2:]
        pair_ids = ((0, 1), (0, 2), (1, 2)) if g == 0 else ((3, 4), (3, 5), (4, 5))
        distances = []
        for i, j in pair_ids:
            delta = part[:, i, 1:4] - part[:, j, 1:4]
            distances.append(np.linalg.norm(delta, axis=1))
        result["groups"].append({
            "particle_indices": idx.tolist(),
            "mean_frequencies_Hz": freq.tolist(),
            "mean_frequency_Hz": mean_freq,
            "relative_frequency_spread": freq_spread,
            "turns_each_after_load": turn.tolist(),
            "minimum_turns_abs": float(np.min(np.abs(turn))),
            "mean_radius_R_each": np.mean(rr, axis=0).tolist(),
            "radius_5_95_percentile_R": np.quantile(rr, [0.05, 0.95], axis=0).tolist(),
            "mean_z_R_each": np.mean(zz, axis=0).tolist(),
            "center_z_mean_R": float(np.mean(np.mean(zz, axis=1))),
            "within_group_z_rms_R": float(np.sqrt(np.mean((zz - np.mean(zz, axis=1, keepdims=True)) ** 2))),
            "mean_Psi3": float(np.mean(psi)),
            "p05_Psi3": float(np.quantile(psi, 0.05)),
            "late_mean_cyclic_gaps_deg": np.rad2deg(np.mean(late_gaps, axis=0)).tolist(),
            "late_max_abs_gap_error_deg": float(np.max(np.abs(np.rad2deg(late_gaps) - 120.0))),
            "relative_phase_late_diagnostics": phase_rows,
            "minimum_internal_pair_distances_R": [float(np.min(d)) for d in distances],
        })
    means = [g["mean_frequency_Hz"] for g in result["groups"]]
    result["opposite_group_frequency_signs"] = means[0] * means[1] < 0.0
    result["all_particles_at_least_10_turns"] = all(g["minimum_turns_abs"] >= 10.0 for g in result["groups"])
    result["late_window_group_signs"] = []
    for w in range(3):
        lo = skip + (len(logs) - 1 - skip) * w // 3
        hi = skip + (len(logs) - 1 - skip) * (w + 1) // 3
        dt_turns = (theta[hi - skip] - theta[lo - skip]) / (2.0 * math.pi)
        result["late_window_group_signs"].append([
            float(np.mean(dt_turns[:3])), float(np.mean(dt_turns[3:]))])
    result["directions_constant_in_three_windows"] = all(a[0] * a[1] < 0 for a in result["late_window_group_signs"])
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p, family, modes, coeff, qratio = nb.load_local15()
    protocols = nb.load_protocols()
    radii_by_group = np.asarray([0.0002, 0.00189], dtype=np.float64)
    particle_radii = radii_by_group[GROUPS]
    pilot = {r["task_id"]: r for r in (json.loads(x) for x in PILOT.read_text().splitlines() if x.strip())}
    selected = [
        "P2_505_angle0_phase0", "P2_505_angle0_phase1",
        "P2_505_angle1_phase0", "P2_505_angle1_phase1",
        "constant_angle0_phase0", "P2_353_angle0_phase0", "P7_292_angle0_phase0",
    ]
    tasks = []
    starts = []
    for source_id in selected:
        source = pilot[source_id]
        protocol_name = source["protocol"]
        protocol = protocols[protocol_name]
        skip = int(math.ceil(nb.ld.protocol_duration_cycles(protocol))) + RELAX_AFTER_LOAD
        total_periods = skip + POST_LOAD_PERIODS
        task_id = source_id + "_long"
        tasks.append((task_id, source_id, protocol_name, np.asarray(source["initial_state"], dtype=np.float64),
                      np.asarray(source["rf_phase_rad"], dtype=np.float64), skip, total_periods))
        starts.extend([np.asarray(source["initial_state"], dtype=np.float64),
                       np.asarray(source["rf_phase_rad"], dtype=np.float64)])

    metadata = {
        "status": "running_or_resumable",
        "purpose": "long post-load verification of P1 ordered candidates; stroboscopic at primary RF period",
        "post_load_primary_RF_periods": POST_LOAD_PERIODS,
        "steps_per_primary_period": STEPS,
        "workers": WORKERS,
        "selected_source_task_ids": selected,
        "start_and_phase_sha256": digest(np.concatenate(starts)),
        "fixed_local15_parameters": "../local15_n6_parameters.json",
        "energy_balance": "RK4 stage-integrated explicit RF power and drag after completed loading, with total kinetic+RF+DC+Coulomb energy",
    }
    (OUT / "long_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    out_jsonl = OUT / "long_runs.jsonl"
    done = set()
    if out_jsonl.exists():
        for line in out_jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if (OUT / f"{row['task_id']}_trajectory.npz").exists():
                    done.add(row["task_id"])
    todo = [t for t in tasks if t[0] not in done]

    def simulate(task):
        task_id, source_id, pname, initial, phase, skip, total = task
        tic = time.perf_counter()
        result = nb.integrate(initial, phase, p, protocols[pname], GROUPS, family, modes, coeff,
                              qratio, particle_radii, total, STEPS, skip, True, True)
        elapsed = time.perf_counter() - tic
        logs, work = result[13], result[14]
        metrics = structural_metrics(logs, skip)
        row = {
            "task_id": task_id, "source_pilot_task_id": source_id, "protocol": pname,
            "initial_state": initial.tolist(), "rf_phase_rad": phase.tolist(),
            "total_periods_requested": total, "completed_periods": int(result[9]),
            "post_load_periods_observed": max(0, int(result[9]) - skip),
            "steps_per_period": STEPS, "retained": bool(result[10]),
            "category": str(result[11]), "failure_status": int(result[12]),
            "turns_each_after_load": np.asarray(result[1]).tolist(),
            "late_window_turns_each": np.asarray(result[2]).tolist(),
            "frequencies_Hz_each": np.asarray(result[3]).tolist(),
            "r_min_R_each": np.asarray(result[4]).tolist(), "r_max_R_each": np.asarray(result[5]).tolist(),
            "max_abs_z_R_each": np.asarray(result[6]).tolist(),
            "minimum_pair_distances_R_i_lt_j": np.asarray(result[8]).tolist(),
            "structure_after_load": metrics,
            "energy_balance": {
                "samples": int(len(work)),
                "initial_total_energy": float(work[0, 1]) if len(work) else None,
                "final_total_energy": float(work[-1, 1]) if len(work) else None,
                "integrated_RF_explicit_work": float(work[-1, 2]) if len(work) else None,
                "integrated_drag_work": float(work[-1, 3]) if len(work) else None,
                "final_balance_residual": float(work[-1, 4]) if len(work) else None,
                "max_abs_balance_residual": float(np.max(np.abs(work[:, 4]))) if len(work) else None,
                "relative_max_residual_vs_RF_work": float(np.max(np.abs(work[:, 4])) / max(abs(work[-1, 2]), 1.0e-30)) if len(work) else None,
            },
            "elapsed_seconds": elapsed,
        }
        return row, logs, work

    with ThreadPoolExecutor(max_workers=WORKERS) as pool, out_jsonl.open("a", encoding="utf-8", buffering=1) as f:
        futures = {pool.submit(simulate, task): task for task in todo}
        for future in as_completed(futures):
            task = futures[future]
            row, logs, work = future.result()
            tmp_path = OUT / f".{task[0]}_trajectory.npz.tmp"
            with tmp_path.open("wb") as tmp:
                np.savez_compressed(tmp, trajectory=logs, energy_balance=work, groups=GROUPS,
                                    start=row["initial_state"], phases=row["rf_phase_rad"],
                                    post_load_skip=task[5])
                tmp.flush(); os.fsync(tmp.fileno())
            tmp_path.replace(OUT / f"{task[0]}_trajectory.npz")
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush(); os.fsync(f.fileno())
            print(task[0], row["category"], "retained", row["retained"],
                  "turns", np.round(row["turns_each_after_load"], 2).tolist(),
                  "freq", np.round(row["frequencies_Hz_each"], 3).tolist(),
                  "Psi", [round(g["mean_Psi3"], 4) for g in row["structure_after_load"]["groups"]],
                  "energy residual", row["energy_balance"]["final_balance_residual"],
                  f"{row['elapsed_seconds']:.1f}s", flush=True)
    metadata["status"] = "complete" if len(done) + len(todo) == len(tasks) else "partial_resumable"
    (OUT / "long_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
