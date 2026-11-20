"""Small resumable P1 scan of symmetric 3+3 rings from rest."""
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
OUT = HERE / "p1_ordered_pilot"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
RING_RADII = (0.45, 0.63)
RING_ZS = (0.0, 0.1)
RELATIVE_ANGLES = (0.0, math.pi / 3.0)
RF_PHASES = ((0.0, 0.0), (0.7, 2.1))
PROTOCOL_NAMES = ("constant", "P2_505", "P2_353", "P7_292")
PERIODS = 3_000
STEPS = 96
RELAX_AFTER_LOAD = 200
SEED = 20261313


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def structural_metrics(logs: np.ndarray, groups: np.ndarray, skip: int) -> dict:
    # logs: period x particle x [tau,x,y,z,vx,vy,vz,theta_unwrapped]
    if logs.shape[0] <= skip:
        return {"groups": [], "opposite_group_frequency_signs": None,
                "note": "contact or loss occurred before the post-load analysis window"}
    part = logs[skip:, :, :]
    theta = part[:, :, 7]
    radius = np.hypot(part[:, :, 1], part[:, :, 2])
    z = part[:, :, 3]
    output = {"groups": []}
    for g in (0, 1):
        indices = np.flatnonzero(groups == g)
        tg = theta[:, indices]
        rg = radius[:, indices]
        zg = z[:, indices]
        tspan = max(1, len(part) - 1)
        mean_turns = (tg[-1] - tg[0]) / (2.0 * math.pi)
        frequencies = mean_turns * nb.F1_HZ / tspan
        freq_mean = float(np.mean(frequencies))
        freq_spread = float(np.std(frequencies) / max(abs(freq_mean), 1e-15))
        relative = tg - tg[:, :1]
        rel_late = relative[len(relative) // 3:]
        phase_range = np.ptp(rel_late, axis=0)
        psi3 = np.abs(np.mean(np.exp(3j * tg), axis=1))
        gaps = []
        for angles in tg:
            ordered = np.sort(np.mod(angles, 2.0 * math.pi))
            gaps.append(np.diff(np.r_[ordered, ordered[0] + 2.0 * math.pi]))
        gaps = np.asarray(gaps)
        distances = []
        for i in range(3):
            for j in range(i + 1, 3):
                dx = part[:, indices[i], 1] - part[:, indices[j], 1]
                dy = part[:, indices[i], 2] - part[:, indices[j], 2]
                dz = part[:, indices[i], 3] - part[:, indices[j], 3]
                distances.append(np.sqrt(dx * dx + dy * dy + dz * dz))
        output["groups"].append({
            "group": g,
            "mean_frequencies_Hz": frequencies.tolist(),
            "mean_frequency_Hz": freq_mean,
            "relative_frequency_spread": freq_spread,
            "mean_turns_each": mean_turns.tolist(),
            "mean_radius_R": np.mean(rg, axis=0).tolist(),
            "mean_z_R": np.mean(zg, axis=0).tolist(),
            "mean_center_z_R": float(np.mean(np.mean(zg, axis=1))),
            "within_group_z_rms_R": float(np.sqrt(np.mean((zg - np.mean(zg, axis=1, keepdims=True)) ** 2))),
            "mean_Psi3": float(np.mean(psi3)),
            "p05_Psi3": float(np.quantile(psi3, 0.05)),
            "mean_cyclic_gaps_deg": np.rad2deg(np.mean(gaps, axis=0)).tolist(),
            "max_abs_gap_error_from_120_deg": float(np.max(np.abs(np.rad2deg(gaps) - 120.0))),
            "late_relative_phase_range_deg": np.rad2deg(phase_range).tolist(),
            "minimum_internal_pair_distance_R": float(min(np.min(d) for d in distances)),
        })
    output["opposite_group_frequency_signs"] = output["groups"][0]["mean_frequency_Hz"] * output["groups"][1]["mean_frequency_Hz"] < 0.0
    return output


def simulate(task, p, protocols, family, modes, coeff, qratio, particle_radii):
    task_id, pname, angle_idx, relative_angle, phase_idx, phase = task
    initial, _ = nb.initialize_rings(RING_RADII, RING_ZS, relative_angle, groups=GROUPS)
    protocol = protocols[pname]
    duration = nb.ld.protocol_duration_cycles(protocol)
    skip = int(math.ceil(duration)) + RELAX_AFTER_LOAD
    start = time.perf_counter()
    result = nb.integrate(initial, np.asarray(phase, dtype=np.float64), p, protocol,
                          GROUPS, family, modes, coeff, qratio, particle_radii,
                          PERIODS, STEPS, skip, True)
    elapsed = time.perf_counter() - start
    logs = result[13]
    metrics = structural_metrics(logs, GROUPS, skip)
    run = {
        "task_id": task_id, "protocol": pname, "relative_angle_rad": relative_angle,
        "rf_phase_rad": list(phase), "initial_state": initial.tolist(),
        "completed_periods": int(result[9]), "retained": bool(result[10]),
        "category": str(result[11]), "failure_status": int(result[12]),
        "turns_each": np.asarray(result[1]).tolist(),
        "late_window_turns": np.asarray(result[2]).tolist(),
        "frequency_Hz_each": np.asarray(result[3]).tolist(),
        "r_min_R_each": np.asarray(result[4]).tolist(),
        "r_max_R_each": np.asarray(result[5]).tolist(),
        "max_abs_z_R_each": np.asarray(result[6]).tolist(),
        "minimum_pair_distances_R": np.asarray(result[8]).tolist(),
        "structural_metrics_after_load": metrics,
        "elapsed_seconds": elapsed,
    }
    return run, logs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p, family, modes, coeff, qratio = nb.load_local15()
    protocols = nb.load_protocols()
    radii_by_group = np.asarray([0.0002, 0.00189], dtype=np.float64)
    particle_radii = radii_by_group[GROUPS]
    metadata = {
        "status": "screening_only",
        "seed": SEED,
        "periods": PERIODS,
        "steps_per_period": STEPS,
        "groups": GROUPS.tolist(),
        "ring_radii_R": list(RING_RADII),
        "ring_z_centers_R": list(RING_ZS),
        "relative_angles_rad": list(RELATIVE_ANGLES),
        "rf_phase_sets_rad": [list(x) for x in RF_PHASES],
        "protocols": list(PROTOCOL_NAMES),
        "warning": "P1 pilot is too short for strict 10-turn validation and is not an attractor claim",
    }
    tasks = []
    for pname in PROTOCOL_NAMES:
        for angle_idx, relative_angle in enumerate(RELATIVE_ANGLES):
            for phase_idx, phase in enumerate(RF_PHASES):
                task_id = f"{pname}_angle{angle_idx}_phase{phase_idx}"
                tasks.append((task_id, pname, angle_idx, relative_angle, phase_idx, phase))
    frozen = []
    for _, _, _, rel, _, phase in tasks:
        frozen.extend([nb.initialize_rings(RING_RADII, RING_ZS, rel, groups=GROUPS)[0],
                       np.asarray(phase, dtype=np.float64)])
    metadata["start_and_phase_hash"] = digest(np.concatenate(frozen))
    (OUT / "pilot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    out_jsonl = OUT / "pilot_runs.jsonl"
    done = set()
    if out_jsonl.exists():
        with out_jsonl.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["task_id"])
    todo = [t for t in tasks if t[0] not in done]
    with ThreadPoolExecutor(max_workers=4) as pool, out_jsonl.open("a", encoding="utf-8", buffering=1) as f:
        futures = {pool.submit(simulate, task, p, protocols, family, modes, coeff, qratio, particle_radii): task
                   for task in todo}
        for future in as_completed(futures):
            task = futures[future]
            run, logs = future.result()
            task_id = task[0]
            f.write(json.dumps(run, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
            np.savez_compressed(OUT / f"{task_id}_trajectory.npz", time_tau=logs[:, 0, 0],
                                trajectory=logs, groups=GROUPS, start=run["initial_state"],
                                phases=np.asarray(task[5]))
            metrics = run["structural_metrics_after_load"]
            print(task_id, run["category"], "retained", run["retained"],
                  "turns", np.round(run["turns_each"], 3).tolist(),
                  "Psi3", [round(x["mean_Psi3"], 3) for x in metrics["groups"]],
                  f"{run['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
