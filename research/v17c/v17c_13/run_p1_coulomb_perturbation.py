"""Compare collective phase recovery at C=0 and C=C0 from perturbed triangles."""
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
from run_p1_long_candidates import structural_metrics

HERE = Path(__file__).resolve().parent
PILOT = HERE / "p1_ordered_pilot" / "pilot_runs.jsonl"
OUT = HERE / "p1_coulomb_perturbation"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
PERTURB_DEG = (10.0, 20.0)
COUPLING_MULTIPLIER = (0.0, 1.0)
POST_LOAD_PERIODS = 15_000
RELAX = 200
STEPS = 96
WORKERS = 4


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def perturbed_state(initial: np.ndarray, degrees: float) -> np.ndarray:
    state = initial.copy().reshape((6, 6))
    delta = math.radians(degrees)
    # Rotate one member of each triangle in opposite directions; preserve radii,
    # z, and zero velocity, and do not add any artificial mechanical impulse.
    for i, sign in ((1, 1.0), (4, -1.0)):
        x, y = state[i, 0], state[i, 1]
        angle = math.atan2(y, x) + sign * delta
        radius = math.hypot(x, y)
        state[i, 0], state[i, 1] = radius * math.cos(angle), radius * math.sin(angle)
    return state.ravel()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p0, family, modes, coeff, qratio = nb.load_local15()
    protocols = nb.load_protocols()
    pilot = {r["task_id"]: r for r in
             (json.loads(s) for s in PILOT.read_text().splitlines() if s.strip())}
    source = pilot["P2_505_angle0_phase0"]
    base = np.asarray(source["initial_state"], dtype=np.float64)
    phase = np.asarray(source["rf_phase_rad"], dtype=np.float64)
    radius_by_group = np.asarray([0.0002, 0.00189], dtype=np.float64)
    radii = radius_by_group[GROUPS]
    protocol = protocols["P2_505"]
    skip = int(math.ceil(nb.ld.protocol_duration_cycles(protocol))) + RELAX
    total = skip + POST_LOAD_PERIODS
    tasks = []
    for degrees in PERTURB_DEG:
        for factor in COUPLING_MULTIPLIER:
            state = perturbed_state(base, degrees)
            tasks.append((f"angle{degrees:g}_C{factor:g}", degrees, factor, state))
    meta = {
        "status": "running_or_resumable", "source_start": "P2_505_angle0_phase0",
        "angle_perturbations_deg": list(PERTURB_DEG), "C_multipliers": list(COUPLING_MULTIPLIER),
        "steps_per_period": STEPS, "post_load_periods": POST_LOAD_PERIODS,
        "initial_states_sha256": sha(np.concatenate([x[3] for x in tasks])),
        "protocol": "fixed P2_505 with continuous RF phases; same for all cases",
        "workers": WORKERS,
    }
    (OUT / "perturbation_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    path = OUT / "perturbation_runs.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if (OUT / f"{row['task_id']}_trajectory.npz").exists():
                    done.add(row["task_id"])
    todo = [x for x in tasks if x[0] not in done]

    def simulate(task):
        task_id, degrees, factor, initial = task
        p = p0.copy()
        p[5] = p0[5] * factor
        tic = time.perf_counter()
        result = nb.integrate(initial, phase, p, protocol, GROUPS, family, modes, coeff,
                              qratio, radii, total, STEPS, skip, True, True)
        elapsed = time.perf_counter() - tic
        logs, work = result[13], result[14]
        metric = structural_metrics(logs, skip)
        row = {
            "task_id": task_id, "initial_angle_perturbation_deg": degrees,
            "C_multiplier": factor, "C_absolute": float(p[5]),
            "initial_state": initial.tolist(), "rf_phase_rad": phase.tolist(),
            "completed_periods": int(result[9]), "retained": bool(result[10]),
            "category": str(result[11]), "failure_status": int(result[12]),
            "turns_each_after_load": np.asarray(result[1]).tolist(),
            "frequencies_Hz_each": np.asarray(result[3]).tolist(),
            "r_min_R_each": np.asarray(result[4]).tolist(), "r_max_R_each": np.asarray(result[5]).tolist(),
            "minimum_pair_distances_R_i_lt_j": np.asarray(result[8]).tolist(),
            "structure_after_load": metric,
            "energy_balance": {
                "RF_work": float(work[-1, 2]), "drag_work": float(work[-1, 3]),
                "final_residual": float(work[-1, 4]),
                "relative_max_residual_vs_RF_work": float(np.max(np.abs(work[:, 4])) / max(abs(work[-1, 2]), 1e-30)),
            },
            "elapsed_seconds": elapsed,
        }
        return row, logs, work

    with ThreadPoolExecutor(max_workers=WORKERS) as pool, path.open("a", encoding="utf-8", buffering=1) as f:
        futures = {pool.submit(simulate, task): task for task in todo}
        for future in as_completed(futures):
            task = futures[future]
            row, logs, work = future.result()
            tmp_path = OUT / f".{task[0]}_trajectory.npz.tmp"
            with tmp_path.open("wb") as tmp:
                np.savez_compressed(tmp, trajectory=logs, energy_balance=work,
                                    start=row["initial_state"], phases=phase, groups=GROUPS,
                                    post_load_skip=skip)
                tmp.flush(); os.fsync(tmp.fileno())
            tmp_path.replace(OUT / f"{task[0]}_trajectory.npz")
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush(); os.fsync(f.fileno())
            metric = row["structure_after_load"]
            print(task_id := row["task_id"], row["category"], "retained", row["retained"],
                  "turns", np.round(row["turns_each_after_load"], 2).tolist(),
                  "Psi", [round(g["mean_Psi3"], 3) for g in metric["groups"]],
                  "phase ranges", [g["late_relative_phase_range_deg"] for g in metric["groups"]],
                  f"{row['elapsed_seconds']:.1f}s", flush=True)
    meta["status"] = "complete"
    (OUT / "perturbation_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
