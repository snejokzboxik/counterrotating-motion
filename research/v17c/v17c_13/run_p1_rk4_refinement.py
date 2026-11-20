"""Resumable full-horizon RK4 step refinement of the #505 N=6 candidate."""
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
SOURCE = HERE / "p1_ordered_pilot" / "pilot_runs.jsonl"
OUT = HERE / "p1_rk4_refinement"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
RESOLUTIONS = (64, 128, 256)
POST_LOAD_PERIODS = 15_000
RELAX = 200
WORKERS = 3


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p, family, modes, coeff, qratio = nb.load_local15()
    protocols = nb.load_protocols()
    rows = {r["task_id"]: r for r in
            (json.loads(x) for x in SOURCE.read_text().splitlines() if x.strip())}
    source = rows["P2_505_angle0_phase0"]
    initial = np.asarray(source["initial_state"], dtype=np.float64)
    phase = np.asarray(source["rf_phase_rad"], dtype=np.float64)
    rad_by_group = np.asarray([0.0002, 0.00189], dtype=np.float64)
    radii = rad_by_group[GROUPS]
    protocol = protocols["P2_505"]
    skip = int(math.ceil(nb.ld.protocol_duration_cycles(protocol))) + RELAX
    total = skip + POST_LOAD_PERIODS
    tasks = [(f"steps_{n}", n) for n in RESOLUTIONS]
    meta = {
        "status": "running_or_resumable", "source_start": "P2_505_angle0_phase0",
        "resolutions_steps_per_period": list(RESOLUTIONS), "post_load_periods": POST_LOAD_PERIODS,
        "workers": WORKERS, "phase_rad": phase.tolist(),
        "purpose": "full-horizon convergence of mean frequency, structure, and integrated energy balance",
    }
    (OUT / "refinement_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    result_path = OUT / "refinement_runs.jsonl"
    done = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if (OUT / f"{row['task_id']}_trajectory.npz").exists():
                    done.add(row["task_id"])
    todo = [t for t in tasks if t[0] not in done]

    def simulate(task):
        task_id, steps = task
        tic = time.perf_counter()
        result = nb.integrate(initial, phase, p, protocol, GROUPS, family, modes, coeff,
                              qratio, radii, total, steps, skip, True, True)
        elapsed = time.perf_counter() - tic
        logs, work = result[13], result[14]
        structure = structural_metrics(logs, skip)
        row = {
            "task_id": task_id, "steps_per_period": steps, "completed_periods": int(result[9]),
            "post_load_periods_observed": max(0, int(result[9]) - skip),
            "retained": bool(result[10]), "category": str(result[11]), "failure_status": int(result[12]),
            "turns_each_after_load": np.asarray(result[1]).tolist(),
            "frequencies_Hz_each": np.asarray(result[3]).tolist(),
            "r_min_R_each": np.asarray(result[4]).tolist(), "r_max_R_each": np.asarray(result[5]).tolist(),
            "minimum_pair_distances_R_i_lt_j": np.asarray(result[8]).tolist(),
            "structure_after_load": structure,
            "energy_balance": {
                "RF_work": float(work[-1, 2]), "drag_work": float(work[-1, 3]),
                "final_residual": float(work[-1, 4]),
                "relative_max_residual_vs_RF_work": float(np.max(np.abs(work[:, 4])) / max(abs(work[-1, 2]), 1e-30)),
            },
            "elapsed_seconds": elapsed,
        }
        return row, logs, work

    with ThreadPoolExecutor(max_workers=WORKERS) as pool, result_path.open("a", encoding="utf-8", buffering=1) as f:
        futures = {pool.submit(simulate, task): task for task in todo}
        for future in as_completed(futures):
            task = futures[future]
            row, logs, work = future.result()
            tmp_path = OUT / f".{task[0]}_trajectory.npz.tmp"
            with tmp_path.open("wb") as tmp:
                np.savez_compressed(tmp, trajectory=logs, energy_balance=work,
                                    start=initial, phases=phase, groups=GROUPS, post_load_skip=skip)
                tmp.flush(); os.fsync(tmp.fileno())
            tmp_path.replace(OUT / f"{task[0]}_trajectory.npz")
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush(); os.fsync(f.fileno())
            print(task_id := row["task_id"], row["category"], "turns min",
                  round(min(abs(x) for x in row["turns_each_after_load"]), 3),
                  "freq", np.round(row["frequencies_Hz_each"], 5).tolist(),
                  "Psi", [round(g["mean_Psi3"], 5) for g in row["structure_after_load"]["groups"]],
                  "energy relative error", row["energy_balance"]["relative_max_residual_vs_RF_work"],
                  f"{row['elapsed_seconds']:.1f}s", flush=True)
    meta["status"] = "complete"
    (OUT / "refinement_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
