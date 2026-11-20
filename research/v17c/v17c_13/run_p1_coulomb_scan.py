"""Resumable Coulomb-strength continuation for the N=6 ordered #505 candidate."""
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
SOURCE = HERE / "p1_ordered_pilot" / "pilot_runs.jsonl"
OUT = HERE / "p1_coulomb_scan"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
FACTORS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
POST_LOAD_PERIODS = 15_000
RELAX = 200
STEPS = 96
WORKERS = 4


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p0, family, modes, coeff, qratio = nb.load_local15()
    protocols = nb.load_protocols()
    source_rows = {r["task_id"]: r for r in
                   (json.loads(x) for x in SOURCE.read_text().splitlines() if x.strip())}
    source = source_rows["P2_505_angle0_phase0"]
    initial = np.asarray(source["initial_state"], dtype=np.float64)
    phase = np.asarray(source["rf_phase_rad"], dtype=np.float64)
    radius_by_group = np.asarray([0.0002, 0.00189], dtype=np.float64)
    radii = radius_by_group[GROUPS]
    protocol = protocols["P2_505"]
    skip = int(math.ceil(nb.ld.protocol_duration_cycles(protocol))) + RELAX
    total = skip + POST_LOAD_PERIODS
    meta = {
        "status": "running_or_resumable", "source_start": "P2_505_angle0_phase0",
        "baseline_C": float(p0[5]), "coupling_multipliers": list(FACTORS),
        "steps_per_primary_period": STEPS, "post_load_periods": POST_LOAD_PERIODS,
        "initial_state_sha256": sha(initial), "rf_phase_rad": phase.tolist(),
        "workers": WORKERS,
        "purpose": "determine whether long-lived triangular phase order depends on Coulomb coupling",
    }
    (OUT / "coulomb_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    path = OUT / "coulomb_runs.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if (OUT / f"{row['task_id']}_trajectory.npz").exists():
                    done.add(row["task_id"])
    tasks = [(f"C_{factor:g}", factor) for factor in FACTORS if f"C_{factor:g}" not in done]

    def simulate(task):
        task_id, factor = task
        p = p0.copy()
        p[5] = p0[5] * factor
        tic = time.perf_counter()
        result = nb.integrate(initial, phase, p, protocol, GROUPS, family, modes, coeff,
                              qratio, radii, total, STEPS, skip, True, True)
        elapsed = time.perf_counter() - tic
        logs, work = result[13], result[14]
        structure = structural_metrics(logs, skip)
        row = {
            "task_id": task_id, "C_multiplier": factor, "C_absolute": float(p[5]),
            "completed_periods": int(result[9]), "post_load_periods_observed": max(0, int(result[9]) - skip),
            "retained": bool(result[10]), "category": str(result[11]), "failure_status": int(result[12]),
            "turns_each_after_load": np.asarray(result[1]).tolist(),
            "frequencies_Hz_each": np.asarray(result[3]).tolist(),
            "r_min_R_each": np.asarray(result[4]).tolist(), "r_max_R_each": np.asarray(result[5]).tolist(),
            "max_abs_z_R_each": np.asarray(result[6]).tolist(),
            "minimum_pair_distances_R_i_lt_j": np.asarray(result[8]).tolist(),
            "structure_after_load": structure,
            "energy_balance": {
                "integrated_RF_explicit_work": float(work[-1, 2]) if len(work) else None,
                "integrated_drag_work": float(work[-1, 3]) if len(work) else None,
                "final_residual": float(work[-1, 4]) if len(work) else None,
                "relative_max_residual_vs_RF_work": float(np.max(np.abs(work[:, 4])) / max(abs(work[-1, 2]), 1e-30)) if len(work) else None,
            },
            "elapsed_seconds": elapsed,
        }
        return row, logs, work

    with ThreadPoolExecutor(max_workers=WORKERS) as pool, path.open("a", encoding="utf-8", buffering=1) as f:
        futures = {pool.submit(simulate, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            row, logs, work = future.result()
            tmp_path = OUT / f".{task[0]}_trajectory.npz.tmp"
            with tmp_path.open("wb") as tmp:
                np.savez_compressed(tmp, trajectory=logs, energy_balance=work,
                                    start=initial, phases=phase, groups=GROUPS,
                                    post_load_skip=skip)
                tmp.flush(); os.fsync(tmp.fileno())
            tmp_path.replace(OUT / f"{task[0]}_trajectory.npz")
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush(); os.fsync(f.fileno())
            print(task[0], "C", row["C_absolute"], row["category"],
                  "retained", row["retained"], "turns", np.round(row["turns_each_after_load"], 2).tolist(),
                  "Psi", [round(g["mean_Psi3"], 3) for g in row["structure_after_load"]["groups"]],
                  "spread%", [round(100 * g["relative_frequency_spread"], 3) for g in row["structure_after_load"]["groups"]],
                  f"{row['elapsed_seconds']:.1f}s", flush=True)
    meta["status"] = "complete"
    (OUT / "coulomb_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
