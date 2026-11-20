"""Test whether a formed 3+3 structure recovers after a position perturbation."""
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
SOURCE = HERE / "p1_long_candidates" / "P2_505_angle0_phase0_long_trajectory.npz"
OUT = HERE / "p1_postformation_perturbations"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
ANGLES = (10.0, 20.0)
COUPLINGS = (0.0, 1.0)
POST_PERTURB_PERIODS = 15_000
SKIP = 200
STEPS = 96
WORKERS = 4
TAU = 2.0 * math.pi


def perturb_positions(state: np.ndarray, degrees: float) -> np.ndarray:
    y = state.copy().reshape((6, 6))
    delta = math.radians(degrees)
    for i, sign in ((1, 1.0), (4, -1.0)):
        x, yy = y[i, 0], y[i, 1]
        radius = math.hypot(x, yy)
        theta = math.atan2(yy, x) + sign * delta
        y[i, 0], y[i, 1] = radius * math.cos(theta), radius * math.sin(theta)
    return y.ravel()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p0, family, modes, coeff, qratio = nb.load_local15()
    protocols = nb.load_protocols()
    with np.load(SOURCE) as archive:
        logs = archive["trajectory"]
        phase_initial = archive["phases"].astype(np.float64)
        load_skip = int(archive["post_load_skip"])
    start_cycle = load_skip + 10_000
    absolute_tau = float(logs[start_cycle, 0, 0])
    formed_state = logs[start_cycle, :, 1:7].ravel().copy()
    # Shift carrier origins so local t=0 is the saved absolute RF phase.
    phases = np.mod(phase_initial + np.asarray([absolute_tau, p0[2] * absolute_tau]), TAU)
    radii = np.asarray([0.0002] * 3 + [0.00189] * 3, dtype=np.float64)
    tasks = []
    for degrees in ANGLES:
        for factor in COUPLINGS:
            tasks.append((f"angle{degrees:g}_C{factor:g}", degrees, factor,
                          perturb_positions(formed_state, degrees)))
    meta = {
        "status": "running_or_resumable",
        "source": SOURCE.name, "source_start_period_after_load": 10_000,
        "absolute_source_tau": absolute_tau,
        "phase_continuity": "effective local phases include Omega_i*t0 modulo 2pi; no RF phase reset",
        "position_perturbations_deg": list(ANGLES), "C_multipliers": list(COUPLINGS),
        "post_perturbation_periods": POST_PERTURB_PERIODS, "steps_per_period": STEPS,
        "skip_for_metrics_and_energy": SKIP, "workers": WORKERS,
        "definition": "rotate one particle in each ring in opposite azimuthal directions while keeping all velocities fixed",
    }
    (OUT / "postformation_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    result_path = OUT / "postformation_runs.jsonl"
    done = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if (OUT / f"{row['task_id']}_trajectory.npz").exists():
                    done.add(row["task_id"])
    todo = [task for task in tasks if task[0] not in done]

    def simulate(task):
        task_id, degrees, factor, initial = task
        p = p0.copy()
        p[5] = p0[5] * factor
        tic = time.perf_counter()
        result = nb.integrate(initial, phases, p, protocols["constant"], GROUPS, family, modes, coeff,
                              qratio, radii, POST_PERTURB_PERIODS, STEPS, SKIP, True, True)
        elapsed = time.perf_counter() - tic
        trajectory, work = result[13], result[14]
        structure = structural_metrics(trajectory, SKIP)
        initial_angles = np.arctan2(initial.reshape((6, 6))[:, 1], initial.reshape((6, 6))[:, 0])
        psi_at_perturb = [float(abs(np.mean(np.exp(3j * initial_angles[g * 3:(g + 1) * 3])))) for g in (0, 1)]
        row = {
            "task_id": task_id, "position_perturbation_deg": degrees,
            "C_multiplier": factor, "C_absolute": float(p[5]),
            "absolute_source_tau": absolute_tau, "effective_local_phases_rad": phases.tolist(),
            "initial_state": initial.tolist(), "Psi3_immediately_after_perturbation": psi_at_perturb,
            "completed_periods": int(result[9]), "retained": bool(result[10]),
            "category": str(result[11]), "failure_status": int(result[12]),
            "turns_each_after_perturbation": np.asarray(result[1]).tolist(),
            "frequencies_Hz_each": np.asarray(result[3]).tolist(),
            "minimum_pair_distances_R_i_lt_j": np.asarray(result[8]).tolist(),
            "r_min_R_each": np.asarray(result[4]).tolist(), "r_max_R_each": np.asarray(result[5]).tolist(),
            "structure_after_perturbation": structure,
            "energy_balance": {
                "RF_work": float(work[-1, 2]), "drag_work": float(work[-1, 3]),
                "final_residual": float(work[-1, 4]),
                "relative_max_residual_vs_RF_work": float(np.max(np.abs(work[:, 4])) / max(abs(work[-1, 2]), 1e-30)),
            },
            "elapsed_seconds": elapsed,
        }
        return row, trajectory, work

    with ThreadPoolExecutor(max_workers=WORKERS) as pool, result_path.open("a", encoding="utf-8", buffering=1) as f:
        futures = {pool.submit(simulate, task): task for task in todo}
        for future in as_completed(futures):
            task = futures[future]
            row, trajectory, work = future.result()
            tmp_path = OUT / f".{task[0]}_trajectory.npz.tmp"
            with tmp_path.open("wb") as tmp:
                np.savez_compressed(tmp, trajectory=trajectory, energy_balance=work,
                                    start=row["initial_state"], phases=phases, groups=GROUPS,
                                    post_perturb_skip=SKIP)
                tmp.flush(); os.fsync(tmp.fileno())
            tmp_path.replace(OUT / f"{task[0]}_trajectory.npz")
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush(); os.fsync(f.fileno())
            structure = row["structure_after_perturbation"]
            print(row["task_id"], row["category"], "retained", row["retained"],
                  "turns min", round(min(abs(x) for x in row["turns_each_after_perturbation"]), 3),
                  "initial/final Psi", row["Psi3_immediately_after_perturbation"],
                  [round(g["mean_Psi3"], 4) for g in structure["groups"]],
                  "late gap", [round(g["late_max_abs_gap_error_deg"], 2) for g in structure["groups"]],
                  flush=True)
    meta["status"] = "complete"
    (OUT / "postformation_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
