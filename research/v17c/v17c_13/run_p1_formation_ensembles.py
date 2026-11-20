"""Resumable N=6 small formation ensembles on the same P2_505 protocol."""
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
OUT = HERE / "p1_formation_ensembles"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
ENSEMBLES = ("ordered", "perturbed", "random_azimuth",
             "random_position_rest", "random_position_moving")
POST_LOAD_PERIODS = 15_000
STEPS_PER_PERIOD = 96
RELAX_AFTER_LOAD = 200
WORKERS = 4


def classify(result, structure: dict, requested_periods: int) -> tuple[str, dict]:
    retained, status = bool(result[10]), int(result[12])
    completed = int(result[9])
    turns = np.asarray(result[1], dtype=np.float64)
    windows = np.asarray(result[2], dtype=np.float64)
    if status == 1:
        return "PARTICLE_CONTACT", {"finite_sphere_contact": True}
    if status == 2:
        return "WALL_CONTACT", {"wall_or_axial_boundary_contact": True}
    if status == 3:
        return "UNRESOLVED", {"point_singularity": True}
    if np.any(np.asarray(result[4]) < nb.RMIN_CLASS):
        return "AXIS_AMBIGUOUS", {"axis_min_radius_R": float(np.min(result[4]))}
    if not retained or completed < requested_periods or len(structure.get("groups", [])) != 2:
        return "UNRESOLVED", {"completed_periods": completed, "requested_periods": requested_periods}

    groups = structure["groups"]
    means = np.asarray([g["mean_frequency_Hz"] for g in groups])
    opposite = bool(means[0] * means[1] < 0.0)
    logged = np.asarray(result[13])
    logged_r = np.hypot(logged[:, :, 1], logged[:, :, 2])
    radial_gap = np.mean(logged_r[:, :3], axis=1) - np.mean(logged_r[:, 3:], axis=1)
    initial_radial_order = bool(radial_gap[0] < 0.0)
    final_radial_order = bool(np.mean(radial_gap[-max(10, len(radial_gap)//10):]) < 0.0)
    ring_exchange = (initial_radial_order != final_radial_order and
                     abs(radial_gap[0]) > 0.05 and
                     abs(float(np.mean(radial_gap[-max(10, len(radial_gap)//10):]))) > 0.05)
    enough_turns = bool(all(g["minimum_turns_abs"] >= 10.0 for g in groups))
    particle_signs_stable = bool(np.all(windows[:, :3] * np.sign(means[0]) > 0.0) and
                                 np.all(windows[:, 3:] * np.sign(means[1]) > 0.0))
    phase_slopes = [abs(x["linear_drift_deg_per_1000_periods"])
                    for g in groups for x in g["relative_phase_late_diagnostics"][1:]]
    phase_ranges = [x["detrended_range_deg"]
                    for g in groups for x in g["relative_phase_late_diagnostics"][1:]]
    phase_slip = max(phase_slopes, default=0.0) > 0.25 or max(phase_ranges, default=0.0) > 20.0
    crystal_group = [
        g["relative_frequency_spread"] < 0.01 and
        g["mean_Psi3"] >= 0.9 and g["p05_Psi3"] >= 0.9 and
        g["late_max_abs_gap_error_deg"] <= 10.0 and
        max((abs(x["linear_drift_deg_per_1000_periods"])
             for x in g["relative_phase_late_diagnostics"][1:]), default=0.0) <= 0.25 and
        max((x["detrended_range_deg"]
             for x in g["relative_phase_late_diagnostics"][1:]), default=0.0) <= 20.0
        for g in groups
    ]
    criteria = {
        "retained_all_six_to_full_horizon": retained and completed == requested_periods,
        "at_least_ten_turns_each": enough_turns,
        "opposite_group_mean_signs": opposite,
        "every_particle_direction_stable_in_three_windows": particle_signs_stable,
        "crystal_group_1": bool(crystal_group[0]),
        "crystal_group_2": bool(crystal_group[1]),
        "phase_slip_flag": bool(phase_slip),
        "radial_ring_order_reversed_from_initial": bool(ring_exchange),
        "all_particle_and_wall_contacts_absent": status == 0,
    }
    if np.max(np.abs(turns)) < 0.25:
        category = "NONROTATING"
    elif opposite and all(crystal_group) and enough_turns and particle_signs_stable:
        category = "COUNTERROTATING_CRYSTALS_3_PLUS_3"
    elif ring_exchange:
        category = "RING_EXCHANGE"
    elif opposite and phase_slip:
        category = "PHASE_SLIP"
    elif opposite:
        category = "COUNTERROTATING_UNORDERED"
    elif means[0] * means[1] > 0.0 and all(crystal_group):
        category = "COROTATING_CRYSTALS"
    elif any(crystal_group):
        category = "PARTIAL_CRYSTALLIZATION"
    elif means[0] * means[1] > 0.0:
        category = "COROTATING_UNORDERED"
    else:
        category = "UNRESOLVED"
    return category, criteria


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p, family, modes, coeff, qratio = nb.load_local15()
    protocols = nb.load_protocols()
    particle_radii = np.asarray([0.0002] * 3 + [0.00189] * 3, dtype=np.float64)
    protocol = protocols["P2_505"]
    skip = int(math.ceil(nb.ld.protocol_duration_cycles(protocol))) + RELAX_AFTER_LOAD
    total_periods = skip + POST_LOAD_PERIODS

    tasks = []
    for ensemble in ENSEMBLES:
        path = OUT / f"{ensemble}_initials.npz"
        metadata_path = OUT / f"{ensemble}_metadata.json"
        if not path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"frozen starts not prepared: {ensemble}; run prepare_p1_crystal_ensembles.py")
        with np.load(path) as data:
            initial = data["initial"].copy()
            phases = data["phases"].copy()
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if initial.shape != (16, 6, 6) or phases.shape != (16, 2):
            raise ValueError(f"bad frozen start shapes for {ensemble}")
        if meta["protocol"] != "P2_505, identical for every initial condition; continuous carriers":
            raise ValueError(f"unexpected protocol in frozen metadata: {ensemble}")
        for index in range(16):
            tasks.append((f"{ensemble}_{index:03d}", ensemble, index,
                          initial[index], phases[index], meta["start_and_phase_sha256"]))

    metadata_path = OUT / "run_metadata.json"
    meta = {
        "status": "running_or_resumable", "protocol": "P2_505",
        "protocol_duration_cycles": float(nb.ld.protocol_duration_cycles(protocol)),
        "relax_after_load_cycles": RELAX_AFTER_LOAD,
        "post_load_periods": POST_LOAD_PERIODS, "steps_per_period": STEPS_PER_PERIOD,
        "total_periods": total_periods, "workers": WORKERS,
        "ensembles": list(ENSEMBLES), "n_per_ensemble": 16,
        "classification": "explicit finite-horizon 3+3 criteria in classify(); failures remain in denominator",
        "energy_balance": "RK4 stage-integrated RF work and drag with kinetic+RF+DC+Coulomb total energy",
    }
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result_path = OUT / "formation_runs.jsonl"
    done = set()
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if (OUT / f"{row['task_id']}_trajectory.npz").exists():
                    done.add(row["task_id"])
    todo = [t for t in tasks if t[0] not in done]
    print(f"N=6 formation: {len(tasks)} starts; {len(done)} checkpointed; {len(todo)} remaining; workers={WORKERS}", flush=True)

    def simulate(task):
        task_id, ensemble, index, initial, phase, start_sha = task
        tic = time.perf_counter()
        result = nb.integrate(initial.ravel(), phase, p, protocol, GROUPS,
                              family, modes, coeff, qratio, particle_radii,
                              total_periods, STEPS_PER_PERIOD, skip, True, True)
        elapsed = time.perf_counter() - tic
        logs, work = result[13], result[14]
        structure = structural_metrics(logs, skip)
        category, criteria = classify(result, structure, total_periods)
        row = {
            "task_id": task_id, "ensemble": ensemble, "initial_index": index,
            "start_file_sha256": start_sha, "initial_state": np.asarray(initial).tolist(),
            "rf_phases_rad": np.asarray(phase).tolist(), "protocol": "P2_505",
            "completed_periods": int(result[9]), "requested_periods": total_periods,
            "post_load_periods_observed": max(0, int(result[9]) - skip),
            "retained": bool(result[10]), "failure_status": int(result[12]),
            "category": category, "criteria": criteria,
            "turns_each_total": np.asarray(result[1]).tolist(),
            "late_window_turns_each": np.asarray(result[2]).tolist(),
            "frequencies_Hz_each": np.asarray(result[3]).tolist(),
            "r_min_R_each": np.asarray(result[4]).tolist(),
            "r_max_R_each": np.asarray(result[5]).tolist(),
            "max_abs_z_R_each": np.asarray(result[6]).tolist(),
            "minimum_pair_distances_R_i_lt_j": np.asarray(result[8]).tolist(),
            "structure_after_load": structure,
            "energy_balance": ({
                "sample_count": int(len(work)), "RF_work": float(work[-1, 2]),
                "drag_work": float(work[-1, 3]), "final_residual": float(work[-1, 4]),
                "relative_max_residual_vs_RF_work": float(np.max(np.abs(work[:, 4])) / max(abs(work[-1, 2]), 1e-30)),
            } if len(work) else None),
            "elapsed_seconds": elapsed,
        }
        return row, logs, work

    with ThreadPoolExecutor(max_workers=WORKERS) as pool, result_path.open("a", encoding="utf-8", buffering=1) as handle:
        futures = {pool.submit(simulate, task): task for task in todo}
        completed = len(done)
        for future in as_completed(futures):
            row, logs, work = future.result()
            task_id = row["task_id"]
            tmp = OUT / f".{task_id}_trajectory.npz.tmp"
            with tmp.open("wb") as f:
                np.savez_compressed(f, trajectory=logs, energy_balance=work,
                                    initial=row["initial_state"], phases=row["rf_phases_rad"],
                                    groups=GROUPS, post_load_skip=skip)
                f.flush(); os.fsync(f.fileno())
            tmp.replace(OUT / f"{task_id}_trajectory.npz")
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush(); os.fsync(handle.fileno())
            completed += 1
            structure_groups = row["structure_after_load"].get("groups", [])
            first_group_frequency = (structure_groups[0].get("mean_frequency_Hz")
                                     if structure_groups else None)
            freq_text = f"{first_group_frequency:.2f}Hz" if first_group_frequency is not None else "n/a"
            print(f"{completed}/{len(tasks)} {task_id} {row['category']} retained={row['retained']} "
                  f"turns_min={min(abs(x) for x in row['turns_each_total']):.2f} "
                  f"freq={freq_text} "
                  f"time={row['elapsed_seconds']:.1f}s", flush=True)
    meta["status"] = "complete"
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
