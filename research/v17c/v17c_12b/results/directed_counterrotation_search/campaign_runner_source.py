"""Directed long-horizon basin search from counterrotating orbital seeds."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
SOURCE = HERE / "results" / "alumina_experimental_50hz_screen"
OUT = HERE / "results" / "directed_counterrotation_search"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MODEL))
import basin_scan as sim  # noqa: E402

SEED = 20261229
CONFIG_IDS = (
    "a10um_exp50_0038", "a10um_exp50_0050", "a12.5um_exp50_0091",
    "a10um_exp50_0106", "a10um_exp50_0006", "a12.5um_exp50_0019",
    "a12.5um_exp50_0104", "a15um_exp50_0115",
)
N_PER_DIRECTION = 8
NSTART = 2 * N_PER_DIRECTION
PERIODS = 15000
STEPS = 64
SKIP = 100
WORKERS = min(11, os.cpu_count() or 1)
R = 0.010


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provisional(raw: str, turns: np.ndarray, windows: np.ndarray,
                retained: bool, axis: bool) -> str:
    if raw == "finite_particle_contact": return "PARTICLE_CONTACT"
    if raw == "finite_particle_wall_contact": return "WALL_OR_AXIAL_ESCAPE"
    if raw == "lost_retention": return "ESCAPE"
    if raw == "point_singularity_unresolved": return "UNRESOLVED"
    if axis: return "AXIS_AMBIGUOUS"
    signs = np.zeros((3, 2), dtype=np.int8)
    for w in range(3):
        for i in range(2):
            if abs(windows[w, i]) >= 0.05:
                signs[w, i] = 1 if windows[w, i] > 0 else -1
    if retained and np.all(signs != 0) and np.all(np.abs(turns) >= 1.0):
        if np.all(signs[:, 0] * signs[:, 1] < 0):
            return "PROVISIONAL_CR_PLUS_MINUS" if turns[0] > 0 else "PROVISIONAL_CR_MINUS_PLUS"
        if np.all(signs[:, 0] * signs[:, 1] > 0):
            return "PROVISIONAL_CO_PLUS_PLUS" if turns[0] > 0 else "PROVISIONAL_CO_MINUS_MINUS"
    if retained and np.all(np.abs(turns) < 0.25): return "NONROTATING"
    return "UNRESOLVED_SHORT"


def build_starts() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    source_meta = json.loads((SOURCE / "campaign_meta.json").read_text(encoding="utf-8"))
    source_configs = {c["id"]: c for c in source_meta["configurations"]}
    initial_all, phase_all, labels = [], [], []
    for cidx, cid in enumerate(CONFIG_IDS):
        c = source_configs[cid]
        radius = c["physical"]["radius_m"] / R
        z_eq = c["physical"]["gravity_equilibrium_z_over_R_if_vertical"]
        if abs(z_eq) + radius >= 2.0:
            raise RuntimeError(f"{cid}: center-of-mass equilibrium outside axial domain")
        u = qmc.Sobol(d=13, scramble=True, seed=SEED + cidx).random_base2(4)
        for i, x in enumerate(u):
            sign1, sign2 = (1, -1) if i < N_PER_DIRECTION else (-1, 1)
            r_inner = 0.25 + 0.20 * x[0]
            r_outer = 0.55 + 0.20 * x[1]
            r1, r2 = (r_inner, r_outer) if x[2] < 0.5 else (r_outer, r_inner)
            theta1, theta2 = 2 * math.pi * x[3], 2 * math.pi * x[4]
            z1 = z_eq + 0.03 * (2*x[5] - 1)
            z2 = z_eq + 0.03 * (2*x[6] - 1)
            w1 = math.exp(math.log(0.003) + x[7] * math.log(100.0))
            w2 = math.exp(math.log(0.003) + x[8] * math.log(100.0))
            state = np.zeros(12, dtype=np.float64)
            state[0:3] = [r1*math.cos(theta1), r1*math.sin(theta1), z1]
            state[3:6] = [-sign1*w1*state[1], sign1*w1*state[0], 0.01*(2*x[11]-1)]
            state[6:9] = [r2*math.cos(theta2), r2*math.sin(theta2), z2]
            state[9:12] = [-sign2*w2*state[7], sign2*w2*state[6], 0.01*(2*x[12]-1)]
            phase = 2 * math.pi * x[9:11]
            initial_all.append(state); phase_all.append(phase)
            labels.append({"config_id": cid, "config_index": cidx,
                "start_index": i, "initial_direction": "PLUS_MINUS" if sign1 > 0 else "MINUS_PLUS",
                "r_inner_orbit_particle1": bool(r1 < r2),
                "angular_rate_dimensionless_particle1": float(w1),
                "angular_rate_dimensionless_particle2": float(w2)})
    return np.asarray(initial_all), np.asarray(phase_all), labels


def main() -> None:
    source_meta = json.loads((SOURCE / "campaign_meta.json").read_text(encoding="utf-8"))
    source_configs = {c["id"]: c for c in source_meta["configurations"]}
    configs = [source_configs[cid] for cid in CONFIG_IDS]
    starts, phases, labels = build_starts()
    starts_path = OUT / "frozen_directed_starts.npz"
    if not starts_path.exists():
        np.savez_compressed(starts_path, initial=starts, phases=phases,
            labels=np.array(json.dumps(labels)), seed=SEED)
    else:
        frozen = np.load(starts_path)
        if not np.array_equal(frozen["initial"], starts) or not np.array_equal(frozen["phases"], phases):
            raise RuntimeError("existing directed start set differs")
    _, family, k, coeff = sim.load_model()
    meta = {"study": "directed counterrotation-seed search for the best retained physical 50 Hz Al2O3 configs",
        "source_campaign": "results/alumina_experimental_50hz_screen",
        "source_runner_sha256": json.loads((SOURCE / "campaign_code_sha256.json").read_text(encoding="utf-8"))["runner_sha256"],
        "configuration_ids": list(CONFIG_IDS), "configurations": configs,
        "starts_per_configuration": NSTART,
        "directed_starts_per_orientation": N_PER_DIRECTION,
        "periods": PERIODS, "steps_per_rf_period": STEPS, "skip_periods": SKIP,
        "seed": SEED, "workers": WORKERS,
        "interpretation": "Directed initial-condition search only; never a basin probability or spontaneous-capture result.",
        "initial_conditions": "Two separated radii 0.25-0.45R and 0.55-0.75R, z near exact CM gravity equilibrium, opposite tangential directions, randomized rates and phases."}
    meta_path = OUT / "campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise RuntimeError("directed campaign metadata differs")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    source_copy = OUT / "campaign_runner_source.py"
    if not source_copy.exists(): shutil.copy2(Path(__file__), source_copy)
    (OUT / "campaign_code_sha256.json").write_text(json.dumps({
        "runner_sha256": sha(Path(__file__)), "saved_runner_sha256": sha(source_copy),
        "integrator_sha256": sha(MODEL / "basin_scan.py"),
        "field_model_sha256": sha(MODEL / "field_model.json"),
        "campaign": OUT.name, "seed": SEED, "configs": len(configs),
        "starts_per_config": NSTART, "periods": PERIODS, "steps": STEPS,
        "workers": WORKERS}, indent=2) + "\n", encoding="utf-8")

    run_path = OUT / "trajectories.jsonl"
    existing = {}
    if run_path.exists():
        with run_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing[(int(row["config_index"]), int(row["start_index"]))] = row
    tasks = [(ci, si) for ci in range(len(configs)) for si in range(NSTART)
        if (ci, si) not in existing]

    def run_one(task):
        ci, si = task
        c = configs[ci]
        ph = c["physical"]
        row_label = labels[ci*NSTART+si]
        raw = sim.integrate_one(starts[ci*NSTART+si], phases[ci*NSTART+si],
            np.asarray(c["p"]), family, k, coeff, PERIODS, STEPS, SKIP,
            qratio=1.0, oseen_Re_per_speed=ph["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["Re_per_dimensionless_speed"], hydro_mode=2,
            a1_over_R=ph["radius_m"] / R, a2_over_R=ph["radius_m"] / R,
            gravity_z=ph["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=ph["Cunningham"])
        end, turns, windows, rmin, rmax, zmax, dtheta, dmin, completed, retained, escaped, singular, cat, freq, vmax = raw
        axis = bool(min(rmin) < sim.RMIN_CLASS)
        return {"config_index": ci, "config_id": c["id"], "start_index": si,
            **row_label, "seed": SEED, "initial": starts[ci*NSTART+si].tolist(),
            "phase": phases[ci*NSTART+si].tolist(), "final_state": end.tolist(),
            "turns": turns.tolist(), "window_turns": windows.tolist(),
            "raw_category": str(cat), "provisional_category": provisional(str(cat), turns, windows, bool(retained), axis),
            "retained": bool(retained), "completed_periods": int(completed),
            "r_min_over_R": rmin.tolist(), "r_max_over_R": rmax.tolist(),
            "z_max_over_R": zmax.tolist(), "distance_min_over_R": float(dmin),
            "max_Re_by_particle": (vmax * ph["Re_per_dimensionless_speed"]).tolist()}

    if tasks:
        ci, si = tasks[0]; c = configs[ci]; ph = c["physical"]
        sim.integrate_one(starts[ci*NSTART+si], phases[ci*NSTART+si], np.asarray(c["p"]),
            family, k, coeff, 1, STEPS, 0, qratio=1.0,
            oseen_Re_per_speed=ph["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["Re_per_dimensionless_speed"], hydro_mode=2,
            a1_over_R=ph["radius_m"] / R, a2_over_R=ph["radius_m"] / R,
            gravity_z=ph["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=ph["Cunningham"])

    print(json.dumps({"configs": len(configs), "starts_per_config": NSTART,
        "trajectories": len(configs)*NSTART, "todo": len(tasks), "workers": WORKERS}, indent=2), flush=True)
    started = time.perf_counter()
    with run_path.open("a", encoding="utf-8", buffering=1) as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            existing[(row["config_index"], row["start_index"])] = row
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if n % 32 == 0 or n == len(tasks):
                f.flush(); os.fsync(f.fileno())
                print(f"checkpoint {len(existing)}/{len(configs)*NSTART}; {time.perf_counter()-started:.1f}s", flush=True)
    counts = Counter(r["provisional_category"] for r in existing.values())
    by_config = defaultdict(Counter); by_direction = defaultdict(Counter)
    for r in existing.values():
        by_config[r["config_id"]][r["provisional_category"]] += 1
        by_direction[r["initial_direction"]][r["provisional_category"]] += 1
    summary = {"study": meta["study"], "rows": len(existing), "expected_rows": len(configs)*NSTART,
        "outcome_counts": dict(sorted(counts.items())),
        "outcomes_by_initial_direction": {k: dict(sorted(v.items())) for k,v in sorted(by_direction.items())},
        "outcomes_by_config": {k: dict(sorted(v.items())) for k,v in sorted(by_config.items())},
        "classification": "Directed initial-condition search only; not basin probabilities.",
        "trajectory_sha256": sha(run_path), "meta_sha256": sha(meta_path),
        "wall_seconds": time.perf_counter()-started}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"rows":summary["rows"],"outcomes":summary["outcome_counts"],
        "outcomes_by_initial_direction":summary["outcomes_by_initial_direction"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
