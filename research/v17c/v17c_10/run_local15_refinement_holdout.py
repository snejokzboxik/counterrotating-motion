"""Fresh strict matched holdout for the leading refinement-screen candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
V17C9 = HERE.parent / "v17c_9"
BASE = V17C9 / "baseline"
OUT = HERE / "local15_refinement_validation"
START_DIR = OUT / "holdout_starts"
sys.path.insert(0, str(BASE))
from basin_scan import load_model, make_initials

GROUPS = ("rest", "weak", "wide", "edge")
N = 128
PERIODS = 15000
STEPS = 64
TRANSIENT = 200
SIZES = (0.0002, 0.00189)
RUN_TAG = "refineholdout20261510"
START_SEEDS = {"rest": 20261510, "weak": 20261511, "wide": 20261512, "edge": 20261513}
EDGE_RMIN, EDGE_RMAX = 0.70, 0.90
SELECTED = (0, 12, 18, 32, 35)
FILE_KEYS = {0: "refine00", 12: "refine12", 18: "refine18", 32: "local15", 35: "candidate52"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_candidates():
    configs = setup_configs = __import__("run_local15_refinement_screen").local_screen_configs()
    by_id = {int(c["config_id"]): c for c in configs}
    if not all(i in by_id for i in SELECTED):
        raise ValueError("selected refinement candidates are missing")
    return {i: by_id[i] for i in SELECTED}


def make_inputs():
    starts, phases = {}, {}
    for group in GROUPS[:3]:
        starts[group], phases[group] = make_initials(group, N, START_SEEDS[group])
    starts["edge"], phases["edge"] = make_initials("wide", N, START_SEEDS["edge"])
    u = qmc.Sobol(d=2, scramble=True, seed=START_SEEDS["edge"] + 7919).random_base2(7)
    radial = np.sqrt(EDGE_RMIN**2 + u[:, 0] * (EDGE_RMAX**2 - EDGE_RMIN**2))
    angle = 2.0 * math.pi * u[:, 1]
    for i in range(N):
        offset = 0 if i % 2 == 0 else 6
        starts["edge"][i, offset] = radial[i] * math.cos(angle[i])
        starts["edge"][i, offset + 1] = radial[i] * math.sin(angle[i])
    return starts, phases


def prepare_starts(starts, phases):
    START_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for group in GROUPS:
        path = START_DIR / f"{group}_N128_seed{START_SEEDS[group]}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (not np.array_equal(saved["initial"], starts[group])
                        or not np.array_equal(saved["phases"], phases[group])
                        or int(saved["seed"]) != START_SEEDS[group]):
                    raise RuntimeError(f"saved strict refinement starts differ: {group}")
        else:
            np.savez_compressed(path, initial=starts[group], phases=phases[group], seed=START_SEEDS[group])
        paths[group] = path
    return paths


def read_rows(path: Path, initial: np.ndarray, phases: np.ndarray):
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            i = int(row["index"])
            if i in rows or not 0 <= i < N:
                raise RuntimeError(f"duplicate or invalid checkpoint index {i}")
            if (not np.array_equal(np.asarray(row["initial"], dtype=float), initial[i])
                    or not np.array_equal(np.asarray(row["phase"], dtype=float), phases[i])):
                raise RuntimeError(f"checkpoint inputs differ from frozen holdout start {i}")
            rows[i] = row
    return rows


def output_row(i, initial, phase, result, seed):
    (end, turns, windows, rmin, rmax, zmax, maxd, dmin, complete, retained,
     escaped, singular, category, freq) = result
    return {
        "index": int(i), "seed": int(seed), "initial": initial.tolist(), "phase": phase.tolist(),
        "final_state": end.tolist(), "turns": turns.tolist(), "window_turns": windows.tolist(),
        "frequency_Hz": freq.tolist(), "r_min": rmin.tolist(), "r_max": rmax.tolist(),
        "z_max": zmax.tolist(), "max_abs_inner_step_dtheta_rad": maxd.tolist(),
        "minimum_separation": float(dmin), "completed_periods": int(complete),
        "retained": bool(retained), "escaped": bool(escaped), "singular_unresolved": bool(singular),
        "category": str(category), "hydro_model": "none", "a1_over_R": SIZES[0], "a2_over_R": SIZES[1],
    }


def run_group(candidate_id, candidate, group, args, module, field_model, starts, phases, start_path):
    key = FILE_KEYS[candidate_id]
    p = np.asarray(candidate["p"], dtype=np.float64)
    qratio = float(candidate["qratio"])
    stem = f"{key}_{group}_matched_strict_none_a0.0002-0.00189_{RUN_TAG}_{N}_{PERIODS}p_{STEPS}s"
    run_path, meta_path = OUT / f"{stem}.jsonl", OUT / f"{stem}.meta.json"
    metadata = {
        "screen_config_id": candidate_id, "candidate_label": candidate["label"],
        "source": candidate["source"], "parameters": p.tolist(), "qratio": qratio,
        "ensemble": group, "n": N, "seed": START_SEEDS[group],
        "periods": PERIODS, "steps": STEPS, "transient": TRANSIENT,
        "starts": str(start_path.relative_to(HERE.parents[2])).replace("/", "\\"),
        "frozen_start_sha256": sha256(start_path), "candidate_source": "V17C.10 strict holdout for P4 refinement screen",
        "run_tag": RUN_TAG, "hydro_mode": "none",
        "a1_over_R": SIZES[0], "a2_over_R": SIZES[1],
        "fixed_field_config_sha256": sha256(module.SOURCE_CFG),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    if meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved != metadata:
            if run_path.exists() and run_path.stat().st_size:
                raise RuntimeError(f"holdout metadata mismatch: {meta_path}")
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    existing = read_rows(run_path, starts[group], phases[group])
    todo = [i for i in range(N) if i not in existing]
    if not todo:
        print(f"{key}/{group}: already complete {N}/{N}", flush=True)
        return
    first = todo[0]
    module.integrate_one(starts[group][first], phases[group][first], p, *field_model[1:],
                         PERIODS, STEPS, TRANSIENT, qratio=qratio,
                         oseen_Re_per_speed=0.0, hydro_mode=0,
                         a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0)
    print(f"{key}/{group}: n={N}, todo={len(todo)}, workers={args.threads}", flush=True)
    tic = time.perf_counter()

    def one(i):
        result = module.integrate_one(starts[group][i], phases[group][i], p, *field_model[1:],
                                      PERIODS, STEPS, TRANSIENT, qratio=qratio,
                                      oseen_Re_per_speed=0.0, hydro_mode=0,
                                      a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0)
        return output_row(i, starts[group][i], phases[group][i], result, START_SEEDS[group])

    with run_path.open("a", encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = [pool.submit(one, i) for i in todo]
            for n, future in enumerate(as_completed(futures), 1):
                row = future.result()
                existing[int(row["index"])] = row
                fout.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                if n % 16 == 0 or n == len(todo):
                    print(f"{key}/{group}: checkpoint {len(existing)}/{N} ({time.perf_counter()-tic:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("--threads must be in [1, 4]")
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    starts, phases = make_inputs()
    start_paths = prepare_starts(starts, phases)
    module = __import__("basin_scan")
    field_model = load_model()
    for candidate_id in SELECTED:
        for group in GROUPS:
            run_group(candidate_id, candidates[candidate_id], group, args, module,
                      field_model, starts, phases, start_paths[group])


if __name__ == "__main__":
    main()
