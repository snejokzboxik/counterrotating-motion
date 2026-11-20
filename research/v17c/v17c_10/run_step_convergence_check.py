"""Same-start 96-step integration check for local15 and candidate52."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
V17C9 = HERE.parent / "v17c_9"
BASE = V17C9 / "baseline"
OUT = HERE / "focused_edge_convergence"
START_DIR = OUT / "convergence_starts"
REPLICATION = HERE / "focused_edge_replication"
sys.path.insert(0, str(BASE))
from basin_scan import load_model

GROUPS = ("rest", "weak", "wide", "edge")
N = 128
PERIODS = 15000
STEPS = 96
TRANSIENT = 200
SIZES = (0.0002, 0.00189)
RUN_TAG = "stepconv20261420"
SELECTED = (15, 36)
FILE_KEYS = {15: "local15", 36: "candidate52"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_candidates():
    ranked = json.loads((HERE / "focused_edge_screen" / "ranked.json").read_text(encoding="utf-8"))
    by_id = {int(row["config_id"]): row for row in ranked}
    return {i: by_id[i] for i in SELECTED}


def prepare_starts():
    START_DIR.mkdir(parents=True, exist_ok=True)
    arrays, paths = {}, {}
    for group in GROUPS:
        source = REPLICATION / "replication_starts" / f"{group}_N256_seed{20261410 + GROUPS.index(group)}.npz"
        with np.load(source, allow_pickle=False) as saved:
            initial, phases = saved["initial"][:N].copy(), saved["phases"][:N].copy()
            seed = int(saved["seed"])
        if seed != 20261410 + GROUPS.index(group):
            raise RuntimeError(f"unexpected source seed for {group}")
        path = START_DIR / f"{group}_N128_subset256_seed{seed}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (not np.array_equal(saved["initial"], initial)
                        or not np.array_equal(saved["phases"], phases)
                        or int(saved["seed"]) != seed):
                    raise RuntimeError(f"saved convergence starts differ from N=256 prefix: {group}")
        else:
            np.savez_compressed(path, initial=initial, phases=phases, seed=seed,
                                source_sha256=sha256(source), source_indices=np.arange(N))
        arrays[group] = (initial, phases, seed)
        paths[group] = path
    return arrays, paths


def read_rows(path: Path, initial: np.ndarray, phases: np.ndarray):
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            i = int(row["index"])
            if i in rows or not 0 <= i < N:
                raise RuntimeError(f"duplicate or invalid convergence index {i}")
            if (not np.array_equal(np.asarray(row["initial"], dtype=float), initial[i])
                    or not np.array_equal(np.asarray(row["phase"], dtype=float), phases[i])):
                raise RuntimeError(f"convergence checkpoint input mismatch: {i}")
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
        "retained": bool(retained), "escaped": bool(escaped),
        "singular_unresolved": bool(singular), "category": str(category),
        "hydro_model": "none", "oseen_Re_per_speed": 0.0, "oseen_Re2_per_speed": 0.0,
        "a1_over_R": SIZES[0], "a2_over_R": SIZES[1],
    }


def run_group(candidate_id, candidate, group, args, module, field_model, data, start_path):
    initial, phases, seed = data
    p = np.asarray(candidate["p"], dtype=np.float64)
    qratio = float(candidate["qratio"])
    key = FILE_KEYS[candidate_id]
    stem = f"{key}_{group}_matched_strict_none_a0.0002-0.00189_{RUN_TAG}_{N}_{PERIODS}p_{STEPS}s"
    run_path, meta_path = OUT / f"{stem}.jsonl", OUT / f"{stem}.meta.json"
    source_start = REPLICATION / "replication_starts" / f"{group}_N256_seed{seed}.npz"
    metadata = {
        "screen_config_id": candidate_id, "candidate_label": candidate["label"],
        "parameters": p.tolist(), "qratio": qratio,
        "ensemble": group, "n": N, "seed": seed,
        "periods": PERIODS, "steps": STEPS, "transient": TRANSIENT,
        "source_start_file": str(source_start.relative_to(HERE.parents[2])).replace("/", "\\"),
        "source_start_sha256": sha256(source_start),
        "frozen_start_sha256": sha256(start_path),
        "source_start_indices": [0, N], "run_tag": RUN_TAG,
        "candidate_source": "V17C.10 same-start numerical sensitivity check",
        "hydro_mode": "none", "a1_over_R": SIZES[0], "a2_over_R": SIZES[1],
        "fixed_field_config_sha256": sha256(module.SOURCE_CFG),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
            if run_path.exists() and run_path.stat().st_size:
                raise RuntimeError(f"convergence metadata mismatch: {meta_path}")
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    existing = read_rows(run_path, initial, phases)
    todo = [i for i in range(N) if i not in existing]
    if not todo:
        print(f"{key}/{group}: already complete {N}/{N}", flush=True)
        return
    first = todo[0]
    module.integrate_one(initial[first], phases[first], p, *field_model[1:],
                         PERIODS, STEPS, TRANSIENT, qratio=qratio,
                         oseen_Re_per_speed=0.0, hydro_mode=0,
                         a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0)
    print(f"{key}/{group}: n={N}, steps={STEPS}, todo={len(todo)}, workers={args.threads}", flush=True)
    tic = time.perf_counter()

    def one(i):
        result = module.integrate_one(initial[i], phases[i], p, *field_model[1:],
                                      PERIODS, STEPS, TRANSIENT, qratio=qratio,
                                      oseen_Re_per_speed=0.0, hydro_mode=0,
                                      a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0)
        return output_row(i, initial[i], phases[i], result, seed)

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
    starts, start_paths = prepare_starts()
    import basin_scan as module
    field_model = load_model()
    for candidate_id in SELECTED:
        for group in GROUPS:
            run_group(candidate_id, candidates[candidate_id], group, args, module,
                      field_model, starts[group], start_paths[group])


if __name__ == "__main__":
    main()
