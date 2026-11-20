"""Fresh strict matched holdout for the refine12-neighborhood screen leaders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import run_local15_refinement_holdout as setup

N = 128
GROUPS = ("rest", "weak", "wide", "edge")
SEEDS = {"rest": 20261810, "weak": 20261811, "wide": 20261812, "edge": 20261813}
RUN_TAG = "refine12holdout20261810"
SIZES = (0.0002, 0.00189)
SELECTED = (17, 10, 23, 32, 34, 35, 36)
FILE_KEYS = {
    17: "refine12_sobol17", 10: "refine12_sobol10", 23: "refine12_sobol23",
    32: "refine12", 34: "refine00", 35: "local15", 36: "candidate52",
}
OUT = HERE / "refine12_neighborhood_validation"
START_DIR = OUT / "holdout_starts"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def configure():
    setup.N = N
    setup.GROUPS = GROUPS
    setup.START_SEEDS = SEEDS
    setup.RUN_TAG = RUN_TAG
    setup.SELECTED = SELECTED
    setup.FILE_KEYS = FILE_KEYS
    setup.OUT = OUT
    setup.START_DIR = START_DIR
    setup.SIZES = SIZES
    setup.__file__ = str(Path(__file__).resolve())


def load_candidates():
    meta = json.loads((HERE / "refine12_neighborhood_screen" / "meta.json").read_text(encoding="utf-8"))
    by_id = {int(c["config_id"]): c for c in meta["candidates"]}
    if not all(i in by_id for i in SELECTED):
        raise ValueError("selected neighborhood-screen candidates are missing")
    out = {}
    for cid in SELECTED:
        c = by_id[cid]
        out[cid] = {**c, "p": np.asarray(c["p"], dtype=np.float64), "qratio": float(c["qratio"])}
    return out


def run_group(candidate_id, candidate, group, threads, module, field_model,
              starts, phases, start_path):
    key = FILE_KEYS[candidate_id]
    p = np.asarray(candidate["p"], dtype=np.float64)
    qratio = float(candidate["qratio"])
    stem = f"{key}_{group}_matched_strict_none_a0.0002-0.00189_{RUN_TAG}_{N}_{setup.PERIODS}p_{setup.STEPS}s"
    run_path, meta_path = OUT / f"{stem}.jsonl", OUT / f"{stem}.meta.json"
    metadata = {
        "screen_config_id": candidate_id, "candidate_label": candidate["label"],
        "source": candidate["source"], "parameters": p.tolist(), "qratio": qratio,
        "ensemble": group, "n": N, "seed": SEEDS[group],
        "periods": setup.PERIODS, "steps": setup.STEPS, "transient": setup.TRANSIENT,
        "starts": str(start_path.relative_to(HERE.parents[2])).replace("/", "\\"),
        "frozen_start_sha256": sha256(start_path),
        "candidate_source": "V17C.10 strict holdout for refine12-neighborhood screen",
        "run_tag": RUN_TAG, "hydro_mode": "none",
        "a1_over_R": SIZES[0], "a2_over_R": SIZES[1],
        "fixed_field_config_sha256": setup.sha256(module.SOURCE_CFG),
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

    existing = setup.read_rows(run_path, starts[group], phases[group])
    todo = [i for i in range(N) if i not in existing]
    if not todo:
        print(f"{key}/{group}: already complete {N}/{N}", flush=True)
        return
    first = todo[0]
    module.integrate_one(starts[group][first], phases[group][first], p, *field_model[1:],
                         setup.PERIODS, setup.STEPS, setup.TRANSIENT,
                         qratio=qratio, oseen_Re_per_speed=0.0, hydro_mode=0,
                         a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0)
    print(f"{key}/{group}: n={N}, todo={len(todo)}, workers={threads}", flush=True)
    tic = time.perf_counter()

    def one(i):
        result = module.integrate_one(starts[group][i], phases[group][i], p, *field_model[1:],
                                      setup.PERIODS, setup.STEPS, setup.TRANSIENT,
                                      qratio=qratio, oseen_Re_per_speed=0.0, hydro_mode=0,
                                      a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0)
        return setup.output_row(i, starts[group][i], phases[group][i], result, SEEDS[group])

    with run_path.open("a", encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = [pool.submit(one, i) for i in todo]
            for n, future in enumerate(as_completed(futures), 1):
                row = future.result()
                existing[int(row["index"])] = row
                fout.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                if n % 16 == 0 or n == len(todo):
                    print(f"{key}/{group}: checkpoint {len(existing)}/{N} ({time.perf_counter()-tic:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=min(11, os.cpu_count() or 1))
    args = parser.parse_args()
    if not 1 <= args.threads <= (os.cpu_count() or 1):
        raise ValueError(f"--threads must be in [1, {os.cpu_count() or 1}]")
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    starts, phases = setup.make_inputs()
    start_paths = setup.prepare_starts(starts, phases)
    module = __import__("basin_scan")
    field_model = setup.load_model()
    execution = {
        "study": "V17C.10 refine12-neighborhood strict matched holdout",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "worker_count": args.threads, "logical_processors_reported": os.cpu_count(),
        "trajectory_parallelism": "ThreadPoolExecutor over independent Numba nogil trajectories",
        "runner_sha256": sha256(Path(__file__).resolve()),
        "sample_size_per_candidate_and_ensemble": N,
        "fresh_start_seeds": SEEDS,
        "same_frozen_starts_and_phases_within_ensemble": True,
    }
    (OUT / "parallel_execution.json").write_text(json.dumps(execution, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(execution, indent=2, ensure_ascii=False), flush=True)
    for cid in SELECTED:
        for group in GROUPS:
            run_group(cid, candidates[cid], group, args.threads, module, field_model,
                      starts, phases, start_paths[group])


if __name__ == "__main__":
    main()
