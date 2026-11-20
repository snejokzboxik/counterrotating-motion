"""Independent-start long validation of early-screen protocol leaders."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "v17c_9" / "baseline"
sys.path.insert(0, str(BASE))
import basin_scan  # noqa: E402
import screen_loading_protocols as screen  # noqa: E402
import loading_dynamics as ld  # noqa: E402

OUT = HERE / "loading_validation"
GROUPS = ("rest", "weak", "wide", "edge")
SEEDS = {"rest": 20261141, "weak": 20261142, "wide": 20261143, "edge": 20261144}
PROTOCOL_IDS = (0, 505, 353, 192, 405, 331, 203, 62, 467)


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def frozen_validation_starts(n):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "validation_starts.npz"
    meta_path = OUT / "validation_starts.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["n"] != n:
            raise RuntimeError("validation start set is frozen at a different N")
        z = np.load(path)
        return {g: (z[f"{g}_initial"], z[f"{g}_phase"]) for g in GROUPS}, meta
    if n & (n - 1):
        raise ValueError("N must be a power of two")
    original = screen.SEEDS.copy()
    try:
        screen.SEEDS = SEEDS.copy()
        data = screen.make_training_starts(n)
    finally:
        screen.SEEDS = original
    arrays = {}
    meta = {"n": n, "seeds": SEEDS,
            "groups": {},
            "source": "Same V17C.10 published start distributions; new independent scrambled-Sobol seeds, disjoint from P2 train set."}
    for group, (initial, phase) in data.items():
        arrays[f"{group}_initial"] = initial
        arrays[f"{group}_phase"] = phase
        meta["groups"][group] = {"initial_sha256": array_sha(initial), "phase_sha256": array_sha(phase)}
    np.savez_compressed(path, **arrays)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return data, meta


def integrate_chunk(args):
    pid, group, ids, initials, phases, protocol, p, family, k, coeff, qratio, periods, steps = args
    skip = min(periods, int(math.ceil(ld.protocol_duration_cycles(protocol))))
    result = []
    for j, initial, phase in zip(ids, initials, phases):
        r = ld.integrate(initial, phase, p, protocol, family, k, coeff,
                         periods, steps, skip, qratio,
                         a1_over_R=.0002, a2_over_R=.00189)
        result.append({
            "protocol_id": pid, "ensemble": group, "start_id": int(j),
            "periods": periods, "steps_per_period": steps, "skip_after_transition": skip,
            "transition_cycles": ld.protocol_duration_cycles(protocol),
            "category": r["category"], "retained": bool(r["retained"]),
            "completed_periods": int(r["completed_periods"]),
            "turns_after_transition": np.asarray(r["turns"]).tolist(),
            "window_turns": np.asarray(r["window_turns"]).tolist(),
            "frequency_Hz": np.asarray(r["frequency_Hz"]).tolist(),
            "r_min": np.asarray(r["r_min"]).tolist(),
            "r_max": np.asarray(r["r_max"]).tolist(),
            "abs_z_max": np.asarray(r["abs_z_max"]).tolist(),
            "minimum_separation": float(r["minimum_separation"]),
        })
    return result


def main():
    n, periods, steps, workers, chunk_size = 64, 3000, 96, 4, 8
    OUT.mkdir(parents=True, exist_ok=True)
    starts, start_meta = frozen_validation_starts(n)
    protocols_all = json.loads((HERE / "loading_screen" / "protocols.json").read_text(encoding="utf-8"))
    protocols = {pid: np.asarray(protocols_all[pid]["protocol"], dtype=np.float64) for pid in PROTOCOL_IDS}
    fixed = json.loads((HERE / "local15_parameters.json").read_text(encoding="utf-8"))
    _p_default, family, k, coeff = basin_scan.load_model()
    p = np.asarray(fixed["candidate"]["p"], dtype=np.float64)
    qratio = float(fixed["candidate"]["q2_over_q1"])
    result_path = OUT / "validation_runs.jsonl"
    done = set()
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line); done.add((x["protocol_id"], x["ensemble"], x["start_id"]))
    tasks = []
    for pid, protocol in protocols.items():
        for group in GROUPS:
            ii, ph = starts[group]
            for begin in range(0, n, chunk_size):
                ids = [j for j in range(begin, min(n, begin+chunk_size))
                       if (pid, group, j) not in done]
                if ids:
                    tasks.append((pid, group, ids,
                                  np.asarray([ii[j] for j in ids]), np.asarray([ph[j] for j in ids]),
                                  protocol, p, family, k, coeff, qratio, periods, steps))
    # compile/cache before parallel processing
    if tasks:
        ld.integrate(starts["rest"][0][0], starts["rest"][1][0], p,
                     ld.constant_protocol(), family, k, coeff, 1, steps, 0, qratio,
                     a1_over_R=.0002, a2_over_R=.00189)
    meta = {"status": "running", "protocol_ids": list(PROTOCOL_IDS), "N_per_ensemble": n,
            "ensembles": list(GROUPS), "periods": periods, "steps_per_period": steps,
            "workers": workers, "chunk_size": chunk_size, "seeds": SEEDS,
            "training_start_hashes": start_meta["groups"],
            "terminal_parameters": p.tolist(), "q2_over_q1": qratio,
            "strict_category_note": "CR_screen requires retention, no axis ambiguity, opposite directions in three late windows, and at least ten turns per particle after the transition. Other outcomes are not CR success."}
    (OUT / "validation_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"validation protocols={len(protocols)}, N={n}*4, horizon={periods} cycles, steps={steps}, pending chunks={len(tasks)}, workers={workers}", flush=True)
    t0 = time.time(); blocks = 0
    with result_path.open("a", encoding="utf-8", buffering=1) as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(integrate_chunk, x) for x in tasks]
        for fut in as_completed(futures):
            for row in fut.result(): out.write(json.dumps(row, separators=(",", ":")) + "\n")
            out.flush(); os.fsync(out.fileno())
            blocks += 1
            if blocks % 8 == 0:
                print(f"checkpoint blocks={blocks}/{len(tasks)} rows={blocks*chunk_size} elapsed={time.time()-t0:.0f}s", flush=True)
    meta["status"] = "complete"; meta["elapsed_seconds"] = time.time()-t0
    (OUT / "validation_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"complete blocks={blocks}; elapsed={meta['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
