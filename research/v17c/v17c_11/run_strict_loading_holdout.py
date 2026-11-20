"""Independent long-horizon holdout for local15 and selected loading schedules."""
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
OUT = HERE / "strict_loading_holdout"
BASE = HERE.parent / "v17c_9" / "baseline"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))
import basin_scan
import loading_dynamics as ld
import screen_loading_protocols as first

GROUPS = ("rest", "weak", "wide", "edge")
SEEDS = {"rest": 20261201, "weak": 20261202, "wide": 20261203, "edge": 20261204}
N = 64
POST_LOAD_PERIODS = 15_000
RELAX_PERIODS = 200
STEPS = 96
WORKERS = 4
BATCH = 4


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def load_protocols():
    p2 = json.loads((HERE / "loading_screen" / "protocols.json").read_text(encoding="utf-8"))
    p6 = json.loads((HERE / "adaptive_loading" / "adaptive_protocols.json").read_text(encoding="utf-8"))
    p7 = json.loads((HERE / "adaptive_loading_v2" / "adaptive_protocols.json").read_text(encoding="utf-8"))
    picked = [
        {"id": 0, "label": "local15_constant", "source_protocol_id": 0,
         "source": "local15 constant protocol", "vector": ld.constant_protocol().tolist()},
        {"id": 1, "label": "P2_505", "source_protocol_id": 505,
         "source": "P2 loading screen", "vector": p2[505]["protocol"]},
        {"id": 2, "label": "P2_353", "source_protocol_id": 353,
         "source": "P2 loading screen", "vector": p2[353]["protocol"]},
        {"id": 3, "label": "P6_40", "source_protocol_id": 40,
         "source": "P6 adaptive loading screen", "vector": p6[40]["protocol"]},
        {"id": 4, "label": "P7_292", "source_protocol_id": 292,
         "source": "P7 adaptive loading screen", "vector": p7[292]["protocol"]},
    ]
    for d in picked:
        prot = np.asarray(d["vector"], dtype=np.float64)
        d["duration_cycles"] = ld.protocol_duration_cycles(prot)
        d["protocol_vector_19"] = prot.tolist()
    return picked


def starts_frozen():
    OUT.mkdir(parents=True, exist_ok=True)
    npz, meta_path = OUT / "holdout_starts.npz", OUT / "holdout_starts.json"
    if npz.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["n"] != N or meta["seeds"] != SEEDS:
            raise RuntimeError("holdout start file does not match requested N/seed")
        z = np.load(npz)
        return {g: (z[f"{g}_initial"], z[f"{g}_phase"]) for g in GROUPS}, meta
    old = first.SEEDS.copy()
    try:
        first.SEEDS = SEEDS.copy()
        data = first.make_training_starts(N)
    finally:
        first.SEEDS = old
    arrays, meta = {}, {"n": N, "seeds": SEEDS, "groups": {},
        "definition": "Independent scrambled Sobol holdout from V17C.10 generator; edge has area-uniform radius [0.70,0.90]R."}
    for g, (initial, phase) in data.items():
        arrays[f"{g}_initial"], arrays[f"{g}_phase"] = initial, phase
        meta["groups"][g] = {"initial_sha256": sha(initial), "phase_sha256": sha(phase)}
    np.savez_compressed(npz, **arrays)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return data, meta


def run_batch(args):
    desc, group, ids, initials, phases, p, field_family, modes, coeff, qratio = args
    prot = np.asarray(desc["protocol_vector_19"], dtype=np.float64)
    duration = float(desc["duration_cycles"])
    skip = int(math.ceil(duration)) + RELAX_PERIODS
    periods = skip + POST_LOAD_PERIODS
    result_rows = []
    for sid, initial, phase in zip(ids, initials, phases):
        r = ld.integrate(initial, phase, p, prot, field_family, modes, coeff,
                         periods, STEPS, skip, qratio,
                         a1_over_R=0.0002, a2_over_R=0.00189)
        result_rows.append({
            "protocol_id": desc["id"], "protocol_label": desc["label"],
            "ensemble": group, "start_id": int(sid),
            "initial_state": initial.tolist(), "initial_rf_phase": phase.tolist(),
            "final_state": r["final_state"].tolist(),
            "category": r["category"], "strict_success": r["category"] == "CR_screen",
            "retained": bool(r["retained"]), "completed_periods": int(r["completed_periods"]),
            "transition_cycles": duration, "relax_periods": RELAX_PERIODS,
            "measured_periods": POST_LOAD_PERIODS, "steps_per_period": STEPS,
            "turns": r["turns"].tolist(), "window_turns": r["window_turns"].tolist(),
            "frequency_Hz": r["frequency_Hz"].tolist(),
            "r_min": r["r_min"].tolist(), "r_max": r["r_max"].tolist(),
            "max_abs_z": r["abs_z_max"].tolist(),
            "minimum_separation": float(r["minimum_separation"]),
            "max_abs_inner_step_dtheta_rad": r["max_abs_inner_step_dtheta_rad"].tolist(),
        })
    return result_rows


def load_checkpoint_keys(path):
    if not path.exists():
        return set()
    data, keys = path.read_bytes(), set()
    lines = data.splitlines(keepends=True)
    offset = 0
    for i, raw in enumerate(lines):
        try:
            row = json.loads(raw.strip())
        except json.JSONDecodeError:
            if any(x.strip() for x in lines[i + 1:]):
                raise RuntimeError(f"corrupt non-final checkpoint row at byte {offset}")
            with path.open("r+b") as f:
                f.truncate(offset)
            break
        key = (row["protocol_id"], row["ensemble"], row["start_id"])
        if key in keys:
            raise RuntimeError(f"duplicate holdout key {key}")
        keys.add(key); offset += len(raw)
    if lines and offset == len(data) and data and not data.endswith(b"\n"):
        with path.open("ab") as f:
            f.write(b"\n")
    return keys


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    starts, start_meta = starts_frozen()
    protocols = load_protocols()
    (OUT / "holdout_protocols.json").write_text(json.dumps(protocols, indent=2) + "\n", encoding="utf-8")
    fixed = json.loads((HERE / "local15_parameters.json").read_text(encoding="utf-8"))
    p = np.asarray(fixed["candidate"]["p"], dtype=np.float64)
    qratio = float(fixed["candidate"]["q2_over_q1"])
    _, field_family, modes, coeff = basin_scan.load_model()
    outpath = OUT / "holdout_runs.jsonl"
    done = load_checkpoint_keys(outpath)
    tasks = []
    for d in protocols:
        for g in GROUPS:
            initials, phases = starts[g]
            remaining = [i for i in range(N) if (d["id"], g, i) not in done]
            for b in range(0, len(remaining), BATCH):
                ids = remaining[b:b+BATCH]
                tasks.append((d, g, ids, initials[ids], phases[ids], p,
                              field_family, modes, coeff, qratio))
    if tasks:
        ld.integrate(starts["rest"][0][0], starts["rest"][1][0], p,
                     ld.constant_protocol(), field_family, modes, coeff, 1, STEPS, 0,
                     qratio, a1_over_R=0.0002, a2_over_R=0.00189)
    meta = {"status": "running", "protocol_count": len(protocols), "N_per_ensemble": N,
        "ensembles": list(GROUPS), "seeds": SEEDS, "start_hashes": start_meta["groups"],
        "RF_steps_per_primary_period": STEPS, "post_load_periods": POST_LOAD_PERIODS,
        "relax_periods": RELAX_PERIODS, "workers": WORKERS, "batch_size": BATCH,
        "candidate_vector": p.tolist(), "q2_over_q1": qratio,
        "strict_success_category": "CR_screen",
        "warning": "This holdout chooses no new parameters; screens only fixed controls and preselected protocol #292."}
    (OUT / "holdout_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"strict holdout protocols={len(protocols)}, N={N}x4, post-load={POST_LOAD_PERIODS}, "
          f"tasks={len(tasks)}, workers={WORKERS}", flush=True)
    t0, completed = time.time(), 0
    with outpath.open("a", encoding="utf-8", buffering=1) as out, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_batch, task) for task in tasks]
        for f in as_completed(futures):
            for row in f.result():
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
            out.flush(); os.fsync(out.fileno()); completed += 1
            if completed % 8 == 0:
                print(f"checkpoint batches={completed}/{len(tasks)} rows={completed*BATCH} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    meta["status"] = "complete"
    meta["elapsed_seconds"] = time.time() - t0
    (OUT / "holdout_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"complete batches={completed}; elapsed={meta['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
