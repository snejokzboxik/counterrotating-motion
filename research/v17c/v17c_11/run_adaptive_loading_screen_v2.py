"""1024-schedule fresh-start screen, checkpointed by protocol/ensemble blocks."""
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
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
OUT = HERE / "adaptive_loading_v2"
BASE = HERE.parent / "v17c_9" / "baseline"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))
import basin_scan
import loading_dynamics as ld
import screen_loading_protocols as first

GROUPS = ("rest", "weak", "wide", "edge")
SEEDS = {"rest": 20261171, "weak": 20261172, "wide": 20261173, "edge": 20261174}
N_PROTOCOLS = 1024
N = 32
PERIODS = 48
STEPS = 96
WORKERS = 4


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def logu(u, lo, hi):
    return float(math.exp(math.log(lo) + u * math.log(hi / lo)))


def gen_protocols():
    p2 = json.loads((HERE / "loading_screen" / "protocols.json").read_text(encoding="utf-8"))
    leader = json.loads((HERE / "best_loading_protocol.json").read_text(encoding="utf-8"))
    out = [
        {"id": 0, "family": "anchor_local15_constant", "protocol": ld.constant_protocol().tolist()},
        {"id": 1, "family": "anchor_P2_505", "protocol": p2[505]["protocol"]},
        {"id": 2, "family": "anchor_P2_353", "protocol": p2[353]["protocol"]},
        {"id": 3, "family": "anchor_P6_40", "protocol": leader["protocol_vector_19"]},
    ]
    for i, d in enumerate(out):
        d["duration_cycles"] = ld.protocol_duration_cycles(np.asarray(d["protocol"], dtype=float))

    u = qmc.Sobol(d=16, scramble=True, seed=20261175).random_base2(10)
    powers = (0.5, 1.0, 2.0, 3.0, 4.0)
    for a in u[: N_PROTOCOLS - len(out)]:
        shape = min(4, int(a[1] * 5))
        power = powers[min(4, int(a[2] * 5))]
        family_u = a[0]
        if family_u < 0.50:  # targeted neighborhood of early leader #40
            prot = ld.ramp_protocol(
                duration=logu(a[3], 0.35, 4.0),
                duration_b=logu(a[4], 0.10, 3.0),
                duration_dc=logu(a[5], 0.10, 4.0),
                delay_dc=float(a[6] * 4.0),
                shape=shape, power=power,
                g0=float(0.015 + 0.335 * a[7]),
                g0_b=float(0.20 + 0.65 * a[8]),
                g0_dc=float(0.01 + 0.44 * a[9]), dc_mode=0,
            )
            label = "local40_delayed_dc_neighborhood"
        elif family_u < 0.68:  # #505-like initially reduced RF, late DC
            prot = ld.ramp_protocol(
                duration=logu(a[3], 0.35, 16.0),
                duration_b=logu(a[4], 0.10, 4.0),
                duration_dc=logu(a[5], 0.10, 6.0),
                delay_dc=float(a[6] * 8.0), shape=shape, power=power,
                g0=float(0.02 + 0.38 * a[7]),
                g0_b=float(0.02 + 0.55 * a[8]),
                g0_dc=float(0.02 + 0.58 * a[9]), dc_mode=0,
            )
            label = "P2_505_delayed_dc_neighborhood"
        elif family_u < 0.80:  # synchronized DC proportional to RF squared
            g0 = float(0.02 + 0.83 * a[7])
            prot = ld.ramp_protocol(duration=logu(a[3], 0.10, 30.0),
                                    shape=shape, power=power, g0=g0,
                                    g0_dc=g0 * g0, dc_mode=1)
            label = "synchronized_RF_DC_squared"
        elif family_u < 0.93:  # two-stage temporary enhanced capture
            prot = ld.capture_protocol(
                capture_a=float(0.80 + 1.10 * a[7]),
                capture_b=float(0.80 + 1.10 * a[8]),
                capture_dc=float(0.50 + 0.90 * a[9]),
                ramp_cycles=logu(a[3], 0.10, 5.0),
                hold_cycles=float(5.0 * a[4]),
                transition_cycles=logu(a[5], 0.10, 20.0),
                shape=shape, transition_shape=min(3, shape), power=power,
                start_a=float(0.01 + 0.50 * a[10]),
                start_b=float(0.01 + 0.50 * a[11]),
                start_dc=float(0.01 + 0.50 * a[12]),
            )
            label = "two_stage_capture"
        else:  # B first, then A and DC, with shared starts for all particles
            prot = ld.b_first_protocol(
                duration_a=logu(a[3], 0.10, 18.0),
                duration_b=logu(a[4], 0.10, 8.0),
                duration_dc=logu(a[5], 0.10, 10.0),
                delay_a=float(a[6] * 6.0), delay_dc=float(a[7] * 5.0),
                shape=shape, power=power, g0=float(0.01 + 0.50 * a[8]), dc_mode=0,
            )
            label = "B_first_shared_envelope"
        out.append({"id": len(out), "family": label,
                    "duration_cycles": ld.protocol_duration_cycles(prot),
                    "protocol": prot.tolist()})
    if len(out) != N_PROTOCOLS:
        raise RuntimeError(f"expected {N_PROTOCOLS} protocols, got {len(out)}")
    if max(d["duration_cycles"] for d in out) > PERIODS:
        raise RuntimeError("at least one protocol does not reach terminal parameters by screen horizon")
    return out


def starts_frozen():
    OUT.mkdir(parents=True, exist_ok=True)
    npz, meta_path = OUT / "training_starts.npz", OUT / "training_starts.json"
    if npz.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["n"] != N or meta["seeds"] != SEEDS:
            raise RuntimeError("frozen V2 start set conflicts with requested N/seeds")
        z = np.load(npz)
        return {g: (z[f"{g}_initial"], z[f"{g}_phase"]) for g in GROUPS}, meta
    old = first.SEEDS.copy()
    try:
        first.SEEDS = SEEDS.copy()
        data = first.make_training_starts(N)
    finally:
        first.SEEDS = old
    arrays, meta = {}, {"n": N, "seeds": SEEDS, "groups": {},
        "definition": "Fresh V17C.10 distributions with independent scrambled Sobol seeds; edge draws one radius area-uniformly in [0.70,0.90]R."}
    for g, (initial, phase) in data.items():
        arrays[f"{g}_initial"], arrays[f"{g}_phase"] = initial, phase
        meta["groups"][g] = {"initial_sha256": sha(initial), "phase_sha256": sha(phase)}
    np.savez_compressed(npz, **arrays)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return data, meta


def block(args):
    pid, group, family_name, ids, initials, phases, protocol, p, field_family, modes, coeff, qratio = args
    duration = ld.protocol_duration_cycles(protocol)
    skip = min(PERIODS, int(math.ceil(duration)))
    rows = []
    for sid, initial, phase in zip(ids, initials, phases):
        r = ld.integrate(initial, phase, p, protocol, field_family, modes, coeff,
                         PERIODS, STEPS, skip, qratio,
                         a1_over_R=0.0002, a2_over_R=0.00189)
        rows.append({
            "protocol_id": pid, "family": family_name, "ensemble": group,
            "start_id": sid, "transition_cycles": duration,
            "transition_complete": duration <= PERIODS,
            "category": r["category"], "retained": bool(r["retained"]),
            "completed_periods": int(r["completed_periods"]),
            "r_min": r["r_min"].tolist(),
            "minimum_separation": float(r["minimum_separation"]),
            "window_turns": r["window_turns"].tolist(), "turns": r["turns"].tolist(),
        })
    return rows


def load_checkpoint_keys(path):
    """Read durable JSONL keys and trim a partial final line after power loss."""
    if not path.exists():
        return set()
    data = path.read_bytes()
    offset = 0
    keys = set()
    lines = data.splitlines(keepends=True)
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            offset += len(raw)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if any(x.strip() for x in lines[i + 1:]):
                raise RuntimeError(f"corrupt non-final checkpoint row in {path} at byte {offset}")
            with path.open("r+b") as f:
                f.truncate(offset)
            break
        key = (row["protocol_id"], row["ensemble"], row["start_id"])
        if key in keys:
            raise RuntimeError(f"duplicate checkpoint key: {key}")
        keys.add(key)
        offset += len(raw)
    if lines and offset == len(data) and not data.endswith(b"\n"):
        with path.open("ab") as f:
            f.write(b"\n")
    return keys


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    starts, start_meta = starts_frozen()
    protocols = gen_protocols()
    (OUT / "adaptive_protocols.json").write_text(json.dumps(protocols, indent=2) + "\n", encoding="utf-8")
    fixed = json.loads((HERE / "local15_parameters.json").read_text(encoding="utf-8"))
    p = np.asarray(fixed["candidate"]["p"], dtype=float)
    qratio = float(fixed["candidate"]["q2_over_q1"])
    _, field_family, modes, coeff = basin_scan.load_model()
    outpath = OUT / "adaptive_runs.jsonl"
    done = load_checkpoint_keys(outpath)
    tasks = []
    for d in protocols:
        protocol = np.asarray(d["protocol"], dtype=float)
        for g in GROUPS:
            initial, phases = starts[g]
            todo = [i for i in range(N) if (d["id"], g, i) not in done]
            if todo:
                tasks.append((d["id"], g, d["family"], todo, initial[todo], phases[todo],
                              protocol, p, field_family, modes, coeff, qratio))
    # Trigger compilation before the parallel worker pool starts.
    if tasks:
        ld.integrate(starts["rest"][0][0], starts["rest"][1][0], p,
                     ld.constant_protocol(), field_family, modes, coeff, 1, STEPS, 0,
                     qratio, a1_over_R=0.0002, a2_over_R=0.00189)
    meta = {"status": "running", "protocols": N_PROTOCOLS, "N_per_ensemble": N,
        "periods": PERIODS, "steps_per_period": STEPS, "workers": WORKERS,
        "ensembles": list(GROUPS), "seeds": SEEDS, "start_hashes": start_meta["groups"],
        "terminal_parameters": p.tolist(), "q2_over_q1": qratio,
        "sobol_seed": 20261175,
        "screen_semantics": "early retention only; not long-term counter-rotation probability"}
    (OUT / "adaptive_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"adaptive V2 protocols={N_PROTOCOLS}, N={N}*4, horizon={PERIODS}, "
          f"tasks={len(tasks)}, workers={WORKERS}", flush=True)
    t0, blocks = time.time(), 0
    with outpath.open("a", encoding="utf-8", buffering=1) as out, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(block, task) for task in tasks]
        for f in as_completed(futures):
            for row in f.result():
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
            out.flush(); os.fsync(out.fileno()); blocks += 1
            if blocks % 64 == 0:
                print(f"checkpoint blocks={blocks}/{len(tasks)} rows={blocks*N} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    meta["status"] = "complete"
    meta["elapsed_seconds"] = time.time() - t0
    (OUT / "adaptive_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"complete blocks={blocks}; elapsed={meta['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
