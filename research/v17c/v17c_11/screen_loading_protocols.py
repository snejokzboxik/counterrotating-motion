"""Checkpointed matched-start early screen for common RF/DC loading profiles."""
from __future__ import annotations

import argparse
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
BASE = HERE.parent / "v17c_9" / "baseline"
sys.path.insert(0, str(BASE))
import basin_scan  # noqa: E402
import loading_dynamics as ld  # noqa: E402

OUT = HERE / "loading_screen"
GROUPS = ("rest", "weak", "wide", "edge")
SEEDS = {"rest": 20261131, "weak": 20261132, "wide": 20261133, "edge": 20261134}
DURATIONS = np.asarray([0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30,
                        50, 100, 200, 300, 500, 1000, 3000], dtype=float)
G0S = np.asarray([0, .05, .10, .20, .35, .50, .70, .85], dtype=float)


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def make_training_starts(n: int):
    if n & (n - 1):
        raise ValueError("N must be a power of two for scrambled Sobol starts")
    starts = {}
    for group in GROUPS:
        initials, phases = basin_scan.make_initials(group if group != "edge" else "wide",
                                                      n, SEEDS[group])
        if group == "edge":
            u = qmc.Sobol(d=2, scramble=True, seed=SEEDS[group] + 7919).random_base2(int(math.log2(n)))
            which = (u[:, 1] >= .5).astype(np.int64)
            radius = np.sqrt(.70**2 + u[:, 0] * (.90**2 - .70**2))
            for j in range(n):
                b = 6 * which[j]
                old_r = math.hypot(initials[j, b], initials[j, b + 1])
                if old_r > 0:
                    initials[j, b:b+2] *= radius[j] / old_r
                else:
                    initials[j, b] = radius[j]
        starts[group] = (initials, phases)
    return starts


def save_starts(n: int):
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / "training_starts.npz"
    sidecar = OUT / "training_starts.json"
    if npz.exists() and sidecar.exists():
        old = json.loads(sidecar.read_text(encoding="utf-8"))
        if old["n"] != n:
            raise RuntimeError(f"training set already frozen at N={old['n']}; refusing regeneration")
        z = np.load(npz)
        return {g: (z[f"{g}_initial"], z[f"{g}_phase"]) for g in GROUPS}, old
    data = make_training_starts(n)
    arrays = {}
    meta = {"n": n, "seeds": SEEDS, "groups": {},
            "definition": "Fresh scrambled Sobol training starts generated from the V17C.9 basin_scan.make_initials; edge draws one particle radius area-uniformly in [0.70,0.90]R. Frozen across protocols."}
    for group, (initial, phase) in data.items():
        arrays[f"{group}_initial"] = initial
        arrays[f"{group}_phase"] = phase
        meta["groups"][group] = {"initial_sha256": sha256(initial), "phase_sha256": sha256(phase)}
    np.savez_compressed(npz, **arrays)
    sidecar.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return data, meta


def duration(u: float) -> float:
    return float(DURATIONS[min(int(u * len(DURATIONS)), len(DURATIONS) - 1)])


def make_protocols(count: int = 512):
    if count < 1 or count > 512:
        raise ValueError("protocol count must be 1..512")
    uu = qmc.Sobol(d=12, scramble=True, seed=20261135).random_base2(9)
    protocols = [ld.constant_protocol()]
    descriptions = [{"id": 0, "family": "constant", "protocol": protocols[0].tolist()}]
    for u in uu[:max(0, count - 1)]:
        shape = min(4, int(u[1] * 5))
        power = [0.5, 1, 2, 3, 4][min(4, int(u[2] * 5))]
        g0 = float(G0S[min(7, int(u[3] * 8))])
        scheme = min(7, int(u[0] * 8))
        da, db, dd = duration(u[4]), duration(u[5]), duration(u[6])
        delay_b = float(u[7] * min(10.0, max(0.0, da)))
        delay_dc = float(u[8] * min(10.0, max(0.0, da)))
        dc_mode = int(u[9] < 0.5)
        if scheme == 0:  # all channels together
            d = duration(u[4]); gdc = g0
            p = ld.ramp_protocol(duration=d, shape=shape, power=power, g0=g0,
                                 g0_dc=gdc, dc_mode=0)
            family = "all_together"
        elif scheme == 1:  # RF together; DC follows RF squared
            d = duration(u[4])
            p = ld.ramp_protocol(duration=d, shape=shape, power=power, g0=g0,
                                 g0_dc=g0*g0, dc_mode=1)
            family = "dc_rf_squared"
        elif scheme in (2, 3):  # independent simultaneous rates, optional DC delay
            p = ld.ramp_protocol(duration=da, duration_b=db, duration_dc=dd,
                                 delay_b=delay_b if scheme == 3 else 0.0,
                                 delay_dc=delay_dc if scheme == 3 else 0.0,
                                 shape=shape, power=power, g0=g0,
                                 g0_b=float(G0S[min(7, int(u[10]*8))]),
                                 g0_dc=float(G0S[min(7, int(u[11]*8))]),
                                 dc_mode=0)
            family = "independent_delayed" if scheme == 3 else "independent_rates"
        elif scheme in (4, 5):  # RF first, delayed DC or DC reduced initially
            p = ld.ramp_protocol(duration=da, duration_b=db, duration_dc=dd,
                                 delay_dc=delay_dc, shape=shape, power=power,
                                 g0=g0, g0_dc=float(G0S[min(7, int(u[10]*8))]),
                                 dc_mode=0)
            family = "rf_then_dc" if scheme == 4 else "reduced_initial_dc"
        elif scheme == 6:  # B first, followed by A
            p = ld.b_first_protocol(duration_a=da, duration_b=db, duration_dc=dd,
                                    delay_a=delay_b, delay_dc=delay_dc,
                                    shape=shape, power=power, g0=g0)
            family = "B_then_A"
        else:  # capture at a common elevated/reduced level, hold, then transition
            cap_a = 0.65 + 1.35*u[4]
            cap_b = 0.65 + 1.35*u[5]
            cap_dc = 0.35 + 1.15*u[6]
            ramp = duration(u[7]); hold = float(duration(u[8]) if u[8] < .55 else 0.0)
            trans = duration(u[9])
            p = ld.capture_protocol(capture_a=cap_a, capture_b=cap_b,
                                    capture_dc=cap_dc, ramp_cycles=ramp,
                                    hold_cycles=hold, transition_cycles=trans,
                                    shape=shape, transition_shape=min(3, shape),
                                    power=power, start_a=g0, start_b=g0,
                                    start_dc=float(G0S[min(7, int(u[10]*8))]))
            family = "capture_hold_transition"
        i = len(protocols)
        protocols.append(p)
        descriptions.append({"id": i, "family": family,
                             "duration_cycles": ld.protocol_duration_cycles(p),
                             "protocol": p.tolist()})
    return protocols, descriptions


def save_protocols(count: int = 512):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocols.json"
    protocols, desc = make_protocols(count)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if len(old) >= count and old[:count] == desc:
            return [np.asarray(d["protocol"], dtype=np.float64) for d in old[:count]], old
        raise RuntimeError("saved protocol sequence differs; refusing to overwrite checkpoint provenance")
    path.write_text(json.dumps(desc, indent=2) + "\n", encoding="utf-8")
    return protocols, desc


def run_block(args):
    protocol_id, group, start_ids, initials, phases, protocol, p, family, k, coeff, qratio, periods, steps = args
    skip = min(periods, int(math.ceil(ld.protocol_duration_cycles(protocol))))
    out = []
    for j, (initial, phase) in zip(start_ids, zip(initials, phases)):
        r = ld.integrate(initial, phase, p, protocol, family, k, coeff,
                         periods, steps, skip, qratio,
                         a1_over_R=.0002, a2_over_R=.00189,
                         log_every=steps)
        last_tau = float(r["log_tau"][-1]) if len(r["log_tau"]) else 0.0
        out.append({
            "protocol_id": protocol_id, "ensemble": group, "start_id": j,
            "screen_periods": periods, "steps_per_period": steps,
            "transition_cycles": ld.protocol_duration_cycles(protocol),
            "transition_completed": ld.protocol_duration_cycles(protocol) <= periods,
            "category": r["category"], "retained": bool(r["retained"]),
            "wall_or_particle_contact": r["category"] in ("finite_particle_wall_contact", "finite_particle_contact"),
            "completed_periods": int(r["completed_periods"]), "last_tau": last_tau,
            "turns_post_transition": r["turns"].tolist(),
            "window_turns": r["window_turns"].tolist(),
            "r_min": r["r_min"].tolist(), "r_max": r["r_max"].tolist(),
            "abs_z_max": r["abs_z_max"].tolist(),
            "minimum_separation": float(r["minimum_separation"]),
            "frequency_Hz_screen": r["frequency_Hz"].tolist(),
        })
    return out


def run_screen(count: int, n: int, periods: int, steps: int, workers: int):
    OUT.mkdir(parents=True, exist_ok=True)
    starts, starts_meta = save_starts(max(16, n))
    protocols, desc = save_protocols(512)
    protocols, desc = protocols[:count], desc[:count]
    fixed = json.loads((HERE / "local15_parameters.json").read_text(encoding="utf-8"))
    _baseline_p, family, k, coeff = basin_scan.load_model()
    p = np.asarray(fixed["candidate"]["p"], dtype=np.float64)
    qratio = float(fixed["candidate"]["q2_over_q1"])
    result_path = OUT / "early_screen.jsonl"
    done = set()
    if result_path.exists():
        with result_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done.add((row["protocol_id"], row["ensemble"], row["start_id"]))
    tasks = []
    for pid, protocol in enumerate(protocols):
        for group in GROUPS:
            ii, pp = starts[group]
            remaining = [(j, ii[j], pp[j]) for j in range(n)
                         if (pid, group, j) not in done]
            if not remaining:
                continue
            # The worker gets any missing tail; a fresh run starts at id zero.
            if [j for j, _, _ in remaining] != list(range(remaining[0][0], n)):
                # Resume supports noncontiguous gaps too; keep explicit IDs in a local wrapper.
                for j, init, ph in remaining:
                    tasks.append((pid, group, [j], np.asarray([init]), np.asarray([ph]), protocol,
                                  p, family, k, coeff, qratio, periods, steps))
            else:
                tasks.append((pid, group, [x[0] for x in remaining], np.asarray([x[1] for x in remaining]),
                              np.asarray([x[2] for x in remaining]), protocol,
                              p, family, k, coeff, qratio, periods, steps,
                              ))

    # Warm the compiler before parallel dispatch; this is a simulation of one frozen
    # start and also verifies the new loading wrapper remains callable.
    if tasks:
        a = starts["rest"][0][0]; ph = starts["rest"][1][0]
        ld.integrate(a, ph, p, ld.constant_protocol(), family, k, coeff,
                     1, steps, 0, qratio, a1_over_R=.0002, a2_over_R=.00189)
    meta = {"protocol_count": count, "N_per_ensemble": n, "ensembles": list(GROUPS),
            "periods": periods, "steps_per_period": steps, "workers": workers,
            "training_start_hashes": starts_meta["groups"],
            "fixed_terminal_configuration": "local15",
            "terminal_parameter_vector": p.tolist(), "q2_over_q1": qratio,
            "protocol_sha256": hashlib.sha256((OUT/"protocols.json").read_bytes()).hexdigest(),
            "status": "running", "start_time_unix": time.time(),
            "note": "Short screen only. A passing early-retention proxy is not a strict CR result."}
    (OUT / "screen_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"protocols={count}, starts={n}*4, tasks={len(tasks)}, horizon={periods}, steps={steps}, workers={workers}", flush=True)
    completed_blocks = 0
    wall0 = time.time()
    with result_path.open("a", encoding="utf-8", buffering=1) as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_block, task) for task in tasks]
        for fut in as_completed(futures):
            rows = fut.result()
            for row in rows:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
            out.flush(); os.fsync(out.fileno())
            completed_blocks += 1
            if completed_blocks % 64 == 0:
                print(f"checkpoint blocks={completed_blocks}/{len(tasks)} rows~{completed_blocks*n} elapsed={time.time()-wall0:.1f}s", flush=True)
    meta["status"] = "complete"
    meta["elapsed_seconds"] = time.time() - wall0
    (OUT / "screen_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"complete blocks={completed_blocks}; elapsed={meta['elapsed_seconds']:.2f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-count", type=int, default=64)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--periods", type=int, default=48)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run_screen(args.protocol_count, args.n, args.periods, args.steps, args.workers)


if __name__ == "__main__":
    main()
