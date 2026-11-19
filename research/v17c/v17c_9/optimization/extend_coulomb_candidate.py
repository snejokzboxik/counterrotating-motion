"""Continue saved candidate-48 trajectories at 2*C0 without replaying the prefix."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "stage3" / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=int, default=25000,
                    help="continuation length from the saved 15,000-cycle state")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--coupling-scale", type=float, default=2.0)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(HERE.parent / "baseline"))
    from basin_scan import integrate_one, load_model

    z = np.load(HERE / "stage2_speed" / "configurations.npz")
    p = np.asarray(z["p"][48], dtype=float).copy()
    p[5] *= a.coupling_scale
    qratio = float(z["qratio"][48])
    _, family, k, coeff = load_model()
    prefix_periods = 15000
    a1, a2 = 0.0002, 0.00189

    tasks = []
    source_files = {}
    for ensemble in ("rest", "weak", "wide"):
        source = RUNS / (
            f"cfg48_{ensemble}_matched_holdout_none_chi0-0_"
            f"a{a1:g}-{a2:g}_Cscale{a.coupling_scale:g}_64_15000p_64s.jsonl"
        )
        source_files[ensemble] = str(source)
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # Continue only the prior strict CR set. It became slow same-sign
            # motion at 2*C0 and missed the ten-turn threshold at 15k cycles.
            if row["category"] == "inconclusive":
                tasks.append((ensemble, row))

    dest = RUNS / (
        f"cfg48_Cscale{a.coupling_scale:g}_continued_from15000_"
        f"{a.periods}p_{a.steps}s.jsonl"
    )
    meta = dest.with_suffix(".meta.json")
    metadata = {
        "candidate_id": 48,
        "coupling_scale": a.coupling_scale,
        "prefix_periods": prefix_periods,
        "extension_periods": a.periods,
        "steps": a.steps,
        "n_tasks": len(tasks),
        "source_files": source_files,
        "phase_continuation": "phase_new = phase_original + [1, nu]*2pi*prefix_periods modulo 2pi",
        "note": "Starts from saved final_state after the 15k-cycle run; no prefix replay.",
    }
    if meta.exists() and json.loads(meta.read_text(encoding="utf-8")) != metadata:
        raise RuntimeError("extension checkpoint metadata mismatch")
    meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    done = set()
    if dest.exists():
        done = {json.loads(s)["key"] for s in dest.read_text(encoding="utf-8").splitlines() if s.strip()}
    todo = [(g, r) for g, r in tasks if f"{g}:{r['index']}" not in done]
    print(f"candidate 48 Cscale={a.coupling_scale:g}: {len(tasks)} continuations, {len(todo)} pending", flush=True)
    t0 = time.perf_counter()

    def one(task):
        ensemble, old = task
        state0 = np.asarray(old["final_state"], dtype=float)
        phase = np.asarray(old["phase"], dtype=float).copy()
        phase[0] += 2.0 * np.pi * prefix_periods
        phase[1] += 2.0 * np.pi * p[2] * prefix_periods
        phase = np.mod(phase, 2.0 * np.pi)
        result = integrate_one(
            state0, phase, p, family, k, coeff,
            a.periods, a.steps, 0, qratio, 0.0, 0,
            a1, a2, -1.0,
        )
        end, turns, windows, rmin, rmax, zmax, maxd, dmin, complete, retained, escaped, singular, category, freq = result
        return {
            "key": f"{ensemble}:{old['index']}",
            "ensemble": ensemble,
            "index": int(old["index"]),
            "previous_category": old["category"],
            "category": str(category),
            "extension_frequency_Hz": freq.tolist(),
            "extension_turns": turns.tolist(),
            "extension_window_turns": windows.tolist(),
            "combined_turns_from_initial": (np.asarray(old["turns"]) + turns).tolist(),
            "retained_through_extension": bool(retained),
            "escaped": bool(escaped),
            "singular_unresolved": bool(singular),
            "completed_extension_periods": int(complete),
            "total_periods": int(prefix_periods + complete),
            "r_min_extension": rmin.tolist(),
            "r_max_extension": rmax.tolist(),
            "z_max_extension": zmax.tolist(),
            "minimum_separation_extension": float(dmin),
            "max_abs_inner_step_dtheta_rad": maxd.tolist(),
            "final_state": end.tolist(),
            "phase_at_extension_start": phase.tolist(),
        }

    with dest.open("a", encoding="utf-8", buffering=1) as f, ThreadPoolExecutor(max_workers=a.threads) as pool:
        futures = [pool.submit(one, task) for task in todo]
        for i, future in enumerate(as_completed(futures), 1):
            f.write(json.dumps(future.result(), separators=(",", ":")) + "\n")
            if i % 16 == 0 or i == len(futures):
                print(f"continuation checkpoint {i}/{len(futures)} ({time.perf_counter()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
