"""Fresh local Sobol screen around the N=256-confirmed refine12 candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_local15_refinement_screen as base

GROUPS = ("rest", "weak", "wide", "edge")
STANDARD_GROUPS = GROUPS[:3]
N_CONFIGS = 32
N_ANCHORS = 5
N_PER_GROUP = 64
PERIODS = 2400
STEPS = 48
TRANSIENT = 200
STRICT_PERIODS = 15000
STRICT_TRANSIENT = 200
STRICT_TURNS = 10.0
SCALED_TURN_MIN = STRICT_TURNS * (PERIODS - TRANSIENT) / (STRICT_PERIODS - STRICT_TRANSIENT)
CONFIG_SEED = 20261720
START_SEEDS = {"rest": 20261721, "weak": 20261722, "wide": 20261723, "edge": 20261724}
SIZES = (0.0002, 0.00189)
EDGE_RMIN, EDGE_RMAX = 0.70, 0.90
OUT = HERE / "refine12_neighborhood_screen"

PARAMETER_RANGES = {
    "rf_scale_vs_candidate52": [0.86, 0.99],
    "B_over_A": [4.30, 5.10],
    "nu": [3.00, 3.42],
    "dc_scale_vs_rf_squared": [0.85, 1.35],
    "mass_ratio": [7.0, 9.2],
    "charge_ratio": [2.40, 3.50],
    "lambda1": [0.16, 0.26],
    "lambda2": [0.10, 0.18],
    "C_eff": [7.5e-6, 1.8e-5],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_base():
    base.GROUPS = GROUPS
    base.STANDARD_GROUPS = STANDARD_GROUPS
    base.N_CONFIGS = N_CONFIGS
    base.N_ANCHORS = N_ANCHORS
    base.N_PER_GROUP = N_PER_GROUP
    base.PERIODS = PERIODS
    base.STEPS = STEPS
    base.TRANSIENT = TRANSIENT
    base.STRICT_PERIODS = STRICT_PERIODS
    base.STRICT_TRANSIENT = STRICT_TRANSIENT
    base.STRICT_TURNS = STRICT_TURNS
    base.SCALED_TURN_MIN = SCALED_TURN_MIN
    base.CONFIG_SEED = CONFIG_SEED
    base.START_SEEDS = START_SEEDS
    base.SIZES = SIZES
    base.EDGE_RMIN, base.EDGE_RMAX = EDGE_RMIN, EDGE_RMAX
    base.OUT = OUT


def local_configs():
    previous_meta = json.loads((HERE / "local15_refinement_screen" / "meta.json").read_text(encoding="utf-8"))
    by_id = {int(c["config_id"]): c for c in previous_meta["candidates"]}
    center = by_id[12]
    stage4 = np.load(base.OPT / "stage4_candidate48" / "configurations.npz")
    base_p = np.asarray(stage4["p"][52], dtype=np.float64)
    u = qmc.Sobol(d=9, scramble=True, seed=CONFIG_SEED).random_base2(5)
    configs = []
    for i, x in enumerate(u):
        rf = 0.86 + 0.13 * x[0]
        phi = 4.30 + 0.80 * x[1]
        nu = 3.00 + 0.42 * x[2]
        dc = 0.85 + 0.50 * x[3]
        mass_ratio = 7.0 + 2.2 * x[4]
        qratio = 2.40 + 1.10 * x[5]
        lam1 = math.exp(math.log(0.16) + x[6] * math.log(0.26 / 0.16))
        lam2 = math.exp(math.log(0.10) + x[7] * math.log(0.18 / 0.10))
        coupling = math.exp(math.log(7.5e-6) + x[8] * math.log(1.8e-5 / 7.5e-6))
        p = base_p.copy()
        p[0] = base_p[0] * rf
        p[1] = p[0] * phi
        p[2] = nu
        p[3] = base_p[3] * rf * rf * dc
        p[4] = 1.0 / mass_ratio
        p[5] = coupling
        p[6], p[7] = lam1, lam2
        configs.append({
            "config_id": i, "label": f"refine12_sobol_{i:02d}",
            "source": "fresh Sobol neighborhood around N=256-confirmed refine12",
            "p": p, "qratio": qratio, "rf_scale_vs_candidate52": rf,
            "harmonic_potential_ratio_B_over_A": phi, "nu": nu,
            "dc_scale_vs_rf_squared": dc, "mass_ratio": mass_ratio,
            "charge_ratio": qratio, "lambda1": lam1, "lambda2": lam2,
            "C_eff": coupling,
        })
    for cid, source_id, label in (
        (32, 12, "refine12_anchor"), (33, 18, "refine18_anchor"),
        (34, 0, "refine00_anchor"), (35, 32, "local15_anchor"),
        (36, 35, "candidate52_anchor"),
    ):
        row = by_id[source_id]
        configs.append({
            "config_id": cid, "label": label, "source": f"P4 refinement anchor {source_id}",
            "p": np.asarray(row["p"], dtype=np.float64), "qratio": float(row["qratio"]),
        })
    center = {**center, "p": np.asarray(center["p"], dtype=np.float64)}
    return configs, center


def make_inputs():
    starts, phases = {}, {}
    for group in STANDARD_GROUPS:
        starts[group], phases[group] = base.make_initials(group, N_PER_GROUP, START_SEEDS[group])
    starts["edge"], phases["edge"] = base.make_edge_initials(N_PER_GROUP, START_SEEDS["edge"])
    return starts, phases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=min(11, os.cpu_count() or 1))
    args = parser.parse_args()
    if not 1 <= args.threads <= (os.cpu_count() or 1):
        raise ValueError(f"--threads must be in [1, {os.cpu_count() or 1}]")
    configure_base()
    OUT.mkdir(parents=True, exist_ok=True)
    configs, center = local_configs()
    starts, phases = make_inputs()
    arrays = {"p": np.asarray([c["p"] for c in configs]),
              "qratio": np.asarray([c["qratio"] for c in configs])}
    for group in GROUPS:
        arrays[f"initial_{group}"] = starts[group]
        arrays[f"phase_{group}"] = phases[group]
    input_path = OUT / "configurations_and_training_starts.npz"
    if input_path.exists():
        with np.load(input_path, allow_pickle=False) as saved:
            if any(not np.array_equal(saved[name], value) for name, value in arrays.items()):
                raise RuntimeError("saved refine12-neighborhood inputs differ from their seeds")
    else:
        np.savez_compressed(input_path, **arrays)

    metadata = {
        "study": "V17C.10 fresh Sobol neighborhood screen around refine12",
        "config_count": len(configs), "sobol_config_count": N_CONFIGS,
        "anchor_config_count": N_ANCHORS, "n_per_group": N_PER_GROUP,
        "groups": list(GROUPS), "periods": PERIODS, "steps": STEPS,
        "transient_periods": TRANSIENT, "config_seed": CONFIG_SEED,
        "start_seeds": START_SEEDS, "finite_radii_over_R": list(SIZES),
        "horizon_scaled_min_abs_turns": SCALED_TURN_MIN,
        "strict_reference": {"periods": STRICT_PERIODS, "transient": STRICT_TRANSIENT,
                             "min_abs_turns": STRICT_TURNS},
        "parameter_ranges": PARAMETER_RANGES,
        "center": {"config_id": 12, "label": center["label"],
                   "p": np.asarray(center["p"], dtype=float).tolist(),
                   "qratio": float(center["qratio"])},
        "edge_ensemble": {
            "n": N_PER_GROUP,
            "definition": "wide starts with one particle radius sampled area-uniformly in [0.70R,0.90R], alternating particle identity",
            "minimum_initial_max_radius_over_R": EDGE_RMIN,
        },
        "candidates": [
            {"config_id": c["config_id"], "label": c["label"], "source": c["source"],
             "p": np.asarray(c["p"], dtype=float).tolist(), "qratio": float(c["qratio"])}
            for c in configs
        ],
        "fixed_field_config_sha256": sha256(base.BASE.parent.parent / "v17c_5" / "input" / "config.json"),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "input_arrays_sha256": sha256(input_path),
        "note": "Exploratory training screen only. The scaled turn cutoff is linear horizon alignment, not a calibrated strict-success predictor. Keep rest, weak, wide, and edge separate.",
    }
    meta_path = OUT / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
        raise RuntimeError("refine12-neighborhood metadata mismatch; refusing incompatible resume")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "parallel_execution.json").write_text(json.dumps({
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "worker_count": args.threads, "logical_processors_reported": os.cpu_count(),
        "trajectory_parallelism": "ThreadPoolExecutor over independent Numba nogil trajectories",
        "runner_sha256": metadata["runner_sha256"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    run_path = OUT / "runs.jsonl"
    existing = {}
    if run_path.exists():
        for line in run_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["config_id"]), row["ensemble"], int(row["start_id"]))
            cid, group, i = key
            if key in existing or cid not in {c["config_id"] for c in configs} or group not in GROUPS or not 0 <= i < N_PER_GROUP:
                raise RuntimeError(f"duplicate/invalid checkpoint key {key}")
            if (not np.array_equal(np.asarray(row["initial"], dtype=float), starts[group][i])
                    or not np.array_equal(np.asarray(row["phase"], dtype=float), phases[group][i])):
                raise RuntimeError(f"checkpoint frozen-input mismatch {key}")
            cfg = configs[cid]
            if (not np.array_equal(np.asarray(row["p"], dtype=float), np.asarray(cfg["p"], dtype=float))
                    or float(row["qratio"]) != float(cfg["qratio"])
                    or int(row["seed"]) != START_SEEDS[group]):
                raise RuntimeError(f"checkpoint parameters or seed mismatch {key}")
            existing[key] = row

    model = base.load_model()
    module = sys.modules["basin_scan"]
    for config in configs:
        cid = int(config["config_id"])
        for group in GROUPS:
            todo = [i for i in range(N_PER_GROUP) if (cid, group, i) not in existing]
            if not todo:
                print(f"config{cid}/{group}: already complete {N_PER_GROUP}/{N_PER_GROUP}", flush=True)
                continue
            first = todo[0]
            module.integrate_one(starts[group][first], phases[group][first], config["p"], *model[1:],
                                 PERIODS, STEPS, TRANSIENT, qratio=config["qratio"],
                                 oseen_Re_per_speed=0.0, hydro_mode=0,
                                 a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0)
            print(f"{config['label']}/{group}: n={N_PER_GROUP}, todo={len(todo)}, workers={args.threads}", flush=True)
            tic = time.perf_counter()

            def one(i):
                result = module.integrate_one(starts[group][i], phases[group][i], config["p"], *model[1:],
                                              PERIODS, STEPS, TRANSIENT, qratio=config["qratio"],
                                              oseen_Re_per_speed=0.0, hydro_mode=0,
                                              a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0)
                return base.output_row(cid, config, group, i, starts[group][i], phases[group][i], result)

            with run_path.open("a", encoding="utf-8", buffering=1) as fout:
                with ThreadPoolExecutor(max_workers=args.threads) as pool:
                    futures = [pool.submit(one, i) for i in todo]
                    for n, future in enumerate(as_completed(futures), 1):
                        row = future.result()
                        existing[(cid, group, int(row["start_id"]))] = row
                        fout.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                        if n % 16 == 0 or n == len(todo):
                            done = sum(k[0] == cid and k[1] == group for k in existing)
                            print(f"{config['label']}/{group}: checkpoint {done}/{N_PER_GROUP} ({time.perf_counter()-tic:.1f}s)", flush=True)

    rows = list(existing.values())
    ranked, _ = base.rank_rows(rows, configs)
    (OUT / "ranked.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "study": "V17C.10 refine12 neighborhood screen summary",
        "total_rows": len(rows), "expected_rows": len(configs) * len(GROUPS) * N_PER_GROUP,
        "scaled_turn_min": SCALED_TURN_MIN,
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "top_10": ranked[:10],
    }
    (OUT / "screen_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"total_rows": len(rows), "expected_rows": summary["expected_rows"],
                      "scaled_turn_min": SCALED_TURN_MIN, "top_10": ranked[:10]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
