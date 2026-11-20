"""Fresh Sobol refinement around local15 using a horizon-scaled turn proxy."""
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
from pathlib import Path

for _name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
V17C9 = HERE.parent / "v17c_9"
BASE = V17C9 / "baseline"
OPT = V17C9 / "optimization"
OUT = HERE / "local15_refinement_screen"
sys.path.insert(0, str(BASE))
from basin_scan import integrate_one, load_model, make_initials

GROUPS = ("rest", "weak", "wide", "edge")
STANDARD_GROUPS = GROUPS[:3]
N_CONFIGS = 32
N_ANCHORS = 4
N_PER_GROUP = 64
PERIODS = 2400
STEPS = 48
TRANSIENT = 200
STRICT_PERIODS = 15000
STRICT_TRANSIENT = 200
STRICT_TURNS = 10.0
SCALED_TURN_MIN = STRICT_TURNS * (PERIODS - TRANSIENT) / (STRICT_PERIODS - STRICT_TRANSIENT)
CONFIG_SEED = 20261430
START_SEEDS = {"rest": 20261440, "weak": 20261441, "wide": 20261442, "edge": 20261443}
SIZES = (0.0002, 0.00189)
EDGE_RMIN, EDGE_RMAX = 0.70, 0.90
ANCHOR_KEYS = {32: "local15", 33: "local06", 34: "config17", 35: "candidate52"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_screen_configs():
    ranked = json.loads((HERE / "focused_edge_screen" / "ranked.json").read_text(encoding="utf-8"))
    by_id = {int(row["config_id"]): row for row in ranked}
    local15 = by_id[15]
    stage4 = np.load(OPT / "stage4_candidate48" / "configurations.npz")
    base = np.asarray(stage4["p"][52], dtype=np.float64)

    # Perturb around local15 while retaining useful coverage beyond the earlier P3 box.
    u = qmc.Sobol(d=9, scramble=True, seed=CONFIG_SEED).random_base2(5)
    configs = []
    for i, x in enumerate(u):
        rf = 0.88 + 0.14 * x[0]
        phi = 3.80 + 1.10 * x[1]
        nu = 2.90 + 0.45 * x[2]
        dc = 0.90 + 0.60 * x[3]
        mass_ratio = 8.0 + 2.5 * x[4]
        qratio = 2.60 + 1.10 * x[5]
        lam1 = math.exp(math.log(0.16) + x[6] * math.log(0.26 / 0.16))
        lam2 = math.exp(math.log(0.09) + x[7] * math.log(0.17 / 0.09))
        coupling = math.exp(math.log(6.0e-6) + x[8] * math.log(1.4e-5 / 6.0e-6))
        p = base.copy()
        p[0] = base[0] * rf
        p[1] = p[0] * phi
        p[2] = nu
        p[3] = base[3] * rf * rf * dc
        p[4] = 1.0 / mass_ratio
        p[5] = coupling
        p[6], p[7] = lam1, lam2
        configs.append({
            "config_id": i, "label": f"refine_sobol_{i:02d}",
            "source": "fresh Sobol refinement around P3 local15",
            "p": p, "qratio": qratio, "rf_scale_vs_candidate52": rf,
            "harmonic_potential_ratio_B_over_A": phi, "nu": nu,
            "dc_scale_vs_rf_squared": dc, "mass_ratio": mass_ratio,
            "charge_ratio": qratio, "lambda1": lam1, "lambda2": lam2,
            "C_eff": coupling,
        })
    for cid, p3id, label in (
        (32, 15, "local15_anchor"), (33, 6, "local06_anchor"),
        (34, 33, "config17_anchor"), (35, 36, "candidate52_anchor"),
    ):
        row = by_id[p3id]
        configs.append({
            "config_id": cid, "label": label, "source": f"P3 focused anchor {p3id}",
            "p": np.asarray(row["p"], dtype=np.float64), "qratio": float(row["qratio"]),
        })
    return configs


def make_edge_initials(count: int, seed: int):
    initial, phases = make_initials("wide", count, seed)
    u = qmc.Sobol(d=2, scramble=True, seed=seed + 7919).random_base2(int(math.log2(count)))
    forced_r = np.sqrt(EDGE_RMIN**2 + u[:, 0] * (EDGE_RMAX**2 - EDGE_RMIN**2))
    forced_phi = 2.0 * math.pi * u[:, 1]
    for i in range(count):
        offset = 0 if i % 2 == 0 else 6
        initial[i, offset] = forced_r[i] * math.cos(forced_phi[i])
        initial[i, offset + 1] = forced_r[i] * math.sin(forced_phi[i])
    return initial, phases


def make_inputs():
    starts, phases = {}, {}
    for group in STANDARD_GROUPS:
        starts[group], phases[group] = make_initials(group, N_PER_GROUP, START_SEEDS[group])
    starts["edge"], phases["edge"] = make_edge_initials(N_PER_GROUP, START_SEEDS["edge"])
    return starts, phases


def proxy_label(row):
    turns = np.asarray(row["turns"], dtype=float)
    windows = np.asarray(row["window_turns"], dtype=float)
    rmin = np.asarray(row["r_min"], dtype=float)
    if not row["retained"] or np.min(rmin) < 0.02 or np.min(np.abs(turns)) < SCALED_TURN_MIN:
        return False
    signs = np.where(np.abs(windows) < 0.05, 0.0, np.sign(windows))
    return bool(np.all(signs != 0.0) and np.all(signs[:, 0] * signs[:, 1] < 0.0))


def read_rows(path: Path, starts: np.ndarray, phases: np.ndarray):
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            i = int(row["start_id"])
            cid = int(row["config_id"])
            key = (cid, i)
            if key in rows or not 0 <= i < N_PER_GROUP:
                raise RuntimeError(f"duplicate/invalid task in checkpoint: {key}")
            if (not np.array_equal(np.asarray(row["initial"], dtype=float), starts[i])
                    or not np.array_equal(np.asarray(row["phase"], dtype=float), phases[i])):
                raise RuntimeError(f"checkpoint input mismatch: {key}")
            rows[key] = row
    return rows


def output_row(cid, config, group, i, initial, phase, result):
    (end, turns, windows, rmin, rmax, zmax, maxd, dmin, complete, retained,
     escaped, singular, category, freq) = result
    row = {
        "config_id": cid, "label": config["label"], "ensemble": group,
        "start_id": int(i), "seed": START_SEEDS[group],
        "p": np.asarray(config["p"], dtype=float).tolist(), "qratio": float(config["qratio"]),
        "initial": initial.tolist(), "phase": phase.tolist(), "final_state": end.tolist(),
        "turns": turns.tolist(), "window_turns": windows.tolist(), "frequency_Hz": freq.tolist(),
        "r_min": rmin.tolist(), "r_max": rmax.tolist(), "z_max": zmax.tolist(),
        "max_abs_inner_step_dtheta_rad": maxd.tolist(), "minimum_separation": float(dmin),
        "completed_periods": int(complete), "retained": bool(retained), "escaped": bool(escaped),
        "singular_unresolved": bool(singular), "category": str(category),
        "proxy_cr_scaled_turns": False,
        "hydro_model": "none", "a1_over_R": SIZES[0], "a2_over_R": SIZES[1],
    }
    row["proxy_cr_scaled_turns"] = proxy_label(row)
    return row


def rank_rows(rows, configs):
    stats = {c["config_id"]: {g: {"n": 0, "cr": 0, "retained": 0, "categories": Counter()}
                              for g in GROUPS} for c in configs}
    for row in rows:
        s = stats[int(row["config_id"])][row["ensemble"]]
        s["n"] += 1
        s["cr"] += bool(row["proxy_cr_scaled_turns"])
        s["retained"] += bool(row["retained"])
        s["categories"][row["category"]] += 1
    ranked = []
    for config in configs:
        by_group = stats[config["config_id"]]
        if any(by_group[g]["n"] != N_PER_GROUP for g in GROUPS):
            continue
        rates = {g: by_group[g]["cr"] / N_PER_GROUP for g in GROUPS}
        retention = {g: by_group[g]["retained"] / N_PER_GROUP for g in GROUPS}
        ranked.append({
            "config_id": config["config_id"], "label": config["label"], "source": config["source"],
            "proxy_by_group": rates, "retention_by_group": retention,
            "categories_by_group": {g: dict(by_group[g]["categories"]) for g in GROUPS},
            "objective": [min(rates.values()), retention["edge"],
                          float(np.mean([rates[g] for g in STANDARD_GROUPS])), min(retention.values())],
            "p": np.asarray(config["p"], dtype=float).tolist(), "qratio": float(config["qratio"]),
        })
    ranked.sort(key=lambda r: tuple(r["objective"]), reverse=True)
    return ranked, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("--threads must be in [1, 4]")
    OUT.mkdir(parents=True, exist_ok=True)
    configs = local_screen_configs()
    starts, phases = make_inputs()
    input_arrays = {"p": np.asarray([c["p"] for c in configs]),
                    "qratio": np.asarray([c["qratio"] for c in configs])}
    for group in GROUPS:
        input_arrays[f"initial_{group}"] = starts[group]
        input_arrays[f"phase_{group}"] = phases[group]
    input_path = OUT / "configurations_and_training_starts.npz"
    if input_path.exists():
        with np.load(input_path, allow_pickle=False) as saved:
            if any(not np.array_equal(saved[name], value) for name, value in input_arrays.items()):
                raise RuntimeError("saved local refinement inputs differ from generated seed set")
    else:
        np.savez_compressed(input_path, **input_arrays)

    metadata = {
        "study": "V17C.10 local refinement screen around local15",
        "config_count": len(configs), "sobol_config_count": N_CONFIGS,
        "anchor_config_count": N_ANCHORS, "n_per_group": N_PER_GROUP,
        "groups": list(GROUPS), "periods": PERIODS, "steps": STEPS,
        "transient_periods": TRANSIENT, "config_seed": CONFIG_SEED,
        "start_seeds": START_SEEDS, "finite_radii_over_R": list(SIZES),
        "horizon_scaled_min_abs_turns": SCALED_TURN_MIN,
        "strict_reference": {"periods": STRICT_PERIODS, "transient": STRICT_TRANSIENT,
                             "min_abs_turns": STRICT_TURNS},
        "parameter_ranges": {
            "rf_scale_vs_candidate52": [0.88, 1.02], "B_over_A": [3.8, 4.9],
            "nu": [2.9, 3.35], "dc_scale_vs_rf_squared": [0.9, 1.5],
            "mass_ratio": [8.0, 10.5], "charge_ratio": [2.6, 3.7],
            "lambda1": [0.16, 0.26], "lambda2": [0.09, 0.17],
            "C_eff": [6e-6, 1.4e-5],
        },
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
        "fixed_field_config_sha256": sha256(BASE.parent.parent / "v17c_5" / "input" / "config.json"),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "input_arrays_sha256": sha256(input_path),
        "note": "Exploratory training screen only. The scaled turn cutoff is linear horizon alignment, not a calibrated strict-success predictor. Preserve all rows and keep four start distributions separate.",
    }
    meta_path = OUT / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
        raise RuntimeError("local refinement metadata mismatch; refusing incompatible resume")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_path = OUT / "runs.jsonl"
    existing = {}
    # Validate each group's checkpoint rows against its own frozen inputs before resuming.
    if run_path.exists():
        for line in run_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            g, i, cid = row["ensemble"], int(row["start_id"]), int(row["config_id"])
            key = (cid, g, i)
            if key in existing or not 0 <= i < N_PER_GROUP:
                raise RuntimeError(f"duplicate/invalid checkpoint key {key}")
            if (not np.array_equal(np.asarray(row["initial"], dtype=float), starts[g][i])
                    or not np.array_equal(np.asarray(row["phase"], dtype=float), phases[g][i])):
                raise RuntimeError(f"checkpoint frozen-input mismatch {key}")
            existing[key] = row

    model = load_model()
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
            print(f"config{cid}/{group}: n={N_PER_GROUP}, todo={len(todo)}, workers={args.threads}", flush=True)
            tic = time.perf_counter()

            def one(i):
                result = module.integrate_one(starts[group][i], phases[group][i], config["p"], *model[1:],
                                              PERIODS, STEPS, TRANSIENT, qratio=config["qratio"],
                                              oseen_Re_per_speed=0.0, hydro_mode=0,
                                              a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0)
                return output_row(cid, config, group, i, starts[group][i], phases[group][i], result)

            with run_path.open("a", encoding="utf-8", buffering=1) as fout:
                with ThreadPoolExecutor(max_workers=args.threads) as pool:
                    futures = [pool.submit(one, i) for i in todo]
                    for n, future in enumerate(as_completed(futures), 1):
                        row = future.result()
                        existing[(cid, group, int(row["start_id"]))] = row
                        fout.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                        if n % 16 == 0 or n == len(todo):
                            print(f"config{cid}/{group}: checkpoint {sum(k[0] == cid and k[1] == group for k in existing)}/{N_PER_GROUP} ({time.perf_counter()-tic:.1f}s)", flush=True)

    rows = list(existing.values())
    ranked, stats = rank_rows(rows, configs)
    if ranked:
        (OUT / "ranked.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "study": "V17C.10 local15 refinement screen summary",
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
