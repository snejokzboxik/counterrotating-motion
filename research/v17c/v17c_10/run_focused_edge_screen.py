"""Fresh-start local Sobol search that weights early radial retention."""
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
OPT = V17C9 / "optimization"
BASE = V17C9 / "baseline"
OUT = HERE / "focused_edge_screen"
sys.path.insert(0, str(BASE))
from basin_scan import integrate_one, load_model, make_initials

GROUPS = ("rest", "weak", "wide", "edge")
STANDARD_GROUPS = GROUPS[:3]
N_CONFIGS = 32
N_ANCHORS = 5
N_PER_GROUP = 32
PERIODS = 2400
STEPS = 48
SKIP = 500
CONFIG_SEED = 20261200
START_SEEDS = {"rest": 20261210, "weak": 20261211, "wide": 20261212, "edge": 20261213}
A1, A2 = 0.0002, 0.00189
P0_TOP_IDS = (12, 17, 28, 49, 64)
EDGE_RMIN, EDGE_RMAX = 0.70, 0.90


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_edge_initials(count: int, seed: int):
    initial, phases = make_initials("wide", count, seed)
    u = qmc.Sobol(d=2, scramble=True, seed=seed+7919).random_base2(int(math.log2(count)))
    forced_r = np.sqrt(EDGE_RMIN**2 + u[:, 0]*(EDGE_RMAX**2-EDGE_RMIN**2))
    forced_phi = 2.0*math.pi*u[:, 1]
    for i in range(count):
        offset = 0 if i % 2 == 0 else 6
        initial[i, offset] = forced_r[i]*math.cos(forced_phi[i])
        initial[i, offset+1] = forced_r[i]*math.sin(forced_phi[i])
    return initial, phases


def make_search_configs(base: np.ndarray, qbase: float, p0):
    u = qmc.Sobol(d=9, scramble=True, seed=CONFIG_SEED).random_base2(5)
    configs = []
    for i, x in enumerate(u):
        rf = 0.92 + 0.16*x[0]
        phi = 3.55 + 0.95*x[1]
        nu = 2.75 + 0.40*x[2]
        dc = 0.85 + 0.50*x[3]
        mass_ratio = 8.0 + 2.5*x[4]
        qratio = 2.50 + 1.40*x[5]
        lam1 = math.exp(math.log(0.14) + x[6]*math.log(0.23/0.14))
        lam2 = math.exp(math.log(0.085) + x[7]*math.log(0.145/0.085))
        coupling = math.exp(math.log(6.5e-6) + x[8]*math.log(1.2e-5/6.5e-6))
        p = base.copy()
        p[0] = base[0]*rf
        p[1] = p[0]*phi
        p[2] = nu
        p[3] = base[3]*rf*rf*dc
        p[4] = 1.0/mass_ratio
        p[5] = coupling
        p[6] = lam1
        p[7] = lam2
        configs.append({
            "config_id": i, "label": f"local_sobol_{i:02d}",
            "source": "fresh Sobol neighborhood spanning candidate52 and config17",
            "p": p, "qratio": qratio,
            "rf_scale_vs_candidate52": rf,
            "harmonic_potential_ratio_B_over_A": phi,
            "nu": nu, "dc_scale_vs_rf_squared": dc,
            "mass_ratio": mass_ratio, "charge_ratio": qratio,
            "lambda1": lam1, "lambda2": lam2, "C_eff": coupling,
        })
    for anchor_index, p0_id in enumerate(P0_TOP_IDS):
        p = np.asarray(p0["p"][p0_id], dtype=np.float64)
        qratio = float(p0["qratio"][p0_id])
        configs.append({
            "config_id": N_CONFIGS+anchor_index,
            "label": {12: "screen_rank2", 17: "config17_validated", 28: "screen_rank3", 49: "screen_rank4", 64: "candidate52_anchor"}[p0_id],
            "source": f"P0 training anchor config {p0_id}",
            "p": p, "qratio": qratio,
            "P0_config_id": p0_id,
        })
    return configs


def make_inputs():
    starts, phases = {}, {}
    for group in STANDARD_GROUPS:
        starts[group], phases[group] = make_initials(group, N_PER_GROUP, START_SEEDS[group])
    starts["edge"], phases["edge"] = make_edge_initials(N_PER_GROUP, START_SEEDS["edge"])
    return starts, phases


def cr_proxy(retained, turns, windows, rmin):
    if not retained or np.min(rmin) < 0.02 or np.min(np.abs(turns)) < 1.05:
        return False
    signs = np.sign(windows)
    return bool(np.all(signs[:, 0]*signs[:, 1] < 0.0) and np.all(np.abs(windows) > 0.05))


def rank_rows(result_path: Path, configs):
    stats = {
        c["config_id"]: {
            g: {"n": 0, "cr": 0, "retained": 0, "categories": {}}
            for g in GROUPS
        }
        for c in configs
    }
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            s = stats[row["config_id"]][row["ensemble"]]
            s["n"] += 1
            s["cr"] += bool(row["cr_proxy"])
            s["retained"] += bool(row["retained"])
            cat = row["category"]
            s["categories"][cat] = s["categories"].get(cat, 0) + 1
    ranked = []
    for config in configs:
        by_group = stats[config["config_id"]]
        if not all(by_group[g]["n"] == N_PER_GROUP for g in GROUPS):
            continue
        cr = {g: by_group[g]["cr"]/N_PER_GROUP for g in GROUPS}
        retention = {g: by_group[g]["retained"]/N_PER_GROUP for g in GROUPS}
        score = {
            "balanced_min_cr_all_four_groups": min(cr.values()),
            "balanced_min_cr_standard_groups": min(cr[g] for g in STANDARD_GROUPS),
            "edge_cr_proxy": cr["edge"],
            "edge_retention": retention["edge"],
            "mean_cr_standard_groups": float(np.mean([cr[g] for g in STANDARD_GROUPS])),
            "balanced_min_retention_all_four_groups": min(retention.values()),
        }
        ranked.append({
            "config_id": config["config_id"], "label": config["label"],
            "source": config["source"], "P_CR_proxy_by_group": cr,
            "P_retained_by_group": retention,
            "categories_by_group": {g: by_group[g]["categories"] for g in GROUPS},
            "objective": score,
            "parameters": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in config.items() if k not in ("p", "source")
            },
            "p": np.asarray(config["p"], dtype=float).tolist(),
            "qratio": float(config["qratio"]),
        })
    ranked.sort(key=lambda r: (
        r["objective"]["balanced_min_cr_all_four_groups"],
        r["objective"]["edge_retention"],
        r["objective"]["mean_cr_standard_groups"],
        r["objective"]["balanced_min_retention_all_four_groups"],
    ), reverse=True)
    (OUT / "ranked.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit-configs", type=int, default=0, help="debug subset; 0 runs all candidates and anchors")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("--threads must be in [1, 4]")
    OUT.mkdir(parents=True, exist_ok=True)
    stage4 = np.load(OPT / "stage4_candidate48" / "configurations.npz")
    p0 = np.load(OPT / "stage5_candidate52" / "configurations.npz")
    base = np.asarray(stage4["p"][52], dtype=np.float64)
    qbase = float(stage4["qratio"][52])
    configs = make_search_configs(base, qbase, p0)
    if args.limit_configs:
        configs = configs[:args.limit_configs]
    starts, phases = make_inputs()

    input_arrays = {"p": np.asarray([c["p"] for c in configs]), "qratio": np.asarray([c["qratio"] for c in configs])}
    for group in GROUPS:
        input_arrays[f"initial_{group}"] = starts[group]
        input_arrays[f"phase_{group}"] = phases[group]
    input_path = OUT / "configurations_and_training_starts.npz"
    if input_path.exists():
        with np.load(input_path, allow_pickle=False) as saved:
            for name, expected in input_arrays.items():
                if not np.array_equal(saved[name], expected):
                    raise RuntimeError(f"saved training inputs differ from seed-generated values: {name}")
    else:
        np.savez_compressed(input_path, **input_arrays)

    configuration_meta = [
        {"config_id": c["config_id"], "label": c["label"], "source": c["source"],
         "p": np.asarray(c["p"], dtype=float).tolist(), "qratio": float(c["qratio"]),
         "search_coordinates": {k: v for k, v in c.items() if k not in ("p", "source")}}
        for c in configs
    ]
    metadata = {
        "study": "V17C.10 focused Sobol screen for early radial retention",
        "config_count": len(configs), "sobol_config_count": N_CONFIGS,
        "anchor_config_count": N_ANCHORS, "N_per_group": N_PER_GROUP,
        "groups": list(GROUPS), "periods": PERIODS, "steps": STEPS,
        "transient_periods": SKIP, "config_seed": CONFIG_SEED,
        "start_seeds": START_SEEDS, "finite_radii_over_R": [A1, A2],
        "objective_order": [
            "maximize minimum CR proxy across rest, weak, wide, and edge",
            "maximize edge retention",
            "maximize mean CR proxy across the standard three ensembles",
            "maximize minimum retention across all four groups",
        ],
        "edge_ensemble": {
            "n": N_PER_GROUP, "definition": "wide starts with one particle radius sampled area-uniformly in [0.70R,0.90R], alternating particle identity",
            "minimum_initial_max_radius_over_R": EDGE_RMIN,
        },
        "parameter_ranges": {
            "rf_scale_vs_candidate52": [0.92, 1.08],
            "B_over_A": [3.55, 4.50], "nu": [2.75, 3.15],
            "dc_scale_vs_rf_squared": [0.85, 1.35],
            "mass_ratio": [8.0, 10.5], "charge_ratio": [2.50, 3.90],
            "lambda1": [0.14, 0.23], "lambda2": [0.085, 0.145],
            "C_eff": [6.5e-6, 1.2e-5],
        },
        "configurations": configuration_meta,
        "note": "Exploratory training screen only. The edge group is an additional targeted distribution, not a replacement for rest/weak/wide or an estimate of their natural weights. Keep all trials in denominators; reserve fresh starts for strict validation.",
    }
    meta_path = OUT / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
        raise RuntimeError("focused-screen metadata mismatch; refusing to resume incompatible checkpoint")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    p_model, family, k, coeff = load_model()
    result_path = OUT / "runs.jsonl"
    done = set()
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((int(row["config_id"]), row["ensemble"], int(row["start_id"])))
    tasks = [
        (config, group, i)
        for config in configs for group in GROUPS for i in range(N_PER_GROUP)
        if (int(config["config_id"]), group, i) not in done
    ]
    if tasks:
        c, g, i = tasks[0]
        integrate_one(starts[g][i], phases[g][i], c["p"], family, k, coeff,
                      2, STEPS, 0, c["qratio"], 0.0, 0, A1, A2, -1.0)
    print(f"focused edge screen: configs={len(configs)}, groups={len(GROUPS)}, N={N_PER_GROUP}, todo={len(tasks)}, workers={args.threads}", flush=True)
    tic = time.perf_counter()

    def one(task):
        config, group, i = task
        result = integrate_one(
            starts[group][i], phases[group][i], config["p"], family, k, coeff,
            PERIODS, STEPS, SKIP, config["qratio"], 0.0, 0, A1, A2, -1.0,
        )
        end, turns, windows, rmin, rmax, zmax, maxd, dmin, complete, retained, escaped, singular, category, freq = result
        return {
            "config_id": int(config["config_id"]), "config_label": config["label"],
            "ensemble": group, "start_id": int(i), "seed": int(START_SEEDS[group]),
            "p": np.asarray(config["p"], dtype=float).tolist(), "qratio": float(config["qratio"]),
            "initial": starts[group][i].tolist(), "phase": phases[group][i].tolist(),
            "turns": turns.tolist(), "window_turns": windows.tolist(),
            "frequency_Hz": freq.tolist(), "r_min": rmin.tolist(),
            "r_max": rmax.tolist(), "z_max": zmax.tolist(),
            "minimum_separation": float(dmin), "retained": bool(retained),
            "escaped": bool(escaped), "singular_unresolved": bool(singular),
            "category": str(category), "cr_proxy": cr_proxy(retained, turns, windows, rmin),
            "completed_periods": int(complete), "final_state": end.tolist(),
        }

    total = len(tasks)
    with result_path.open("a", encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {pool.submit(one, task): task for task in tasks}
            for n, future in enumerate(as_completed(futures), 1):
                row = future.result()
                fout.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False)+"\n")
                if n % 128 == 0 or n == total:
                    ranked = rank_rows(result_path, configs)
                    best = [(r["config_id"], round(r["objective"]["balanced_min_cr_all_four_groups"], 3), round(r["objective"]["edge_retention"], 3)) for r in ranked[:5]]
                    print(f"checkpoint {n}/{total} elapsed={time.perf_counter()-tic:.1f}s ranked_complete={len(ranked)} top={best}", flush=True)
    rank_rows(result_path, configs)


if __name__ == "__main__":
    main()
