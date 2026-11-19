"""Fresh-start finite-sphere Sobol refinement around the long-validated candidate 48."""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = HERE / "stage4_candidate48"
sys.path.insert(0, str(BASE / "baseline"))
from basin_scan import integrate_one, load_model, make_initials

NCFG = 64
N_PER_ENSEMBLE = 16
PERIODS = 1800
STEPS = 48
SKIP = 400
CONFIG_SEED = 20261138
START_SEEDS = {"rest": 20261139, "weak": 20261140, "wide": 20261141}
A1, A2 = 0.0002, 0.00189


def make_configs(base, base_qratio):
    u = qmc.Sobol(d=9, scramble=True, seed=CONFIG_SEED).random_base2(6)
    configs = []
    for i, x in enumerate(u):
        rf = 0.72 + 0.45*x[0]
        phi = 2.4 + 1.6*x[1]
        nu = 2.3 + 1.0*x[2]
        dc = 0.55 + 1.10*x[3]
        mass_ratio = 5.0 + 8.0*x[4]
        qratio = 2.2 + 3.3*x[5]
        lam1 = math.exp(math.log(0.05) + x[6]*math.log(4.0))
        lam2 = math.exp(math.log(0.05) + x[7]*math.log(4.0))
        coupling = math.exp(math.log(8e-6) + x[8]*math.log(5.0))
        p = base.copy()
        p[0] = base[0]*rf
        p[1] = p[0]*phi
        p[2] = nu
        p[3] = base[3]*rf*rf*dc
        p[4] = 1.0/mass_ratio
        p[5] = coupling
        p[6] = lam1
        p[7] = lam2
        configs.append({"config_id": i, "source": "fresh Sobol around candidate48", "p": p,
            "qratio": qratio, "rf_scale_vs_candidate48": rf,
            "harmonic_potential_ratio_B_over_A": phi, "nu": nu,
            "dc_scale_vs_rf_squared": dc, "mass_ratio": mass_ratio,
            "charge_ratio": qratio, "lambda1": lam1, "lambda2": lam2,
            "C_eff": coupling})
    configs.append({"config_id": NCFG, "source": "candidate48 anchor", "p": base.copy(),
        "qratio": base_qratio, "rf_scale_vs_candidate48": 1.0,
        "harmonic_potential_ratio_B_over_A": base[1]/base[0], "nu": base[2],
        "dc_scale_vs_rf_squared": 1.0, "mass_ratio": 1.0/base[4],
        "charge_ratio": base_qratio, "lambda1": base[6],
        "lambda2": base[7], "C_eff": base[5]})
    return configs


def cr_proxy(retained, turns, windows, rmin):
    if not retained or np.min(rmin) < 0.02 or np.min(np.abs(turns)) < 1.05:
        return False
    signs = np.sign(windows)
    return bool(np.all(signs[:, 0]*signs[:, 1] < 0.0) and np.all(np.abs(windows) > 0.05))


def rank_rows(path, configs):
    stats = {c["config_id"]: {g: {"n": 0, "cr": 0, "ret": 0} for g in ("rest", "weak", "wide")}
             for c in configs}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            s = stats[r["config_id"]][r["ensemble"]]
            s["n"] += 1
            s["cr"] += bool(r["cr_proxy"])
            s["ret"] += bool(r["retained"])
    ranked = []
    for c in configs:
        groups = stats[c["config_id"]]
        if not all(groups[g]["n"] == N_PER_ENSEMBLE for g in groups):
            continue
        cr = [groups[g]["cr"]/groups[g]["n"] for g in ("rest", "weak", "wide")]
        retention = [groups[g]["ret"]/groups[g]["n"] for g in ("rest", "weak", "wide")]
        ranked.append({"config_id": c["config_id"], "source": c["source"],
            "P_CR_proxy_by_ensemble": dict(zip(("rest", "weak", "wide"), cr)),
            "P_retained_by_ensemble": dict(zip(("rest", "weak", "wide"), retention)),
            "balanced_min_CR_proxy": min(cr), "balanced_mean_CR_proxy": float(np.mean(cr)),
            "balanced_min_retention": min(retention),
            "parameters": {k: v for k, v in c.items() if k not in ("p", "source")}})
    ranked.sort(key=lambda r: (r["balanced_min_CR_proxy"], r["balanced_mean_CR_proxy"],
                               r["balanced_min_retention"]), reverse=True)
    (OUT / "ranked.json").write_text(json.dumps(ranked, indent=2)+"\n", encoding="utf-8")
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source = np.load(HERE / "stage2_speed" / "configurations.npz")
    base = np.asarray(source["p"][48], dtype=float)
    base_qratio = float(source["qratio"][48])
    configs = make_configs(base, base_qratio)
    configs = configs[:a.limit or len(configs)]
    starts, phases = {}, {}
    for g, seed in START_SEEDS.items():
        starts[g], phases[g] = make_initials(g, N_PER_ENSEMBLE, seed)
    p_arr = np.array([c["p"] for c in configs])
    qr_arr = np.array([c["qratio"] for c in configs])
    np.savez_compressed(OUT / "configurations.npz", p=p_arr, qratio=qr_arr,
        initial_rest=starts["rest"], phase_rest=phases["rest"],
        initial_weak=starts["weak"], phase_weak=phases["weak"],
        initial_wide=starts["wide"], phase_wide=phases["wide"])
    metadata = {"n_configs": len(configs), "N_per_ensemble": N_PER_ENSEMBLE,
        "ensembles": ["rest", "weak", "wide"], "periods": PERIODS, "steps": STEPS,
        "transient_cycles": SKIP, "config_seed": CONFIG_SEED, "start_seeds": START_SEEDS,
        "finite_spheres_a_over_R": [A1, A2],
        "objective": "maximize minimum CR proxy across rest/weak/wide while retaining all trials in denominators",
        "parameter_ranges": {"rf_scale_vs_candidate48": [0.72, 1.17], "B_over_A": [2.4, 4.0],
            "nu": [2.3, 3.3], "dc_scale_vs_rf_squared": [0.55, 1.65],
            "mass_ratio": [5.0, 13.0], "charge_ratio": [2.2, 5.5],
            "lambda1_lambda2": [0.05, 0.20], "C_eff": [8e-6, 4e-5]},
        "note": "Short screen only. Keep an untouched new-seed holdout for final selection."}
    mp = OUT / "meta.json"
    if mp.exists() and json.loads(mp.read_text(encoding="utf-8")) != metadata:
        raise RuntimeError("stage4 metadata mismatch; refusing to overwrite checkpoint")
    mp.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")

    _, family, k, coeff = load_model()
    result_path = OUT / "runs.jsonl"
    done = set()
    if result_path.exists():
        done = {(json.loads(s)["config_id"], json.loads(s)["ensemble"], json.loads(s)["start_id"])
                for s in result_path.read_text(encoding="utf-8").splitlines() if s.strip()}
    tasks = [(c, g, i) for c in configs for g in ("rest", "weak", "wide")
             for i in range(N_PER_ENSEMBLE) if (c["config_id"], g, i) not in done]
    if tasks:
        c, g, i = tasks[0]
        integrate_one(starts[g][i], phases[g][i], c["p"], family, k, coeff,
                      2, STEPS, 0, c["qratio"], 0.0, 0, A1, A2, -1.0)
    print(f"candidate48 finite-sphere refinement: configs={len(configs)}, trials={len(tasks)}, workers={a.threads}", flush=True)
    tic = time.perf_counter()

    def one(task):
        c, g, i = task
        r = integrate_one(starts[g][i], phases[g][i], c["p"], family, k, coeff,
                          PERIODS, STEPS, SKIP, c["qratio"], 0.0, 0, A1, A2, -1.0)
        end, turns, windows, rmin, rmax, zmax, maxd, dmin, complete, retained, escaped, singular, cat, freq = r
        return {"config_id": c["config_id"], "ensemble": g, "start_id": i,
            "seed": START_SEEDS[g], "initial": starts[g][i].tolist(), "phase": phases[g][i].tolist(),
            "turns": turns.tolist(), "window_turns": windows.tolist(), "frequency_Hz": freq.tolist(),
            "r_min": rmin.tolist(), "r_max": rmax.tolist(), "z_max": zmax.tolist(),
            "minimum_separation": float(dmin), "retained": bool(retained), "escaped": bool(escaped),
            "category": str(cat), "cr_proxy": cr_proxy(retained, turns, windows, rmin),
            "completed_periods": int(complete), "final_state": end.tolist()}

    with result_path.open("a", encoding="utf-8", buffering=1) as f, ThreadPoolExecutor(max_workers=a.threads) as pool:
        futures = [pool.submit(one, t) for t in tasks]
        for j, future in enumerate(as_completed(futures), 1):
            f.write(json.dumps(future.result(), separators=(",", ":"))+"\n")
            if j % 128 == 0 or j == len(futures):
                ranked = rank_rows(result_path, configs)
                print(f"checkpoint {j}/{len(tasks)} wall={time.perf_counter()-tic:.1f}s top="
                      f"{[(r['config_id'],round(r['balanced_min_CR_proxy'],3),round(r['balanced_mean_CR_proxy'],3)) for r in ranked[:5]]}", flush=True)


if __name__ == "__main__":
    main()
