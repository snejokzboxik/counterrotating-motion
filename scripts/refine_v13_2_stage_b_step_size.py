"""Half-step sensitivity check on representative Stage B/local-map outcomes."""
from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability/local_candidate_refinement/step_halving"
BASE = OUT.parent / "trajectory_results.csv"
ALPHAS = (0.025142857142857144, 0.026571428571428572)
BETAS = (0.1572222222222222,)
PHASES = (0.0, math.pi / 2)
ETA = -0.05
N_INITIAL = 32
N_PERIODS = 100
STEP = 2 * math.pi / 128
RTOL, ATOL = 2e-10, 2e-12


def run(task: tuple) -> dict:
    alpha, beta, phase, model, index, state = task
    from three_phase_trap.mass_filter_v13 import acceleration
    kappa = 0. if model == "Q" else ETA * beta
    y0 = np.asarray(state, float)
    def rhs(t, y):
        return np.r_[y[2:], acceleration(y[0], y[1], t + phase, kappa, beta, alpha,
                                          standing_octupole=model == "SO")]
    def aperture(t, y): return float(np.hypot(y[0], y[1]) - 1.)
    aperture.terminal = True; aperture.direction = 1
    sol = solve_ivp(rhs, (0., 2 * math.pi * N_PERIODS), y0, method="DOP853",
                    rtol=RTOL, atol=ATOL, max_step=STEP, dense_output=True, events=aperture)
    status = "numerical_error" if not sol.success else ("escaped" if len(sol.t_events[0]) else "retained")
    exitp = float(sol.t_events[0][0] / (2 * math.pi)) if status == "escaped" else None
    rho = np.hypot(sol.y[0], sol.y[1])
    if len(sol.t) > 1:
        mid = sol.sol((sol.t[:-1] + sol.t[1:]) / 2)
        rho_max = float(max(np.max(rho), np.max(np.hypot(mid[0], mid[1]))))
    else: rho_max = float(np.max(rho))
    return {"alpha": alpha, "beta": beta, "eta": ETA, "phase_rad": phase,
            "phase_deg": float(np.degrees(phase)), "model": model, "index": index,
            "status_100_h128": status, "exit_periods_100_h128": exitp,
            "max_rho_100_h128": max(1., rho_max) if status == "escaped" else rho_max,
            "max_step_tau": STEP, "rtol": RTOL, "atol": ATOL, "nfev": int(sol.nfev)}


def key(r: dict) -> tuple:
    return (round(float(r["alpha"]), 12), round(float(r["beta"]), 12),
            round(float(r["phase_rad"]), 12), r["model"], int(r["index"]))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=8); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = list(csv.DictReader(BASE.open(newline="", encoding="utf-8")))
    initials = list(csv.DictReader((OUT.parent / "initial_conditions.csv").open(newline="", encoding="utf-8")))
    states = {int(r["index"]): [float(r[k]) for k in ("x0_rho", "y0_rho", "vx0_rho_per_tau", "vy0_rho_per_tau")] for r in initials}
    baseline = {key({"alpha": r["alpha"], "beta": r["beta"], "phase_rad": math.radians(float(r["phase_deg"])),
                     "model": r["model"], "index": r["index"]}): r for r in base
                if abs(float(r["alpha"]) - ALPHAS[0]) < 1e-10 or abs(float(r["alpha"]) - ALPHAS[1]) < 1e-10}
    tasks = [(a, b, phase, model, i, states[i]) for a in ALPHAS for b in BETAS for phase in PHASES
             for model in ("Q", "SO", "RO") for i in range(N_INITIAL)]
    params = {"purpose": "local V13.2 ideal-field trajectory step-halving check",
              "cells": [{"alpha": a, "beta": b} for a in ALPHAS for b in BETAS],
              "models": ["Q", "SO", "RO"], "eta": ETA, "phases_rad": list(PHASES),
              "n_common_initial": N_INITIAL, "horizon_periods": N_PERIODS,
              "baseline_max_step_tau": 2 * math.pi / 64, "refined_max_step_tau": STEP,
              "integrator": {"method": "DOP853", "rtol": RTOL, "atol": ATOL}, "workers": args.workers}
    (OUT / "parameters.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    ck = OUT / "checkpoint.jsonl"
    done = [json.loads(x) for x in ck.read_text(encoding="utf-8").splitlines() if x.strip()] if ck.exists() else []
    seen = {key(r) for r in done}
    pending = [t for t in tasks if (round(t[0], 12), round(t[1], 12), round(t[2], 12), t[3], t[4]) not in seen]
    print(f"Step-halving: {len(pending)} remaining of {len(tasks)}; workers={args.workers}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool, ck.open("a", encoding="utf-8") as f:
            futures = [pool.submit(run, t) for t in pending]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result(); done.append(row)
                f.write(json.dumps(row, separators=(",", ":")) + "\n"); f.flush()
                if i % 192 == 0 or i == len(futures): print(f"Step-halving completed {i}/{len(futures)}", flush=True)
    refined = {key(r): r for r in done}
    comparisons = []
    for t in tasks:
        k = (round(t[0], 12), round(t[1], 12), round(t[2], 12), t[3], t[4])
        old, new = baseline[k], refined[k]
        comparisons.append({"alpha": t[0], "beta": t[1], "phase_deg": float(np.degrees(t[2])),
                            "model": t[3], "index": t[4], "status_h64": old["status_100"],
                            "status_h128": new["status_100_h128"],
                            "classification_changed": old["status_100"] != new["status_100_h128"],
                            "exit_periods_h64": old["exit_periods_100"],
                            "exit_periods_h128": new["exit_periods_100_h128"],
                            "max_rho_h64": old["max_rho_100"], "max_rho_h128": new["max_rho_100_h128"]})
    with (OUT / "trajectory_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comparisons[0])); w.writeheader(); w.writerows(comparisons)
    changed = sum(r["classification_changed"] for r in comparisons)
    summary = {"n": len(comparisons), "classification_changes": changed,
               "max_exit_time_difference_periods": max(abs(float(r["exit_periods_h64"]) - float(r["exit_periods_h128"]))
                   for r in comparisons if r["exit_periods_h64"] not in (None, "") and r["exit_periods_h128"] not in (None, ""))}
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
