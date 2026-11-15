"""Parallel, stratified max-step convergence audit for the V13.1 ideal pilot."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1"
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("v131pilot", ROOT / "scripts/run_v13_1_ideal_pilot.py")
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def run(task):
    return MOD.advance(task)


def main() -> None:
    old = list(csv.DictReader((BASE / "ideal_pilot/trajectory_results.csv").open(encoding="utf-8", newline="")))
    initials = list(csv.DictReader((BASE / "ideal_pilot/initial_conditions.csv").open(encoding="utf-8", newline="")))
    seeds = {int(r["index"]): np.array([float(r[k]) for k in ("x0_rho", "y0_rho", "vx0_rho_per_tau", "vy0_rho_per_tau")])
             for r in initials}
    selected = []
    for r in old:
        mu = float(r["mu"]); phase = float(r["phase_rad"]); eta = float(r["eta"]); model = r["model"]; idx = int(r["index"])
        main_band = (mu in (.975, 1.0, 1.025, 1.05) and model == "Q") or (
            mu in (.975, 1.0, 1.025, 1.05) and model in ("SO", "RO") and eta == -.05)
        positive = (mu in (1.0, 1.05) and phase in (0.0, np.pi / 2) and model in ("SO", "RO") and eta == .05)
        if idx < 2 and (main_band or positive):
            selected.append(r)

    tasks = []
    for r in selected:
        key = (float(r["mu"]), float(r["phase_rad"]), r["model"], float(r["eta"]), int(r["index"]))
        state = seeds[key[4]]
        for fraction in (1 / 64, 1 / 128):
            tasks.append((*key, state, fraction))

    completed = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            k = (result["mu"], result["phase_rad"], result["model"], result["eta"],
                 result["index"], result["max_step_tau"])
            completed[k] = result

    rows = []
    for r in selected:
        mu = float(r["mu"]); phase = float(r["phase_rad"]); eta = float(r["eta"]); model = r["model"]; idx = int(r["index"])
        a = completed[(mu, phase, model, eta, idx, 2 * np.pi / 64)]
        b = completed[(mu, phase, model, eta, idx, 2 * np.pi / 128)]
        rows.append({
            "mu": mu, "phase_deg": phase * 180 / np.pi, "model": model, "eta": eta, "index": idx,
            "original_max_step_tau": 2 * np.pi / 64,
            "validated_step64_status_100": a["status_100"], "validated_step128_status_100": b["status_100"],
            "validated_step64_status_300": a["status_300"], "validated_step128_status_300": b["status_300"],
            "same_100": a["status_100"] == b["status_100"], "same_300": a["status_300"] == b["status_300"],
            "exit64": a["exit_periods_300"], "exit128": b["exit_periods_300"],
            "delta_exit_periods": (b["exit_periods_300"] - a["exit_periods_300"])
                if a["exit_periods_300"] is not None and b["exit_periods_300"] is not None else None,
            "maxrho64": a["max_rho_300"], "maxrho128": b["max_rho_300"],
        })

    out = BASE / "ideal_pilot/step_refinement"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "step_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    resolved_times = [abs(r["delta_exit_periods"]) for r in rows if r["delta_exit_periods"] is not None]
    meta = {
        "n_selected": len(rows), "workers": 4,
        "steps_tau": [2 * np.pi / 64, 2 * np.pi / 128],
        "original_max_step_tau": 2 * np.pi / 64,
        "status_disagreements_100": sum(not r["same_100"] for r in rows),
        "status_disagreements_300": sum(not r["same_300"] for r in rows),
        "max_abs_exit_time_difference_periods": max(resolved_times) if resolved_times else None,
        "note": "Original trajectories used max_step_tau=2π/64. Corrected audit explicitly tests 2π/64 versus 2π/128 using max_step_rf fractions 1/64 and 1/128; cases are stratified by mass, phase, model and outcome regime.",
    }
    (out / "step_refinement.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
