"""Longer adaptive-window exact-RF refinement of a sign-change bracket.

Uses the same ideal two-mode RF RHS and frozen panel as the 3,000-period
screen, extending the horizon at selected frequencies without changing the
initial conditions or phases. Each frequency/species is a separate process.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rotation_sign_scan_local15 as base  # noqa: E402


def initialize_worker(periods: int, steps: int, window: int):
    # ProcessPool uses spawn on Windows, so configure the imported worker module.
    base.PERIODS = periods
    base.STEPS = steps
    base.WINDOW = window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", nargs="+", type=float, required=True)
    parser.add_argument("--periods", type=int, default=15000)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--species", nargs="+", choices=("species1", "species2"),
                        default=("species1",))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if args.periods % 500 or args.window <= 0 or args.window > args.periods:
        raise SystemExit("periods must be divisible by 500 and 0 < window <= periods")
    base.OUT.mkdir(parents=True, exist_ok=True)
    with base.PANEL.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    panel_sha = hashlib.sha256(base.PANEL.read_bytes()).hexdigest()
    state0 = np.array([[float(r[k]) for k in ("x0", "y0", "vx0", "vy0")] for r in rows])
    phase1 = np.array([float(r["phase1"]) for r in rows])
    phase2 = np.array([float(r["phase2"]) for r in rows])
    tasks = [(nu, species, rows, state0, phase1, phase2)
             for nu in args.nu for species in args.species]
    workers = args.workers or len(tasks)
    if workers > 11:
        raise SystemExit("worker cap is 11 for this remote-compute profile")
    with cf.ProcessPoolExecutor(max_workers=workers,
            initializer=initialize_worker,
            initargs=(args.periods, args.steps, args.window)) as pool:
        results = list(pool.map(base.run_one, tasks))
    results.sort(key=lambda r: (r["nu"], r["species"]))
    flat = [row for result in results for row in result["rows"]]
    nu_tag = "_".join(f"{nu:.6f}".replace(".", "p") for nu in args.nu)
    species_tag = "_".join(args.species)
    stem = f"local15_sign_refinement_{args.periods}p_{species_tag}_{nu_tag}"
    doc = {"status": "completed", "model_scope": "exact ideal k=+2,-4 Local15; C=0; linear drag; no full MFS",
        "parameters": base.PARAM, "nu_values": args.nu, "species": args.species,
        "periods": args.periods, "steps_per_primary_period": args.steps,
        "measurement_window_periods": args.window, "panel_seed": 15152026,
        "panel_sha256": panel_sha, "panel_is_iid_probability_sample": False,
        "worker_count": workers, "results": results, "rows": flat}
    (base.OUT / f"{stem}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with (base.OUT / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(json.dumps({"status": doc["status"], "tasks": len(results),
        "trajectories": len(flat), "periods": args.periods, "workers": workers,
        "output": stem}))


if __name__ == "__main__":
    main()
