"""Extend saved exact-RF sign-transition ensembles to 50k periods.

The species-1 boundary frequencies resume their 15k checkpoints; the
species-2 control at nu=2.25 resumes its 3k checkpoint. Original files are
copied into a separate checkpoint directory and remain unchanged.
"""
from __future__ import annotations

import concurrent.futures as cf
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rotation_sign_scan_local15 as base  # noqa: E402

TARGET_PERIODS = 50_000
STEPS = 96
WINDOW = 500
CASES = (
    (2.2125, "species1", 15_000),
    (2.23125, "species1", 15_000),
    (2.240625, "species1", 15_000),
    (2.25, "species1", 15_000),
    (2.25, "species2", 3_000),
)
WORKERS = len(CASES)
CHECKPOINT_DIR = base.OUT / "extended_sign_transition_50000_checkpoints"
RESULT_JSON = base.OUT / "local15_sign_transition_50000.json"
RESULT_CSV = base.OUT / "local15_sign_transition_50000.csv"
HORIZONS = (5_000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000)


def initialize_worker(periods: int, steps: int, window: int, output: str) -> None:
    base.PERIODS = periods
    base.STEPS = steps
    base.WINDOW = window
    base.OUT = Path(output)


def seed_checkpoint(nu: float, species: str, source_periods: int) -> None:
    name = f"nu_{nu:.3f}_{species}.npz"
    source = base.OUT / name
    target = CHECKPOINT_DIR / name
    if target.exists():
        with np.load(target, allow_pickle=False) as data:
            if (int(data["step"]) >= TARGET_PERIODS * STEPS and
                    float(data["nu"]) == nu and str(data["species"].item()) == species):
                return
            if (int(data["step"]) >= source_periods * STEPS and
                    float(data["nu"]) == nu and str(data["species"].item()) == species):
                return
        raise ValueError(f"invalid continuation checkpoint: {target}")
    with np.load(source, allow_pickle=False) as data:
        if int(data["step"]) != source_periods * STEPS:
            raise ValueError(f"expected {source_periods} period checkpoint: {source}")
        if float(data["nu"]) != nu or str(data["species"].item()) != species:
            raise ValueError(f"checkpoint metadata mismatch: {source}")
        if int(data["steps_per_period"]) != STEPS:
            raise ValueError(f"step scheme mismatch: {source}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def summarize_checkpoint(nu: float, species: str, source_periods: int,
                         result: dict) -> dict:
    path = CHECKPOINT_DIR / result["checkpoint"]
    with np.load(path, allow_pickle=False) as data:
        radii = data["rho_samples"]
        angles = data["angle_samples"]
        escape_periods = data["escape_period"].copy()
        maxrho = data["maxrho"].copy()

    rows = []
    for index, start in enumerate(result["rows"]):
        history = []
        for horizon in HORIZONS:
            rho_mean = float(np.mean(radii[horizon-WINDOW:horizon+1, index]))
            omega = float((angles[horizon, index] - angles[horizon-WINDOW, index]) /
                          (2 * np.pi * WINDOW))
            escaped = bool(np.isfinite(escape_periods[index]) and
                           escape_periods[index] <= horizon)
            history.append({
                "horizon_periods": horizon,
                "rho_mean_last_500": rho_mean,
                "omega_last_500": omega,
                "escaped_by_horizon": escaped,
                "escape_period": (None if not np.isfinite(escape_periods[index])
                                  else float(escape_periods[index])),
                "sign": 0 if escaped else int(np.sign(omega)),
            })
        rows.append({"species": species, "nu": nu, "start_id": start["start_id"],
                     "max_rho": float(maxrho[index]),
                     "escape_period": (None if not np.isfinite(escape_periods[index])
                                       else float(escape_periods[index])),
                     "omega_mean_15k_to_50k": float((angles[50_000, index] -
                         angles[15_000, index]) / (2 * np.pi * 35_000)),
                     "horizons": history})

    sign_counts = {}
    for horizon in HORIZONS:
        outcomes = [row["horizons"][HORIZONS.index(horizon)] for row in rows]
        sign_counts[str(horizon)] = {
            "positive": sum(item["sign"] > 0 for item in outcomes),
            "negative": sum(item["sign"] < 0 for item in outcomes),
            "escaped": sum(item["escaped_by_horizon"] for item in outcomes),
            "n_starts": len(rows),
        }
    late = [row for row in rows if not (row["escape_period"] is not None and
                                         row["escape_period"] <= 50_000)]
    late_values = [row["omega_mean_15k_to_50k"] for row in late]
    return {
        "nu": nu,
        "species": species,
        "source_horizon_periods": source_periods,
        "target_horizon_periods": TARGET_PERIODS,
        "sign_counts": sign_counts,
        "cumulative_15k_to_50k": {
            "positive": sum(value > 0 for value in late_values),
            "negative": sum(value < 0 for value in late_values),
            "escaped": len(rows) - len(late),
            "n_starts": len(rows),
            "retained_min_median_max": ([float(min(late_values)),
                float(np.median(late_values)), float(max(late_values))]
                if late_values else []),
        },
        "leading_roots": result["leading_roots"],
        "trajectory_records": rows,
        "checkpoint": f"results/rotation_sign_scan/extended_sign_transition_50000_checkpoints/{result['checkpoint']}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, help="run only this frequency node")
    parser.add_argument("--species", choices=("species1", "species2"),
                        help="run only this particle species")
    parser.add_argument("--workers", type=int, help="override the process count")
    args = parser.parse_args()
    cases = tuple(case for case in CASES
                  if (args.nu is None or case[0] == args.nu)
                  and (args.species is None or case[1] == args.species))
    workers = args.workers or len(cases)
    if not cases or not 1 <= workers <= len(cases):
        raise SystemExit("select at least one case and use 1..number_of_cases workers")
    with base.PANEL.open(newline="", encoding="utf-8") as f:
        panel_rows = list(csv.DictReader(f))
    panel_sha = hashlib.sha256(base.PANEL.read_bytes()).hexdigest()
    state0 = np.array([[float(row[k]) for k in ("x0", "y0", "vx0", "vy0")]
                       for row in panel_rows])
    phase1 = np.array([float(row["phase1"]) for row in panel_rows])
    phase2 = np.array([float(row["phase2"]) for row in panel_rows])
    tasks = []
    for nu, species, source_periods in cases:
        seed_checkpoint(nu, species, source_periods)
        tasks.append((nu, species, panel_rows, state0, phase1, phase2))

    started = time.perf_counter()
    print(f"extending {len(tasks)} frequency/species ensembles on {workers} workers "
          f"to {TARGET_PERIODS} periods", flush=True)
    with cf.ProcessPoolExecutor(max_workers=workers,
            initializer=initialize_worker,
            initargs=(TARGET_PERIODS, STEPS, WINDOW, str(CHECKPOINT_DIR))) as pool:
        results = list(pool.map(base.run_one, tasks))
    results.sort(key=lambda row: (row["nu"], row["species"]))
    source_periods_by_case = {(nu, species): source for nu, species, source in cases}
    summary = [summarize_checkpoint(row["nu"], row["species"],
                                    source_periods_by_case[(row["nu"], row["species"])], row)
               for row in results
               ]
    flat = []
    for ensemble in summary:
        for row in ensemble["trajectory_records"]:
            for horizon in row["horizons"]:
                flat.append({"species": row["species"], "nu": row["nu"],
                             "start_id": row["start_id"],
                             "omega_mean_15k_to_50k": row["omega_mean_15k_to_50k"],
                             **horizon})

    extension_segments = len(cases) * len(panel_rows)
    initial_particle_periods = sum(len(panel_rows) * (TARGET_PERIODS - source)
                                   for _, _, source in cases)
    report = {
        "status": "completed",
        "model_scope": "exact ideal k=+2,-4 Local15 RF; C=0; linear drag; no full MFS",
        "purpose": "finite-horizon continuation of the fixed species-1 sign-transition panel and one species-2 control",
        "panel_seed": 15152026,
        "panel_sha256": panel_sha,
        "parameters": base.PARAM,
        "panel_is_iid_probability_sample": False,
        "rf_phases_fixed": True,
        "steps_per_primary_period": STEPS,
        "measurement_window_periods": WINDOW,
        "horizons": list(HORIZONS),
        "worker_count_history_by_invocation": [5, 1, 2],
        "final_recovery_pool_workers": workers,
        "summary_reconstruction_wall_seconds": time.perf_counter() - started,
        "frequency_species_jobs": len(cases),
        "continued_trajectory_segments": extension_segments,
        "additional_particle_periods": initial_particle_periods,
        "particle_period_count_policy": "nominal trajectory continuation from the source checkpoints to 50k; any duplicated work during checkpoint recovery is excluded and is not reconstructable from the final checkpoints",
        "ensembles": summary,
        "limitations": [
            "these are continuations of a fixed deterministic panel, not basin probabilities",
            "only species 1 is continued at the three nested transition nodes; species 2 is an additional nu=2.25 control",
            "finite-time rotation signs do not establish asymptotic attractors or a unique critical frequency",
            "RF phases are fixed; no Coulomb or MFS higher spatial harmonics are included",
        ],
    }
    RESULT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"status": "completed", "continued_trajectory_segments": extension_segments,
                      "additional_particle_periods": initial_particle_periods,
                      "recovery_pool_workers": workers,
                      "summary_reconstruction_wall_seconds": report["summary_reconstruction_wall_seconds"],
                      "output": str(RESULT_JSON)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
