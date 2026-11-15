"""Half-step check for boundary-near full-P4 FEM trajectories."""
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
SPEC = importlib.util.spec_from_file_location("v131fempilot", ROOT / "scripts/run_v13_1_fem_pilot.py")
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def run(task):
    return MOD.advance(task)


def initialize():
    MOD.initialize()


def main() -> None:
    files = [BASE / "fem_h50_p4_mu100_105/trajectory_results.csv",
             BASE / "fem_h50_p4_mu110/trajectory_results.csv"]
    data = []
    for path in files:
        data.extend(csv.DictReader(path.open(encoding="utf-8", newline="")))

    selected = []
    for mu in (1.0, 1.05, 1.10):
        for phase in (0.0, 90.0):
            for model, eta in (("Q", 0.0), ("Qopt", 0.0), ("SO", -0.05), ("RO", -0.05)):
                group = [r for r in data if float(r["mu"]) == mu and float(r["phase_deg"]) == phase
                         and r["model"] == model and float(r["eta"]) == eta]
                # Select one marginal trajectory per group: a retained orbit with largest rho,
                # or a loss occurring closest to the 300-cycle horizon.
                retained = [r for r in group if r["status_300"] == "retained"]
                escaped = [r for r in group if r["status_300"] == "escaped"]
                if retained:
                    chosen = max(retained, key=lambda r: float(r["max_rho_300"]))
                    boundary_metric = "largest_subaperture_rho"
                    metric_value = float(chosen["max_rho_300"])
                elif escaped:
                    chosen = max(escaped, key=lambda r: float(r["exit_periods_300"] or -1.0))
                    boundary_metric = "latest_exit_period"
                    metric_value = float(chosen["exit_periods_300"] or 0.0)
                else:
                    continue
                selected.append((chosen, boundary_metric, metric_value))

    tasks = []
    for row, _, _ in selected:
        state = np.array([float(row[k]) for k in ("x0", "y0", "vx0", "vy0")])
        tasks.append((float(row["mu"]), float(row["phase_rad"]), row["model"], float(row["eta"]),
                      int(row["index"]), state, 1 / 128))

    fine = {}
    with ProcessPoolExecutor(max_workers=3, initializer=initialize) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            key = (result["mu"], result["phase_rad"], result["model"], result["eta"], result["index"])
            fine[key] = result

    out_rows = []
    for old, metric, value in selected:
        key = (float(old["mu"]), float(old["phase_rad"]), old["model"], float(old["eta"]), int(old["index"]))
        refined = fine[key]
        e100 = float(old["exit_periods_100"]) if old["exit_periods_100"] else None
        e300 = float(old["exit_periods_300"]) if old["exit_periods_300"] else None
        f100 = refined["exit_periods_100"]
        f300 = refined["exit_periods_300"]
        out_rows.append({
            "mu": old["mu"], "phase_deg": old["phase_deg"], "model": old["model"], "eta": old["eta"],
            "initial_index": old["index"], "selection_metric": metric, "selection_value": value,
            "coarse_max_step_tau": 2 * np.pi / 64, "fine_max_step_tau": 2 * np.pi / 128,
            "status100_coarse": old["status_100"], "status100_fine": refined["status_100"],
            "status300_coarse": old["status_300"], "status300_fine": refined["status_300"],
            "same100": old["status_100"] == refined["status_100"],
            "same300": old["status_300"] == refined["status_300"],
            "exit100_difference_periods": f100 - e100 if f100 is not None and e100 is not None else None,
            "exit300_difference_periods": f300 - e300 if f300 is not None and e300 is not None else None,
            "maxrho300_coarse": old["max_rho_300"], "maxrho300_fine": refined["max_rho_300"],
            "fine_error": refined.get("error", ""),
        })

    out = BASE / "fem_step_refinement"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "step_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    exits = [abs(r["exit300_difference_periods"]) for r in out_rows if r["exit300_difference_periods"] is not None]
    meta = {
        "n_selected": len(out_rows), "workers": 3, "field": "full native h50 P4 FEM m4+m2",
        "mass_ratios": [1.0, 1.05, 1.10], "phase_deg": [0, 90],
        "selection": "One trajectory per mass/phase/model group, closest to aperture boundary among retained trajectories or latest exit among escaped trajectories.",
        "max_step_tau": [2 * np.pi / 64, 2 * np.pi / 128],
        "status_disagreements_100": sum(not r["same100"] for r in out_rows),
        "status_disagreements_300": sum(not r["same300"] for r in out_rows),
        "max_abs_exit_time_difference_periods": max(exits) if exits else None,
        "unresolved_fine": sum(bool(r["fine_error"]) or r["status300_fine"] == "unresolved" for r in out_rows),
    }
    (out / "step_refinement.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
