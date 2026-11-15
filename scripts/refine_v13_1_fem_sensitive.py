"""Refine the FEM half-step cases that changed status at max_step_tau=2π/128."""
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
    audit = list(csv.DictReader((BASE / "fem_step_refinement/step_comparison.csv").open(encoding="utf-8", newline="")))
    changed = [r for r in audit if r["same100"] == "False" or r["same300"] == "False"]
    source_paths = [BASE / "fem_h50_p4_mu100_105/trajectory_results.csv",
                    BASE / "fem_h50_p4_mu110/trajectory_results.csv"]
    original = []
    for path in source_paths:
        original.extend(csv.DictReader(path.open(encoding="utf-8", newline="")))
    selected = []
    for a in changed:
        r = next(r for r in original if float(r["mu"]) == float(a["mu"])
                 and float(r["phase_deg"]) == float(a["phase_deg"])
                 and r["model"] == a["model"] and float(r["eta"]) == float(a["eta"])
                 and int(r["index"]) == int(a["initial_index"]))
        selected.append((a, r))

    tasks = []
    for a, r in selected:
        state = np.array([float(r[k]) for k in ("x0", "y0", "vx0", "vy0")])
        for fraction in (1 / 256, 1 / 512):
            tasks.append((float(r["mu"]), float(r["phase_rad"]), r["model"], float(r["eta"]),
                          int(r["index"]), state, fraction))

    results = {}
    with ProcessPoolExecutor(max_workers=2, initializer=initialize) as pool:
        for future in as_completed([pool.submit(run, t) for t in tasks]):
            z = future.result()
            key = (z["mu"], z["phase_rad"], z["model"], z["eta"], z["index"], z["max_step_tau"])
            results[key] = z

    rows = []
    for a, old in selected:
        base_key = (float(old["mu"]), float(old["phase_rad"]), old["model"], float(old["eta"]), int(old["index"]))
        row = {"mu": old["mu"], "phase_deg": old["phase_deg"], "model": old["model"], "eta": old["eta"],
               "initial_index": old["index"], "source_status_tau64": old["status_300"],
               "audit_status_tau128": a["status300_fine"]}
        for fraction in (1 / 256, 1 / 512):
            step = 2 * np.pi * fraction
            z = results[(*base_key, step)]
            tag = str(256 if fraction == 1 / 256 else 512)
            row[f"status100_tau{tag}"] = z["status_100"]
            row[f"status300_tau{tag}"] = z["status_300"]
            row[f"exit300_tau{tag}"] = z["exit_periods_300"]
            row[f"maxrho300_tau{tag}"] = z["max_rho_300"]
            row[f"error_tau{tag}"] = z.get("error", "")
        rows.append(row)

    out = BASE / "fem_step_refinement"
    with (out / "sensitive_refinement.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    meta = {"n_refined": len(rows), "field": "full h50 P4 FEM",
            "max_step_tau": [2 * np.pi / 64, 2 * np.pi / 128, 2 * np.pi / 256, 2 * np.pi / 512],
            "protocol": "Only trajectories whose status changed between tau step 2π/64 and 2π/128; rerun to 2π/256 and 2π/512.",
            "unresolved_errors": sum(bool(r.get("error_tau256")) or bool(r.get("error_tau512")) for r in rows)}
    (out / "sensitive_refinement.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
