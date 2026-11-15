"""High-accuracy follow-up for the one still step-sensitive Qopt FEM orbit."""
from __future__ import annotations

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


def main() -> None:
    path = BASE / "fem_h50_p4_mu110/trajectory_results.csv"
    row = next(r for r in csv.DictReader(path.open(encoding="utf-8", newline=""))
               if float(r["mu"]) == 1.1 and float(r["phase_deg"]) == 0.0 and r["model"] == "Qopt"
               and int(r["index"]) == 14)
    state = np.array([float(row[k]) for k in ("x0", "y0", "vx0", "vy0")])
    MOD.initialize()
    # max_step_tau=2π/2048 and tighter tolerances than the main pilot.
    result = MOD.advance((1.1, 0.0, "Qopt", 0.0, 14, state, 1 / 2048, 3e-10, 3e-12))
    out = BASE / "fem_step_refinement"
    with (out / "high_accuracy_sensitive_2048.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result)); w.writeheader(); w.writerow(result)
    meta = {"case": "mu=1.10, phase=0, Qopt, index=14", "field": "full h50 P4 FEM",
            "max_step_tau": 2 * np.pi / 2048, "rtol": 3e-10, "atol": 3e-12,
            "purpose": "refine the step-sensitive 300-cycle status after runs at 2π/256, 2π/512, and 2π/1024"}
    (out / "high_accuracy_sensitive_2048.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**meta, "status_100": result["status_100"], "status_300": result["status_300"],
                      "exit_periods_300": result["exit_periods_300"], "max_rho_300": result["max_rho_300"]}, indent=2))


if __name__ == "__main__":
    main()
