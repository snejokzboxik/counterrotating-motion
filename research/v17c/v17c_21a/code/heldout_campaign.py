"""Full six-balance M3 re-solves at the four published V21 hold-out points."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "v17c" / "v17c_21" / "code"))
import analytical_model as legacy  # noqa: E402
from m3_harmonic_balance import HBConfig  # noqa: E402
from m3_system import solve_m3_root  # noqa: E402


def main():
    outdir = BASE / "results"
    grid = [128, 12, 12]
    campaign = json.loads((outdir / "full_grid_campaign.json").read_text(encoding="utf-8"))
    base = next(x for x in campaign["stages"] if x["grid"] == [128, 12, 12])
    seed0 = np.asarray(base["root_vector"], dtype=float)
    with np.load(outdir / "xi_seed_192_16.npz") as f:
        xi_seed0 = [f["xi1"], f["xi2"]]
    reference = json.loads((ROOT / "research" / "v17c" / "v17c_21" / "results" /
                            "heldout_validation.json").read_text(encoding="utf-8"))
    cases = [("A", 1.05), ("B", 0.95), ("nu", 1.02), ("beta", 1.05)]
    output = {
        "study": "V17C.21.1 M3 hold-out re-solves",
        "grid": grid,
        "reference_trajectories": "reused V20B/V21 prepublished 15000-period RK4 hold-outs; no rerun",
        "root_seed_source": "V17C.21.1 full 128x12x12 re-solve",
        "rows": [],
    }
    for key, factor in cases:
        data = legacy.load_inputs()
        if key in ("A", "B"):
            family = 0 if key == "A" else 1
            data["modes"] = [replace(m, amplitude=m.amplitude * factor) if m.family == family else m
                             for m in data["modes"]]
        elif key == "nu":
            data["modes"] = [replace(m, frequency=m.frequency * factor) if m.family == 1 else m
                             for m in data["modes"]]
        else:
            data["beta"] *= factor
        cfg = HBConfig(*grid[:2], cutoff=0.05, oversample=1, method="newton_krylov",
                       tolerance=1e-7, max_iter=30, relaxation=0.5, history=6)
        print(f"START holdout {key}*{factor:g}", flush=True)
        root = solve_m3_root(data, seed0, cfg, max_nfev=160,
                             scaled_balance_tolerance=2e-6, xi_seed=xi_seed0)
        ref = next(row for row in reference["rows"]
                   if row["parameter"] == key and row["factor"] == factor)
        row = {
            "parameter": key, "factor": factor, "grid": grid,
            "status": root["status"], "root": root["root"],
            "scaled_balance_residual": root["scaled_balance_residual"],
            "scaled_balance_max_norm": root["scaled_balance_max_norm"],
            "outer_nfev": root["outer_nfev"], "wall_time_s": root["wall_time_s"],
            "inner": root["inner"],
            "RK4_reference": {k: ref[k] for k in
                              ("mean_radius_R", "frequency_Hz", "late_window_frequency_Hz",
                               "horizon_periods") if k in ref},
            "radius_error_percent": [100 * (a - b) / b for a, b in
                                     zip(root["root"]["radii_R"], ref["mean_radius_R"])],
            "frequency_error_percent": [100 * (a - b) / abs(b) for a, b in
                                        zip(root["root"]["frequency_Hz"], ref["frequency_Hz"])],
            "qualification": "M3 full six-balance root at this grid; comparison uses the archived finite-window RK4 estimate.",
        }
        output["rows"].append(row)
        (outdir / "heldout_m3_resolved.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        np.savez_compressed(outdir / f"heldout_{key}_{factor:g}_xi.npz",
                            xi1=root["_xi_grids"][0], xi2=root["_xi_grids"][1])
        print(json.dumps({"case": f"{key}*{factor:g}", "status": root["status"],
                          "root": root["root"], "radius_error_percent": row["radius_error_percent"],
                          "frequency_error_percent": row["frequency_error_percent"]},
                         ensure_ascii=False), flush=True)
    print("COMPLETE all four holdouts", flush=True)


if __name__ == "__main__":
    main()
