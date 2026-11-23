"""Re-solve the full nonlinear M3 root for each fast/slow cutoff."""
from __future__ import annotations

import json
import sys
import time
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
    campaign = json.loads((outdir / "full_grid_campaign.json").read_text(encoding="utf-8"))
    baseline = next(x for x in campaign["stages"] if x["grid"] == [192, 16, 16])
    seed = np.asarray(baseline["root_vector"], dtype=float)
    with np.load(outdir / "xi_seed_192_16.npz") as f:
        xi_seed = [f["xi1"], f["xi2"]]
    data = legacy.load_inputs()
    output = {
        "study": "V17C.21.1 complete M3 re-solves by fast/slow Fourier cutoff",
        "grid": [192, 16, 16],
        "cutoffs": [0.02, 0.035, 0.05, 0.075, 0.10],
        "reference_root_used_only_as_initial_seed": True,
        "stages": [],
    }
    for cutoff in output["cutoffs"]:
        cfg = HBConfig(192, 16, cutoff=cutoff, oversample=1, method="newton_krylov",
                       tolerance=1e-7, max_iter=30, relaxation=0.5, history=6)
        print(f"START cutoff={cutoff:g}", flush=True)
        t0 = time.perf_counter()
        if cutoff == 0.05:
            row = {k: v for k, v in baseline.items() if k != "grid"}
            row.update({"cutoff": cutoff, "grid": [192, 16, 16], "source": "full_grid_campaign"})
            output["stages"].append(row)
            (outdir / "cutoff_sensitivity.json").write_text(
                json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            continue
        try:
            result = solve_m3_root(data, seed, cfg, max_nfev=180,
                                   scaled_balance_tolerance=2e-6, xi_seed=xi_seed)
            xi_seed = result.pop("_xi_grids")
            result.pop("_xi_fourier")
            root = result["root"]
            seed = np.array([root["radii_R"][0], root["omega_over_Omega1"][0],
                             root["radii_R"][1], root["omega_over_Omega1"][1],
                             root["mean_z_R"][0], root["mean_z_R"][1]])
            row = {**result, "cutoff": cutoff, "grid": [192, 16, 16],
                   "elapsed_s": time.perf_counter() - t0}
            output["stages"].append(row)
            np.savez_compressed(outdir / f"xi_cutoff_{cutoff:g}.npz", xi1=xi_seed[0], xi2=xi_seed[1])
            print(json.dumps({"cutoff": cutoff, "status": row["status"], "root": root,
                              "excluded_modes": [p.get("number_of_excluded_rf_combination_modes",
                                                       len(p.get("excluded_rf_combination_modes", [])))
                                                 for p in row["inner"]],
                              "min_fast_denominator": [p["minimum_fast_mode_denominator"]
                                                       for p in row["inner"]],
                              "max_balance_residual": row["scaled_balance_max_norm"],
                              "elapsed_s": row["elapsed_s"]}, ensure_ascii=False), flush=True)
        except Exception as exc:
            row = {"cutoff": cutoff, "grid": [192, 16, 16],
                   "status": "FAILED_OR_UNRESOLVED",
                   "error": f"{type(exc).__name__}: {exc}",
                   "elapsed_s": time.perf_counter() - t0}
            output["stages"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        (outdir / "cutoff_sensitivity.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("COMPLETE cutoff sweep", flush=True)


if __name__ == "__main__":
    main()
