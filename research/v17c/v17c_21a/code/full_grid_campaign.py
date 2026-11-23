"""Full six-balance M3 re-solves on the required Fourier phase grids.

Each checkpoint records a complete external root solve and its two verified
inner harmonic-balance solves.  Coarse-grid roots are seeds only; no archived
root is accepted as the new grid result.
"""
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
from m3_system import initial_root, solve_m3_root  # noqa: E402


def json_safe(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def main():
    outdir = BASE / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    data = legacy.load_inputs()
    pred = json.loads((ROOT / "research" / "v17c" / "v17c_21" / "results" /
                       "analytical_predictions.json").read_text(encoding="utf-8"))
    archived = pred["models"]["M3_full_MFS_SN_torus_Galerkin"]
    root_seed = initial_root(archived)
    grids = [(96, 8), (96, 12), (128, 12), (192, 16)]
    summary_path = outdir / "full_grid_campaign.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        prior = {
            "study": "V17C.21.1 full nonlinear M3 grid convergence",
            "source_commit": "05a8d8ce811c9858ab1e74b67b15139c557018b1",
            "reference_root_used_only_as_initial_seed": True,
            "stages": [],
        }
    if prior.get("campaign_schema_version") != 6:
        prior = {
            "study": "V17C.21.1 full nonlinear M3 grid convergence",
            "source_commit": "05a8d8ce811c9858ab1e74b67b15139c557018b1",
            "reference_root_used_only_as_initial_seed": True,
            "campaign_schema_version": 6,
            "stages": [],
        }
    done = {tuple(row["grid"]) for row in prior["stages"] if row.get("status") == "VERIFIED_ROOT"}
    seed = root_seed.copy()
    xi_seed = None
    for ntheta, nphase in grids:
        grid = (ntheta, nphase, nphase)
        if grid in done:
            row = next(s for s in prior["stages"] if tuple(s["grid"]) == grid)
            seed = np.asarray(row["root_vector"], dtype=float)
            xi_path = outdir / f"xi_seed_{ntheta}_{nphase}.npz"
            if xi_path.exists():
                with np.load(xi_path) as f:
                    xi_seed = [f["xi1"], f["xi2"]]
            continue
        cfg = HBConfig(ntheta, nphase, cutoff=0.05, oversample=1, method="newton_krylov",
                       tolerance=1e-7, max_iter=30, relaxation=0.5, history=6)
        print(f"START full M3 re-solve grid={grid}", flush=True)
        t0 = time.perf_counter()
        try:
            result = solve_m3_root(data, seed, cfg, max_nfev=160,
                                   scaled_balance_tolerance=2e-6, xi_seed=xi_seed)
        except Exception as exc:
            failure = {
                "grid": list(grid), "status": "FAILED_OR_UNRESOLVED",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": time.perf_counter() - t0,
                "seed_root_vector": seed.tolist(),
                "config": cfg.__dict__,
            }
            prior["stages"].append(failure)
            summary_path.write_text(json.dumps(prior, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
            print(json.dumps(failure, ensure_ascii=False), flush=True)
            raise
        xi_seed = result.pop("_xi_grids")
        coeffs = result.pop("_xi_fourier")
        root = result["root"]
        seed = np.array([root["radii_R"][0], root["omega_over_Omega1"][0],
                         root["radii_R"][1], root["omega_over_Omega1"][1],
                         root["mean_z_R"][0], root["mean_z_R"][1]], dtype=float)
        np.savez_compressed(outdir / f"xi_seed_{ntheta}_{nphase}.npz",
                            xi1=xi_seed[0], xi2=xi_seed[1],
                            xi1_hat=coeffs[0], xi2_hat=coeffs[1])
        row = {k: v for k, v in result.items() if k != "inner"}
        row["grid"] = list(grid)
        row["status"] = result["status"]
        row["seed_root_vector"] = root_seed.tolist() if not prior["stages"] else None
        row["root_vector"] = seed.tolist()
        row["elapsed_s"] = time.perf_counter() - t0
        row["inner"] = result["inner"]
        row["archived_root_delta"] = {
            "radii_percent": [100 * (a - b) / b for a, b in zip(root["radii_R"], archived["radii_R"])],
            "frequency_percent": [100 * (a - b) / abs(b) for a, b in
                                  zip(root["frequency_Hz"], archived["orbital_frequency_Hz"])],
            "mean_z_delta_R": [a - b for a, b in zip(root["mean_z_R"], archived["mean_z_R"])],
        }
        prior["stages"].append(row)
        summary_path.write_text(json.dumps(prior, indent=2, ensure_ascii=False, default=json_safe) + "\n",
                                encoding="utf-8")
        print(json.dumps({"grid": grid, "status": row["status"], "root": root,
                          "scaled_balance_max_norm": row["scaled_balance_max_norm"],
                          "nfev": row["outer_nfev"], "elapsed_s": row["elapsed_s"]},
                         ensure_ascii=False), flush=True)
        if result["status"] != "VERIFIED_ROOT":
            raise RuntimeError(f"external M3 root not verified for grid {grid}")
    print(f"COMPLETE all required grids; checkpoint={summary_path}", flush=True)


if __name__ == "__main__":
    main()
