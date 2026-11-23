"""Coarse branch-tracked 2-D maps for the three priority M3 parameter pairs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))

import m3_continuation as one_d  # noqa: E402
from m3_harmonic_balance import HBConfig  # noqa: E402
from m3_system import solve_m3_root  # noqa: E402

FACTORS = (0.80, 0.90, 1.00, 1.10, 1.20)
PAIRS = (("A", "B"), ("A", "beta"), ("B", "nu"))
SEED_JSON = ROOT / "research" / "v17c" / "v17c_21a" / "results" / "full_grid_campaign.json"
SEED_NPZ = ROOT / "research" / "v17c" / "v17c_21a" / "results" / "xi_seed_192_16.npz"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _nominal(data: dict, name: str) -> float:
    return one_d._parameter_value(data, name)


def _set_pair(base: dict, pair: tuple[str, str], factors: tuple[float, float]) -> dict:
    out = base
    for name, factor in zip(pair, factors):
        out = one_d.change_field_parameter(out, name, _nominal(base, name) * factor)
    return out


def _solve_point(base: dict, pair: tuple[str, str], factors: tuple[float, float],
                 seed: np.ndarray, xi: list[np.ndarray], cfg: HBConfig,
                 row_label: str) -> tuple[dict, np.ndarray, list[np.ndarray]]:
    data = _set_pair(base, pair, factors)
    start = time.perf_counter()
    result = solve_m3_root(data, seed, cfg, max_nfev=140,
                           scaled_balance_tolerance=2e-6,
                           require_convergence=True, xi_seed=xi)
    xi_out = [q.copy() for q in result.pop("_xi_grids")]
    result.pop("_xi_fourier", None)
    label, clearance = one_d._classify(result, data)
    row = {
        "parameter_pair": list(pair), "factors_from_nominal": list(factors),
        "values": [_nominal(base, n) * f for n, f in zip(pair, factors)],
        "row_continuation": row_label, "status": result["status"],
        "classification": label, "root": result["root"],
        "scaled_balance_residual": result["scaled_balance_residual"],
        "scaled_balance_max_norm": result["scaled_balance_max_norm"],
        "outer_solver_success": result["outer_solver_success"],
        "outer_nfev": result["outer_nfev"],
        "balance_evaluations": result["balance_evaluations"],
        "wall_time_s": time.perf_counter() - start,
        "inner": [{k: v for k, v in q.items() if k in {
            "status", "converged", "iterations", "relative_nonlinear_residual",
            "relative_original_fast_equation_residual", "rms_unresolved_slow_force",
            "rms_unresolved_fast_force", "minimum_fast_mode_denominator",
            "rms_micromotion", "max_micromotion", "max_speed", "mean_Re_estimate",
            "max_Re_estimate", "number_of_excluded_rf_combination_modes"
        }} for q in result.get("inner", [])],
        "geometry": clearance,
        "configuration": {"grid": [cfg.ntheta, cfg.nphase, cfg.nphase],
                          "cutoff": cfg.cutoff, "method": cfg.method,
                          "hb_tolerance": cfg.tolerance,
                          "hb_equation_tolerance": cfg.equation_tolerance},
    }
    x = np.array([row["root"]["radii_R"][0], row["root"]["omega_over_Omega1"][0],
                  row["root"]["radii_R"][1], row["root"]["omega_over_Omega1"][1],
                  row["root"]["mean_z_R"][0], row["root"]["mean_z_R"][1]])
    return row, x, xi_out


def _run_row(pair: tuple[str, str], factor_y: float) -> dict:
    base, root, xi_seed = one_d._baseline()
    cfg = HBConfig(96, 12, cutoff=0.05, oversample=1, method="newton_krylov",
                   tolerance=1e-7, equation_tolerance=5e-5, max_iter=30,
                   relaxation=0.5, history=6)
    filename = "map_" + "_".join(pair)
    path = BASE / "results" / f"{filename}_y{factor_y:.2f}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("status") in {"ROW_COMPLETE", "CENTER_UNRESOLVED", "PARTIAL_UNRESOLVED"}:
            return {"pair": list(pair), "second_factor": factor_y,
                    "status": "ROW_ALREADY_PRESENT",
                    "points": len(saved.get("points", []))}
    row_result = {
        "study": "V17C.21.2 coarse branch-tracked M3 parameter map",
        "source_commit": "94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8",
        "parameter_pair": list(pair), "second_factor": factor_y,
        "factor_grid_x": list(FACTORS), "factor_grid_y": list(FACTORS),
        "solver_config": cfg.__dict__, "points": [], "status": "ROW_IN_PROGRESS",
        "fixed_particle_parameters": True,
        "scope": "Existence/classification of a tracked local M3 branch. Stability spectra are computed only at selected points; no basin probability is implied.",
    }
    center_factors = (1.0, factor_y)
    try:
        center, x_center, xi_center = _solve_point(base, pair, center_factors,
                                                    root, xi_seed, cfg, "row_center")
        row_result["points"].append(center)
    except Exception as exc:
        row_result["points"].append({"parameter_pair": list(pair),
            "factors_from_nominal": list(center_factors), "status": "NUMERICALLY_UNRESOLVED",
            "failure_type": type(exc).__name__, "message": str(exc)})
        row_result["status"] = "CENTER_UNRESOLVED"
        _write(path, row_result)
        return {"pair": list(pair), "second_factor": factor_y,
                "status": row_result["status"], "points": len(row_result["points"])}

    for side, factors in (("lower_x", [f for f in FACTORS if f < 1.0][::-1]),
                          ("upper_x", [f for f in FACTORS if f > 1.0])):
        xprev, xiseed = x_center.copy(), [q.copy() for q in xi_center]
        for fx in factors:
            try:
                point, xnew, xinew = _solve_point(base, pair, (fx, factor_y),
                                                   xprev, xiseed, cfg, side)
                row_result["points"].append(point)
                xprev, xiseed = xnew, xinew
                _write(path, row_result)
                print(f"M3 map {pair[0]}x{pair[1]} y={factor_y:.2f} x={fx:.2f} "
                      f"{point['classification']} residual={point['scaled_balance_max_norm']:.2e}",
                      flush=True)
            except Exception as exc:
                row_result["points"].append({"parameter_pair": list(pair),
                    "factors_from_nominal": [fx, factor_y], "row_continuation": side,
                    "status": "NUMERICALLY_UNRESOLVED",
                    "failure_type": type(exc).__name__, "message": str(exc)})
                row_result["status"] = "PARTIAL_UNRESOLVED"
                _write(path, row_result)
                break
    if row_result["status"] == "ROW_IN_PROGRESS":
        row_result["status"] = "ROW_COMPLETE"
    _write(path, row_result)
    return {"pair": list(pair), "second_factor": factor_y, "status": row_result["status"],
            "points": len(row_result["points"])}


def main() -> None:
    jobs = [(pair, fy) for pair in PAIRS for fy in FACTORS]
    max_workers = min(4, os.cpu_count() or 1)
    print(f"Starting {len(jobs)} map rows in up to {max_workers} workers", flush=True)
    # Checkpointing occurs per completed point; rows are independent, but a
    # single process owns each pair's JSON file to avoid write races.
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_row, pair, fy): (pair, fy) for pair, fy in jobs}
        for future in as_completed(futures):
            pair, fy = futures[future]
            try:
                print(json.dumps(future.result(), ensure_ascii=True, allow_nan=False), flush=True)
            except Exception as exc:
                print(f"MAP ROW FAILED {pair} y={fy}: {type(exc).__name__}: {exc}", flush=True)

    for pair in PAIRS:
        filename = "map_" + "_".join(pair)
        rows = []
        for fy in FACTORS:
            row_path = BASE / "results" / f"{filename}_y{fy:.2f}.json"
            if row_path.exists():
                rows.append(json.loads(row_path.read_text(encoding="utf-8")))
        payload = {
            "study": "V17C.21.2 coarse branch-tracked M3 parameter map",
            "source_commit": "94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8",
            "parameter_pair": list(pair), "factor_grid": list(FACTORS),
            "solver_config": {"grid": [96, 12, 12], "cutoff": 0.05,
                              "method": "newton_krylov", "tolerance": 1e-7},
            "map_rows": rows, "fixed_particle_parameters": True,
            "stability_spectra": "computed only at selected points documented separately",
            "status": "COARSE_MAP_COMPLETE" if all(row["status"] == "ROW_COMPLETE"
                for row in rows) and len(rows) == len(FACTORS) else "COARSE_MAP_PARTIAL",
            "scope": "Continuation-tracked local M3 roots; not capture probabilities or a complete global bifurcation diagram.",
        }
        _write(BASE / "results" / f"{filename}.json", payload)


if __name__ == "__main__":
    main()
