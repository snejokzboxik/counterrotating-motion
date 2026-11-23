"""One-parameter continuation of the V17C.21.1 M3 averaged roots.

Each parameter direction is a separate worker.  The known V21.1 M3 root is
used only as a predictor/initial guess; all reported continuation points solve
the complete six averaged balances and both inner harmonic-balance systems.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
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
V21 = ROOT / "research" / "v17c" / "v17c_21" / "code"
V21A = ROOT / "research" / "v17c" / "v17c_21a" / "code"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V21A))
sys.path.insert(0, str(V21))

import analytical_model as legacy  # noqa: E402
from m3_harmonic_balance import HBConfig  # noqa: E402
from m3_system import initial_root, solve_m3_root  # noqa: E402

SEED_JSON = ROOT / "research" / "v17c" / "v17c_21a" / "results" / "full_grid_campaign.json"
SEED_NPZ = ROOT / "research" / "v17c" / "v17c_21a" / "results" / "xi_seed_192_16.npz"
FACTORS = (0.70, 0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20, 1.30)
X_SCALE = np.array([0.55, 0.0011, 0.75, 0.0010, 0.33, 0.86])


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def _baseline() -> tuple[dict, np.ndarray, list[np.ndarray]]:
    data = legacy.load_inputs()
    summary = json.loads(SEED_JSON.read_text(encoding="utf-8"))
    fine = next(row for row in summary["stages"] if row.get("grid") == [192, 16, 16]
                and row.get("status") == "VERIFIED_ROOT")
    with np.load(SEED_NPZ) as f:
        xi = [f["xi1"].copy(), f["xi2"].copy()]
    return data, initial_root(fine["root"]), xi


def change_field_parameter(data: dict, name: str, value: float) -> dict:
    """Copy field data and update a whole RF family consistently."""
    d = dict(data)
    d["modes"] = [replace(m, amplitude=float(value) if (name == "A" and m.family == 0)
                          else float(value) if (name == "B" and m.family == 1)
                          else m.amplitude,
                          frequency=float(value) if (name == "nu" and m.family == 1)
                          else 1.0 if (name == "nu" and m.family == 0)
                          else m.frequency)
                 for m in data["modes"]]
    if name == "beta":
        d["beta"] = float(value)
    return d


def _parameter_value(data: dict, name: str) -> float:
    if name == "A":
        return float(next(m.amplitude for m in data["modes"] if m.family == 0))
    if name == "B":
        return float(next(m.amplitude for m in data["modes"] if m.family == 1))
    if name == "nu":
        return float(next(m.frequency for m in data["modes"] if m.family == 1))
    return float(data["beta"])


def _classify(result: dict, data: dict) -> tuple[str, dict]:
    root = result["root"]
    r = root["radii_R"]
    w = root["omega_over_Omega1"]
    z = root["mean_z_R"]
    p = data["candidate"]["physical_parameter_mapping_from_V20B"]
    R = float(p["trap_radius_m"])
    a = [float(p[f"particle{i}"]["radius_m"]) / R for i in (1, 2)]
    radial_clearance = min(1.0 - r[i] - a[i] for i in range(2))
    axial_clearance = min(2.0 - abs(z[i]) - a[i] for i in range(2))
    pair_clearance = float(np.hypot(r[0] - r[1], z[0] - z[1]) - a[0] - a[1])
    geometric = radial_clearance > 0.0 and axial_clearance > 0.0 and pair_clearance > 0.0
    tol = 1e-6
    if not geometric:
        label = "GEOMETRICALLY_INVALID"
    elif w[0] > tol and w[1] < -tol:
        label = "CR_PLUS_MINUS"
    elif w[0] < -tol and w[1] > tol:
        label = "CR_MINUS_PLUS"
    elif w[0] > tol and w[1] > tol:
        label = "CO_PLUS_PLUS"
    elif w[0] < -tol and w[1] < -tol:
        label = "CO_MINUS_MINUS"
    elif abs(w[0]) <= tol and abs(w[1]) <= tol:
        label = "BOUND_NONROTATING"
    else:
        label = "MIXED_OR_NEAR_ZERO"
    return label, {"radial_wall_clearance_R": radial_clearance,
                   "ideal_axial_envelope_clearance_R": axial_clearance,
                   "pair_surface_clearance_R": pair_clearance,
                   "omega_zero_tolerance": tol}


def _root_row(name: str, factor: float, value: float, x: np.ndarray,
              xi: list[np.ndarray], base_data: dict, config: HBConfig,
              direction: str) -> tuple[dict, np.ndarray, list[np.ndarray]]:
    data = change_field_parameter(base_data, name, value)
    started = time.perf_counter()
    result = solve_m3_root(data, x, config, max_nfev=140,
                           scaled_balance_tolerance=2e-6,
                           require_convergence=True, xi_seed=xi)
    xi_out = [q.copy() for q in result.pop("_xi_grids")]
    result.pop("_xi_fourier", None)
    label, clearances = _classify(result, data)
    inner = result.get("inner", [])
    row = {
        "parameter": name, "factor_from_nominal": factor, "value": value,
        "continuation_direction": direction, "status": result["status"],
        "classification": label, "root": result["root"],
        "scaled_balance_residual": result["scaled_balance_residual"],
        "scaled_balance_max_norm": result["scaled_balance_max_norm"],
        "outer_solver_success": result["outer_solver_success"],
        "outer_message": result["outer_message"], "outer_nfev": result["outer_nfev"],
        "balance_evaluations": result["balance_evaluations"],
        "wall_time_s": time.perf_counter() - started,
        "inner": [{k: v for k, v in q.items() if k in {
            "status", "converged", "iterations", "relative_nonlinear_residual",
            "relative_original_fast_equation_residual", "rms_unresolved_slow_force",
            "rms_unresolved_fast_force", "minimum_fast_mode_denominator",
            "rms_micromotion", "max_micromotion", "mean_speed", "max_speed",
            "mean_Re_estimate", "max_Re_estimate", "rms_nonlinear_force_tail",
            "relative_nonlinear_force_tail", "number_of_excluded_rf_combination_modes",
            "number_fast_modes", "number_of_fast_modes"
        }} for q in inner],
        "geometry": clearances,
        "configuration": {"grid": [config.ntheta, config.nphase, config.nphase],
                          "cutoff": config.cutoff, "method": config.method,
                          "hb_tolerance": config.tolerance,
                          "hb_equation_tolerance": config.equation_tolerance},
    }
    return row, np.asarray([result["root"]["radii_R"][0],
                            result["root"]["omega_over_Omega1"][0],
                            result["root"]["radii_R"][1],
                            result["root"]["omega_over_Omega1"][1],
                            result["root"]["mean_z_R"][0],
                            result["root"]["mean_z_R"][1]]), xi_out


def _run_parameter(name: str) -> dict:
    base_data, seed, seed_xi = _baseline()
    nominal = _parameter_value(base_data, name)
    config = HBConfig(96, 12, cutoff=0.05, oversample=1, method="newton_krylov",
                      tolerance=1e-7, equation_tolerance=5e-5, max_iter=30,
                      relaxation=0.5, history=6)
    output_path = BASE / "results" / f"continuation_{name}.json"
    checkpoint_path = BASE / "results" / f"continuation_{name}_checkpoint.npz"
    result = {
        "study": "V17C.21.2 one-parameter M3 continuation",
        "source_commit": "94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8",
        "parameter": name, "nominal_value": nominal,
        "particle_parameters_fixed": True,
        "whole_rf_family_amplitudes_scaled": name in {"A", "B"},
        "family2_temporal_frequency_updated": name == "nu",
        "holdout_reference_not_used_as_objective": True,
        "solver_config": config.__dict__, "factor_schedule": list(FACTORS),
        "directions": {}, "status": "IN_PROGRESS",
    }

    # Baseline is freshly resolved at the campaign's working grid; the historic
    # fine-grid root only supplies the first predictor and micromotion seed.
    try:
        row0, x0, xi0 = _root_row(name, 1.0, nominal, seed, seed_xi,
                                  base_data, config, "baseline")
    except Exception as exc:
        result["status"] = "BASELINE_REPRODUCTION_FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(output_path, result)
        return result
    result["baseline"] = row0
    result["directions"]["lower"] = []
    result["directions"]["upper"] = []
    _write_json(output_path, result)

    for direction, factors in (("lower", [f for f in FACTORS if f < 1.0][::-1]),
                               ("upper", [f for f in FACTORS if f > 1.0])):
        x, xi = x0.copy(), [q.copy() for q in xi0]
        last_factor = 1.0
        for factor in factors:
            value = nominal * factor
            try:
                row, x_new, xi_new = _root_row(name, factor, value, x, xi,
                                                base_data, config, direction)
                row["step_from_previous_factor"] = factor - last_factor
                result["directions"][direction].append(row)
                x, xi, last_factor = x_new, xi_new, factor
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(checkpoint_path, root=x, xi1=xi[0], xi2=xi[1],
                                    parameter=name, value=value, direction=direction)
                _write_json(output_path, result)
                print(f"M3 continuation {name} {direction} factor={factor:.4g} "
                      f"class={row['classification']} residual={row['scaled_balance_max_norm']:.2e}",
                      flush=True)
            except Exception as exc:
                failure = {"parameter": name, "factor_from_nominal": factor,
                           "value": value, "continuation_direction": direction,
                           "status": "NUMERICALLY_UNRESOLVED",
                           "failure_type": type(exc).__name__, "message": str(exc)}
                result["directions"][direction].append(failure)
                # Preserve the failed point. It is not a bifurcation label.
                _write_json(output_path, result)
                break
            _write_json(output_path, result)
    result["status"] = "COARSE_CONTINUATION_COMPLETE"
    result["interpretation"] = (
        "This is a traced local branch on the declared factor schedule. A failed corrector is "
        "NUMERICALLY_UNRESOLVED, not evidence for a fold or absence of another root."
    )
    _write_json(output_path, result)
    return result


def main() -> None:
    params = ["A", "B", "nu", "beta"]
    max_workers = min(4, len(params), os.cpu_count() or 1)
    print(f"Starting four independent continuation paths in {max_workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_parameter, name): name for name in params}
        for future in as_completed(futures):
            name = futures[future]
            try:
                row = future.result()
                print(f"CONTINUATION {name}: {row['status']}", flush=True)
            except Exception as exc:
                _write_json(BASE / "results" / f"continuation_{name}.json", {
                    "parameter": name, "status": "WORKER_FAILED",
                    "failure_type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"CONTINUATION {name} worker failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
