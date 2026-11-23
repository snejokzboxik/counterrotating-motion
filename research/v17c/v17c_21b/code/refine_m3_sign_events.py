"""Refine omega_1=0 events along three M3 continuation branches."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))

import m3_continuation as one_d  # noqa: E402
from m3_harmonic_balance import HBConfig  # noqa: E402
from m3_system import M3PairSystem, solve_m3_root  # noqa: E402

PARAMETERS = ("A", "nu", "beta")
SEED_NPZ = ROOT / "research" / "v17c" / "v17c_21a" / "results" / "xi_seed_192_16.npz"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _vec(row: dict) -> np.ndarray:
    r, w, z = row["root"]["radii_R"], row["root"]["omega_over_Omega1"], row["root"]["mean_z_R"]
    return np.array([r[0], w[0], r[1], w[1], z[0], z[1]], dtype=float)


def _find_bracket(name: str, data: dict) -> dict:
    rows = [data["baseline"]]
    rows += [x for seq in data["directions"].values() for x in seq if x.get("status") == "VERIFIED_ROOT"]
    rows.sort(key=lambda row: row["factor_from_nominal"])
    for left, right in zip(rows[:-1], rows[1:]):
        wl = float(left["root"]["omega_over_Omega1"][0])
        wr = float(right["root"]["omega_over_Omega1"][0])
        if wl == 0 or wr == 0 or np.sign(wl) != np.sign(wr):
            return {"parameter": name, "low_factor": float(left["factor_from_nominal"]),
                    "high_factor": float(right["factor_from_nominal"]),
                    "low_row": left, "high_row": right,
                    "initial_omega1": [wl, wr]}
    raise RuntimeError(f"No omega_1 sign-change bracket found for {name}")


def _run_event(name: str) -> dict:
    series_path = BASE / "results" / f"continuation_{name}.json"
    series = json.loads(series_path.read_text(encoding="utf-8"))
    bracket = _find_bracket(name, series)
    base, _, _ = one_d._baseline()
    nominal = one_d._parameter_value(base, name)
    cfg = HBConfig(96, 12, cutoff=0.05, oversample=1, method="newton_krylov",
                   tolerance=1e-7, equation_tolerance=5e-5, max_iter=30,
                   relaxation=0.5, history=6)
    with np.load(SEED_NPZ) as f:
        xi_seed = [f["xi1"].copy(), f["xi2"].copy()]
    lo, hi = bracket["low_factor"], bracket["high_factor"]
    rowlo, rowhi = bracket["low_row"], bracket["high_row"]
    xlo, xhi = _vec(rowlo), _vec(rowhi)
    wlo, whi = bracket["initial_omega1"]
    output = {
        "study": "V17C.21.2 refined M3 direction-change event",
        "source_commit": "94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8",
        "parameter": name, "nominal_value": nominal,
        "event": "omega1_over_Omega1 = 0 on a tracked M3 equilibrium branch",
        "initial_bracket": {"factors": [lo, hi], "omega1": [wlo, whi]},
        "solver_config": cfg.__dict__, "bisection_target_factor_width": 0.001,
        "iterations": [], "status": "IN_PROGRESS",
        "interpretation": "Zero-frequency branch event; not by itself a fold, Hopf event, or full-RF bifurcation.",
    }
    outpath = BASE / "results" / f"event_omega1_zero_{name}.json"
    _write(outpath, output)

    while abs(hi - lo) > 0.001:
        mid = 0.5 * (lo + hi)
        frac = (mid - lo) / (hi - lo)
        seed_x = (1.0 - frac) * xlo + frac * xhi
        start = time.perf_counter()
        try:
            row, xmid, _ = one_d._root_row(name, mid, nominal * mid, seed_x,
                                           xi_seed, base, cfg, "event_bisection")
        except Exception as exc:
            failure = {"factor": mid, "status": "NUMERICALLY_UNRESOLVED",
                       "failure_type": type(exc).__name__, "message": str(exc),
                       "wall_time_s": time.perf_counter() - start}
            output["iterations"].append(failure)
            output["status"] = "BISECTION_UNRESOLVED"
            _write(outpath, output)
            return output
        wmid = float(xmid[1])
        row["iteration_factor_bracket_before"] = [lo, hi]
        row["wall_time_s"] = time.perf_counter() - start
        output["iterations"].append(row)
        if np.sign(wmid) == np.sign(wlo):
            lo, xlo, wlo, rowlo = mid, xmid, wmid, row
        else:
            hi, xhi, whi, rowhi = mid, xmid, wmid, row
        output["current_bracket"] = {"factors": [lo, hi], "omega1": [wlo, whi]}
        _write(outpath, output)
        print(f"M3 event {name}: [{lo:.6f},{hi:.6f}] w1=[{wlo:+.3e},{whi:+.3e}]", flush=True)

    secant = lo - wlo * (hi - lo) / (whi - wlo)
    x0 = xlo + ((secant - lo) / (hi - lo)) * (xhi - xlo)
    system = M3PairSystem(one_d.change_field_parameter(base, name, nominal * secant),
                          cfg, xi_seed=xi_seed)
    xscale = np.array([0.55, 0.0011, 0.75, 0.0010, 0.33, 0.86, max(0.05, abs(secant))])
    lower = np.array([0.1, -0.05, 0.1, -0.05, -2.0, -2.0, 0.6])
    upper = np.array([0.99, 0.05, 0.99, 0.05, 2.0, 2.0, 1.4])

    def event_residual(y):
        x = np.asarray(y[:6], dtype=float)
        factor = float(y[6])
        system.data = one_d.change_field_parameter(base, name, nominal * factor)
        residual, _ = system.balance(x, require_inner=True)
        return np.r_[residual, x[1] * 1000.0]

    started = time.perf_counter()
    try:
        aug = least_squares(event_residual, np.r_[x0, secant], bounds=(lower, upper),
                            x_scale=xscale, xtol=2e-10, ftol=2e-10, gtol=2e-10,
                            max_nfev=60, diff_step=1e-5)
        residual = event_residual(aug.x)
        xevent, factor = np.asarray(aug.x[:6]), float(aug.x[6])
        final_row, final_x, xi_final = one_d._root_row(name, factor, nominal * factor,
                                                        xevent, xi_seed, base, cfg,
                                                        "augmented_zero_frequency")
        event_norm = abs(float(final_x[1]))
        verified = bool(aug.success and final_row["scaled_balance_max_norm"] <= 2e-6
                        and event_norm <= 2e-8)
        output["augmented_event_solve"] = {
            "status": "VERIFIED_EVENT" if verified else "EVENT_NOT_VERIFIED",
            "outer_solver_success": bool(aug.success), "outer_message": str(aug.message),
            "factor": factor, "parameter_value": nominal * factor,
            "root": final_row["root"],
            "scaled_balance_residual": final_row["scaled_balance_residual"],
            "scaled_balance_max_norm": final_row["scaled_balance_max_norm"],
            "omega1_over_Omega1": float(final_x[1]),
            "event_residual_scaled": float(residual[-1]),
            "all_augmented_residuals_inf_norm": float(np.max(np.abs(residual))),
            "wall_time_s": time.perf_counter() - started,
            "evaluation_count": system.evaluations,
        }
        output["final_bisection_bracket"] = {"factors": [lo, hi], "omega1": [wlo, whi],
                                              "secant_factor": secant}
        output["final_event_row"] = final_row
        np.savez_compressed(BASE / "results" / f"event_omega1_zero_{name}_xi.npz",
                            xi1=xi_final[0], xi2=xi_final[1])
        output["status"] = "EVENT_REFINED" if verified else "EVENT_REFINEMENT_UNRESOLVED"
    except Exception as exc:
        output["status"] = "AUGMENTED_EVENT_SOLVE_FAILED"
        output["augmented_failure"] = {"failure_type": type(exc).__name__,
                                       "message": str(exc)}
    _write(outpath, output)
    return output


def main() -> None:
    max_workers = len(PARAMETERS)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_event, name): name for name in PARAMETERS}
        for future in as_completed(futures):
            name = futures[future]
            try:
                row = future.result()
                print(f"M3 event {name}: {row['status']}", flush=True)
            except Exception as exc:
                _write(BASE / "results" / f"event_omega1_zero_{name}.json", {
                    "parameter": name, "status": "WORKER_FAILED",
                    "failure_type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(),
                })
                print(f"M3 event {name} failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
