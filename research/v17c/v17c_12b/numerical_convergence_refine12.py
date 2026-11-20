"""RK4 step-size and independent DOP853 check for one long-lived CR trajectory."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
RESULTS = HERE / "results"
SOURCE = RESULTS / "rescaled_legacy_candidate_validation"
LONG = RESULTS / "long_horizon_refine12_240k_validation"
OUT = RESULTS / "numerical_convergence_refine12"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MODEL))
import physical_mixed_pair_search as base  # noqa: E402
import updated_objective_validation as labels  # noqa: E402

CONFIG_ID = "scaled_refine12_sobol17_f50hz"
PERIODS = 30_000
SKIP = 10_000
RK_STEPS = (64, 96, 128)
DOP_RTL = 1e-9
DOP_ATL = 1e-11


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metric_from_samples(t: np.ndarray, y: np.ndarray, p: dict) -> dict:
    period = 2.0 * np.pi
    turns = []
    windows = np.zeros((3, 2))
    theta = [np.unwrap(np.arctan2(y[1], y[0])), np.unwrap(np.arctan2(y[7], y[6]))]
    post_transient = int(round(SKIP * 8))
    for i in range(2):
        turns.append(float((theta[i][-1] - theta[i][post_transient]) / (2.0 * np.pi)))
        edges = np.linspace(SKIP * period, PERIODS * period, 4)
        for w in range(3):
            lo = int(np.searchsorted(t, edges[w], side="left"))
            hi = int(np.searchsorted(t, edges[w + 1], side="right")) - 1
            windows[w, i] = (theta[i][hi] - theta[i][lo]) / (2.0 * np.pi)
    rmin = [float(np.min(np.hypot(y[0], y[1]))), float(np.min(np.hypot(y[6], y[7])))]
    rmax = [float(np.max(np.hypot(y[0], y[1]))), float(np.max(np.hypot(y[6], y[7])))]
    zmax = [float(np.max(np.abs(y[2]))), float(np.max(np.abs(y[8])))]
    dist = np.sqrt((y[0]-y[6])**2 + (y[1]-y[7])**2 + (y[2]-y[8])**2)
    freq = [turns[i] / max(1, PERIODS-SKIP) for i in range(2)]
    strict = labels.strict_late_label("ok", np.asarray(turns), windows, True, min(rmin) < base.sim.RMIN_CLASS)
    return {"turns": turns, "window_turns": windows.tolist(), "frequency_over_rf": freq,
            "strict_label": strict, "r_min_over_R": rmin, "r_max_over_R": rmax,
            "abs_z_max_over_R": zmax, "min_pair_distance_over_R_sampled": float(np.min(dist))}


def main() -> None:
    meta = json.loads((SOURCE / "campaign_meta.json").read_text(encoding="utf-8"))
    cfg = next(c for c in meta["candidates"] if c["id"] == CONFIG_ID)
    physical = cfg["physical"]
    long_rows = [json.loads(line) for line in (LONG / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = [r for r in long_rows if r["strict_late_category"].startswith("STRICT_CR_")
                and min(abs(float(t)) for t in r["turns"]) >= 100]
    if not eligible:
        raise RuntimeError("no audited long-horizon >=100-turn trajectory is available")
    median = float(np.median([min(abs(float(t)) for t in r["turns"]) for r in eligible]))
    selected = min(eligible, key=lambda r: abs(min(abs(float(t)) for t in r["turns"]) - median))
    start = np.asarray(selected["initial"], dtype=np.float64)
    phase = np.asarray(selected["phase"], dtype=np.float64)
    p = np.asarray(cfg["p"], dtype=np.float64)
    _, family, k, coeff = base.sim.load_model()
    re1 = physical["particle1"]["Re_per_dimensionless_speed"]
    re2 = physical["particle2"]["Re_per_dimensionless_speed"]
    slip = 0.5 * (physical["particle1"]["Cunningham"] + physical["particle2"]["Cunningham"])
    common = {"qratio": cfg["qratio"], "oseen_Re_per_speed": re1, "oseen_Re2_per_speed": re2,
              "hydro_mode": 2, "a1_over_R": physical["a1_over_R"], "a2_over_R": physical["a2_over_R"],
              "gravity_z": physical["gravity_z_dimensionless"], "drag_mode": 2, "slip_correction": slip}
    results = {}
    started = time.perf_counter()
    for steps in RK_STEPS:
        raw = base.sim.integrate_one(start, phase, p, family, k, coeff, PERIODS, steps, SKIP, **common)
        end, turns, windows, rmin, rmax, zmax, dtheta, dmin, completed, retained, escaped, singular, category, freq, vmax = raw
        strict = labels.strict_late_label(str(category), turns, windows, bool(retained), bool(min(rmin) < base.sim.RMIN_CLASS))
        results[f"RK4_{steps}"] = {"turns": turns.tolist(), "window_turns": windows.tolist(),
            "frequency_over_rf": freq.tolist(), "strict_label": strict, "raw_category": str(category),
            "retained": bool(retained), "completed_periods": int(completed), "r_min_over_R": rmin.tolist(),
            "r_max_over_R": rmax.tolist(), "abs_z_max_over_R": zmax.tolist(),
            "min_pair_distance_over_R": float(dmin), "max_Re_by_particle": (vmax*np.asarray([re1,re2])).tolist()}
    rk_seconds = time.perf_counter() - started

    # DOP853 integrates the same non-autonomous RHS but does not share the RK4 update loop.
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        dy = np.empty(12, dtype=np.float64)
        base.sim._rhs(t, y, p, phase, family, k, coeff, cfg["qratio"], dy,
                      re1, 2, physical["a1_over_R"], physical["a2_over_R"], re2,
                      physical["gravity_z_dimensionless"], 2, slip)
        return dy

    t_end = 2.0 * np.pi * PERIODS
    sample_count = PERIODS * 8 + 1
    t_eval = np.linspace(0.0, t_end, sample_count)
    dop_start = time.perf_counter()
    dop = solve_ivp(rhs, (0.0, t_end), start, method="DOP853", t_eval=t_eval,
                    rtol=DOP_RTL, atol=DOP_ATL, max_step=2.0*np.pi/64.0)
    dop_seconds = time.perf_counter() - dop_start
    if not dop.success or dop.y.shape[1] != sample_count:
        raise RuntimeError(f"DOP853 failed: {dop.message}")
    results["DOP853"] = {**metric_from_samples(dop.t, dop.y, physical),
        "success": bool(dop.success), "nfev": int(dop.nfev), "njev": int(dop.njev),
        "nlu": int(dop.nlu), "rtol": DOP_RTL, "atol": DOP_ATL,
        "max_step_over_main_rf_period": 1/64, "sample_points_per_main_rf_period": 8}
    np.savez_compressed(OUT / "dop853_representative_trajectory.npz", t=dop.t, y=dop.y,
                        initial=start, phase=phase)
    output = {"study": "V17C.12B RK4 step refinement and independent DOP853 check",
        "configuration": CONFIG_ID, "selected_start": {"ensemble": selected["ensemble"],
            "start_index": selected["start_index"], "source_240k_turns": selected["turns"],
            "initial": start.tolist(), "phase": phase.tolist()},
        "integration_horizon_periods": PERIODS, "skip_periods": SKIP,
        "dimensionless_parameters": cfg["p"], "physical_drag_mode": "Schiller-Naumann with mean Cunningham slip correction",
        "results": results, "runtime_seconds": {"RK4_all": rk_seconds, "DOP853": dop_seconds},
        "interpretation": "Comparison of orbit metrics/classification for a representative long-lived trajectory; not a basin statistic.",
        "source_hashes": {"runner": sha(Path(__file__)), "integrator": sha(MODEL/"basin_scan.py"),
            "field_model": sha(MODEL/"field_model.json"), "source_campaign_meta": sha(SOURCE/"campaign_meta.json"),
            "long_horizon_summary": sha(LONG/"summary.json")}}
    path = OUT / "summary.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(path), "results": results, "runtime_seconds": output["runtime_seconds"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
