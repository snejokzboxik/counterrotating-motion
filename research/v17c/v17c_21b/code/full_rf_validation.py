"""Independent full 12-state RF validation at predeclared M3 points.

Uses the published V17C.17 RK4/Numba pair integrator and V20B corrected
refine12 physical parameters. No M3 fitting to the full trajectories is done.
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
V20 = ROOT / "research" / "v17c" / "v17c_20b" / "code"
V21 = ROOT / "research" / "v17c" / "v17c_21" / "code"
sys.path.insert(0, str(V20))
sys.path.insert(0, str(V21))

import attractor_search as src  # noqa: E402


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def read_roots() -> dict[str, dict]:
    choices = {
        "M3_CR_nominal": ("A", 1.0),
        "M3_CO_A_low": ("A", 0.80),
        "M3_CO_beta_high": ("beta", 1.30),
    }
    rows = {}
    for label, (parameter, factor) in choices.items():
        payload = json.loads((BASE / "results" / f"continuation_{parameter}.json").read_text(encoding="utf-8"))
        candidate = payload["baseline"] if factor == 1.0 else next(
            row for seq in payload["directions"].values() for row in seq
            if abs(row["factor_from_nominal"] - factor) < 1e-10)
        rows[label] = {"parameter": parameter, "factor": factor, "row": candidate}
    event = json.loads((BASE / "results" / "event_omega1_zero_A.json").read_text(encoding="utf-8"))
    bracket = event["final_bisection_bracket"]
    iter_rows = event["iterations"]
    for label, idx in (("M3_A_sign_before", 0), ("M3_A_sign_after", 1)):
        factor = bracket["factors"][idx]
        candidate = next(row for row in iter_rows if abs(row["factor_from_nominal"] - factor) < 1e-12)
        rows[label] = {"parameter": "A", "factor": factor, "row": candidate}
    return rows


def build_start(row: dict) -> np.ndarray:
    root = row["root"]
    r, w, z = root["radii_R"], root["omega_over_Omega1"], root["mean_z_R"]
    state = np.zeros(12)
    for i, j in enumerate((0, 6)):
        state[j], state[j + 2], state[j + 4] = r[i], z[i], r[i] * w[i]
    return state


def _run_case(label: str, spec: dict, case: dict, start_cfg: dict, field: dict,
              periods: int, steps: int, skip: int) -> dict:
    parameter, factor, row = spec["parameter"], spec["factor"], spec["row"]
    p, qratio = src.arrays(case, field)
    if parameter == "A":
        p[0] *= factor
    elif parameter == "B":
        p[1] *= factor
    elif parameter == "nu":
        p[2] *= factor
    elif parameter == "beta":
        p[3] *= factor
    phase = np.asarray(start_cfg["phase"], dtype=float)
    initial = build_start(row)
    started = time.perf_counter()
    raw = src.PAIR._integrate_trace(
        initial, phase, p, field["family"], field["k"], field["coeff"], qratio,
        case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"], case["re2_per_speed"],
        case["gravity_z"], periods, steps, skip, steps * 10)
    end, states, idx, work, turns, windows, rmin, rmax, zmax, vmax, dmin, meanr, meand, meanu, winr, wind, moment, completed, status = raw
    freq = np.asarray(turns, dtype=float) * case["f1_Hz"] / max(1, completed - skip)
    signs = [[0 if abs(v) < 0.05 else int(np.sign(v)) for v in w] for w in windows]
    if status != 0:
        classification = {1: "COLLISION", 2: "ESCAPE", 3: "NUMERIC_OR_CLOSE_PAIR"}.get(int(status), "UNRESOLVED")
    elif min(abs(np.asarray(turns))) < 10 or any(0 in s for s in signs):
        classification = "UNRESOLVED"
    elif np.all(np.asarray(windows)[:, 0] > 0) and np.all(np.asarray(windows)[:, 1] < 0):
        classification = "CR_PLUS_MINUS"
    elif np.all(np.asarray(windows)[:, 0] < 0) and np.all(np.asarray(windows)[:, 1] > 0):
        classification = "CR_MINUS_PLUS"
    elif np.all(np.asarray(windows)[:, 0] > 0) and np.all(np.asarray(windows)[:, 1] > 0):
        classification = "CO_PLUS_PLUS"
    elif np.all(np.asarray(windows)[:, 0] < 0) and np.all(np.asarray(windows)[:, 1] < 0):
        classification = "CO_MINUS_MINUS"
    else:
        classification = "UNRESOLVED"
    out = {
        "label": label, "parameter": parameter, "factor": factor,
        "m3_classification": row["classification"], "m3_root": row["root"],
        "initial_state_from_m3_circular_mean": initial.tolist(), "initial_rf_phases_rad": phase.tolist(),
        "field_parameters": {"A": float(p[0]), "B": float(p[1]), "nu": float(p[2]), "beta": float(p[3])},
        "model": "V20B corrected refine12; complete 8-mode 2D MFS transverse field; 3D 12-state RK4; Coulomb, gravity, ideal DC quadrupole, Cunningham-corrected Stokes x Schiller-Naumann; polarization off; infinite rods",
        "numerics": {"integrator": "published V17C.17 Numba RK4 _integrate_trace", "periods_requested": periods,
                     "periods_completed": int(completed), "steps_per_RF1_period": steps, "skip_periods": skip,
                     "saved_stride_steps": steps * 10, "wall_time_s": time.perf_counter() - started},
        "status_code": int(status), "classification": classification,
        "turns_late": np.asarray(turns).tolist(), "window_turns_late": np.asarray(windows).tolist(),
        "window_signs": signs, "frequency_Hz_from_turns": freq.tolist(),
        "mean_radius_R": np.asarray(meanr).tolist(), "r_min_R": np.asarray(rmin).tolist(),
        "r_max_R": np.asarray(rmax).tolist(), "max_abs_z_R": np.asarray(zmax).tolist(),
        "max_speed_dimensionless": np.asarray(vmax).tolist(), "minimum_center_distance_R": float(dmin),
        "mean_center_distance_R": float(meand), "mean_Coulomb_potential_proxy": float(meanu),
        "M3_prediction_is_not_retargeted": True,
    }
    stem = label + f"_rk4{steps}_{periods}p"
    np.savez_compressed(BASE / "results" / f"{stem}.npz", final_state=end, states=states,
                        step_index=idx, work_history=work, turns=turns, window_turns=windows,
                        initial=initial, phase=phase, p=p, qratio=qratio,
                        periods=periods, steps=steps, skip=skip)
    write_json(BASE / "results" / f"{stem}.json", out)
    print(f"full RF {label}: class={classification}, turns={np.round(turns,2)}, status={status}", flush=True)
    return out


def main() -> None:
    field = src.load_field()
    case = src.load_candidates()["scaled_refine12_sobol17_f50hz"]
    start_cfg = src.choose_refine_long_start()
    roots = read_roots()
    prereg = {"study": "V17C.21.2 full RF holdout validation", "frozen_before_integration": True,
              "source_commit": "94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8",
              "selection_rule": "M3-predicted nominal, inside-CR, near sign boundary, and post-boundary CO; no full RF outcomes viewed during selection",
              "initial_state_source": "M3 circular mean with zero radial/axial velocities; full RF MFS microvelocity develops dynamically",
              "initial_rf_phases_rad": start_cfg["phase"],
              "cases": {k: {"parameter": v["parameter"], "factor": v["factor"],
                             "M3_classification": v["row"]["classification"], "root": v["row"]["root"],
                             "initial_state": build_start(v["row"]).tolist()}
                        for k, v in roots.items()}}
    write_json(BASE / "results" / "PRE_REGISTERED_RF_POINTS.json", prereg)
    runs = []
    for label, periods, steps in (("M3_CR_nominal", 15000, 64), ("M3_CO_A_low", 15000, 64),
                                  ("M3_A_sign_before", 15000, 64), ("M3_A_sign_after", 15000, 64),
                                  ("M3_CO_beta_high", 15000, 64), ("M3_CR_nominal", 15000, 96)):
        runs.append(_run_case(label, roots[label], case, start_cfg, field, periods, steps, 1000))
        write_json(BASE / "results" / "full_rf_validation.json", {"status": "IN_PROGRESS", "rows": runs})
    write_json(BASE / "results" / "full_rf_validation.json", {"status": "COMPLETE", "rows": runs})


if __name__ == "__main__":
    main()
