"""Short independent DOP853 perturbation check for the V17C.22 baseline.

This deliberately reuses the V20B field/candidate registries and implements
the same 12D equations as V20B's independent DOP853 audit.  It does not alter
any source data.  A short horizon diagnoses local return; it is not used to
classify an attractor or long-term counterrotation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[4]
V20B = ROOT / "research" / "v17c" / "v17c_20b"
OUT = ROOT / "research" / "v17c" / "v17c_22" / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rhs(t: float, y: np.ndarray, phase: np.ndarray, p: np.ndarray,
        field: dict, qratio: float, re1: float, re2: float,
        gravity_z: float) -> np.ndarray:
    """V20B full 8-mode 12D RF RHS, vectorized over state columns."""
    scalar = y.ndim == 1
    Y = y.reshape(12, -1)
    dY = np.zeros_like(Y)
    A, B, nu, beta, mass_ratio, C, lam1, lam2 = map(float, p)
    carrier1 = A * np.exp(-1j * (t + phase[0]))
    carrier2 = B * np.exp(-1j * (nu * t + phase[1]))
    delta = Y[:3] - Y[6:9]
    d2 = np.sum(delta * delta, axis=0)
    inv_d3 = np.zeros_like(d2)
    valid = (C != 0.0) & (d2 > 1e-28)
    inv_d3[valid] = 1.0 / (d2[valid] * np.sqrt(d2[valid]))
    for part, j in ((0, 0), (1, 6)):
        x, yy, z = Y[j], Y[j + 1], Y[j + 2]
        vx, vy, vz = Y[j + 3], Y[j + 4], Y[j + 5]
        pos = x + 1j * yy
        ex = np.zeros(Y.shape[1]); ey = np.zeros(Y.shape[1])
        for family, mode, coeff in zip(field["family"], field["k"], field["coeff"]):
            degree = abs(int(mode))
            base = pos if mode > 0 else np.conjugate(pos)
            carrier = carrier1 if family == 0 else carrier2
            grad = carrier * coeff * degree * base ** (degree - 1)
            ex -= grad.real
            ey += grad.imag if mode > 0 else -grad.imag
        q_over_m = 1.0 if part == 0 else mass_ratio * qratio
        coulomb_mass = 1.0 if part == 0 else mass_ratio
        sign = 1.0 if part == 0 else -1.0
        lam = lam1 if part == 0 else lam2
        chi = re1 if part == 0 else re2
        speed = np.sqrt(vx * vx + vy * vy + vz * vz)
        re = np.maximum(0.0, chi * speed)
        drag = lam * (1.0 + 0.15 * re ** 0.687)
        dY[j:j + 3] = Y[j + 3:j + 6]
        dY[j + 3] = q_over_m * (ex + beta * x) + sign * coulomb_mass * C * inv_d3 * delta[0] - drag * vx
        dY[j + 4] = q_over_m * (ey + beta * yy) + sign * coulomb_mass * C * inv_d3 * delta[1] - drag * vy
        dY[j + 5] = q_over_m * (-2.0 * beta * z) + sign * coulomb_mass * C * inv_d3 * delta[2] - drag * vz + gravity_z
    return dY[:, 0] if scalar else dY


def phase_at(phase0: np.ndarray, nu: float, period_index: int) -> np.ndarray:
    """Exact base-torus phase at archived RF1 period, without time reset."""
    return np.mod(np.asarray(phase0) + 2.0 * math.pi * np.array([1.0, nu]) * period_index,
                  2.0 * math.pi)


def make_perturbations(y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    eps = 1.0e-4
    states = [np.asarray(y, float).copy()]
    names = ["reference"]
    radial = states[0].copy()
    radial[:2] += eps * radial[:2] / np.linalg.norm(radial[:2])
    states.append(radial); names.append("p1_radial_position_1e-4R")
    azimuth = states[0].copy()
    x, yy = azimuth[0], azimuth[1]
    azimuth[0] = x * math.cos(eps) - yy * math.sin(eps)
    azimuth[1] = x * math.sin(eps) + yy * math.cos(eps)
    states.append(azimuth); names.append("p1_azimuth_1e-4rad")
    axial = states[0].copy(); axial[2] += eps
    states.append(axial); names.append("p1_axial_position_1e-4R")
    tangential = states[0].copy()
    speed_scale = 0.0038162658249025747
    norm = np.linalg.norm(tangential[:2])
    tangential[3] += eps * speed_scale * (-tangential[1] / norm)
    tangential[4] += eps * speed_scale * ( tangential[0] / norm)
    states.append(tangential); names.append("p1_tangential_velocity_1e-4scale")
    return np.column_stack(states), names


def observables(periods: np.ndarray, states: np.ndarray, f1: float,
                phase_start: np.ndarray, phase_end: np.ndarray,
                names: list[str], radius_threshold: float) -> dict:
    # states shape: time x 12 x case
    rows = []
    for k, name in enumerate(names):
        y = states[:, :, k]
        theta = np.unwrap(np.arctan2(y[:, [1, 7]], y[:, [0, 6]]), axis=0)
        late = periods >= periods[-1] - 100.0
        turns = (theta[-1] - theta[np.flatnonzero(late)[0]]) / (2.0 * math.pi)
        freq = turns * f1 / (periods[-1] - periods[np.flatnonzero(late)[0]])
        rr = np.sqrt(y[:, [0, 6]] ** 2 + y[:, [1, 7]] ** 2)
        pair = np.linalg.norm(y[:, :3] - y[:, 6:9], axis=1)
        ref = states[:, :, 0]
        delta = y - ref
        position_idx = [0, 1, 2, 6, 7, 8]
        velocity_idx = [3, 4, 5, 9, 10, 11]
        vscale = np.array([0.0038162658249025747] * 3 +
                          [0.004575900271655408] * 3)
        metric = np.sqrt(np.sum(delta[:, position_idx] ** 2, axis=1) +
                         np.sum((delta[:, velocity_idx] / vscale) ** 2, axis=1))
        rows.append({
            "case": name,
            "late_window_periods": 100,
            "late_turns": turns.tolist(),
            "late_frequency_Hz": freq.tolist(),
            "late_mean_radius_R": rr[late].mean(axis=0).tolist(),
            "late_radius_std_R": rr[late].std(axis=0).tolist(),
            "late_mean_z_R": y[late][:, [2, 8]].mean(axis=0).tolist(),
            "minimum_pair_center_distance_R": float(pair.min()),
            "pair_surface_clearance_R": float(pair.min() - radius_threshold),
            "maximum_phase_locked_state_distance_metric": float(metric.max()),
            "final_phase_locked_state_distance_metric": float(metric[-1]),
            "classification": "SHORT_HORIZON_ONLY_NOT_ATTRACTOR_EVIDENCE",
        })
    return {"phase_start_rad": phase_start.tolist(), "phase_end_rad": phase_end.tolist(), "rows": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((V20B / "results" / "candidate_parameters.json").read_text(encoding="utf-8"))
    case = next(c for c in cfg["candidates"] if c["legacy_candidate"] == "refine12_sobol17")
    field_doc = json.loads((ROOT / "research/v17c/v17c_12b/model/field_model.json").read_text(encoding="utf-8"))
    field = {"family": np.asarray(field_doc["family"], int), "k": np.asarray(field_doc["k"], int),
             "coeff": np.asarray(field_doc["coeff"], float)[:, 0] + 1j * np.asarray(field_doc["coeff"], float)[:, 1]}
    p = np.asarray(case["p_corrected"], float); qratio = float(case["qratio"])
    tr_path = V20B / "results" / "refine12_240k_corrected_drag_trajectory.npz"
    tr = np.load(tr_path)
    phase0 = np.asarray(tr["phase"], float)
    seed_periods = (238000, 240000)
    out_rows = []
    all_times = []; all_states = []
    started = time.perf_counter()
    horizon = 300
    t_eval = np.arange(horizon + 1, dtype=float) * 2.0 * math.pi
    for pidx in seed_periods:
        hit = np.flatnonzero(np.asarray(tr["step_index"]) == pidx * int(tr["steps"]))
        if not len(hit):
            raise ValueError(f"archive lacks exact period {pidx}")
        base = np.asarray(tr["states"][hit[0]], float)
        y0, names = make_perturbations(base)
        phase_start = phase_at(phase0, float(p[2]), pidx)
        args = (phase_start, p, field, qratio, float(case["re1_per_speed"]),
                float(case["re2_per_speed"]), float(case["gravity_z"]))
        ncase = y0.shape[1]
        sol = solve_ivp(lambda t, y: rhs(t, y.reshape(12, ncase), *args).ravel(),
                        (0.0, horizon * 2.0 * math.pi), y0.ravel(), method="DOP853",
                        t_eval=t_eval, rtol=1e-9, atol=1e-11,
                        max_step=2.0 * math.pi / 96.0)
        if not sol.success:
            raise RuntimeError(sol.message)
        state_cube = sol.y.reshape(12, len(names), -1).transpose(2, 0, 1)
        phase_end = phase_at(phase_start, float(p[2]), horizon)
        run = observables(t_eval / (2.0 * math.pi), state_cube, float(case["f1_Hz"]),
                          phase_start, phase_end, names,
                          float(case["a1_over_R"] + case["a2_over_R"]))
        run.update({"source_period": pidx, "horizon_rf1_periods": horizon,
                    "steps_per_rf1_period_maximum": 96, "rtol": 1e-9, "atol": 1e-11,
                    "nfev": int(sol.nfev), "success": bool(sol.success),
                    "phase_rule": "phase0 + 2*pi*(1,nu)*absolute_RF1_period; local solve time starts at zero only after rebasing both RF phases to the archived absolute time"})
        out_rows.append(run); all_times.append(t_eval / (2.0 * math.pi)); all_states.append(state_cube)
        print(f"period={pidx}: nfev={sol.nfev}, success={sol.success}, final perturb distances=" +
              ",".join(f"{x['final_phase_locked_state_distance_metric']:.3e}" for x in run["rows"][1:]), flush=True)
    np.savez_compressed(OUT / "short_dop853_perturbation_trajectories.npz",
                        periods=np.asarray(all_times), states=np.asarray(all_states),
                        source_periods=np.asarray(seed_periods), p=p, qratio=qratio,
                        base_phase=phase0, steps_per_rf1_period_max=96)
    result = {
        "study": "V17C.22 short full-RF perturbation recheck",
        "source_commit": "894c6896ce5927371708037ffa4a4021189d2b54",
        "source_candidate_registry": "research/v17c/v17c_20b/results/candidate_parameters.json",
        "trajectory_source": "research/v17c/v17c_20b/results/refine12_240k_corrected_drag_trajectory.npz",
        "trajectory_sha256": sha256(tr_path),
        "model": "Full 12D V20B equations, complete eight-mode MFS field, two RF phases, Coulomb, ideal DC quadrupole, gravity, Cunningham-corrected Stokes and Schiller-Naumann drag; polarization off; infinite rods.",
        "parameters_unchanged": case["p_corrected"], "qratio": qratio,
        "protocol": "Two exact archived late RF1 states; reference plus radial, azimuthal, axial, and tangential-velocity perturbations at fixed RF phases; 300-period DOP853 local diagnostic.",
        "elapsed_wall_seconds": time.perf_counter() - started,
        "runs": out_rows,
        "interpretation_limit": "The horizon is too short for long-term frequency classification and no invariant set is fitted. These trajectories test short-time response and integrator execution only; existing V20A 5000-period perturbation data remain the long-horizon evidence.",
    }
    out = OUT / "short_dop853_perturbation_summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"saved {out}; elapsed={result['elapsed_wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
