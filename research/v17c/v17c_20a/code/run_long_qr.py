from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[4]
V20B = ROOT / "research" / "v17c" / "v17c_20b"
V20A = ROOT / "research" / "v17c" / "v17c_20a"
sys.path.insert(0, str(V20B / "code"))
import attractor_search as a  # noqa: E402
import variational_qr as vq  # noqa: E402

OUT = V20A / "results" / "lyapunov"


@njit(cache=True)
def drag_trace(y, p, re1, re2):
    total = 0.0
    for part in range(2):
        j = 6 * part
        vx, vy, vz = y[j + 3], y[j + 4], y[j + 5]
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        chi = re1 if part == 0 else re2
        lam = p[6] if part == 0 else p[7]
        fac = 1.0 + 0.15 * (chi * speed) ** 0.687
        derivative = 0.0
        if speed > 1e-30:
            derivative = 0.15 * 0.687 * (chi ** 0.687) * (speed ** (-1.313))
        total -= lam * (3.0 * fac + derivative * speed * speed)
    return total


@njit(cache=True)
def advance_with_divergence(y, M, t0, dt, nsteps, p, phase, family, modes,
                            coeff, qratio, re1, re2, gravity_z, a1, a2):
    """RK4 state/tangent step and matching RK4 integral of tr(J)."""
    y = y.copy()
    M = M.copy()
    t = t0
    div_integral = 0.0
    for _ in range(nsteps):
        k1y, k1m = vq.augmented_rhs(t, y, M, p, phase, family, modes,
                                    coeff, qratio, re1, re2, gravity_z, a1, a2)
        d1 = drag_trace(y, p, re1, re2)

        y2 = y + dt * k1y / 2.0
        m2 = M + dt * k1m / 2.0
        k2y, k2m = vq.augmented_rhs(t + dt / 2.0, y2, m2, p, phase,
                                    family, modes, coeff, qratio, re1, re2,
                                    gravity_z, a1, a2)
        d2 = drag_trace(y2, p, re1, re2)

        y3 = y + dt * k2y / 2.0
        m3 = M + dt * k2m / 2.0
        k3y, k3m = vq.augmented_rhs(t + dt / 2.0, y3, m3, p, phase,
                                    family, modes, coeff, qratio, re1, re2,
                                    gravity_z, a1, a2)
        d3 = drag_trace(y3, p, re1, re2)

        y4 = y + dt * k3y
        m4 = M + dt * k3m
        k4y, k4m = vq.augmented_rhs(t + dt, y4, m4, p, phase, family,
                                    modes, coeff, qratio, re1, re2,
                                    gravity_z, a1, a2)
        d4 = drag_trace(y4, p, re1, re2)

        y = y + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        M = M + dt * (k1m + 2.0 * k2m + 2.0 * k3m + k4m) / 6.0
        div_integral += dt * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
        t += dt
    return y, M, t, div_integral


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_late_seed(candidate_id: str, source_period: int | None = None):
    config = a.read_json(V20B / "results" / "candidate_parameters.json")
    case = next(c for c in config["candidates"] if c["id"] == candidate_id)
    archive_names = {
        "scaled_candidate52_f45hz": "candidate52_50000_common_corrected_trajectory.npz",
        "scaled_local15_f50hz": "local15_50000_common_corrected_trajectory.npz",
    }
    trajectory = V20B / "results" / archive_names.get(
        candidate_id, f"{candidate_id}_50000_common_corrected_trajectory.npz")
    # Refine12 uses its historical start (index 54), while the 50k archives use
    # the matched common start.  Resolve by the exact archived filenames.
    if candidate_id.startswith("scaled_refine12"):
        trajectory = V20B / "results" / "refine12_240k_corrected_drag_trajectory.npz"
        total = 240000
    else:
        total = 50000
    if not trajectory.exists():
        raise FileNotFoundError(trajectory)
    z = np.load(trajectory)
    periods = int(z["periods"])
    if source_period is None:
        source_period = periods
    if source_period < 0 or source_period > periods:
        raise ValueError(f"source_period must be between 0 and {periods}")
    step = int(z["steps"])
    indices = np.asarray(z["step_index"], dtype=np.int64)
    target = source_period * step
    matches = np.flatnonzero(indices == target)
    if not len(matches):
        raise ValueError(f"archived trajectory has no checkpoint at {source_period} periods")
    state = np.asarray(z["states"][matches[0]], dtype=np.float64)
    phase0 = np.asarray(z["phase"], dtype=np.float64)
    p, qratio = a.arrays(case, a.load_field())
    phase = (phase0 + np.array([1.0, float(p[2])]) * (2.0 * math.pi * source_period)) % (2.0 * math.pi)
    return config, case, p, qratio, state, phase, source_period, trajectory


def transition(case, p, qratio, state, phase, periods, steps):
    if periods == 0:
        return state.copy(), phase.copy()
    field = a.load_field()
    a.PAIR._energy_gz = float(case["gravity_z"])
    raw = a.PAIR._integrate_trace(
        state, phase, p, field["family"], field["k"], field["coeff"], qratio,
        case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"],
        case["re2_per_speed"], case["gravity_z"], periods, steps, 0,
        steps * 100)
    end, completed, status = raw[0], raw[-2], raw[-1]
    if int(status) != 0 or int(completed) != periods:
        raise RuntimeError(f"transition failed: status={status}, completed={completed}")
    phase = (phase + np.array([1.0, float(p[2])]) * (2.0 * math.pi * periods)) % (2.0 * math.pi)
    return np.asarray(end, dtype=np.float64), phase


def run(args):
    candidate_ids = {
        "candidate52": "scaled_candidate52_f45hz",
        "local15": "scaled_local15_f50hz",
        "refine12_sobol17": "scaled_refine12_sobol17_f50hz",
    }
    cid = candidate_ids[args.candidate]
    config, case, p, qratio, seed, phase, source_period, trajectory = load_late_seed(cid, args.source_period)
    field = a.load_field()
    start_sha = hashlib.sha256(seed.tobytes()).hexdigest()
    run_id = f"{args.candidate}_{source_period}p_seed__{args.transition}p_transient__{args.periods}p_qr__s{args.steps}_q{args.qr_periods}"
    workdir = OUT / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint = workdir / "checkpoint.json"
    out_path = workdir / "result.json"
    dt = 2.0 * math.pi / args.steps
    y = seed.copy()
    phase_forcing = phase.copy()
    elapsed_steps = 0
    logs = np.zeros(12)
    div_integral = 0.0
    M = np.eye(12)
    rows = []
    if checkpoint.exists() and not args.restart:
        ck = json.loads(checkpoint.read_text(encoding="utf-8"))
        expected = {"run_id": run_id, "seed_sha256": start_sha}
        if any(ck.get(k) != v for k, v in expected.items()):
            raise RuntimeError("checkpoint does not match this run configuration")
        y = np.asarray(ck["state"], dtype=np.float64)
        M = np.asarray(ck["Q"], dtype=np.float64)
        phase_forcing = np.asarray(ck["phase_at_qr_start_rad"], dtype=np.float64)
        logs = np.asarray(ck["log_sums"], dtype=np.float64)
        div_integral = float(ck["divergence_integral"])
        elapsed_steps = int(ck["elapsed_steps"])
        rows = ck["rows"]

    t0 = time.perf_counter()
    if elapsed_steps == 0:
        y, phase_forcing = transition(case, p, qratio, y, phase_forcing,
                                      args.transition, args.steps)
    elapsed_periods = elapsed_steps / args.steps
    qr_steps = args.qr_periods * args.steps
    report_steps = 100 * args.steps
    checkpoint_steps = 500 * args.steps
    end_steps = args.periods * args.steps
    last_report = elapsed_steps
    while elapsed_steps < end_steps:
        n = min(qr_steps, end_steps - elapsed_steps)
        y, M, tau, ddiv = advance_with_divergence(
            y, M, elapsed_steps * dt, dt, n, p, phase_forcing,
            field["family"], field["k"], field["coeff"], qratio,
            case["re1_per_speed"], case["re2_per_speed"], case["gravity_z"],
            case["a1_over_R"], case["a2_over_R"])
        Q, R = np.linalg.qr(M)
        logs += np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
        M = Q
        div_integral += float(ddiv)
        elapsed_steps += n
        elapsed_periods = elapsed_steps / args.steps
        if elapsed_steps % report_steps == 0 or elapsed_steps == end_steps:
            ex = logs / (elapsed_steps * dt)
            rows.append({
                "qr_periods_after_transient": float(elapsed_periods),
                "spectrum_per_tau_descending": np.sort(ex)[::-1].tolist(),
                "spectrum_per_rf_period_descending": (np.sort(ex)[::-1] * 2.0 * math.pi).tolist(),
                "sum_exponents_per_tau": float(np.sum(ex)),
                "integrated_divergence_per_tau": float(div_integral / (elapsed_steps * dt)),
                "volume_balance_abs_error_per_tau": float(abs(np.sum(ex) - div_integral / (elapsed_steps * dt))),
                "state": y.tolist(),
                "forcing_phase": ((phase_forcing + np.array([1.0, float(p[2])]) * (elapsed_steps * dt)) % (2.0 * math.pi)).tolist(),
            })
            last_report = elapsed_steps
        if elapsed_steps % checkpoint_steps == 0 or elapsed_steps == end_steps:
            atomic_json(checkpoint, {
                "run_id": run_id,
                "candidate": args.candidate,
                "seed_sha256": start_sha,
                "trajectory_source": str(trajectory.relative_to(ROOT)).replace("\\", "/"),
                "source_period": source_period,
                "transition_periods_discarded": args.transition,
                "phase_at_seed_rad": phase.tolist(),
                "phase_at_qr_start_rad": phase_forcing.tolist(),
                "periods_completed": elapsed_periods,
                "elapsed_steps": elapsed_steps,
                "state": y.tolist(),
                "Q": M.tolist(),
                "log_sums": logs.tolist(),
                "divergence_integral": div_integral,
                "rows": rows,
            })
            print(f"{run_id}: {elapsed_periods:.0f}/{args.periods} QR periods; wall={time.perf_counter()-t0:.1f}s", flush=True)

    spectrum = np.sort(logs / (elapsed_steps * dt))[::-1]
    result = {
        "run_id": run_id,
        "candidate": args.candidate,
        "source_period": source_period,
        "trajectory_source": str(trajectory.relative_to(ROOT)).replace("\\", "/"),
        "trajectory_sha256": source_digest(trajectory),
        "seed_sha256": start_sha,
        "base_v20b_sha": "d1bf252649e174166becf464269318b5f97acd5d",
        "parameters": {"p_corrected": np.asarray(p).tolist(), "qratio": float(qratio), "nu": float(p[2]),
                       "re1_per_speed": case["re1_per_speed"], "re2_per_speed": case["re2_per_speed"],
                       "gravity_z": case["gravity_z"]},
        "integration": {"steps_per_rf_period": args.steps, "dt_tau": dt,
                        "qr_interval_periods": args.qr_periods, "transition_periods_discarded": args.transition,
                        "qr_horizon_periods": args.periods, "completed_periods": elapsed_periods,
                        "time_unit": "dimensionless tau=Omega1*t", "elapsed_wall_seconds": time.perf_counter()-t0},
        "forcing_phase_convention": "RF phases are initialized from the archived absolute period and advanced continuously; the carrier time origin is continuous through this QR segment; both forcing phases stay fixed in the conditional tangent perturbations.",
        "lyapunov_spectrum_per_tau_descending": spectrum.tolist(),
        "lyapunov_spectrum_per_rf_period_descending": (spectrum * 2.0 * math.pi).tolist(),
        "sum_exponents_per_tau": float(np.sum(spectrum)),
        "integrated_divergence_per_tau": float(div_integral / (elapsed_steps * dt)),
        "volume_balance_abs_error_per_tau": float(abs(np.sum(spectrum) - div_integral / (elapsed_steps * dt))),
        "rows": rows,
        "scope": "Conditional mechanical QR spectrum with the two independent external RF phases prescribed identically; this is not the full autonomous 14D spectrum.",
    }
    atomic_json(out_path, result)
    print(json.dumps({"run_id": run_id, "spectrum": spectrum.tolist(),
                      "volume_error": result["volume_balance_abs_error_per_tau"],
                      "wall_s": result["integration"]["elapsed_wall_seconds"]}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=["candidate52", "local15", "refine12_sobol17"])
    ap.add_argument("--periods", type=int, default=2000)
    ap.add_argument("--source-period", type=int)
    ap.add_argument("--transition", type=int, default=100)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--qr-periods", type=int, default=20)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    if min(args.periods, args.steps, args.qr_periods) <= 0 or args.transition < 0:
        ap.error("periods, steps and qr-periods must be positive; transition nonnegative")
    if args.steps % 1 or args.periods < args.qr_periods:
        ap.error("invalid step/horizon combination")
    run(args)


if __name__ == "__main__":
    main()
