from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[4]
V20B = ROOT / "research" / "v17c" / "v17c_20b"
OUT = ROOT / "research" / "v17c" / "v17c_20a" / "results" / "extended_qr"
sys.path.insert(0, str(V20B / "code"))
import variational_qr as vq  # noqa: E402
import run_long_qr as cond  # noqa: E402


@njit(cache=True)
def advance_extended_with_divergence(y, phase, M, dt, nsteps, p, family,
                                    modes, coeff, qratio, re1, re2,
                                    gravity_z, a1, a2):
    y = y.copy()
    phase = phase.copy()
    M = M.copy()
    div_int = 0.0
    nu = p[2]
    for _ in range(nsteps):
        k1y, k1p, k1m = vq.extended_rhs(y, phase, M, p, nu, family, modes,
                                       coeff, qratio, re1, re2, gravity_z)
        d1 = cond.drag_trace(y, p, re1, re2)
        y2 = y + dt * k1y / 2.0
        p2 = phase + dt * k1p / 2.0
        m2 = M + dt * k1m / 2.0
        k2y, k2p, k2m = vq.extended_rhs(y2, p2, m2, p, nu, family, modes,
                                       coeff, qratio, re1, re2, gravity_z)
        d2 = cond.drag_trace(y2, p, re1, re2)
        y3 = y + dt * k2y / 2.0
        p3 = phase + dt * k2p / 2.0
        m3 = M + dt * k2m / 2.0
        k3y, k3p, k3m = vq.extended_rhs(y3, p3, m3, p, nu, family, modes,
                                       coeff, qratio, re1, re2, gravity_z)
        d3 = cond.drag_trace(y3, p, re1, re2)
        y4 = y + dt * k3y
        p4 = phase + dt * k3p
        m4 = M + dt * k3m
        k4y, k4p, k4m = vq.extended_rhs(y4, p4, m4, p, nu, family, modes,
                                       coeff, qratio, re1, re2, gravity_z)
        d4 = cond.drag_trace(y4, p, re1, re2)
        y += dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        phase += dt * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0
        M += dt * (k1m + 2.0 * k2m + 2.0 * k3m + k4m) / 6.0
        div_int += dt * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
    return y, phase, M, div_int


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def run(args):
    cid = {"candidate52": "scaled_candidate52_f45hz", "local15": "scaled_local15_f50hz",
           "refine12_sobol17": "scaled_refine12_sobol17_f50hz"}[args.candidate]
    config, case, p, qratio, y, phase, source, trajectory = cond.load_late_seed(cid, args.source_period)
    field = cond.a.load_field()
    run_id = f"{args.candidate}_{source}p_seed__{args.transition}p_transient__{args.periods}p_extqr__s{args.steps}_q{args.qr_periods}"
    folder = OUT / run_id
    checkpoint = folder / "checkpoint.json"
    dt = 2.0 * math.pi / args.steps
    M = np.eye(14)
    logs = np.zeros(14)
    div_int = 0.0
    done_steps = 0
    rows = []
    if checkpoint.exists() and not args.restart:
        ck = json.loads(checkpoint.read_text(encoding="utf-8"))
        y = np.asarray(ck["state"], float)
        phase = np.asarray(ck["phase"], float)
        M = np.asarray(ck["Q"], float)
        logs = np.asarray(ck["log_sums"], float)
        div_int = float(ck["divergence_integral"])
        done_steps = int(ck["elapsed_steps"])
        rows = ck["rows"]
    t0 = time.perf_counter()
    if done_steps == 0:
        y, phase = cond.transition(case, p, qratio, y, phase, args.transition, args.steps)
    target_steps = args.periods * args.steps
    qr_steps = args.qr_periods * args.steps
    chunk_steps = 500 * args.steps
    report_steps = 100 * args.steps
    while done_steps < target_steps:
        n = min(qr_steps, target_steps - done_steps)
        y, phase, M, ddiv = advance_extended_with_divergence(
            y, phase, M, dt, n, p, field["family"], field["k"], field["coeff"],
            qratio, case["re1_per_speed"], case["re2_per_speed"],
            case["gravity_z"], case["a1_over_R"], case["a2_over_R"])
        Q, R = np.linalg.qr(M)
        logs += np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
        M = Q
        div_int += float(ddiv)
        done_steps += n
        if done_steps % report_steps == 0 or done_steps == target_steps:
            ex = np.sort(logs / (done_steps * dt))[::-1]
            rows.append({"periods_after_transient": done_steps / args.steps,
                         "spectrum_per_tau_descending": ex.tolist(),
                         "external_phase_rad": (phase % (2.0 * math.pi)).tolist(),
                         "sum_exponents_per_tau": float(ex.sum()),
                         "integrated_divergence_per_tau": float(div_int / (done_steps * dt)),
                         "volume_balance_abs_error_per_tau": float(abs(ex.sum() - div_int / (done_steps * dt)))})
        if done_steps % chunk_steps == 0 or done_steps == target_steps:
            atomic_json(checkpoint, {"run_id": run_id, "state": y.tolist(), "phase": phase.tolist(),
                                     "Q": M.tolist(), "log_sums": logs.tolist(),
                                     "divergence_integral": div_int, "elapsed_steps": done_steps,
                                     "rows": rows, "source_period": source,
                                     "transition_periods_discarded": args.transition})
            print(f"{run_id}: {done_steps/args.steps:.0f}/{args.periods} periods; wall={time.perf_counter()-t0:.1f}s", flush=True)
    ex = np.sort(logs / (done_steps * dt))[::-1]
    result = {"run_id": run_id, "candidate": args.candidate, "parameters": np.asarray(p).tolist(),
              "source_period": source, "trajectory_source": str(trajectory.relative_to(ROOT)).replace("\\", "/"),
              "steps_per_rf_period": args.steps, "qr_interval_periods": args.qr_periods,
              "periods_after_transient": args.periods, "transition_periods_discarded": args.transition,
              "phase_rates": [1.0, float(p[2])], "phase_note": "Both independent forcing phases are explicit tangent coordinates; two exact neutral phase directions are expected by construction.",
              "spectrum_per_tau_descending": ex.tolist(), "sum_exponents_per_tau": float(ex.sum()),
              "integrated_divergence_per_tau": float(div_int / (done_steps * dt)),
              "volume_balance_abs_error_per_tau": float(abs(ex.sum() - div_int / (done_steps * dt))),
              "rows": rows,
              "scope": "Numerical Lyapunov spectrum of the 14D autonomous skew-product extension; finite-time evidence only."}
    atomic_json(folder / "result.json", result)
    print(json.dumps({"run_id": run_id, "spectrum": ex.tolist(),
                      "volume_error": result["volume_balance_abs_error_per_tau"],
                      "wall_s": time.perf_counter()-t0}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=["candidate52", "local15", "refine12_sobol17"], default="refine12_sobol17")
    ap.add_argument("--periods", type=int, default=20000)
    ap.add_argument("--source-period", type=int)
    ap.add_argument("--transition", type=int, default=100)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--qr-periods", type=int, default=20)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    if min(args.periods, args.steps, args.qr_periods) <= 0 or args.transition < 0:
        ap.error("invalid horizon, step, QR interval or transient")
    run(args)


if __name__ == "__main__":
    main()
