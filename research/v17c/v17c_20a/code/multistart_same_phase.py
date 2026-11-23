from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
V12B = ROOT / "research" / "v17c" / "v17c_12b"
BASE = ROOT / "research" / "v17c" / "v17c_20a"
sys.path.insert(0, str(BASE / "code"))
import run_long_qr as qr  # noqa: E402

OUT = BASE / "results" / "multistart"
STARTS_FILE = V12B / "results" / "rescaled_legacy_candidate_validation" / "frozen_starts.npz"


def candidate_case(name):
    cid = {"candidate52": "scaled_candidate52_f45hz", "local15": "scaled_local15_f50hz",
           "refine12_sobol17": "scaled_refine12_sobol17_f50hz"}[name]
    _, case, p, qratio, seed_state, phase, source, traj = qr.load_late_seed(cid)
    return case, p, qratio, seed_state, phase, source, traj


def select_starts(n_each=2, seed=20261123):
    archive = np.load(STARTS_FILE)
    rng = np.random.default_rng(seed)
    rows = []
    for ensemble in ("rest", "weak", "wide", "edge"):
        ids = np.flatnonzero(archive["ensemble"] == ensemble)
        chosen = np.sort(rng.choice(ids, size=n_each, replace=False))
        for idx in chosen:
            rows.append({"start_index": int(idx), "ensemble": ensemble,
                         "initial": archive["initial"][idx].tolist(),
                         "original_phase_rad": archive["phases"][idx].tolist()})
    return rows


def run_one(task):
    name = task["candidate"]
    start = task["start"]
    case, p, qratio, late_seed, phase, source, traj = candidate_case(name)
    p = np.asarray(p, dtype=float).copy()
    for i, scale in enumerate(task.get("multipliers", [1.0, 1.0, 1.0, 1.0])):
        p[i] *= float(scale)
    field = qr.a.load_field()
    initial = np.asarray(start["initial"], dtype=np.float64)
    phase = phase.copy()  # same two RF phases for every mechanical start in one candidate panel
    trans = 100
    first = qr.a.PAIR._integrate_trace(
        initial, phase, p, field["family"], field["k"], field["coeff"], qratio,
        case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"],
        case["re2_per_speed"], case["gravity_z"], trans, task["steps"], 0,
        task["steps"] * 100)
    y, ph = first[0], phase.copy()
    transition_status, transition_done = int(first[-1]), int(first[-2])
    if transition_status != 0 or transition_done != trans:
        outcome = "COLLISION" if transition_status == 1 else ("ESCAPE" if transition_status == 2 else "UNRESOLVED")
        return {"run_id": task["run_id"], "candidate": name, "start_index": start["start_index"],
                "ensemble": start["ensemble"], "initial_state": start["initial"],
                "original_phase_rad_ignored": start["original_phase_rad"], "fixed_phase_rad": phase.tolist(),
                "same_phase_for_all_starts_within_candidate": True,
                "parameters_p_corrected": np.asarray(p).tolist(), "qratio": float(qratio),
                "source_state_period": source, "transition_periods": trans,
                "transition_completed_periods": transition_done, "transition_status": transition_status,
                "periods": task["periods"], "skip_periods_for_late_metrics": task["skip"],
                "steps_per_period": task["steps"], "completed_periods": 0,
                "status": transition_status, "outcome": outcome, "late_turns": [0.0, 0.0],
                "late_window_turns": [[0.0, 0.0]]*3, "late_frequency_Hz": [None, None],
                "late_mean_radius_R": None, "late_radius_std_R": None,
                "relative_radius_drift_first_vs_last_fifth": None, "end_state": np.asarray(y).tolist()}
    ph = (ph + np.array([1.0, float(p[2])]) * (2.0 * np.pi * trans)) % (2.0 * np.pi)
    raw = qr.a.PAIR._integrate_trace(
        y, ph, p, field["family"], field["k"], field["coeff"], qratio,
        case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"],
        case["re2_per_speed"], case["gravity_z"], task["periods"], task["steps"],
        task["skip"], task["steps"] * 100)
    end, states, indices = raw[0], raw[1], raw[2]
    turns = np.asarray(raw[4], dtype=float)
    windows = np.asarray(raw[5], dtype=float)
    status, done = int(raw[-1]), int(raw[-2])
    if status == 1:
        outcome = "COLLISION"
    elif status == 2:
        outcome = "ESCAPE"
    elif status != 0 or done != task["periods"]:
        outcome = "UNRESOLVED"
    elif np.all(np.abs(turns) >= task["min_turns"]):
        if turns[0] * turns[1] < 0:
            outcome = "CR(+−)" if turns[0] > 0 else "CR(−+)"
        else:
            outcome = "CO(++)" if turns[0] > 0 else "CO(−−)"
    else:
        outcome = "BOUND_OR_UNRESOLVED_FEW_TURNS"
    period_index = np.asarray(indices, dtype=float) / task["steps"]
    late_mask = period_index >= task["skip"]
    late = np.asarray(states)[late_mask]
    late_p = period_index[late_mask]
    radii = np.column_stack((np.hypot(late[:, 0], late[:, 1]), np.hypot(late[:, 6], late[:, 7]))) if len(late) else np.empty((0, 2))
    freq = []
    for j in (0, 6):
        if len(late) >= 3:
            theta = np.unwrap(np.arctan2(late[:, j+1], late[:, j]))
            slope = np.polyfit(late_p, theta, 1)[0]
            freq.append(float(slope * case["f1_Hz"] / (2.0 * np.pi)))
        else:
            freq.append(None)
    nlate = len(radii)
    nwin = max(2, nlate // 5)
    drift = []
    if nlate:
        for j in range(2):
            first = float(np.mean(radii[:nwin, j])); last = float(np.mean(radii[-nwin:, j]))
            drift.append((last-first)/max(abs(first), 1e-12))
    return {
        "run_id": task["run_id"], "candidate": name, "start_index": start["start_index"],
        "ensemble": start["ensemble"], "initial_state": start["initial"],
        "original_phase_rad_ignored": start["original_phase_rad"], "fixed_phase_rad": ph.tolist(),
        "same_phase_for_all_starts_within_candidate": True,
        "parameters_p_corrected": np.asarray(p).tolist(), "qratio": float(qratio),
        "source_state_period": source, "transition_periods": trans,
        "periods": task["periods"], "skip_periods_for_late_metrics": task["skip"],
        "steps_per_period": task["steps"], "completed_periods": done, "status": status,
        "outcome": outcome, "late_turns": turns.tolist(), "late_window_turns": windows.tolist(),
        "late_frequency_Hz": freq,
        "late_mean_radius_R": radii.mean(axis=0).tolist() if nlate else None,
        "late_radius_std_R": radii.std(axis=0).tolist() if nlate else None,
        "relative_radius_drift_first_vs_last_fifth": drift if drift else None,
        "end_state": np.asarray(end).tolist(),
    }


def wilson(k, n, z=1.959963984540054):
    if n == 0: return [None, None]
    p = k / n; den = 1.0 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [float(center-half), float(center+half)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-each-ensemble", type=int, default=2)
    ap.add_argument("--periods", type=int, default=15000)
    ap.add_argument("--skip", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--candidates", nargs="+", choices=["candidate52", "local15", "refine12_sobol17"],
                    default=["candidate52", "local15", "refine12_sobol17"])
    ap.add_argument("--A-scale", type=float, default=1.0)
    ap.add_argument("--B-scale", type=float, default=1.0)
    ap.add_argument("--nu-scale", type=float, default=1.0)
    ap.add_argument("--beta-scale", type=float, default=1.0)
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    starts = select_starts(args.n_each_ensemble)
    multipliers = [args.A_scale, args.B_scale, args.nu_scale, args.beta_scale]
    tasks = []
    for candidate in args.candidates:
        for start in starts:
            tasks.append({"candidate": candidate, "start": start, "multipliers": multipliers,
                          "tag": args.tag, "periods": args.periods,
                          "skip": args.skip, "steps": args.steps, "min_turns": 5.0,
                          "run_id": f"{candidate}|{args.tag}|{start['ensemble']}|{start['start_index']}"})
    checkpoint = OUT / f"same_phase_multistart_{args.tag}.jsonl"
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); done[row["run_id"]] = row
    todo = [t for t in tasks if t["run_id"] not in done]
    t0 = time.perf_counter()
    with checkpoint.open("a", encoding="utf-8", buffering=1) as f:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, task) for task in todo]
            for future in as_completed(futures):
                row = future.result(); done[row["run_id"]] = row
                f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                f.flush()
                print(f"{row['run_id']}: {row['outcome'].replace('−', '-') } turns={row['late_turns']}", flush=True)
    grouped = []
    for candidate in ("candidate52", "local15", "refine12_sobol17"):
        rows = [r for r in done.values() if r["candidate"] == candidate]
        cr = sum(r["outcome"].startswith("CR") for r in rows)
        grouped.append({"candidate": candidate, "n": len(rows), "outcomes": {k: sum(r["outcome"] == k for r in rows) for k in sorted({x["outcome"] for x in rows})},
                        "counterrotation_count": cr, "counterrotation_fraction": cr/len(rows) if rows else None,
                        "wilson95_descriptive": wilson(cr, len(rows)),
                        "note": "Exploratory fixed-phase start panel, not a universal basin probability."})
    summary = {"source_branch_commit": "d1bf252649e174166becf464269318b5f97acd5d",
               "tag": args.tag, "field_multipliers_A_B_nu_beta": multipliers,
               "selected_start_seed": 20261123, "starts_per_ensemble": args.n_each_ensemble,
               "ensembles": ["rest", "weak", "wide", "edge"], "runs_total": len(done),
               "periods": args.periods, "skip_periods": args.skip, "steps_per_period": args.steps,
               "phase_policy": "Each candidate uses its archived endpoint RF phases, shared identically by every mechanical start for that candidate; original per-start phases are ignored and recorded.",
               "candidate_results": grouped,
               "results": [done[k] for k in sorted(done)],
               "elapsed_wall_seconds_this_invocation": time.perf_counter()-t0,
               "interpretation": "Finite sample of initial mechanical conditions. Outcome fractions describe only this selected panel and horizon; they are not probabilities over an unspecified loading distribution."}
    summary_path = OUT / f"same_phase_multistart_{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.tag == "baseline":
        (OUT / "same_phase_multistart_summary.json").write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
