from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research" / "v17c" / "v17c_20a"
sys.path.insert(0, str(BASE / "code"))
import run_long_qr as qr  # noqa: E402

OUT = BASE / "results" / "rf_parameter_screen"


def unique_qr_candidates():
    d = json.loads((OUT / "screen_summary.json").read_text(encoding="utf-8"))
    rows = d["rows"]
    # Locations chosen from the 5x5 local panels: baseline plus interior CR
    # points with a full 8-neighbour CR neighborhood in at least one slice.
    targets = [
        (1.00, 1.00, 1.00, 1.00),
        (1.00, 1.15, 1.00, 1.00),
        (1.00, 1.00, 1.00, 0.85),
        (1.00, 1.15, 1.00, 0.85),
        (1.15, 1.00, 1.00, 1.00),
        (1.15, 1.15, 1.00, 1.00),
    ]
    selected = []
    for vals in targets:
        match = next((r for r in rows if all(abs(r["multipliers"][k]-v) < 1e-8 for k, v in zip(("A","B","nu","beta"), vals))), None)
        if match is None:
            raise ValueError(f"field point absent from screen: {vals}")
        selected.append({"run_id": match["run_id"], "multipliers": match["multipliers"],
                         "screen_outcome": match["outcome"], "screen_turns": match["turns"],
                         "screen_min_pair_distance_R": match.get("min_pair_distance_R"),
                         "slices": match.get("slices", [])})
    return selected


def worker(task):
    _, case, p0, qratio, seed, phase0, source, trajectory = qr.load_late_seed("scaled_refine12_sobol17_f50hz", 240000)
    p = np.asarray(p0, dtype=np.float64).copy()
    mult = task["multipliers"]
    for i, k in enumerate(("A", "B", "nu", "beta")):
        p[i] *= mult[k]
    field = qr.a.load_field()
    transition = 100
    tr = qr.a.PAIR._integrate_trace(seed, phase0, p, field["family"], field["k"], field["coeff"], qratio,
        case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"], case["re2_per_speed"], case["gravity_z"],
        transition, 32, 0, 32*100)
    state, status, done = tr[0], int(tr[-1]), int(tr[-2])
    if status != 0 or done != transition:
        return {"run_id": task["run_id"], "outcome": "TRANSIENT_ESCAPE_OR_COLLISION", "transition_status": status,
                "multipliers": mult, "screen_outcome": task["screen_outcome"]}
    phase = (phase0 + np.array([1.0, p[2]]) * (2.0*np.pi*transition)) % (2.0*np.pi)
    state = np.asarray(state, dtype=np.float64)
    M = np.eye(12); logs = np.zeros(12); div_int = 0.0
    steps, horizon, qr_periods = 32, task["periods"], 20
    dt = 2.0*np.pi/steps
    t0 = time.perf_counter()
    for completed in range(0, horizon, qr_periods):
        block = min(qr_periods, horizon-completed)
        state, M, _, ddiv = qr.advance_with_divergence(state, M, completed*steps*dt, dt,
            block*steps, p, phase, field["family"], field["k"], field["coeff"], qratio,
            case["re1_per_speed"], case["re2_per_speed"], case["gravity_z"], case["a1_over_R"], case["a2_over_R"])
        Q, R = np.linalg.qr(M)
        logs += np.log(np.maximum(np.abs(np.diag(R)), 1e-300)); M = Q; div_int += float(ddiv)
    spectrum = np.sort(logs/(horizon*steps*dt))[::-1]
    # Independently replay the same field point with event-resolving pair integrator.
    replay = qr.a.PAIR._integrate_trace(np.asarray(tr[0]), phase, p, field["family"], field["k"], field["coeff"], qratio,
        case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"], case["re2_per_speed"], case["gravity_z"],
        horizon, steps, 0, steps*100)
    replay_end, states, indices = replay[0], replay[1], replay[2]
    actual_status, actual_done = int(replay[-1]), int(replay[-2])
    turns = np.asarray(replay[4], dtype=float)
    rows = []
    for target in sorted(set(x for x in (2000, 5000, horizon) if x <= horizon)):
        i = min(target//100, len(states)-1)
        y = states[i]
        raw_idx = int(indices[i])
        rows.append({"periods": target, "state_sample_step": raw_idx,
                     "state_mean_radius_R": [float(np.hypot(y[0],y[1])),float(np.hypot(y[6],y[7]))]})
    no_contact_clearance = float(np.linalg.norm(replay_end[:3]-replay_end[6:9]))
    return {
        "run_id": task["run_id"], "multipliers": mult, "p_corrected": p.tolist(),
        "screen_outcome": task["screen_outcome"], "screen_turns": task["screen_turns"],
        "source_state_period": source, "source_trajectory": str(trajectory.relative_to(ROOT)).replace("\\", "/"),
        "fixed_phase_at_seed_rad": phase0.tolist(), "phase_after_100p_transient_rad": phase.tolist(),
        "periods_after_transient": horizon, "steps_per_rf_period": steps, "qr_interval_periods": qr_periods,
        "conditional_spectrum_per_tau_descending": spectrum.tolist(),
        "sum_exponents_per_tau": float(spectrum.sum()),
        "integrated_divergence_per_tau": float(div_int/(horizon*steps*dt)),
        "volume_balance_abs_error_per_tau": float(abs(spectrum.sum()-div_int/(horizon*steps*dt))),
        "event_resolved_replay_status": actual_status, "event_resolved_replay_completed_periods": actual_done,
        "event_resolved_turns": turns.tolist(), "event_resolved_min_pair_distance_R": float(replay[10]),
        "event_resolved_mean_radius_R": np.asarray(replay[11]).tolist(),
        "event_resolved_rmax_R": np.asarray(replay[7]).tolist(), "event_resolved_zmax_R": np.asarray(replay[8]).tolist(),
        "QR_base_vs_event_replay_final_state_norm": float(np.linalg.norm(np.asarray(state)-np.asarray(replay_end))),
        "late_state_samples": rows, "elapsed_seconds": time.perf_counter()-t0,
        "scope": "5000-period conditional QR at one frozen refine12 late state; phase fixed in tangent perturbations; independent event-resolving RK4 replay. It does not prove a global attractor or basin size.",
    }


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=6)
    ap.add_argument("--periods",type=int,default=5000)
    ap.add_argument("--select-run-id",action="append")
    args=ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = unique_qr_candidates()
    if args.select_run_id:
        selected=[x for x in selected if x["run_id"] in set(args.select_run_id)]
        if not selected:raise ValueError("no requested screen run_id matched the selected robust points")
    for x in selected:
        x["periods"]=args.periods
        x["run_id"]=f"{x['run_id']}_{args.periods}pQR"
    checkpoint = OUT / f"selected_qr_{args.periods}p_checkpoint.jsonl"
    results = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x=json.loads(line);results[x["run_id"]]=x
    todo=[x for x in selected if x["run_id"] not in results]
    started=time.perf_counter()
    with checkpoint.open("a",encoding="utf-8",buffering=1) as f:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(worker,x):x for x in todo}
            for fut in as_completed(futures):
                row=fut.result();results[row["run_id"]]=row
                f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n");f.flush()
                print(f"{row['run_id']} qr_lambda1={row.get('conditional_spectrum_per_tau_descending',[None])[0]} outcome={row['event_resolved_replay_status'] if 'event_resolved_replay_status' in row else row['outcome']}",flush=True)
    final={"selected_by": "local CR neighborhood; six parameter points incl baseline", "results":[results[x["run_id"]] for x in selected],
           "elapsed_seconds":time.perf_counter()-started}
    (OUT/f"selected_qr_{args.periods}p_results.json").write_text(json.dumps(final,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps([{k:r.get(k) for k in ("run_id","multipliers","conditional_spectrum_per_tau_descending","event_resolved_replay_status","event_resolved_turns","event_resolved_min_pair_distance_R")} for r in final["results"]],indent=2))


if __name__=="__main__":main()
