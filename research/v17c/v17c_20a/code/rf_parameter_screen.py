from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research" / "v17c" / "v17c_20a"
sys.path.insert(0, str(BASE / "code"))
import run_long_qr as qr  # noqa: E402

OUT = BASE / "results" / "rf_parameter_screen"
VARIABLES = ("A", "B", "nu", "beta")
LEVELS = {"A": [0.70, 0.85, 1.0, 1.15, 1.30],
          "B": [0.70, 0.85, 1.0, 1.15, 1.30],
          "nu": [0.80, 0.90, 1.0, 1.10, 1.20],
          "beta": [0.70, 0.85, 1.0, 1.15, 1.30]}


def point_key(mult):
    return tuple(round(float(x), 8) for x in mult)


def make_tasks(n_sobol=32, seed=20261123):
    gathered = {}
    for i, x in enumerate(VARIABLES):
        for y in VARIABLES[i+1:]:
            for fx in LEVELS[x]:
                for fy in LEVELS[y]:
                    mult = [1.0] * 4
                    mult[VARIABLES.index(x)] = fx
                    mult[VARIABLES.index(y)] = fy
                    key = point_key(mult)
                    refs = gathered.setdefault(key, {"multipliers": list(key), "slices": [], "origin": "pairwise_grid"})
                    refs["slices"].append({"x": x, "y": y, "multiplier_x": fx, "multiplier_y": fy})
    # This union includes each baseline and all six 5x5 pairwise slices.
    sobol = qmc.Sobol(d=4, scramble=True, seed=seed).random(n_sobol)
    lows = np.array([0.70, 0.70, 0.80, 0.70])
    highs = np.array([1.30, 1.30, 1.20, 1.30])
    for row in sobol:
        mult = lows + row * (highs - lows)
        key = point_key(mult)
        if key not in gathered:
            gathered[key] = {"multipliers": list(key), "slices": [], "origin": "sobol", "sobol_seed": seed}
        else:
            gathered[key].setdefault("also_sobol", True)
    tasks = []
    for key, value in sorted(gathered.items()):
        ident = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
        value["run_id"] = f"refine12_{ident}"
        tasks.append(value)
    return tasks


def worker(task):
    _, case, p0, qratio, state, phase, source, traj = qr.load_late_seed("scaled_refine12_sobol17_f50hz", 240000)
    p = np.asarray(p0, dtype=np.float64).copy()
    mult = np.asarray(task["multipliers"], dtype=float)
    p[[0, 1, 2, 3]] *= mult
    field = qr.a.load_field()
    transition = 100
    first = qr.a.PAIR._integrate_trace(
        state, phase, p, field["family"], field["k"], field["coeff"], qratio,
        case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"],
        case["re2_per_speed"], case["gravity_z"], transition, task["steps"], 0,
        task["steps"] * 100)
    state = np.asarray(first[0], dtype=float)
    transition_status, transition_done = int(first[-1]), int(first[-2])
    multiplier_record = dict(zip(VARIABLES, mult.tolist()))
    if transition_status != 0 or transition_done != transition:
        outcome = "COLLISION" if transition_status == 1 else ("ESCAPE_OR_WALL" if transition_status == 2 else "UNRESOLVED")
        return {"run_id": task["run_id"], "status": transition_status,
                "completed_periods": 0, "transition_completed_periods": transition_done,
                "outcome": outcome, "multipliers": multiplier_record,
                "slices": task["slices"], "origin": task["origin"], "p_corrected": p.tolist(),
                "fixed_initial_RF_phases_rad": phase.tolist(), "starting_state_source": str(traj.relative_to(ROOT)).replace("\\", "/")}
    phase = (phase + np.array([1.0, float(p[2])]) * (2.0 * math.pi * transition)) % (2.0 * math.pi)
    raw = qr.a.PAIR._integrate_trace(
        state, phase, p, field["family"], field["k"], field["coeff"], qratio,
        case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"],
        case["re2_per_speed"], case["gravity_z"], task["periods"], task["steps"],
        0, task["steps"] * 100)
    turns = np.asarray(raw[4], dtype=float)
    windows = np.asarray(raw[5], dtype=float)
    rmin, rmax, zmax = (np.asarray(raw[k], dtype=float) for k in (6, 7, 8))
    avgr = np.asarray(raw[11], dtype=float)
    dmin = float(raw[10])
    status, completed = int(raw[-1]), int(raw[-2])
    if status == 1:
        outcome = "COLLISION"
    elif status == 2:
        outcome = "ESCAPE_OR_WALL"
    elif status != 0 or completed != task["periods"]:
        outcome = "UNRESOLVED"
    elif abs(turns[0]) >= task["min_turns"] and abs(turns[1]) >= task["min_turns"] and turns[0] * turns[1] < 0:
        outcome = "CR(+−)" if turns[0] > 0 else "CR(−+)"
    elif abs(turns[0]) >= task["min_turns"] and abs(turns[1]) >= task["min_turns"] and turns[0] * turns[1] > 0:
        outcome = "CO(++)" if turns[0] > 0 else "CO(−−)"
    elif max(abs(turns[0]), abs(turns[1])) < 0.05:
        outcome = "BOUND_NONROTATING_SCREEN"
    elif turns[0] * turns[1] < 0:
        outcome = "CR_SIGN_ONLY_FEW_TURNS"
    else:
        outcome = "UNRESOLVED_SHORT_TURN_COUNT"
    return {
        "run_id": task["run_id"], "origin": task["origin"], "slices": task["slices"],
        "multipliers": multiplier_record,
        "p_corrected": p.tolist(), "qratio": float(qratio), "nu": float(p[2]),
        "starting_state_source": str(traj.relative_to(ROOT)).replace("\\", "/"),
        "source_period": source, "same_initial_mechanical_state": True,
        "fixed_initial_RF_phases_rad": phase.tolist(), "transition_periods": transition,
        "periods": task["periods"], "steps_per_period": task["steps"],
        "completed_periods": completed, "status": status, "outcome": outcome,
        "turns": turns.tolist(), "late_window_turns": windows.tolist(),
        "frequency_Hz_screen": (turns * float(case["f1_Hz"]) / task["periods"]).tolist(),
        "mean_radius_R": avgr.tolist(), "rmin_R": rmin.tolist(), "rmax_R": rmax.tolist(),
        "zmax_R": zmax.tolist(), "min_pair_distance_R": dmin,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--periods", type=int, default=3000)
    ap.add_argument("--transition", type=int, default=100)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--min-turns", type=float, default=1.0)
    ap.add_argument("--sobol", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(args.sobol, args.seed)
    for task in tasks:
        task.update({"periods": args.periods, "transition": args.transition,
                     "steps": args.steps, "min_turns": args.min_turns})
    cp = OUT / "screen_checkpoint.jsonl"
    existing = {}
    if cp.exists():
        for line in cp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line); existing[x["run_id"]] = x
    todo = [t for t in tasks if t["run_id"] not in existing]
    started = time.perf_counter()
    with cp.open("a", encoding="utf-8", buffering=1) as f:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(worker, t): t for t in todo}
            for fut in as_completed(futures):
                row = fut.result(); existing[row["run_id"]] = row
                f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                f.flush()
                print(f"{row['run_id']} {row['origin']} {row['outcome']} turns={row.get('turns')}", flush=True)
    rows = [existing[t["run_id"]] for t in tasks]
    counts = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    slice_rows = []
    for i, x in enumerate(VARIABLES):
        for y in VARIABLES[i+1:]:
            points = [r for r in rows if any(s["x"] == x and s["y"] == y for s in r.get("slices", []))]
            slice_rows.append({"slice": f"{x}__{y}", "grid_shape": [len(LEVELS[x]), len(LEVELS[y])],
                               "points": len(points), "resolved": sum(r["outcome"] != "UNRESOLVED" for r in points),
                               "outcomes": {o: sum(r["outcome"] == o for r in points) for o in sorted({r["outcome"] for r in points})},
                               "multipliers_x": LEVELS[x], "multipliers_y": LEVELS[y]})
    coverage = {
        "source_sha": "d1bf252649e174166becf464269318b5f97acd5d",
        "baseline": "refine12_sobol17, late state at 240000 periods; same mechanical state and phases for all points before a 100-period field-specific transient.",
        "variables": [{"name": v, "definition": "multiplier of baseline dimensionless V20B p[0..3] (A, B, nu, beta)",
                       "units": "dimensionless", "multipliers": LEVELS[v], "changed_independently": True} for v in VARIABLES],
        "pairwise_slices": slice_rows,
        "total_unique_points": len(rows), "pairwise_grid_unique_points": sum(r["origin"] == "pairwise_grid" for r in rows),
        "sobol_requested": args.sobol, "sobol_seed": args.seed,
        "sobol_points_present": sum(r["origin"] == "sobol" or r.get("also_sobol", False) for r in tasks),
        "sobol_scored_rows": sum(r["origin"] == "sobol" for r in rows),
        "screen_periods": args.periods, "transition_periods_discarded": args.transition,
        "steps_per_rf_period": args.steps, "minimum_turns_for_CR_label": args.min_turns,
        "classification_counts": counts,
        "classification_limit": "This is a same-late-state, short-horizon field response screen, not an initial-condition basin measure or proof of an attractor. Full MFS harmonic coefficients are retained; A/B scale their respective whole RF families, nu changes the second carrier, beta changes the ideal DC quadrupole.",
    }
    (OUT / "PARAMETER_COVERAGE.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "screen_summary.json").write_text(json.dumps({"rows": rows, "outcome_counts": counts,
        "elapsed_seconds_this_run": time.perf_counter()-started}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "screen_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "origin", "multipliers", "outcome", "status",
            "completed_periods", "turns", "frequency_Hz_screen", "mean_radius_R", "rmin_R", "rmax_R",
            "zmax_R", "min_pair_distance_R", "slices"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (dict, list)) else row.get(k)
                             for k in writer.fieldnames})
    print(json.dumps({"points": len(rows), "counts": counts, "slices": slice_rows}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = {"CR(+−)": "#0072B2", "CR(−+)": "#56B4E9", "CO(++)": "#D55E00", "CO(−−)": "#E69F00",
              "COLLISION": "#CC79A7", "ESCAPE_OR_WALL": "#B22222", "BOUND_NONROTATING_SCREEN": "#009E73",
              "CR_SIGN_ONLY_FEW_TURNS": "#8EC9E8", "UNRESOLVED_SHORT_TURN_COUNT": "#BBBBBB", "UNRESOLVED": "#555555",
              "UNRESOLVED_NONFINITE_TRANSITION": "#333333"}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), constrained_layout=True)
    axes = axes.flat
    for ax, item in zip(axes, slice_rows):
        x, y = item["slice"].split("__")
        sub = [r for r in rows if any(s["x"] == x and s["y"] == y for s in r.get("slices", []))]
        for outcome in sorted({r["outcome"] for r in sub}):
            pts = [r for r in sub if r["outcome"] == outcome]
            ax.scatter([r["multipliers"][x] for r in pts], [r["multipliers"][y] for r in pts],
                       color=colors.get(outcome, "#555555"), s=40, edgecolors="black", linewidths=0.2,
                       label=outcome)
        ax.set_xlabel(f"{x} / {x}₀")
        ax.set_ylabel(f"{y} / {y}₀")
        ax.set_title(f"{x} × {y}; one frozen late start")
        ax.grid(True, alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Local RF-parameter response of the refine12 late state (short screen)")
    fig.subplots_adjust(bottom=0.14)
    figdir = BASE / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figdir / f"FIG07_RF_parameter_local_slices.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
