from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rf_parameter_screen import worker

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "rf_parameter_screen"
VARIABLES = ["A", "B", "nu", "beta"]
LEVELS = {"A": [0.70, 0.85, 1.0, 1.15, 1.30],
          "B": [0.70, 0.85, 1.0, 1.15, 1.30],
          "nu": [0.80, 0.90, 1.0, 1.10, 1.20],
          "beta": [0.70, 0.85, 1.0, 1.15, 1.30]}


def make_tasks():
    summary = json.loads((RES / "screen_summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    tasks = []
    for ia, x in enumerate(VARIABLES):
        for y in VARIABLES[ia + 1:]:
            grid = {}
            for r in rows:
                if r.get("origin") != "pairwise_grid":
                    continue
                ref = next((s for s in r.get("slices", []) if s["x"] == x and s["y"] == y), None)
                if ref is not None:
                    grid[(round(r["multipliers"][x], 8), round(r["multipliers"][y], 8))] = r
            edges = []
            for ix, xv in enumerate(LEVELS[x]):
                for iy, yv in enumerate(LEVELS[y]):
                    for dx, dy in ((1, 0), (0, 1)):
                        jx, jy = ix + dx, iy + dy
                        if jx >= len(LEVELS[x]) or jy >= len(LEVELS[y]):
                            continue
                        xn, yn = LEVELS[x][jx], LEVELS[y][jy]
                        r1 = grid.get((round(xv, 8), round(yv, 8)))
                        r2 = grid.get((round(xn, 8), round(yn, 8)))
                        if r1 is None or r2 is None or r1["outcome"] == r2["outcome"]:
                            continue
                        midpoint = {"A": 1.0, "B": 1.0, "nu": 1.0, "beta": 1.0}
                        midpoint[x] = (xv + xn) / 2.0
                        midpoint[y] = (yv + yn) / 2.0
                        distance = abs(midpoint[x] - 1.0) + abs(midpoint[y] - 1.0)
                        labels = {r1["outcome"], r2["outcome"]}
                        high_value = bool(labels & {"CR(+−)", "CO(−−)", "CO(++)", "CR(−+)"}) and len(labels) > 1
                        # Prefer observed CR/non-CR boundaries, then stay close to baseline.
                        priority = 0 if high_value and any(v.startswith("CR(") for v in labels) else 1
                        edges.append((priority, distance, ix + iy, xv, yv, xn, yn, r1, r2, midpoint))
            edges.sort(key=lambda e: (e[0], e[1], e[2], e[3], e[4]))
            selected = edges[:3]
            for edge in selected:
                _, _, _, xv, yv, xn, yn, r1, r2, midpoint = edge
                key = tuple(round(midpoint[v], 8) for v in VARIABLES)
                ident = hashlib.sha256(json.dumps([x, y, key]).encode()).hexdigest()[:12]
                tasks.append({"run_id": f"refine12_adaptive_{ident}",
                              "multipliers": list(key),
                              "origin": "adaptive_boundary_midpoint",
                              "slices": [{"x": x, "y": y,
                                          "parent_edge": [[xv, yv], [xn, yn]],
                                          "parent_outcomes": [r1["outcome"], r2["outcome"]]}],
                              "edge_distance_L1_from_baseline": abs((xv+xn)/2-1)+abs((yv+yn)/2-1)})
    unique = {}
    for task in tasks:
        key = tuple(round(v, 8) for v in task["multipliers"])
        if key not in unique:
            unique[key] = task
        else:
            unique[key]["slices"].extend(task["slices"])
    result = []
    for item in unique.values():
        item["periods"] = 3000
        item["steps"] = 32
        item["min_turns"] = 1.0
        item["transition"] = 100
        result.append(item)
    return result


def main():
    tasks = make_tasks()
    checkpoint = RES / "adaptive_boundary_checkpoint.jsonl"
    old = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                old[row["run_id"]] = row
    todo = [t for t in tasks if t["run_id"] not in old]
    with checkpoint.open("a", encoding="utf-8", buffering=1) as f:
        with ProcessPoolExecutor(max_workers=min(6, max(1, len(todo)))) as pool:
            futures = {pool.submit(worker, t): t for t in todo}
            for future in as_completed(futures):
                row = future.result()
                row["adaptive_parent_edges"] = futures[future]["slices"]
                row["midpoint_multipliers"] = futures[future]["multipliers"]
                old[row["run_id"]] = row
                f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
                outcome_ascii = row["outcome"].encode("ascii", "backslashreplace").decode("ascii")
                print(row["run_id"], outcome_ascii, row.get("turns"), flush=True)
    rows = [old[t["run_id"]] for t in tasks]
    out = {"source": "six 5x5 pairwise slices in PARAMETER_COVERAGE.json",
           "method": "Select up to three adjacent classification transitions per slice, ranked to prefer CR/non-CR boundaries and proximity to baseline; run the exact midpoint with the same 3000-period screen, late state, RF phases and classifier.",
           "screen_protocol": {"periods": 3000, "transition_periods": 100,
                               "steps_per_period": 32, "minimum_turns": 1.0,
                               "same_refine12_240k_seed_and_RF_phase": True},
           "max_selected_per_slice": 3, "unique_points": len(rows),
           "results": rows,
           "interpretation_limit": "A midpoint class is a new sampled outcome, not a located bifurcation boundary; short-horizon labels are not attractor classifications."}
    (RES / "adaptive_boundary_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"unique_points": len(rows),
                      "outcomes": {o: sum(r["outcome"] == o for r in rows) for o in sorted({r["outcome"] for r in rows})},
                      "note": "ASCII-escaped JSON for Windows console compatibility"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
