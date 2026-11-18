"""Reclassify legacy G1/G2 screens against the ten-turn criterion.

The original exploratory screen labelled a run "counter" from two late
window signs alone. This postprocessor preserves the raw CSVs and emits a
separate strict classification with turn-count and late-window thresholds.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from rf_dc_core import RESULTS

ROOT = RESULTS / "g1_g2_joint"
OUT = ROOT / "strict_classification"
OUT.mkdir(parents=True, exist_ok=True)


def truth(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def classify(row: dict[str, str], broad: bool) -> tuple[bool, bool]:
    if broad:
        w1 = json.loads(row["late_windows1"])
        w2 = json.loads(row["late_windows2"])
    else:
        w1 = json.loads(row["windows1"])
        w2 = json.loads(row["windows2"])
    stable = not truth(row["escaped"]) and not truth(row["collided"])
    s1, s2 = sum(w1[-2:]), sum(w2[-2:])
    late = bool(stable and s1*s2 < 0
                and all(v*s1 > 0 for v in w1[-2:])
                and all(v*s2 > 0 for v in w2[-2:]))
    total_min = min(abs(float(row["turns1"])), abs(float(row["turns2"])))
    final1, final2 = abs(w1[-1]), abs(w2[-1])
    ten = bool(late and total_min >= 10.0 and final1 >= 1.0 and final2 >= 1.0)
    return late, ten


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(label: str, rows: list[dict[str, str]], broad: bool) -> dict:
    per_geometry = {}
    for geometry in ("G1", "G2"):
        group = [r for r in rows if r["geometry"] == geometry]
        classes = [classify(r, broad) for r in group]
        min_turns = [min(abs(float(r["turns1"])), abs(float(r["turns2"]))) for r in group]
        durations = [float(r["periods"]) for r in group]
        per_geometry[geometry] = {
            "runs": len(group),
            "full_horizon_survivors": sum(not truth(r["escaped"]) and not truth(r["collided"]) for r in group),
            "late_opposite_sign_only": sum(x[0] for x in classes),
            "ten_turn_screen_candidates": sum(x[1] for x in classes),
            "max_minimum_total_turns": max(min_turns) if group else None,
            "max_duration_rf_periods": max(durations) if group else None,
        }
    return {"dataset": label, "criteria": {
        "late_opposite_sign_only": "survived; each particle has same nonzero sign in each of its final two windows; the signs differ between particles",
        "ten_turn_screen_candidate": "late_opposite_sign_only, at least 10 total turns for each particle, and at least 1 turn by each particle in its final window",
        "scope": "screen only; a candidate would still require perturbation, parameter robustness, and independent integrator validation",
    }, "per_geometry": per_geometry}


def main() -> None:
    broad = read_csv(ROOT / "screen_320RF_n900_seed17032026" / "G1_G2_screening_all.csv")
    barrier = read_csv(ROOT / "barrier_targeted_600RF_n600_seed17032027" / "G1_G2_barrier_targeted_all.csv")
    broad_out = []
    for row in broad:
        late, ten = classify(row, True)
        broad_out.append({**row, "reclassified_late_opposite_sign_only": late,
                          "reclassified_10turn_screen_candidate": ten})
    barrier_out = []
    for row in barrier:
        late, ten = classify(row, False)
        barrier_out.append({**row, "reclassified_late_opposite_sign_only": late,
                            "reclassified_10turn_screen_candidate": ten})
    write_csv(OUT / "broad_320RF_strict_reclassification.csv", broad_out)
    write_csv(OUT / "barrier_targeted_600RF_strict_reclassification.csv", barrier_out)
    result = {
        "broad_screen": summarize("900 per geometry, 320 RF periods, 32 RK4 steps/period", broad, True),
        "barrier_targeted": summarize("600 per geometry, up to 600 RF periods, 48 RK4 steps/period", barrier, False),
        "legacy_false_positive_diagnostics": [
            {k: r[k] for k in ("run_id", "geometry", "turns1", "turns2", "late_windows1",
                               "late_windows2", "r1_mean", "r2_mean", "z1_rms", "z2_rms",
                               "chi", "coulomb_force_rf_ratio")}
            for r in broad if truth(r.get("counter_rotating_late", "false"))
        ],
    }
    (OUT / "strict_reclassification_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
