"""Independent audit for long-horizon selected-start persistence checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
SOURCE = RESULTS / "rescaled_legacy_candidate_validation"
GROUPS = ("rest", "weak", "wide", "edge")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-name", default="long_horizon_refine12_validation")
    args = parser.parse_args()
    out = RESULTS / args.results_name
    meta = json.loads((out / "campaign_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    source_summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    source_rows = [json.loads(line) for line in (SOURCE / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    source_by_key = {(r["ensemble"], int(r["start_index"])): r for r in source_rows
                     if r["configuration"] == meta["configuration"]}
    frozen = np.load(SOURCE / "frozen_starts.npz", allow_pickle=True)
    starts, phases = frozen["initial"], frozen["phases"]
    offsets = {g: i * 128 for i, g in enumerate(GROUPS)}
    selected = meta["selected_starts"]
    rows = [json.loads(line) for line in (out / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [r["key"] for r in rows]
    expected = {f"{x['ensemble']}:{x['start_index']}" for x in selected}
    selected_match = True
    row_match = True
    horizon = True
    for item in selected:
        key = (item["ensemble"], int(item["start_index"]))
        src = source_by_key.get(key)
        idx = offsets[item["ensemble"]] + int(item["start_index"])
        selected_match &= src is not None and src["strict_late_category"].startswith("STRICT_CR_")
        selected_match &= np.array_equal(np.asarray(item["initial"]), starts[idx])
        selected_match &= np.array_equal(np.asarray(item["phase"]), phases[idx])
        selected_match &= src is not None and item["source_strict_label"] == src["strict_late_category"]
    for row in rows:
        key = (row["ensemble"], int(row["start_index"]))
        src = source_by_key.get(key)
        row_match &= src is not None and src["strict_late_category"].startswith("STRICT_CR_")
        row_match &= src is not None and row["source_strict_label"] == src["strict_late_category"]
        horizon &= (bool(row["retained"]) and int(row["completed_periods"]) == int(meta["periods"])) or (not bool(row["retained"]) and int(row["completed_periods"]) < int(meta["periods"]))
        horizon &= np.array_equal(np.asarray(row["initial"]), starts[offsets[row["ensemble"]] + int(row["start_index"])])
        horizon &= np.array_equal(np.asarray(row["phase"]), phases[offsets[row["ensemble"]] + int(row["start_index"])])
    counts = {g: dict(Counter(r["strict_late_category"] for r in rows if r["ensemble"] == g)) for g in GROUPS}
    at100 = sum(min(abs(float(t)) for t in r["turns"]) >= 100 for r in rows)
    cr_at100 = sum(r["strict_late_category"].startswith("STRICT_CR_") and min(abs(float(t)) for t in r["turns"]) >= 100 for r in rows)
    checks = {
        "expected_and_unique_keys": len(rows) == len(expected) and set(keys) == expected and len(set(keys)) == len(keys),
        "selected_starts_match_source_and_frozen_arrays": bool(selected_match),
        "rows_match_15000_period_CR_source": bool(row_match),
        "all_rows_complete_full_horizon_and_match_start_phase": bool(horizon),
        "ensemble_outcomes_reconstruct_summary": counts == summary["outcomes_by_ensemble"],
        "long_horizon_counts_reconstruct_summary": cr_at100 == summary["long_horizon_CR_count"] and at100 == summary["at_least_100_turns_per_particle_count"],
        "trajectory_hash_matches_summary": sha(out / "trajectories.jsonl") == summary["trajectory_sha256"],
        "source_trajectory_hash_matches": meta["source_trajectory_sha256"] == source_summary["trajectory_sha256"],
        "integrator_hash_matches_saved_source": sha(ROOT / "model" / "basin_scan.py") == sha(out / "campaign_integrator_source.py") == meta["integrator_sha256"],
        "field_hash_matches": sha(ROOT / "model" / "field_model.json") == meta["field_model_sha256"],
        "runner_hash_matches_saved_source": sha(out / "campaign_runner_source.py") == meta["runner_sha256"],
    }
    report = {"audit": args.results_name, "integrity_pass": all(checks.values()), "checks": checks,
              "rows": len(rows), "unique_keys": len(set(keys)), "outcomes_by_ensemble": counts,
              "long_horizon_CR_at_100_turns": cr_at100, "trajectories_at_100_turns": at100,
              "trajectory_sha256": sha(out / "trajectories.jsonl"), "audit_script_sha256": sha(Path(__file__))}
    (out / "independent_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
