"""Independent integrity audit for the directed CR initial-condition search."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "directed_counterrotation_search"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    meta = json.loads((OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    code = json.loads((OUT / "campaign_code_sha256.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    frozen = np.load(OUT / "frozen_directed_starts.npz")
    starts, phases = frozen["initial"], frozen["phases"]
    labels = json.loads(str(frozen["labels"]))
    rows = [json.loads(line) for line in (OUT / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line]
    keys = [(int(r["config_index"]), int(r["start_index"])) for r in rows]
    expected = {(ci, si) for ci in range(len(meta["configuration_ids"])) for si in range(meta["starts_per_configuration"])}
    unique = set(keys)
    by_config = Counter(r["config_id"] for r in rows)
    by_direction = Counter(r["initial_direction"] for r in rows)
    by_outcome = Counter(r["provisional_category"] for r in rows)
    max_periods = int(meta["periods"])
    match_starts = True
    initial_opposite = True
    full_horizon = True
    retained = True
    for r in rows:
        ci, si = int(r["config_index"]), int(r["start_index"])
        idx = ci * meta["starts_per_configuration"] + si
        match_starts &= np.allclose(np.asarray(r["initial"]), starts[idx], rtol=0.0, atol=1e-14)
        match_starts &= np.allclose(np.asarray(r["phase"]), phases[idx], rtol=0.0, atol=1e-14)
        s = np.asarray(r["initial"], dtype=np.float64)
        w1 = (s[0] * s[4] - s[1] * s[3]) / (s[0] ** 2 + s[1] ** 2)
        w2 = (s[6] * s[10] - s[7] * s[9]) / (s[6] ** 2 + s[7] ** 2)
        initial_opposite &= w1 * w2 < 0
        full_horizon &= int(r["completed_periods"]) == max_periods
        retained &= bool(r["retained"])
    runner = HERE / "directed_counterrotation_search.py"
    integrator = HERE / "model" / "basin_scan.py"
    field = HERE / "model" / "field_model.json"
    checks = {
        "expected_row_count": len(rows) == len(expected),
        "unique_complete_config_start_keys": unique == expected,
        "frozen_starts_and_phases_match": bool(match_starts),
        "all_initial_velocity_pairs_are_counterrotating": bool(initial_opposite),
        "all_trajectories_reached_full_horizon": bool(full_horizon),
        "all_pairs_retained": bool(retained),
        "runner_hash_matches_saved_copy_and_campaign": sha(runner) == sha(OUT / "campaign_runner_source.py") == code["runner_sha256"],
        "integrator_hash_matches_campaign": sha(integrator) == code["integrator_sha256"],
        "field_hash_matches_campaign": sha(field) == code["field_model_sha256"],
        "summary_counts_match_rows": summary["outcome_counts"] == dict(sorted(by_outcome.items())),
        "summary_trajectory_hash_matches": summary["trajectory_sha256"] == sha(OUT / "trajectories.jsonl"),
        "per_config_row_counts_match": all(by_config[cid] == meta["starts_per_configuration"] for cid in meta["configuration_ids"]),
        "initial_orientation_balance_match": by_direction == {"PLUS_MINUS": 64, "MINUS_PLUS": 64},
    }
    result = {
        "audit": "directed_counterrotation_search",
        "integrity_pass": all(checks.values()),
        "checks": checks,
        "rows": len(rows),
        "unique_keys": len(unique),
        "configuration_count": len(meta["configuration_ids"]),
        "rows_per_configuration": dict(sorted(by_config.items())),
        "initial_orientation_counts": dict(sorted(by_direction.items())),
        "outcome_counts": dict(sorted(by_outcome.items())),
        "classification_caveat": "Long-horizon window labels remain provisional dynamical classifications, not basin probabilities or proof of a global attractor.",
        "trajectory_sha256": sha(OUT / "trajectories.jsonl"),
        "audit_script_sha256": sha(Path(__file__)),
    }
    (OUT / "independent_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
