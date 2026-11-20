"""Independent integrity and aggregate audit for the physical Sobol screen."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "identical_physical_sobol_screen_corrected"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    meta = json.loads((OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    starts_data = np.load(OUT / "frozen_rest_starts.npz")
    starts = starts_data["initial"]
    phases = starts_data["phases"]
    rows = []
    with (OUT / "trajectories.jsonl").open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            row["_line"] = line_no
            rows.append(row)

    expected = {(ci, si) for ci in range(512) for si in range(8)}
    seen = [(int(r["screen_index"]), int(r["start_index"])) for r in rows]
    counts = Counter(r["provisional_category"] for r in rows)
    by_config: dict[int, Counter] = defaultdict(Counter)
    config_ids = {}
    start_errors = []
    for r in rows:
        ci, si = int(r["screen_index"]), int(r["start_index"])
        by_config[ci][r["provisional_category"]] += 1
        config_ids[ci] = r["config_id"]
        if not np.array_equal(np.asarray(r["initial"]), starts[si]):
            start_errors.append([ci, si, "initial"])
        if not np.array_equal(np.asarray(r["phase"]), phases[si]):
            start_errors.append([ci, si, "phase"])

    duplicates = [list(k) for k, n in Counter(seen).items() if n != 1]
    missing = [list(k) for k in sorted(expected - set(seen))]
    extras = [list(k) for k in seen if k not in expected]
    selected = meta["configurations"]
    physical = [c["physical"] for c in selected]
    outcome_by_config = []
    for ci in range(512):
        z = by_config[ci]
        outcome_by_config.append({"screen_index": ci,
            "config_id": config_ids.get(ci, selected[ci]["id"]),
            "n": sum(z.values()), "counts": dict(sorted(z.items())),
            "both_provisional_CR_directions":
                z.get("PROVISIONAL_CR_PLUS_MINUS", 0) > 0 and
                z.get("PROVISIONAL_CR_MINUS_PLUS", 0) > 0})

    runner = ROOT / "sobol_identical_search.py"
    integrator = ROOT / "model" / "basin_scan.py"
    field = ROOT / "model" / "field_model.json"
    code_hashes = {"runner_sha256": sha(runner),
                   "saved_runner_sha256": sha(OUT / "campaign_runner_source.py"),
                   "integrator_sha256": sha(integrator),
                   "field_model_sha256": sha(field)}
    audit = {
        "audit": "integrity and aggregate audit; short screen is discovery only",
        "expected_trajectory_rows": len(expected), "actual_rows": len(rows),
        "expected_configurations": 512, "actual_configurations_in_metadata": len(selected),
        "unique_keys": len(set(seen)), "missing_keys": missing,
        "duplicate_keys": duplicates, "unexpected_keys": extras,
        "frozen_start_or_phase_mismatches": start_errors,
        "outcome_counts": dict(sorted(counts.items())),
        "outcome_fractions_of_all_starts": {k: v / len(rows) for k, v in sorted(counts.items())},
        "configurations_with_both_provisional_CR_directions": sum(
            x["both_provisional_CR_directions"] for x in outcome_by_config),
        "configurations_by_outcome": outcome_by_config,
        "physical_screen_counts": {
            "selected": len(selected),
            "selected_charge_e_min": min(x["charge_e_each"] for x in physical),
            "selected_charge_e_max": max(x["charge_e_each"] for x in physical),
            "selected_electrode_sum_peak_V_min": min(x["U1_peak_electrode_V"] + x["U2_peak_electrode_V"] for x in physical),
            "selected_electrode_sum_peak_V_max": max(x["U1_peak_electrode_V"] + x["U2_peak_electrode_V"] for x in physical),
            "n_preferred_charge_window": sum(c["preferred_charge_window"] for c in selected),
            "n_under_flat_Paschen_screen": sum(c["flat_Paschen_screen_under_unity"] for c in selected)},
        "code_sha256": code_hashes,
        "integrity_pass": (len(rows) == len(expected) and len(set(seen)) == len(expected)
            and not missing and not duplicates and not extras and not start_errors
            and len(selected) == 512 and code_hashes["runner_sha256"] == code_hashes["saved_runner_sha256"]
            and len(outcome_by_config) == 512 and all(x["n"] == 8 for x in outcome_by_config))}
    path = OUT / "independent_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {k: audit[k] for k in ("expected_trajectory_rows", "actual_rows",
        "unique_keys", "outcome_counts", "configurations_with_both_provisional_CR_directions",
        "physical_screen_counts", "code_sha256", "integrity_pass")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not audit["integrity_pass"]:
        raise SystemExit("audit failed; see independent_audit.json")


if __name__ == "__main__":
    main()
