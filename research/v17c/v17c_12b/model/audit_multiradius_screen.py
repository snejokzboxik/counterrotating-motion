"""Integrity audit for the multi-radius Al2O3 discovery screen."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "alumina_multiradius_sobol_screen"
ESCAPE_R = 1.0
ESCAPE_Z = 2.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    meta = json.loads((OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    configs = meta["configurations"]
    nconfig, nstart = len(configs), int(meta["n_starts_per_config"])
    frozen = np.load(OUT / "frozen_rest_starts.npz")
    starts, phases = frozen["initial"], frozen["phases"]
    rows = []
    with (OUT / "trajectories.jsonl").open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line); r["_line"] = line_no; rows.append(r)

    expected = {(ci, si) for ci in range(nconfig) for si in range(nstart)}
    seen = [(int(r["screen_index"]), int(r["start_index"])) for r in rows]
    counts = Counter(r["provisional_category"] for r in rows)
    by_radius = defaultdict(Counter)
    retained_by_radius = Counter()
    mechanism_by_radius = defaultdict(Counter)
    by_config = defaultdict(Counter)
    identities = {}
    start_errors, radius_errors = [], []
    radial_losses = axial_losses = both_losses = 0
    for r in rows:
        ci, si = int(r["screen_index"]), int(r["start_index"])
        c = configs[ci]
        identities[ci] = c["id"]
        by_config[ci][r["provisional_category"]] += 1
        by_radius[str(c["radius_um"])][r["provisional_category"]] += 1
        retained_by_radius[str(c["radius_um"])] += int(bool(r["retained"]))
        if not np.array_equal(np.asarray(r["initial"]), starts[si]):
            start_errors.append([ci, si, "initial"])
        if not np.array_equal(np.asarray(r["phase"]), phases[si]):
            start_errors.append([ci, si, "phase"])
        if float(r["radius_um"]) != float(c["radius_um"]):
            radius_errors.append([ci, si, r["radius_um"], c["radius_um"]])
        if r["provisional_category"] == "WALL_OR_AXIAL_ESCAPE":
            a_over_R = c["physical"]["radius_m"] / meta["trap"]["R_m"]
            radial = max(r["r_max_over_R"]) + a_over_R >= ESCAPE_R
            axial = max(r["z_max_over_R"]) + a_over_R >= ESCAPE_Z
            radial_losses += int(radial); axial_losses += int(axial)
            both_losses += int(radial and axial)
            mechanism_by_radius[str(c["radius_um"])]["radial"] += int(radial)
            mechanism_by_radius[str(c["radius_um"])]["axial"] += int(axial)
            mechanism_by_radius[str(c["radius_um"])]["both"] += int(radial and axial)

    duplicates = [list(k) for k, n in Counter(seen).items() if n != 1]
    missing = [list(k) for k in sorted(expected - set(seen))]
    extras = [list(k) for k in seen if k not in expected]
    code_path = ROOT / "sobol_alumina_multiradius_search.py"
    code_copy = OUT / "campaign_runner_source.py"
    hashes = {"runner_sha256": sha(code_path), "saved_runner_sha256": sha(code_copy),
        "integrator_sha256": sha(ROOT / "model" / "basin_scan.py"),
        "field_model_sha256": sha(ROOT / "model" / "field_model.json")}
    audit = {
        "audit": "integrity and physical-discovery aggregation; not a final probability study",
        "expected_rows": len(expected), "actual_rows": len(rows),
        "unique_keys": len(set(seen)), "missing_keys": missing,
        "duplicate_keys": duplicates, "unexpected_keys": extras,
        "frozen_start_or_phase_mismatches": start_errors,
        "row_radius_metadata_mismatches": radius_errors,
        "configurations": nconfig, "starts_per_configuration": nstart,
        "configuration_count_by_radius": dict(Counter(str(c["radius_um"]) for c in configs)),
        "outcome_counts": dict(sorted(counts.items())),
        "outcome_fractions_of_all_search_rows": {k: v / len(rows) for k, v in sorted(counts.items())},
        "outcomes_by_radius": {k: {"counts": dict(sorted(v.items())),
            "pair_retained_count": retained_by_radius[k], "pair_retained_fraction": retained_by_radius[k] / sum(v.values()),
            "loss_mechanisms": dict(mechanism_by_radius[k])}
            for k, v in sorted(by_radius.items(), key=lambda x: float(x[0]))},
        "wall_or_axial_escape_mechanism_counts": {"radial": radial_losses,
            "axial": axial_losses, "both": both_losses},
        "configs_with_both_CR_directions": sum(
            z.get("PROVISIONAL_CR_PLUS_MINUS", 0) > 0 and
            z.get("PROVISIONAL_CR_MINUS_PLUS", 0) > 0 for z in by_config.values()),
        "physical_screen_diagnostics": meta["physical_screen_diagnostics"],
        "code_sha256": hashes,
        "integrity_pass": (len(rows) == len(expected) and len(set(seen)) == len(expected)
            and not missing and not duplicates and not extras and not start_errors
            and not radius_errors and len(configs) == nconfig
            and all(len(by_config[i]) >= 0 and sum(by_config[i].values()) == nstart for i in range(nconfig))
            and hashes["runner_sha256"] == hashes["saved_runner_sha256"])}
    (OUT / "independent_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("expected_rows", "actual_rows", "unique_keys",
        "outcome_counts", "outcomes_by_radius", "wall_or_axial_escape_mechanism_counts",
        "configs_with_both_CR_directions", "code_sha256", "integrity_pass")},
        indent=2, ensure_ascii=False))
    if not audit["integrity_pass"]:
        raise SystemExit("audit failed; see independent_audit.json")


if __name__ == "__main__":
    main()
