"""Independent integrity audit for the experiment-anchored 50 Hz screen."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "alumina_experimental_50hz_screen"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    meta = json.loads((OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    configs = meta["configurations"]
    nconfig = len(configs)
    starts_per_ensemble = int(meta["n_per_ensemble_per_config"])
    frozen = np.load(OUT / "frozen_starts.npz")
    starts, phases = frozen["initial"], frozen["phases"]
    ensembles, speed_data = frozen["ensemble"], frozen["injection_speed_m_s"]
    rows = []
    with (OUT / "trajectories.jsonl").open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line); r["_line"] = line_no; rows.append(r)

    nstarts = len(starts)
    expected = {(ci, si) for ci in range(nconfig) for si in range(nstarts)}
    seen = [(int(r["screen_index"]), int(r["start_index"])) for r in rows]
    counts = Counter(r["provisional_category"] for r in rows)
    by_ensemble = defaultdict(Counter)
    by_radius_ensemble = defaultdict(Counter)
    by_config = defaultdict(Counter)
    retained_ensemble = Counter()
    retained_radius_ensemble = Counter()
    start_errors, ensemble_errors, speed_errors, radius_errors = [], [], [], []
    radial_losses = axial_losses = both_losses = 0
    mechanisms_by_radius_ensemble = defaultdict(Counter)
    for r in rows:
        ci, si = int(r["screen_index"]), int(r["start_index"])
        c = configs[ci]
        ensemble = str(ensembles[si])
        key = (str(c["radius_um"]), ensemble)
        by_ensemble[ensemble][r["provisional_category"]] += 1
        by_radius_ensemble[key][r["provisional_category"]] += 1
        by_config[ci][r["provisional_category"]] += 1
        retained_ensemble[ensemble] += int(bool(r["retained"]))
        retained_radius_ensemble[key] += int(bool(r["retained"]))
        if not np.array_equal(np.asarray(r["initial"]), starts[si]):
            start_errors.append([ci, si, "initial"])
        if not np.array_equal(np.asarray(r["phase"]), phases[si]):
            start_errors.append([ci, si, "phase"])
        if r["ensemble"] != ensemble:
            ensemble_errors.append([ci, si, r["ensemble"], ensemble])
        if not np.allclose(np.asarray(r["injection_speed_m_s"]), speed_data[si], rtol=0, atol=0):
            speed_errors.append([ci, si])
        if float(r["radius_um"]) != float(c["radius_um"]):
            radius_errors.append([ci, si, r["radius_um"], c["radius_um"]])
        if r["provisional_category"] == "WALL_OR_AXIAL_ESCAPE":
            a_over_R = c["physical"]["radius_m"] / meta["physical_model"]["R_m"]
            radial = max(r["r_max_over_R"]) + a_over_R >= 1.0
            axial = max(r["z_max_over_R"]) + a_over_R >= 2.0
            radial_losses += int(radial); axial_losses += int(axial)
            both_losses += int(radial and axial)
            mechanisms_by_radius_ensemble[key]["radial"] += int(radial)
            mechanisms_by_radius_ensemble[key]["axial"] += int(axial)
            mechanisms_by_radius_ensemble[key]["both"] += int(radial and axial)

    duplicates = [list(k) for k, n in Counter(seen).items() if n != 1]
    missing = [list(k) for k in sorted(expected - set(seen))]
    extras = [list(k) for k in seen if k not in expected]
    source = ROOT / "sobol_alumina_experimental_50hz_search.py"
    source_copy = OUT / "campaign_runner_source.py"
    hashes = {"runner_sha256": sha(source), "saved_runner_sha256": sha(source_copy),
        "integrator_sha256": sha(ROOT / "model" / "basin_scan.py"),
        "field_model_sha256": sha(ROOT / "model" / "field_model.json")}
    config_summary = {"n_configs": nconfig,
        "n_per_radius": dict(Counter(str(c["radius_um"]) for c in configs)),
        "f1_Hz_values": sorted(set(c["physical"]["f1_Hz"] for c in configs)),
        "charge_e_min": min(c["physical"]["charge_e_each"] for c in configs),
        "charge_e_max": max(c["physical"]["charge_e_each"] for c in configs),
        "U_sum_peak_max_V": max(c["physical"]["U_sum_peak_bound_V"] for c in configs),
        "MFS_triangle_bound_max_V_m": max(c["physical"]["MFS_separate_basis_triangle_bound_V_m"] for c in configs)}
    outcomes_by_radius_ensemble = {}
    for (radius, ensemble), z in sorted(by_radius_ensemble.items(), key=lambda x: (float(x[0][0]), x[0][1])):
        n = sum(z.values())
        outcomes_by_radius_ensemble[f"{radius}um_{ensemble}"] = {
            "rows": n, "counts": dict(sorted(z.items())),
            "pair_retained_count": retained_radius_ensemble[(radius, ensemble)],
            "pair_retained_fraction": retained_radius_ensemble[(radius, ensemble)] / n,
            "loss_mechanisms": dict(mechanisms_by_radius_ensemble[(radius, ensemble)])}
    audit = {
        "audit": "integrity and grouped outcome audit; short screen is not probability validation",
        "expected_rows": len(expected), "actual_rows": len(rows), "unique_keys": len(set(seen)),
        "missing_keys": missing, "duplicate_keys": duplicates, "unexpected_keys": extras,
        "frozen_start_or_phase_mismatches": start_errors,
        "ensemble_mismatches": ensemble_errors, "injection_speed_mismatches": speed_errors,
        "row_radius_mismatches": radius_errors,
        "outcome_counts": dict(sorted(counts.items())),
        "outcomes_by_ensemble": {k: dict(sorted(v.items())) for k, v in sorted(by_ensemble.items())},
        "pair_retained_counts_by_ensemble": dict(retained_ensemble),
        "outcomes_by_radius_and_ensemble": outcomes_by_radius_ensemble,
        "wall_or_axial_escape_mechanisms": {"radial": radial_losses, "axial": axial_losses, "both": both_losses},
        "configurations_with_both_CR_directions": sum(
            z.get("PROVISIONAL_CR_PLUS_MINUS", 0) > 0 and
            z.get("PROVISIONAL_CR_MINUS_PLUS", 0) > 0 for z in by_config.values()),
        "physical_config_summary": config_summary, "code_sha256": hashes,
        "integrity_pass": (len(rows) == len(expected) and len(set(seen)) == len(expected)
            and not missing and not duplicates and not extras and not start_errors
            and not ensemble_errors and not speed_errors and not radius_errors
            and all(sum(by_config[i].values()) == nstarts for i in range(nconfig))
            and hashes["runner_sha256"] == hashes["saved_runner_sha256"]
            and config_summary["f1_Hz_values"] == [50.0]
            and config_summary["charge_e_min"] >= 1e5
            and config_summary["charge_e_max"] <= 5e5
            and config_summary["U_sum_peak_max_V"] <= 4400.0)}
    (OUT / "independent_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("expected_rows", "actual_rows", "unique_keys",
        "outcome_counts", "outcomes_by_ensemble", "pair_retained_counts_by_ensemble",
        "configurations_with_both_CR_directions", "physical_config_summary", "code_sha256", "integrity_pass")},
        indent=2, ensure_ascii=False))
    if not audit["integrity_pass"]:
        raise SystemExit("audit failed; see independent_audit.json")


if __name__ == "__main__":
    main()
