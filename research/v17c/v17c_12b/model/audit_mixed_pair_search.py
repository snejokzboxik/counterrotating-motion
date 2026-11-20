"""Integrity and dimensional-consistency audit for the mixed-particle screen."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mixed_alumina_k1_candidate"
EPS0 = 8.8541878128e-12
ETA = 1.81e-5
R = 0.010


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
    frozen = np.load(OUT / "frozen_starts.npz")
    starts, phases = frozen["initial"], frozen["phases"]
    labels = frozen["ensemble"].astype(str).tolist()
    rows = [json.loads(line) for line in (OUT / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ens_names = meta["ensembles"]
    n_each = int(meta["starts_per_ensemble"])
    ph = meta["configuration"]["physical"]
    expected = {(e, i) for e in ens_names for i in range(n_each)}
    keys = [(str(r["ensemble"]), int(r["start_index"])) for r in rows]
    unique = set(keys)
    categories = Counter(r["provisional_category"] for r in rows)
    by_ensemble = defaultdict(Counter)
    match_starts = True
    retained_full_horizon = True
    early_exit_consistent = True
    retained_flags = True
    radial_losses = axial_losses = both_losses = 0
    for r in rows:
        e, si = str(r["ensemble"]), int(r["start_index"])
        idx = ens_names.index(e) * n_each + si
        match_starts &= labels[idx] == e
        match_starts &= np.allclose(np.asarray(r["initial"]), starts[idx], rtol=0.0, atol=1e-14)
        match_starts &= np.allclose(np.asarray(r["phase"]), phases[idx], rtol=0.0, atol=1e-14)
        retained_full_horizon &= (not bool(r["retained"])) or int(r["completed_periods"]) == int(meta["periods"])
        early_exit_consistent &= bool(r["retained"]) or int(r["completed_periods"]) < int(meta["periods"])
        retained_flags &= bool(r["retained"]) == (int(r["completed_periods"]) == int(meta["periods"]))
        by_ensemble[e][r["provisional_category"]] += 1
        if r["provisional_category"] == "WALL_OR_AXIAL_ESCAPE":
            radial = max(r["r_max_over_R"][0] + ph["a1_over_R"], r["r_max_over_R"][1] + ph["a2_over_R"]) >= 1.0
            axial = max(r["z_max_over_R"][0] + ph["a1_over_R"], r["z_max_over_R"][1] + ph["a2_over_R"]) >= 2.0
            radial_losses += int(radial); axial_losses += int(axial); both_losses += int(radial and axial)

    cfg = meta["configuration"]
    p = cfg["p"]
    ph = cfg["physical"]
    p1, p2 = ph["particle1"], ph["particle2"]
    m1, m2 = p1["mass_kg"], p2["mass_kg"]
    q1, q2 = p1["charge_C"], p2["charge_C"]
    a1, a2 = p1["radius_m"], p2["radius_m"]
    omega = ph["omega1_rad_s"]
    calculated = {
        "mass_ratio": m1 / m2,
        "charge_ratio": q2 / q1,
        "coupling": q1*q2/(4*np.pi*EPS0*m1*R**3*omega**2),
        "lambda1": 6*np.pi*ETA*a1/(m1*omega),
        "lambda2": 6*np.pi*ETA*a2/(m2*omega),
    }
    reconstruction_checks = {
        "mass_ratio_matches_dimensionless_parameter": np.isclose(calculated["mass_ratio"], p[4], rtol=1e-12),
        "charge_ratio_matches_integrator": np.isclose(calculated["charge_ratio"], cfg["qratio"], rtol=1e-12),
        "C_matches_dimensional_equation": np.isclose(calculated["coupling"], p[5], rtol=1e-12),
        "lambda1_matches_stokes": np.isclose(calculated["lambda1"], p[6], rtol=1e-12),
        "lambda2_matches_stokes": np.isclose(calculated["lambda2"], p[7], rtol=1e-12),
        "U1_reconstructs_A": np.isclose(q1*ph["U1_peak_electrode_V"]*0.870316891091623/(m1*R**2*omega**2), p[0], rtol=1e-12),
        "U2_reconstructs_B": np.isclose(q1*ph["U2_peak_electrode_V"]*0.7982172624273378/(m1*R**2*omega**2), p[1], rtol=1e-12),
        "DC_coefficient_reconstructs_beta": np.isclose(q1*ph["V2_ideal_DC_coefficient_V_not_endcap"]/(m1*R**2*omega**2), p[3], rtol=1e-12),
        "derived_K1_density_within_spec": 100 <= p2["density_kg_m3"] <= 140,
        "both_charges_within_2013_range": 1e5 <= p1["charge_e"] <= 1e6 and 1e5 <= p2["charge_e"] <= 1e6,
    }
    reconstruction_checks = {key: bool(value) for key, value in reconstruction_checks.items()}
    runner = ROOT / "physical_mixed_pair_search.py"
    checks = {
        "expected_row_count": len(rows) == len(expected),
        "unique_complete_ensemble_start_keys": unique == expected,
        "frozen_starts_phases_and_ensemble_labels_match": bool(match_starts),
        "all_retained_trajectories_reached_full_horizon": bool(retained_full_horizon),
        "early_exit_categories_have_short_horizons": bool(early_exit_consistent),
        "retention_flags_consistent_with_terminal_categories": bool(retained_flags),
        "runner_hash_matches_saved_copy_and_campaign": sha(runner) == sha(OUT / "campaign_runner_source.py") == code["runner_sha256"],
        "integrator_hash_matches_archived_campaign_source": sha(OUT / "campaign_integrator_source.py") == code["integrator_sha256"] == code["saved_integrator_sha256"],
        "field_hash_matches_campaign": sha(ROOT / "model" / "field_model.json") == code["field_model_sha256"],
        "summary_outcomes_match_rows": summary["outcome_counts"] == dict(sorted(categories.items())),
        "summary_trajectory_hash_matches": summary["trajectory_sha256"] == sha(OUT / "trajectories.jsonl"),
        "summary_per_ensemble_counts_match": summary["outcomes_by_ensemble"] == {k: dict(sorted(v.items())) for k,v in sorted(by_ensemble.items())},
        **reconstruction_checks,
    }
    report = {
        "audit": "physical_mixed_pair_search",
        "integrity_pass": all(checks.values()),
        "checks": checks,
        "reconstructed_dimensionless_values": calculated,
        "rows": len(rows), "unique_keys": len(unique),
        "rows_per_ensemble": {k: sum(v.values()) for k,v in sorted(by_ensemble.items())},
        "loss_mechanisms": {"radial": radial_losses, "axial": axial_losses, "both": both_losses},
        "outcome_counts": dict(sorted(categories.items())),
        "outcomes_by_ensemble": {k: dict(sorted(v.items())) for k,v in sorted(by_ensemble.items())},
        "interpretation": "Literature-constrained fixed-pair scenario; not a directly measured joint particle record and not a final basin-probability study.",
        "trajectory_sha256": sha(OUT / "trajectories.jsonl"),
        "audit_script_sha256": sha(Path(__file__)),
    }
    (OUT / "independent_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
