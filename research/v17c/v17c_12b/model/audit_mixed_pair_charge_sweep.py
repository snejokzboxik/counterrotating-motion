"""Audit the paired charge/frequency screen and dimensional rescaling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mixed_pair_charge_sweep"
BASE = ROOT / "results" / "mixed_alumina_k1_candidate"
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
    codes = json.loads((OUT / "campaign_code_sha256.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    base_meta = json.loads((BASE / "campaign_meta.json").read_text(encoding="utf-8"))
    frozen = np.load(OUT / "frozen_starts.npz", allow_pickle=True)
    base_frozen = np.load(BASE / "frozen_starts.npz", allow_pickle=True)
    rows = [json.loads(line) for line in (OUT / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line]
    configs = meta["configurations"]
    groups = ["rest", "weak", "wide", "edge"]
    n_each = 128
    expected = {(c, g, i) for c in configs for g in groups for i in range(n_each)}
    keys = {(str(r["configuration"]), str(r["ensemble"]), int(r["start_index"])) for r in rows}
    starts_match = np.array_equal(frozen["initial"], base_frozen["initial"]) and np.array_equal(frozen["phases"], base_frozen["phases"])
    labels_match = np.array_equal(frozen["ensemble"], base_frozen["ensemble"])
    each_start_match = True
    full_horizons = True
    short_exits = True
    loss_mechanisms = {name: {"radial": 0, "axial": 0, "both": 0} for name in configs}
    reconstruction = {}
    for name, cfg in configs.items():
        p = cfg["p"]
        ph = cfg["physical"]
        a = ph["particle1"]
        b = ph["particle2"]
        om = ph["omega1_rad_s"]
        reconstruction[name] = {
            "mass_ratio": a["mass_kg"] / b["mass_kg"],
            "charge_ratio": b["charge_C"] / a["charge_C"],
            "C": a["charge_C"] * b["charge_C"] / (4*np.pi*EPS0*a["mass_kg"]*R**3*om**2),
            "lambda1": 6*np.pi*ETA*a["radius_m"] / (a["mass_kg"]*om),
            "lambda2": 6*np.pi*ETA*b["radius_m"] / (b["mass_kg"]*om),
            "q1_e": a["charge_e"], "q2_e": b["charge_e"],
            "f1_Hz": ph["frequency_f1_Hz"], "f2_Hz": ph["frequency_f2_Hz"],
            "U1_V": ph["U1_peak_electrode_V"], "U2_V": ph["U2_peak_electrode_V"],
            "Re1_dimless_speed": a["Re_per_dimensionless_speed"],
            "Re2_dimless_speed": b["Re_per_dimensionless_speed"],
        }
        rec = reconstruction[name]
        cfg_checks = {
            "mass_ratio": np.isclose(rec["mass_ratio"], p[4], rtol=1e-12),
            "charge_ratio": np.isclose(rec["charge_ratio"], cfg["qratio"], rtol=1e-12),
            "C": np.isclose(rec["C"], p[5], rtol=1e-12),
            "lambda1": np.isclose(rec["lambda1"], p[6], rtol=1e-12),
            "lambda2": np.isclose(rec["lambda2"], p[7], rtol=1e-12),
            "f1_scale_matches_charge": np.isclose(ph["frequency_f1_Hz"] / 50.0, a["charge_e"] / 100000.0, rtol=1e-12),
            "f2_nu_relation": np.isclose(ph["frequency_f2_Hz"] / ph["frequency_f1_Hz"], p[2], rtol=1e-12),
            "charges_in_declared_range": 1e5 <= a["charge_e"] <= 1e6 and 1e5 <= b["charge_e"] <= 1e6,
        }
        reconstruction[name]["checks"] = {k: bool(v) for k, v in cfg_checks.items()}
    for row in rows:
        c, group, si = row["configuration"], row["ensemble"], int(row["start_index"])
        idx = groups.index(group)*n_each + si
        each_start_match &= np.allclose(row["initial"], frozen["initial"][idx], rtol=0, atol=1e-14)
        each_start_match &= np.allclose(row["phase"], frozen["phases"][idx], rtol=0, atol=1e-14)
        full_horizons &= (not row["retained"]) or row["completed_periods"] == meta["periods"]
        short_exits &= row["retained"] or row["completed_periods"] < meta["periods"]
        if not row["retained"]:
            ph = configs[c]["physical"]
            radial = max(row["r_max_over_R"][0] + ph["a1_over_R"], row["r_max_over_R"][1] + ph["a2_over_R"]) >= 1.0
            axial = max(row["z_max_over_R"][0] + ph["a1_over_R"], row["z_max_over_R"][1] + ph["a2_over_R"]) >= 2.0
            loss_mechanisms[c]["radial"] += int(radial)
            loss_mechanisms[c]["axial"] += int(axial)
            loss_mechanisms[c]["both"] += int(radial and axial)
    expected_outcomes = {}
    for name in configs:
        expected_outcomes[name] = {
            "rows": sum(r["configuration"] == name for r in rows),
            "outcomes": {},
            "retained_by_ensemble": {},
        }
        local = [r for r in rows if r["configuration"] == name]
        from collections import Counter
        expected_outcomes[name]["outcomes"] = dict(sorted(Counter(r["provisional_category"] for r in local).items()))
        expected_outcomes[name]["retained_by_ensemble"] = {g: sum(bool(r["retained"]) for r in local if r["ensemble"] == g) for g in groups}
    checks = {
        "expected_row_count": len(rows) == len(expected),
        "unique_complete_config_ensemble_start_keys": keys == expected,
        "starts_exactly_reused_from_parent": bool(starts_match and labels_match and each_start_match),
        "retained_full_horizons_and_early_exits_consistent": bool(full_horizons and short_exits),
        "summary_matches_rows": expected_outcomes == summary["by_configuration"],
        "summary_total_matches": summary["rows"] == len(rows) == meta["total_trajectories"],
        "trajectory_sha_matches_summary": sha(OUT / "trajectories.jsonl") == summary["trajectory_sha256"],
        "runner_sha_matches_archived_copy": sha(ROOT / "physical_mixed_pair_charge_frequency_sweep.py") == sha(OUT / "campaign_runner_source.py") == codes["runner_sha256"] == codes["saved_runner_sha256"],
        "integrator_sha_matches_archived_copy": sha(ROOT / "model" / "basin_scan.py") == sha(OUT / "campaign_integrator_source.py") == codes["integrator_sha256"] == codes["saved_integrator_sha256"],
        "field_model_sha_matches": sha(ROOT / "model" / "field_model.json") == codes["field_model_sha256"],
        "parent_start_sha_matches": sha(BASE / "frozen_starts.npz") == codes["parent_starts_sha256"] == meta["parent_starts_sha256"] == base_meta["starts_sha256"],
        "all_dimensional_reconstructions_pass": all(all(x["checks"].values()) for x in reconstruction.values()),
        "parent_and_child_start_arrays_match": bool(starts_match and labels_match),
        "all_observed_losses_are_radial": all(x["radial"] == expected_outcomes[name]["rows"] - sum(expected_outcomes[name]["retained_by_ensemble"].values()) and x["axial"] == 0 for name, x in loss_mechanisms.items()),
        "only_provisional_co_minus_minus_or_wall_losses": all(set(expected_outcomes[name]["outcomes"]).issubset({"PROVISIONAL_CO_MINUS_MINUS", "WALL_OR_AXIAL_ESCAPE"}) for name in configs),
    }
    report = {"audit": "mixed_pair_charge_frequency_sweep", "integrity_pass": all(checks.values()),
        "checks": checks, "rows": len(rows), "unique_keys": len(keys),
        "by_configuration": expected_outcomes, "loss_mechanisms": loss_mechanisms,
        "reconstructed_values": reconstruction,
        "interpretation": "Paired screening with shared frozen starts; no IID basin-probability inference.",
        "trajectory_sha256": sha(OUT / "trajectories.jsonl"),
        "audit_script_sha256": sha(Path(__file__))}
    (OUT / "independent_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
