"""Strict checks for V17C.22 JSON, plots, and mandatory result artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research/v17c/v17c_22"


def main() -> None:
    json_files = list(BASE.rglob("*.json"))
    assert json_files, "no JSON artifacts found"
    for path in json_files:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        # allow_nan=False rejects Infinity/NaN on canonical reserialization.
        json.dumps(obj, ensure_ascii=False, allow_nan=False)
    required = [
        "results/stability_evidence_audit.json",
        "results/short_dop853_perturbation_summary.json",
        "results/short_dop853_perturbation_trajectories.npz",
        "results/full_rf_jacobian_fd_test.json",
        "results/conditional_qr_horizon_spectrum.csv",
        "results/phase_locked_perturbation_reanalysis.csv",
    ]
    for rel in required:
        p = BASE / rel
        assert p.exists() and p.stat().st_size > 0, f"missing/empty {rel}"
    stems = [
        "FIG01_lyapunov_and_m3_spectra",
        "FIG02_phase_locked_perturbation_growth",
        "FIG03_phase_dephasing_vs_observables",
        "FIG04_timescale_and_cutoff_audit",
        "FIG05_short_full_rf_perturbation_response",
        "FIG06_observed_invariant_set_projection",
        "FIG07_m3_vs_full_rf_stability_spectra",
    ]
    for stem in stems:
        for ext in ("pdf", "svg", "png"):
            p = BASE / "figures" / f"{stem}.{ext}"
            assert p.exists() and p.stat().st_size > 1024, f"missing/empty figure {p.name}"
    test = json.loads((BASE / "results/full_rf_jacobian_fd_test.json").read_text(encoding="utf-8"))
    assert test["pass"] and test["max_scaled_error"] < 2e-7
    audit = json.loads((BASE / "results/stability_evidence_audit.json").read_text(encoding="utf-8"))
    assert audit["numerical_conclusion"]["attractor_or_torus_proved"] is False
    assert audit["conditional_qr"]["sum_trace_absolute_difference_per_tau"] < 1e-4
    print(f"PASS: {len(json_files)} strict JSON files, {len(required)} data artifacts, {len(stems)*3} figure files")


if __name__ == "__main__":
    main()
