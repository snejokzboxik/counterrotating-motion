"""Independent audit of the RK4/DOP853 representative convergence run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "numerical_convergence_refine12"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    results = summary["results"]
    keys = ("RK4_64", "RK4_96", "RK4_128", "DOP853")
    methods_present = all(k in results for k in keys)
    labels = {k: results[k]["strict_label"] for k in keys if k in results}
    turns = {k: np.asarray(results[k]["turns"], dtype=float) for k in keys if k in results}
    finite = all(np.all(np.isfinite(v)) for v in turns.values())
    max_delta_64_128 = float(np.max(np.abs(turns["RK4_64"] - turns["RK4_128"]))) if methods_present else float("nan")
    max_delta_128_dop = float(np.max(np.abs(turns["RK4_128"] - turns["DOP853"]))) if methods_present else float("nan")
    max_window_delta_128_dop = float(np.max(np.abs(np.asarray(results["RK4_128"]["window_turns"]) - np.asarray(results["DOP853"]["window_turns"])))) if methods_present else float("nan")
    data = np.load(OUT / "dop853_representative_trajectory.npz")
    archive_shape = list(data["y"].shape)
    start_match = np.array_equal(data["initial"], np.asarray(summary["selected_start"]["initial"])) and np.array_equal(data["phase"], np.asarray(summary["selected_start"]["phase"]))
    post_idx = int(summary["skip_periods"] * results["DOP853"]["sample_points_per_main_rf_period"])
    dop_theta = [np.unwrap(np.arctan2(data["y"][1], data["y"][0])), np.unwrap(np.arctan2(data["y"][7], data["y"][6]))]
    reconstructed_dop_turns = np.asarray([(dop_theta[i][-1] - dop_theta[i][post_idx])/(2*np.pi) for i in range(2)])
    source = summary["source_hashes"]
    checks = {
        "all_four_integrators_present": methods_present,
        "all_metrics_finite": bool(finite),
        "all_methods_same_strict_CR_label": len(set(labels.values())) == 1 and next(iter(labels.values())) == "STRICT_CR_PLUS_MINUS",
        "all_RK4_runs_complete_horizon": all(results[k]["retained"] and results[k]["completed_periods"] == summary["integration_horizon_periods"] for k in ("RK4_64", "RK4_96", "RK4_128")) if methods_present else False,
        "DOP853_successful_and_full_samples": results.get("DOP853", {}).get("success", False) and archive_shape[1] == summary["integration_horizon_periods"] * 8 + 1,
        "DOP853_archive_initial_and_phase_match": bool(start_match),
        "DOP853_post_transient_turns_reconstruct_from_raw_archive": bool(np.allclose(reconstructed_dop_turns, turns["DOP853"], rtol=0, atol=1e-10)),
        "RK4_64_96_128_turn_metrics_converge": max_delta_64_128 < 0.01,
        "RK4_128_and_DOP853_total_and_window_metrics_agree": max_delta_128_dop < 0.01 and max_window_delta_128_dop < 0.05,
        "integrator_hash_matches": source["integrator"] == sha(ROOT / "model" / "basin_scan.py"),
        "field_hash_matches": source["field_model"] == sha(ROOT / "model" / "field_model.json"),
        "source_long_horizon_summary_hash_matches": source["long_horizon_summary"] == sha(ROOT / "results" / "long_horizon_refine12_240k_validation" / "summary.json"),
        "executed_runner_hash_matches_saved_copy": source["executed_runner_sha256"] == sha(OUT / "campaign_runner_source.py"),
        "corrected_runner_hash_matches": source["runner"] == sha(ROOT / "numerical_convergence_refine12.py"),
        "metric_correction_hash_matches": source["metric_correction_script_sha256"] == sha(ROOT / "model" / "correct_numerical_convergence_refine12_metrics.py"),
    }
    report = {"audit": "numerical_convergence_refine12", "integrity_pass": all(checks.values()), "checks": checks,
              "method_labels": labels, "max_turn_difference_RK64_vs_RK128": max_delta_64_128,
              "max_turn_difference_RK128_vs_DOP853": max_delta_128_dop,
              "max_late_window_turn_difference_RK128_vs_DOP853": max_window_delta_128_dop,
              "DOP853_trajectory_shape": archive_shape,
              "trajectory_sha256": sha(OUT / "dop853_representative_trajectory.npz"),
              "audit_script_sha256": sha(Path(__file__))}
    (OUT / "independent_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
