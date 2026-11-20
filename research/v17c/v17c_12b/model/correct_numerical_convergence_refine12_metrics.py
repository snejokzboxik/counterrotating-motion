"""Reconstruct post-transient DOP853 orbit metrics from its saved raw trajectory."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "numerical_convergence_refine12"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
import physical_mixed_pair_search as base  # noqa: E402
import updated_objective_validation as labels  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    archive_path = OUT / "dop853_representative_trajectory.npz"
    archive = np.load(archive_path)
    correction_path = OUT / "metric_correction.json"
    t, y = archive["t"], archive["y"]
    periods = int(summary["integration_horizon_periods"])
    skip = int(summary["skip_periods"])
    samples_per_period = int(summary["results"]["DOP853"]["sample_points_per_main_rf_period"])
    post_idx = skip * samples_per_period
    theta = [np.unwrap(np.arctan2(y[1], y[0])), np.unwrap(np.arctan2(y[7], y[6]))]
    corrected = [(float(theta[i][-1] - theta[i][post_idx]) / (2*np.pi)) for i in range(2)]
    windows = np.asarray(summary["results"]["DOP853"]["window_turns"], dtype=float)
    frequencies = [x / max(1, periods - skip) for x in corrected]
    source_meta = json.loads((ROOT / "results" / "rescaled_legacy_candidate_validation" / "campaign_meta.json").read_text(encoding="utf-8"))
    candidate = next(c for c in source_meta["candidates"] if c["id"] == summary["configuration"])
    re_factors = [candidate["physical"][f"particle{i+1}"]["Re_per_dimensionless_speed"] for i in range(2)]
    speed_max = [float(np.max(np.sqrt(y[3]**2+y[4]**2+y[5]**2)),),
                 float(np.max(np.sqrt(y[9]**2+y[10]**2+y[11]**2)),)]
    re_max_sampled = [speed_max[i] * re_factors[i] for i in range(2)]
    rmin = summary["results"]["DOP853"]["r_min_over_R"]
    strict = labels.strict_late_label("ok", np.asarray(corrected), windows, True,
                                      min(rmin) < base.sim.RMIN_CLASS)
    old_runner = summary["source_hashes"].get("executed_runner_sha256", summary["source_hashes"]["runner"])
    if correction_path.exists():
        old_turns = json.loads(correction_path.read_text(encoding="utf-8"))["previous_DOP853_turns_including_transient"]
    else:
        old_turns = summary["results"]["DOP853"]["turns"]
    summary["results"]["DOP853"]["turns"] = corrected
    summary["results"]["DOP853"]["frequency_over_rf"] = frequencies
    summary["results"]["DOP853"]["strict_label"] = strict
    summary["results"]["DOP853"]["turn_accounting"] = "post-transient only; matches RK4 skip=10000 periods"
    summary["results"]["DOP853"]["max_dimensionless_speed_sampled"] = speed_max
    summary["results"]["DOP853"]["max_Re_by_particle_sampled"] = re_max_sampled
    summary["source_hashes"]["executed_runner_sha256"] = old_runner
    summary["source_hashes"]["runner"] = sha(ROOT / "numerical_convergence_refine12.py")
    summary["source_hashes"]["metric_correction_script_sha256"] = sha(Path(__file__))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    correction = {
        "correction": "DOP853 total turns originally included the first 10000 transient RF periods, while RK4 turn totals excluded them.",
        "trajectory_reintegrated": False,
        "raw_DOP853_trajectory_sha256": sha(archive_path),
        "previous_DOP853_turns_including_transient": old_turns,
        "corrected_post_transient_DOP853_turns": corrected,
        "corrected_post_transient_frequencies_over_rf": frequencies,
        "sampled_max_Re_by_particle": re_max_sampled,
        "window_turns_unchanged": windows.tolist(),
        "strict_label_after_correction": strict,
        "executed_runner_sha256": old_runner,
        "corrected_runner_sha256": summary["source_hashes"]["runner"],
        "correction_script_sha256": sha(Path(__file__)),
    }
    correction_path.write_text(json.dumps(correction, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(correction, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
