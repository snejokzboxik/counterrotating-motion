"""Audit full-horizon RK4 refinement and independent late DOP853 evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    ref_dir = HERE / "p1_rk4_refinement"
    rows = [json.loads(line) for line in
            (ref_dir / "refinement_runs.jsonl").read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["steps_per_period"])
    expected = [64, 128, 256]
    assert [r["steps_per_period"] for r in rows] == expected
    for row in rows:
        assert row["retained"] and row["failure_status"] == 0
        assert row["completed_periods"] == 15203
        assert row["post_load_periods_observed"] == 15000
        assert row["structure_after_load"]["opposite_group_frequency_signs"]
        assert row["structure_after_load"]["all_particles_at_least_10_turns"]
        assert row["structure_after_load"]["directions_constant_in_three_windows"]
        file = ref_dir / f"{row['task_id']}_trajectory.npz"
        assert file.exists() and file.stat().st_size > 1_000_000
        with np.load(file) as data:
            assert data["trajectory"].shape[0] == 15204
            assert np.isfinite(data["trajectory"]).all()
            assert np.isfinite(data["energy_balance"]).all()

    dop = json.loads((HERE / "p1_dop853" / "dop853_summary.json").read_text())
    assert dop["status"] == "passed"
    assert dop["segment_primary_periods"] == 3000
    assert dop["independent_rhs_max_abs_difference_at_start"] < 1e-11
    assert dop["max_abs_frequency_difference_Hz"] < 0.01
    assert max(dop["dop853_metrics"]["turns_each"]) > 0
    assert min(dop["dop853_metrics"]["turns_each"]) < 0

    medium, fine = rows[1], rows[2]
    freq128 = np.asarray([g["mean_frequency_Hz"] for g in medium["structure_after_load"]["groups"]])
    freq256 = np.asarray([g["mean_frequency_Hz"] for g in fine["structure_after_load"]["groups"]])
    result = {
        "status": "passed",
        "rk4_full_horizon": {
            "post_load_periods": 15000,
            "resolutions_steps_per_period": expected,
            "all_six_retained_at_each_resolution": True,
            "counterrotation_in_three_late_windows_at_each_resolution": True,
            "minimum_turns_over_all_particles_by_resolution": {
                str(r["steps_per_period"]): min(abs(x) for x in r["turns_each_after_load"])
                for r in rows
            },
            "mean_frequency_Hz_by_group": {
                str(r["steps_per_period"]): [g["mean_frequency_Hz"]
                                              for g in r["structure_after_load"]["groups"]]
                for r in rows
            },
            "max_abs_frequency_change_128_to_256_Hz": float(np.max(np.abs(freq256 - freq128))),
            "mean_radius_R_by_group": {
                str(r["steps_per_period"]): [float(np.mean(g["mean_radius_R_each"]))
                                              for g in r["structure_after_load"]["groups"]]
                for r in rows
            },
            "mean_Psi3_by_group": {
                str(r["steps_per_period"]): [g["mean_Psi3"]
                                              for g in r["structure_after_load"]["groups"]]
                for r in rows
            },
            "global_minimum_pair_distance_R_by_resolution": {
                str(r["steps_per_period"]): min(r["minimum_pair_distances_R_i_lt_j"])
                for r in rows
            },
            "relative_energy_residual_by_resolution": {
                str(r["steps_per_period"]): r["energy_balance"]["relative_max_residual_vs_RF_work"]
                for r in rows
            },
            "integrated_energy_residual_decreases_with_refinement": (
                rows[0]["energy_balance"]["relative_max_residual_vs_RF_work"] >
                rows[1]["energy_balance"]["relative_max_residual_vs_RF_work"] >
                rows[2]["energy_balance"]["relative_max_residual_vs_RF_work"]
            ),
        },
        "independent_dop853_late_segment": {
            "status": dop["status"],
            "segment_primary_periods": dop["segment_primary_periods"],
            "nfev": dop["nfev"],
            "rtol": dop["rtol"], "atol": dop["atol"],
            "max_step_tau": dop["max_step_tau"],
            "independent_rhs_max_abs_difference_at_start": dop["independent_rhs_max_abs_difference_at_start"],
            "max_abs_state_difference_over_segment": dop["max_abs_state_difference_over_sampled_segment"],
            "max_abs_frequency_difference_Hz": dop["max_abs_frequency_difference_Hz"],
            "rk4_late_mean_frequency_Hz_by_group": dop["rk4_reference_metrics"]["mean_frequency_Hz_by_group"],
            "dop853_late_mean_frequency_Hz_by_group": dop["dop853_metrics"]["mean_frequency_Hz_by_group"],
            "dop853_turns_each_over_late_segment": dop["dop853_metrics"]["turns_each"],
            "dop853_mean_radius_R_by_particle": dop["dop853_metrics"]["mean_radius_R_by_particle"],
            "dop853_mean_Psi3_by_group": dop["dop853_metrics"]["mean_Psi3_by_group"],
        },
        "limitations": [
            "DOP853 independently checks only a late 3000-period segment, not the whole 15000-period horizon.",
            "The full-horizon RK4 frequency statistics are converged at 128/256 steps; RK4/64 shows a visible finite-step frequency bias.",
            "The DOP853 comparison begins from the same saved RK4/96 state, so it validates local continuation rather than independent loading history.",
        ],
    }
    result["status"] = "passed" if result["rk4_full_horizon"]["integrated_energy_residual_decreases_with_refinement"] else "failed"
    (HERE / "p1_numerical_validation_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "max_abs_frequency_change_128_to_256_Hz": result["rk4_full_horizon"]["max_abs_frequency_change_128_to_256_Hz"],
        "energy_residuals": result["rk4_full_horizon"]["relative_energy_residual_by_resolution"],
        "dop853_frequency_diff_Hz": dop["max_abs_frequency_difference_Hz"],
        "dop853_state_diff": dop["max_abs_state_difference_over_sampled_segment"],
    }, indent=2))


if __name__ == "__main__":
    main()
