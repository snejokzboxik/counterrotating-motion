"""Independent numeric controls for the closed leading-harmonic formulas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/v17c/v17c_21/code"))
from analytical_model import load_inputs, mfs_rf_means  # noqa: E402
from theory.radial_localization import leading_radial_force  # noqa: E402
from theory.torque_analysis import leading_mode_torques  # noqa: E402


def test_closed_torque_and_radial_force_against_archived_m2():
    data = load_inputs()
    for i in (0, 1):
        alpha, lam, beta = data["alpha"][i], data["lambda"][i], data["beta"]
        for r in (0.42, 0.58, 0.74):
            for mode in [m for m in data["modes"] if m.k in (2, -4)]:
                force, components = mfs_rf_means(r, 0.0, alpha, lam, beta, [mode], [0])
                component = components[0]
                analytic = leading_mode_torques(r, alpha, lam, beta,
                                                mode.amplitude, mode.coefficient,
                                                mode.frequency, mode.k)
                for computed_key, expected_key in (
                    ("mean_total_RF_torque", "total_RF_torque"),
                    ("mean_rf_torque_about_mean_orbit", "mean_orbit_RF_torque"),
                    ("micromotion_RF_torque", "micromotion_RF_torque"),
                    ("micromotion_angular_momentum", "micromotion_angular_momentum"),
                ):
                    assert np.isclose(component[computed_key], analytic[expected_key],
                                      rtol=3e-13, atol=1e-18)
                radial = leading_radial_force(r, alpha, lam, beta, mode.amplitude,
                                              mode.frequency, mode.n, mode.coefficient)
                assert np.isclose(force[0], radial, rtol=3e-13, atol=1e-18)


def test_fresh_m3_decomposition_and_holdout_radial_prediction():
    d = json.loads((ROOT / "data/m3_nominal_torque_decomposition.json").read_text())
    radial = json.loads((ROOT / "data/radial_localization_validation.json").read_text())
    for p in d["particles"]:
        assert p["m3_inner_converged"]
        assert p["m3_inner_relative_residual"] < 5e-5
        assert abs(sum(x["torque"] for x in p["m3_modes"]) - p["m3_RF_total_torque"]) < 2e-13
        assert abs(p["m3_RF_total_torque"] + p["m3_drag_total_torque"]) < 4e-9
        assert p["m3_RF_family_torques"][0] > 0 > p["m3_RF_family_torques"][1]
    for row in radial["nominal"]:
        assert abs(row["M3_radial_sum"]) < 2e-10
        assert abs(row["leading_vs_full_RF_error_percent"]) < 0.35
    for row in radial["published_holdouts"]:
        assert max(abs(x) for x in row["leading_vs_full_RF_error_percent"]) < 0.35
