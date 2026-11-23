from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research" / "v17c" / "v17c_20a"
V20B = ROOT / "research" / "v17c" / "v17c_20b"
sys.path.insert(0, str(BASE / "code"))
sys.path.insert(0, str(V20B / "code"))
import run_long_qr as qr  # noqa: E402
import variational_qr as vq  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg = json.loads((V20B / "results" / "candidate_parameters.json").read_text(encoding="utf-8"))
    src = cfg["model_source"]
    source_checks = {}
    for key, filekey in (("field_sha256", "field"), ("integrator_sha256", "integrator"),
                         ("verified_pair_integrator_sha256", "verified_pair_integrator")):
        path = ROOT / src[filekey]
        found = sha(path)
        assert found == src[key], (key, found, src[key])
        source_checks[key] = found

    field = qr.a.load_field()
    jacobian_checks = []
    for case in cfg["candidates"]:
        j = vq.validate_jacobian(case)
        assert j["max_relative_jacobian_error"] < 1e-7
        jacobian_checks.append({"candidate": case["legacy_candidate"],
                                "max_relative_jacobian_error": j["max_relative_jacobian_error"],
                                "samples": j["samples"]})

    # Independently compare the new divergence-accumulating RK4 tangent step
    # against the frozen V20B RK4 augmented integrator over one RF1 period.
    _, case, p, q, y, phase, source_period, _ = qr.load_late_seed("scaled_refine12_sobol17_f50hz", 240000)
    M = np.eye(12); dt = 2.0 * np.pi / 32
    y1, m1, t1, div = qr.advance_with_divergence(y, M, 0.0, dt, 32, p, phase,
        field["family"], field["k"], field["coeff"], q, case["re1_per_speed"],
        case["re2_per_speed"], case["gravity_z"], case["a1_over_R"], case["a2_over_R"])
    y2, m2, t2 = vq.advance_augmented(y, M, 0.0, dt, 32, p, phase,
        field["family"], field["k"], field["coeff"], q, case["re1_per_speed"],
        case["re2_per_speed"], case["gravity_z"], case["a1_over_R"], case["a2_over_R"])
    rhs_agreement = {"state_max_abs": float(np.max(np.abs(y1-y2))),
                     "tangent_max_abs": float(np.max(np.abs(m1-m2))),
                     "time_abs": abs(t1-t2), "divergence_integral_finite": bool(np.isfinite(div))}
    assert rhs_agreement["state_max_abs"] < 1e-13 and rhs_agreement["tangent_max_abs"] < 1e-12

    lyap = json.loads((BASE / "results" / "lyapunov_horizon_fits.json").read_text(encoding="utf-8"))
    for candidate in ("candidate52", "local15", "refine12_sobol17"):
        horizons = lyap["fits"][candidate]["horizons_periods"]
        assert horizons == [2000, 5000, 10000, 20000]
        assert len(lyap["fits"][candidate]["extra_50k"]) == 12
        for r in lyap["fits"][candidate]["divergence_check"]:
            assert r["volume_balance_abs_error_per_tau"] < 1e-4
    ext = json.loads((BASE / "results" / "extended_qr" /
        "refine12_sobol17_240000p_seed__100p_transient__50000p_extqr__s32_q20" / "result.json").read_text(encoding="utf-8"))
    assert len(ext["spectrum_per_tau_descending"]) == 14
    assert abs(ext["spectrum_per_tau_descending"][0]) < 1e-12
    assert abs(ext["spectrum_per_tau_descending"][1]) < 1e-12
    assert sum(abs(x) < 1e-4 for x in ext["spectrum_per_tau_descending"]) == 4
    assert ext["volume_balance_abs_error_per_tau"] < 1e-4

    strobe_checks = []
    for candidate in ("candidate52", "local15", "refine12_sobol17"):
        files = list((BASE / "results" / "stroboscopic").glob(f"{candidate}_*every_rf1.npz"))
        assert len(files) == 1
        z = np.load(files[0])
        assert len(z["states"]) == int(z["period_index"][-1]) + 1
        meta = json.loads(files[0].with_suffix(".json").read_text(encoding="utf-8"))
        assert sum(n > 0 for n in meta["phase2_bin_histogram_n"]) == 36
        strobe_checks.append({"candidate": candidate, "samples": len(z["states"]), "phase_bins": 36})

    multistart = json.loads((BASE / "results" / "multistart" / "same_phase_multistart_summary.json").read_text(encoding="utf-8"))
    assert multistart["runs_total"] == 24
    assert sum(x["counterrotation_count"] for x in multistart["candidate_results"]) == 23
    perturb = list((BASE / "results" / "perturbations").glob("*_5000p_phase_locked.json"))
    assert len(perturb) == 3
    setdist = json.loads((BASE / "results" / "torus_diagnostics" / "phase_matched_set_distance.json").read_text(encoding="utf-8"))
    assert setdist["reference_samples"] == 20001
    coverage = json.loads((BASE / "results" / "rf_parameter_screen" / "PARAMETER_COVERAGE.json").read_text(encoding="utf-8"))
    assert coverage["total_unique_points"] == 145 and coverage["sobol_scored_rows"] == 32
    assert len(coverage["pairwise_slices"]) == 6
    adaptive = json.loads((BASE / "results" / "rf_parameter_screen" / "adaptive_boundary_results.json").read_text(encoding="utf-8"))
    assert adaptive["unique_points"] == 12 and len(adaptive["results"]) == 12
    qr_dir = BASE / "results" / "rf_parameter_screen"
    selected5_path = qr_dir / "selected_qr_5000p_results.json"
    # Preserve the original run artifact name used by the first completed batch.
    if not selected5_path.exists():
        selected5_path = qr_dir / "selected_qr_results.json"
    selected5 = json.loads(selected5_path.read_text(encoding="utf-8"))
    assert len(selected5["results"]) == 6
    assert all(r["event_resolved_replay_status"] == 0 and r["event_resolved_replay_completed_periods"] == 5000 for r in selected5["results"])
    if (BASE / "results" / "rf_parameter_screen" / "selected_qr_20000p_results.json").exists():
        selected20 = json.loads((BASE / "results" / "rf_parameter_screen" / "selected_qr_20000p_results.json").read_text(encoding="utf-8"))
        assert len(selected20["results"]) == 1 and selected20["results"][0]["event_resolved_replay_status"] == 0

    candidates = json.loads((BASE / "ATTRACTOR_CANDIDATES.json").read_text(encoding="utf-8"))
    assert candidates["base_commit"] == "d1bf252649e174166becf464269318b5f97acd5d"
    assert len(candidates["candidates"]) == 4
    coverage_root = json.loads((BASE / "PARAMETER_COVERAGE.json").read_text(encoding="utf-8"))
    assert coverage_root["variables_and_pairwise_screen"]["total_unique_points"] == 145
    required_reports = ["RF_ATTRACTORS_AND_INVARIANT_TORI_RU.md", "LYAPUNOV_LONG_HORIZON_RU.md",
                        "TORUS_DIAGNOSTICS_RU.md", "MULTISTABILITY_RU.md",
                        "RF_PARAMETER_SEARCH_RU.md", "NUMERICAL_VALIDATION_RU.md", "SESSION_HANDOFF.md"]
    assert all((BASE / name).is_file() for name in required_reports)
    required_figures = [f"FIG{i:02d}" for i in range(1, 9)]
    figure_files = {}
    for figure in required_figures:
        matches = {ext: (BASE / "figures" / f).is_file() for ext in ("png", "pdf", "svg")
                   for f in [p.name for p in (BASE / "figures").glob(f"{figure}*{'.'+ext}")]}
        assert all(matches.values()) and len(matches) == 3, (figure, matches)
        figure_files[figure] = sorted(matches)
    b115_multistart = json.loads((BASE / "results" / "multistart" / "same_phase_multistart_B115_summary.json").read_text(encoding="utf-8"))
    b115_group = next(x for x in b115_multistart["candidate_results"] if x["candidate"] == "refine12_sobol17")
    assert b115_group["n"] == 8 and b115_group["counterrotation_count"] == 6

    result = {"base_commit": "d1bf252649e174166becf464269318b5f97acd5d",
              "source_integrity": source_checks, "finite_difference_jacobian_audit": jacobian_checks,
              "new_integrator_vs_frozen_v20b_rk4_one_period": rhs_agreement,
              "conditional_qr_horizons": [2000, 5000, 10000, 20000, 50000],
              "extended_qr_spectrum_length": 14, "extended_near_zero_count_at_1e-4": 4,
              "stroboscopic_checks": strobe_checks,
              "same_phase_multistart_runs": multistart["runs_total"],
              "same_phase_multistart_CR_count": 23,
              "perturbation_panels": len(perturb),
              "phase_matched_reference_samples": setdist["reference_samples"],
              "rf_parameter_points": coverage["total_unique_points"],
              "adaptive_transition_midpoints": adaptive["unique_points"],
              "adaptive_outcome_counts": {o: sum(r["outcome"] == o for r in adaptive["results"])
                                           for o in sorted({r["outcome"] for r in adaptive["results"]})},
              "selected_qr_points_5k": len(selected5["results"]),
              "transferable_candidates": len(candidates["candidates"]),
              "b115_multistart": {"n": b115_group["n"], "CR": b115_group["counterrotation_count"]},
              "figure_formats_verified": figure_files,
              "status": "PASS"}
    out = BASE / "results" / "validation_audit.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
