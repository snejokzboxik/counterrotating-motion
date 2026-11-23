from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[2]
V20B = ROOT / "research" / "v17c" / "v17c_20b"
sys.path.insert(0, str(BASE / "code"))
sys.path.insert(0, str(V20B / "code"))
import run_long_qr as qr  # noqa: E402
import attractor_search as a  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = json.loads((V20B / "results" / "candidate_parameters.json").read_text(encoding="utf-8"))
    aliases = {"candidate52": "scaled_candidate52_f45hz",
               "local15": "scaled_local15_f50hz",
               "refine12_sobol17": "scaled_refine12_sobol17_f50hz"}
    field = a.load_field()
    qr_base = BASE / "results" / "lyapunov"
    qr_dirs = {
        "candidate52": next(qr_base.glob("candidate52*50000p_qr__s32_q20")),
        "local15": next(qr_base.glob("local15*50000p_qr__s32_q20")),
        "refine12_sobol17": next(qr_base.glob("refine12_sobol17*50000p_qr__s32_q20")),
    }
    baseline_summary_names = {
        "candidate52": "candidate52_50000_common_corrected_summary.json",
        "local15": "local15_50000_common_corrected_summary.json",
        "refine12_sobol17": "refine12_240k_corrected_drag_summary.json",
    }
    specs = []
    for name, cid in aliases.items():
        _, case, p, qratio, seed, phase, source_period, trajectory = qr.load_late_seed(cid)
        seed_after, phase_after = qr.transition(case, p, qratio, seed, phase, 100, 32)
        j = json.loads((qr_dirs[name] / "result.json").read_text(encoding="utf-8"))
        late = json.loads((V20B / "results" / baseline_summary_names[name]).read_text(encoding="utf-8"))
        specs.append({
            "id": name,
            "classification": ("ATTRACTOR_CANDIDATE; two near-neutral conditional modes, ten contracting modes; invariant-set attraction not fully resolved"
                               if name == "refine12_sobol17" else
                               "LONG_LIVED_ONLY; near-neutral conditional modes; attractor type unresolved"),
            "parameters": {
                "vector_definition": ["A", "B", "nu", "beta", "m1/m2", "C", "lambda1_Cunningham_corrected", "lambda2_Cunningham_corrected"],
                "p_corrected": np.asarray(p, float).tolist(), "q2_over_q1": float(qratio),
                "alpha1": float(case["alpha1"]), "alpha2": float(case["alpha2"]),
                "mu_m2_over_m1": float(case["mu"]), "gravity_z": float(case["gravity_z"]),
                "a1_over_R": float(case["a1_over_R"]), "a2_over_R": float(case["a2_over_R"]),
                "re1_per_speed": float(case["re1_per_speed"]), "re2_per_speed": float(case["re2_per_speed"]),
                "f1_Hz": float(case["f1_Hz"]), "f2_Hz": float(case["f2_Hz"]),
            },
            "initial_condition": {
                "source_trajectory": trajectory.relative_to(ROOT).as_posix(),
                "source_sha256": j["trajectory_sha256"], "source_period": int(source_period),
                "mechanical_state_at_seed_12D": seed.tolist(),
                "rf_phases_at_archived_period_rad": phase.tolist(),
                "discarded_transition_periods": 100,
                "mechanical_state_after_transition_12D": seed_after.tolist(),
                "rf_phases_after_transition_rad": phase_after.tolist(),
                "initial_state_sha256": hashlib.sha256(seed.tobytes()).hexdigest(),
            },
            "physical_parameter_mapping_from_V20B": case.get("physical"),
            "qr": {
                "conditional_spectrum_per_tau_descending_50k": j["lyapunov_spectrum_per_tau_descending"],
                "horizon_RF1_periods": j["integration"]["completed_periods"],
                "steps_per_RF1_period": j["integration"]["steps_per_rf_period"],
                "qr_interval_periods": j["integration"]["qr_interval_periods"],
                "volume_balance_abs_error_per_tau": j["volume_balance_abs_error_per_tau"],
            },
            "published_long_replay_observables": {
                "source_summary": f"research/v17c/v17c_20b/results/{baseline_summary_names[name]}",
                "periods": late["completed_periods"], "late_turns": late["late_turns"],
                "late_frequency_Hz": late["late_frequency_Hz"],
                "late_mean_radius_R": late["late_mean_radius_R"],
                "minimum_pair_distance_R": late["minimum_pair_distance_R"],
                "all_late_chunks_counterrotating": late["all_late_chunks_counterrotating"],
            },
            "evidence_source": {
                "stroboscopic_data": f"results/stroboscopic/{name}_..._every_rf1.npz",
                "fixed_phase_multistart": "results/multistart/same_phase_multistart_summary.json",
                "classification_limit": "Finite starts and windows; not a basin probability or asymptotic proof.",
            },
            "scientific_note": "External RF phases are carried continuously from the archived absolute time; no phase reset. Conditional QR holds phase perturbations fixed."
        })

    # B-up variant: same particle and initial conditions, field B is increased 15%.
    refine = specs[2]
    b115_p = list(refine["parameters"]["p_corrected"])
    b115_p[1] *= 1.15
    b115_summary = json.loads((BASE / "results" / "multistart" / "same_phase_multistart_B115_summary.json").read_text(encoding="utf-8"))
    b115_qr = json.loads((BASE / "results" / "rf_parameter_screen" / "selected_qr_20000p_results.json").read_text(encoding="utf-8"))["results"][0]
    specs.append({
        "id": "refine12_B115",
        "classification": "ATTRACTOR_CANDIDATE; same near-neutral signature, start-panel escapes; no evidence of stronger attraction",
        "parameters": {**refine["parameters"], "p_corrected": b115_p,
                       "B_multiplier": 1.15, "field_multipliers_A_B_nu_beta": [1.0, 1.15, 1.0, 1.0]},
        "initial_condition": refine["initial_condition"],
        "evidence": {
            "fixed_phase_multistart": {"n": 8, "CR_retained": 6, "escapes": 2,
                                       "horizon_periods": 15000, "seed": 20261123},
            "conditional_qr_top2_20k_per_tau": b115_qr["conditional_spectrum_per_tau_descending"][:2],
            "conditional_qr_horizon_periods": 20000,
            "event_resolved_turns": b115_qr["event_resolved_turns"],
            "minimum_pair_distance_R": b115_qr["event_resolved_min_pair_distance_R"],
            "volume_balance_abs_error_per_tau": b115_qr["volume_balance_abs_error_per_tau"],
            "multistart_outcomes": b115_summary["candidate_results"],
        },
        "scientific_note": "Same-phase fixed-start panel; a finite panel does not estimate a universal basin probability."
    })

    sha_paths = {
        "field_basis": ROOT / "research" / "v17c" / "v17c_12b" / "model" / "field_model.json",
        "campaign_integrator": ROOT / "research" / "v17c" / "v17c_12b" / "results" / "rescaled_legacy_candidate_validation" / "campaign_integrator_source.py",
        "pair_integrator": ROOT / "research" / "v17c" / "v17c_17" / "run_pair_coulomb_scan.py",
        "two_particle_rhs_and_field_diagnostics": V20B / "code" / "attractor_search.py",
        "variational_rhs": V20B / "code" / "variational_qr.py",
        "qr_driver": BASE / "code" / "run_long_qr.py",
        "extended_qr_driver": BASE / "code" / "run_extended_qr.py",
        "rf_screen_driver": BASE / "code" / "rf_parameter_screen.py",
        "adaptive_refinement_driver": BASE / "code" / "adaptive_boundary_refinement.py",
        "multistart_driver": BASE / "code" / "multistart_same_phase.py",
    }
    out = {
        "study": "V17C.20A",
        "base_commit": "d1bf252649e174166becf464269318b5f97acd5d",
        "model_scope": "Two particles, 3D mechanics; eight-harmonic transverse MFS field for infinite electrodes; ideal DC quadrupole; Cunningham-corrected Stokes plus Schiller-Naumann drag; gravity and Coulomb; polarization disabled.",
        "shared_field_basis": {"MFS_family_index": field["family"].tolist(),
                               "azimuthal_modes_k": field["k"].tolist(),
                               "complex_coefficients_real_imag": [[float(c.real), float(c.imag)] for c in field["coeff"]],
                               "note": "V20B full 8-complex-harmonic basis; A/B scale the corresponding RF family; phases are independent."},
        "forcing_phase_policy": "Two independent RF phases carry their absolute phase from the archived state and advance continuously; identical phases are used for paired perturbations/multistarts.",
        "code_sha256": {k: sha(v) for k, v in sha_paths.items()},
        "candidates": specs,
    }
    (BASE / "ATTRACTOR_CANDIDATES.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
