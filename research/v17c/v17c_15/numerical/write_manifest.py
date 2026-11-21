"""Write a reproducibility manifest for the V17C.15 subtree."""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy
import scipy

ROOT = Path(__file__).resolve().parents[1]
FILES = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path.name == "MANIFEST.json":
        continue
    if ("__pycache__" in path.parts or ".pytest_cache" in path.parts or
            path.suffix in {".pyc", ".pyo"}):
        continue
    data = path.read_bytes()
    FILES.append({
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    })

manifest = {
    "campaign": "V17C.15",
    "branch": "research/v17c-theory-validation",
    "base_sha": "37c477fbacfccd55e79d6092e6e70d563c559ac6",
    "status": "in_progress",
    "baseline_policy": "V17C.14 and V17C.12B historical results are immutable",
    "claim_policy": {
        "exact": "identities derived from the stated point-particle model",
        "asymptotic": "only with explicit smallness/limit assumptions",
        "numerical": "fixed parameters, initial panel, integrator, horizon, retained losses",
        "hypothesis": "not stated as established until independently checked",
    },
    "environment": {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "logical_cpu_count": 12,
        "observed_total_ram_gib": 15.42,
        "observed_available_ram_gib_at_initial_check": 4.68,
        "parallelism_note": "Long single-particle campaigns used bounded process pools; Cunningham pair used 4 workers, not a concurrent heavy campaign.",
    },
    "compute_inventory": {
        "archived_trajectory_horizon_records": 584,
        "archived_trajectory_horizon_records_definition": "initial-condition trajectory records from the campaigns through the QR panel; resumed sign-transition segments are counted separately",
        "conditional_qr_trajectories": 44,
        "conditional_qr_particle_periods": 940000,
        "conditional_qr_horizons": [5000, 10000, 15000, 30000, 50000],
        "conditional_qr_panel_starts_per_species": 22,
        "conditional_qr_new_panel_trajectories": 42,
        "conditional_qr_extended_trajectories_to_50000": 8,
        "conditional_qr_panel_workers": 11,
        "conditional_qr_extension_workers": 8,
        "full_integrations_from_initial_state": 656,
        "continuation_segments": 176,
        "total_solver_job_segments": 832,
        "total_particle_periods": 10619000,
        "archived_baseline_particle_periods": 1135000,
        "repeated_failed_export_runs": 96,
        "repeated_failed_export_periods": 288000,
        "field_parameter_sets": 1,
        "field_parameter_set_name": "Local15",
        "coarse_sign_scan_nu_values": [2.1, 2.175, 2.25, 2.325, 2.4],
        "distinct_exact_rf_sign_scan_nu_values": [2.1, 2.175, 2.2125, 2.23125, 2.240625, 2.25, 2.325, 2.4],
        "base_local15_nu": 3.1118029452860356,
        "coarse_sign_scan_species": ["species1", "species2"],
        "coarse_sign_scan_trajectories": 240,
        "long_species1_nu_values": [2.2125, 2.23125, 2.240625, 2.25],
        "long_species1_window_horizons": [5000, 10000, 15000, 20000, 30000, 40000, 50000],
        "drag_conventions_in_paired_check": ["legacy_multiply", "corrected_divide"],
        "static_ring_N_checks": [2, 3, 6, 8, 12, 16, 24, 32],
        "ring_checks_are_trajectories": False,
        "sign_transition_extension_frequency_species_ensembles": 5,
        "sign_transition_extension_starts_per_ensemble": 24,
        "sign_transition_extension_segments": 120,
        "sign_transition_extension_particle_periods": 4488000,
        "sign_transition_extension_source_horizons": {"species1": 15000, "species2_control": 3000},
        "sign_transition_extension_target_horizon": 50000,
        "sign_transition_extension_windows": [5000, 10000, 15000, 20000, 30000, 40000, 50000],
        "sign_transition_extension_worker_count_history": [5, 1, 2],
        "mixed_node_120k_extension": {
            "species": "species1",
            "nu": 2.23125,
            "starts": 24,
            "source_horizon": 50000,
            "target_horizon": 120000,
            "continuation_segments": 24,
            "nominal_particle_periods": 1680000,
            "workers": 8,
            "steps_per_primary_period": 96,
            "window_horizons": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
            "last_window_signs_at_120k_retained_negative_positive_escaped": [9, 12, 3],
            "cumulative_50k_to_120k_retained_negative_positive_escaped": [21, 0, 3],
            "restart_note": "one stalled worker was restarted from the latest 500-period checkpoint; unreconstructable duplicate work is excluded from nominal inventory",
        },
        "inventory_policy": "continuation segments and particle-periods count nominal resumed intervals; unreconstructable repeated work during checkpoint recovery is excluded; static ring checks are not trajectories",
    },
    "artifacts": FILES,
    "unresolved": [
        "exact-RF continuation beyond 50,000 periods at nodes other than the species1 mixed node nu=2.23125; individual transition points, long-term branch stability, and asymptotic classification remain unresolved at all nodes",
        "quantitative error map versus epsilon_mu over RF and drag parameters",
        "independent numerical validation of moving-orbit and higher-order averaging corrections",
        "full MFS higher-spatial-harmonic comparison and real-electrode transfer validation",
        "asymptotic classification beyond the conditional mechanical QR panel: substantially longer horizons, RF phase tangent directions, exact periodic Floquet analysis where applicable, and N-body variational spectra",
        "basin measures under explicitly sampled initial coordinates, velocities, RF phases, and loading protocols; the retained QR panel is fixed and not an iid basin sample",
        "direct RF-N-body continuation of the conditional C/C0=3.0787336 collective threshold and stable multi-ring solutions",
        "paired rerun of the historical physical V17C.12B campaign with individual corrected Cunningham coefficients",
        "general 3D MFS/FEM field adapter, electrode breakdown and physical feasibility analysis",
        "rendered and visually verified PDF of the LaTeX article (no local TeX compiler was available)",
    ],
}
(ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"manifest: {len(FILES)} artifact files")
