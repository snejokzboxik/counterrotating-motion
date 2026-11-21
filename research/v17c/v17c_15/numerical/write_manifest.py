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
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
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
        "archived_trajectory_horizon_records": 542,
        "conditional_qr_trajectories": 2,
        "conditional_qr_particle_periods": 30000,
        "conditional_qr_horizons": [5000, 10000, 15000],
        "full_integrations_from_initial_state": 614,
        "continuation_segments": 24,
        "total_solver_job_segments": 638,
        "total_particle_periods": 3541000,
        "archived_baseline_particle_periods": 1135000,
        "repeated_failed_export_runs": 96,
        "repeated_failed_export_periods": 288000,
        "field_parameter_sets": 1,
        "field_parameter_set_name": "Local15",
        "coarse_sign_scan_nu_values": [2.1, 2.175, 2.25, 2.325, 2.4],
        "coarse_sign_scan_species": ["species1", "species2"],
        "coarse_sign_scan_trajectories": 240,
        "long_species1_nu_values": [2.2125, 2.23125, 2.240625, 2.25],
        "long_species1_window_horizons": [5000, 10000, 15000],
        "drag_conventions_in_paired_check": ["legacy_multiply", "corrected_divide"],
        "static_ring_N_checks": [2, 3, 6, 8, 12, 16, 24, 32],
        "ring_checks_are_trajectories": False,
    },
    "artifacts": FILES,
    "unresolved": [
        "exact-RF continuation beyond 15,000 periods around the mixed-sign transition and long-term branch stability",
        "quantitative error map versus epsilon_mu over RF and drag parameters",
        "independent numerical validation of moving-orbit and higher-order averaging corrections",
        "full MFS higher-spatial-harmonic comparison and real-electrode transfer validation",
        "full exact-RF stability beyond one conditional S01 QR trajectory per species: long-horizon sign, phase-augmented flow, additional starts, and basin measures",
        "direct RF-N-body continuation of the conditional C/C0=3.0787336 collective threshold and stable multi-ring solutions",
        "paired rerun of the historical physical V17C.12B campaign with individual corrected Cunningham coefficients",
        "general 3D MFS/FEM field adapter, electrode breakdown and physical feasibility analysis",
        "rendered and visually verified PDF of the LaTeX article (no local TeX compiler was available)",
    ],
}
(ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"manifest: {len(FILES)} artifact files")
