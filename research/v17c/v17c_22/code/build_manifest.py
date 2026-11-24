"""Build deterministic SHA-256 manifest for V17C.22 deliverables and sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research/v17c/v17c_22"
OUT = BASE / "MANIFEST.json"
BASE_SHA = "894c6896ce5927371708037ffa4a4021189d2b54"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    sources = [
        "research/v17c/v17c_20b/code/dop853_independent.py",
        "research/v17c/v17c_20b/code/variational_qr.py",
        "research/v17c/v17c_20a/results/lyapunov_horizon_fits.json",
        "research/v17c/v17c_20a/results/lyapunov/refine12_sobol17_240000p_seed__100p_transient__50000p_qr__s32_q20/result.json",
        "research/v17c/v17c_20a/results/extended_qr/refine12_sobol17_240000p_seed__100p_transient__20000p_extqr__s32_q20/result.json",
        "research/v17c/v17c_20a/results/perturbations/refine12_sobol17_240000p_seed_5000p_phase_locked.json",
        "research/v17c/v17c_20a/results/stroboscopic/refine12_sobol17_240000p_seed_100p_transient_20000p_every_rf1.npz",
        "research/v17c/v17c_21a/results/m3_slowflow_stability.json",
        "research/v17c/v17c_21a/code/m3_harmonic_balance.py",
        "research/v17c/v17c_21a/code/m3_system.py",
        "research/v17c/v17c_21a/code/m3_adiabatic_slow_flow.py",
        "research/v17c/v17c_21b/FULL_RF_BOUNDARY_VALIDATION_RU.md",
        "research/v17c/v17c_21b/results/full_rf_validation.json",
        "research/v17c/v17c_20b/results/candidate_parameters.json",
        "research/v17c/v17c_20b/results/refine12_240k_corrected_drag_trajectory.npz",
        "research/v17c/v17c_12b/model/field_model.json",
    ]
    entries = []
    for p in sorted(BASE.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json" and "__pycache__" not in p.parts:
            rel = p.relative_to(ROOT).as_posix()
            entries.append({"path": rel, "bytes": p.stat().st_size, "sha256": digest(p)})
    source_entries = []
    for rel in sorted(set(sources)):
        p = ROOT / rel
        source_entries.append({"path": rel, "bytes": p.stat().st_size, "sha256": digest(p)})
    manifest = {
        "study": "V17C.22 improved analytical theory for stable counterrotation",
        "branch": "research/v17c22-counterrotation-attraction-theory",
        "base_branch": "research/v17c21b-m3-bifurcation-diagrams",
        "base_commit": BASE_SHA,
        "nominal_parameters_unchanged": True,
        "scientific_status": "LONG_LIVED_COUNTERROTATION; transverse attraction and invariant torus unresolved",
        "deliverables": entries,
        "source_inputs": source_entries,
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"manifest entries={len(entries)}, source inputs={len(source_entries)}")


if __name__ == "__main__":
    main()
