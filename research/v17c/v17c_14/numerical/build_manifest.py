"""Build SHA256 and provenance manifest for the V17C.14 package."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"


def main() -> None:
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        rel = path.relative_to(ROOT)
        if any(part in {".venv", "__pycache__"} for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        payload = path.read_bytes()
        entries.append({
            "path": rel.as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    figure_sets = sorted({p.stem.rsplit(".", 1)[0] for p in (ROOT / "figures").glob("*.pdf")})
    data = {
        "project": "V17C.14",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": "https://github.com/snejokzboxik/three-phase-12-electrode-trap",
        "publication_branch": "research/v17c-theory",
        "base_branch": "research/v17c",
        "base_commit": "fc1d29ecc1067e0191a4cf597189b4973546864f",
        "source_data_branch": "research/v17c-crystals",
        "source_data_commit": "5438b1cdd79c47abab45c6cbaf99c6daec9fbb60",
        "additional_source_audit": {
            "V17C.12B_branch": "research/v17c",
            "V17C.12B_commit": "801ceba8b8555443bd81a9591b2e532f7c36a0e6",
            "V17C.13_summary_files": [
                "research/v17c/v17c_13/BASIN_ANALYSIS_RU.md",
                "research/v17c/v17c_13/PHASE_SYNCHRONIZATION_RU.md",
                "research/v17c/v17c_13/COULOMB_COUPLING_RU.md"
            ]
        },
        "new_trajectory_integrations": 0,
        "analytic_candidate_multipole_configurations": 28,
        "individual_species_root_analyses": 56,
        "leading_branch_beta_nodes_per_species": 301,
        "reduced_phase_lock_model_cases": 7,
        "source_coulomb_sweep_points": 10,
        "imported_n6_source_records": 11,
        "referenced_n6_formation_starts": 80,
        "referenced_n6_selected_long_continuations": 5,
        "ensemble_outcomes_are_aggregate_source_summary": True,
        "figure_sets": figure_sets,
        "manifest_excludes_itself": True,
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest: {len(entries)} files; {len(figure_sets)} figure sets")


if __name__ == "__main__":
    main()
