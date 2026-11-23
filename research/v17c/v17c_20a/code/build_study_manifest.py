from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    source = json.loads((RESULTS / "rf_parameter_screen" / "PARAMETER_COVERAGE.json").read_text(encoding="utf-8"))
    adaptive_path = RESULTS / "rf_parameter_screen" / "adaptive_boundary_results.json"
    adaptive = json.loads(adaptive_path.read_text(encoding="utf-8")) if adaptive_path.exists() else None
    out = {
        "study": "V17C.20A",
        "source_commit": "d1bf252649e174166becf464269318b5f97acd5d",
        "model_group": "refine12_sobol17 particles fixed; field parameters varied",
        "variables_and_pairwise_screen": source,
        "additional_validation": {
            "baseline_candidates": ["candidate52", "Local15", "refine12_sobol17"],
            "conditional_qr_horizons_periods": [2000, 5000, 10000, 20000, 50000],
            "mechanical_dimension": 12,
            "extended_skew_product_dimension": 14,
            "direct_perturbation_cases": 41,
            "fixed_phase_multistart": {"baseline_runs": 24, "B115_runs": 8,
                                        "horizon_periods": 15000, "seed": 20261123},
            "stroboscopic_samples": {"candidate52": 5001, "local15": 5001, "refine12_sobol17": 20001},
            "phase_matched_reference_points": 20001,
            "selected_rf_qr_points_5k": 6,
            "selected_rf_qr_points_20k": 1,
            "adaptive_transition_midpoints": adaptive["unique_points"] if adaptive else 0,
            "adaptive_midpoint_outcomes": ({o: sum(r["outcome"] == o for r in adaptive["results"])
                                             for o in sorted({r["outcome"] for r in adaptive["results"]})}
                                            if adaptive else {}),
            "figure_count": 8,
            "figure_formats": ["png 600 dpi", "pdf", "svg"],
        },
        "interpretation_limits": [
            "Single late-state start for each RF screen point; not a basin probability.",
            "Fixed selected mechanical starts and fixed RF phases; not a universal distribution.",
            "Finite-time Lyapunov values do not prove exact asymptotic signs or torus dimension.",
        ],
    }
    (BASE / "PARAMETER_COVERAGE.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    files = []
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json" or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append({"path": path.relative_to(BASE).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {
        "study": "V17C.20A",
        "base_commit": "d1bf252649e174166becf464269318b5f97acd5d",
        "hash_algorithm": "SHA-256",
        "manifest_self_hash": "excluded; file list hashes all other files in this study directory",
        "file_count": len(files),
        "files": files,
    }
    (BASE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"files_hashed": len(files), "manifest": str(BASE / "MANIFEST.json"),
                      "coverage": str(BASE / "PARAMETER_COVERAGE.json")}, indent=2))


if __name__ == "__main__":
    main()
