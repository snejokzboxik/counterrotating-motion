"""Refresh checksums for V17C.11 products and the fixed published inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / "MANIFEST.json"
SOURCE_INPUTS = [
    "research/v17c/v17c_10/SESSION_HANDOFF.md",
    "research/v17c/v17c_10/FOCUSED_EDGE_VALIDATION_RU.md",
    "research/v17c/v17c_10/FOCUSED_EDGE_REPLICATION_RU.md",
    "research/v17c/v17c_10/NUMERICAL_CONVERGENCE_RU.md",
    "research/v17c/v17c_10/FAILURE_ANALYSIS_RU.md",
    "research/v17c/v17c_10/focused_edge_screen/ranked.json",
    "research/v17c/v17c_10/focused_edge_replication/replication_validation_summary.json",
    "research/v17c/v17c_10/focused_edge_replication/local15_rest_matched_strict_none_a0.0002-0.00189_replicate20261410_256_15000p_64s.meta.json",
    "research/v17c/v17c_10/focused_edge_replication/candidate52_rest_matched_strict_none_a0.0002-0.00189_replicate20261410_256_15000p_64s.meta.json",
    "research/v17c/v17c_5/input/config.json",
    "research/v17c/v17c_9/baseline/basin_scan.py",
    "research/v17c/v17c_9/baseline/verify_candidate_dop853.py",
]


def entry(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    outputs = sorted(p for p in HERE.rglob("*") if p.is_file()
                     and p != MANIFEST and "__pycache__" not in p.parts)
    doc = {
        "study": "V17C.11 shared RF-DC loading protocol optimization",
        "source_commit": "57e197bb744a3af890d466f418a76a95cf7758b4",
        "working_branch": "research/v17c-loading",
        "status": "P0-P5 validated; P6/P7 early-retention screens and P8 N=64 strict independent holdout complete; P9 DOP853/step convergence, larger holdout, robustness, and transferability pending",
        "outputs": [entry(p) for p in outputs],
        "fixed_sources": [entry(ROOT / p) for p in SOURCE_INPUTS],
    }
    MANIFEST.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest covers {len(outputs)} outputs and {len(SOURCE_INPUTS)} fixed sources")


if __name__ == "__main__":
    main()
