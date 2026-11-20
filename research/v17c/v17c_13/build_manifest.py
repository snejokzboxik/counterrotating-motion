"""Build a SHA-256 manifest for V17C.13 artifacts and fixed V17C.11 inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / "MANIFEST.json"
SOURCES = [
    "research/v17c/v17c_11/REPORT_RU.md",
    "research/v17c/v17c_11/SESSION_HANDOFF.md",
    "research/v17c/v17c_11/local15_parameters.json",
    "research/v17c/v17c_11/best_loading_protocol.json",
    "research/v17c/v17c_11/loading_dynamics.py",
    "research/v17c/v17c_11/strict_loading_holdout_n256/holdout_protocols.json",
    "research/v17c/v17c_11/strict_loading_holdout_n256/holdout_runs.jsonl",
    "research/v17c/v17c_9/baseline/basin_scan.py",
]


def item(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    outputs = sorted(p for p in HERE.rglob("*") if p.is_file()
                     and p != MANIFEST and "__pycache__" not in p.parts
                     and ".nbc" not in p.suffix and ".nbi" not in p.suffix)
    doc = {
        "study": "V17C.13 collective dynamics of two three-particle Coulomb crystals",
        "base_commit": "d40a25b8fc073b43e12ed7d15fe07caa7502c988",
        "branch": "research/v17c-crystals",
        "status": "P0 validated; P1 has 6 finite-horizon ordered crystal candidates; P4 ten-point C continuation brackets corotation transition between 2 and 5 C0; four initial and four post-formation perturbation tests complete; RK4 full-horizon refinement and independent DOP853 pending",
        "outputs": [item(p) for p in outputs],
        "fixed_sources": [item(ROOT / p) for p in SOURCES],
    }
    MANIFEST.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {len(outputs)} outputs, {len(SOURCES)} fixed sources")


if __name__ == "__main__":
    main()
