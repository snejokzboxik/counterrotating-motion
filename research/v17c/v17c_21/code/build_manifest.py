"""Build the V17C.21 SHA-256 manifest from tracked research artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MANIFEST = BASE / "MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append({
            "path": path.relative_to(BASE).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    manifest = {
        "study": "V17C.21 analytical counterrotation theory",
        "branch": "research/v17c21-analytical-counterrotation",
        "base_commit": "43c84bf72ce0807d244475518588770fbaf0fb69",
        "source_commits": {
            "V20A": "43c84bf72ce0807d244475518588770fbaf0fb69",
            "V20B": "d1bf252649e174166becf464269318b5f97acd5d",
        },
        "remote_master_sha_before_publication": "e5e6b2473bf519c5670ee1c48882183fbcfc8f2d",
        "baseline": {
            "candidate": "refine12_sobol17_frequency_rescaled",
            "A": 0.03492413282400757,
            "B": 0.15631667332941065,
            "nu": 3.2196773638017477,
            "beta": 0.015388559140477825,
            "m1_over_m2": 0.10963763906505944,
            "C12": 9.747265456858355e-06,
            "lambda1": 0.24452307123064748,
            "lambda2": 0.17736598731050257,
        },
        "trajectory_accounting": {
            "new_joint_pair_integrations": 5,
            "new_RF1_periods": 61000,
            "reused_archived_pair_trajectories": 2,
            "reused_archived_RF1_periods": 245000,
            "pair_trajectory_records_considered": 7,
            "total_RF1_period_horizon_considered": 306000,
            "distinct_pair_parameter_sets": 6,
        },
        "manifest_self_hash_excluded": True,
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in written["files"]:
        if sha256(BASE / item["path"]) != item["sha256"]:
            raise RuntimeError(f"SHA-256 verification failed for {item['path']}")
    print(f"Wrote {MANIFEST.relative_to(BASE.parent.parent.parent)} with {len(files)} artifact hashes")


if __name__ == "__main__":
    main()
