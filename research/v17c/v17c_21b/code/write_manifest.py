"""Write a deterministic SHA-256 inventory for the V17C.21.2 study files."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "MANIFEST.json"
EXCLUDE_DIRS = {"__pycache__"}
EXCLUDE_SUFFIXES = {".pyc", ".tmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = {}
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or path == TARGET or any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix in EXCLUDE_SUFFIXES:
            continue
        files[path.relative_to(BASE).as_posix()] = sha256(path)
    payload = {
        "study": "V17C.21.2 M3 bifurcation and full-RF validation",
        "source_branch": "research/v17c21a-m3-convergence-stability",
        "source_commit": "94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8",
        "output_branch": "research/v17c21b-m3-bifurcation-diagrams",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "SHA-256",
        "manifest_self_hash": "excluded to avoid recursive hashing",
        "file_count": len(files),
        "files": files,
    }
    TARGET.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {TARGET.name} with {len(files)} file hashes.")


if __name__ == "__main__":
    main()
