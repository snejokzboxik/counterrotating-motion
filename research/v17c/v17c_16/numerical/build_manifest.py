"""Build a SHA-256 inventory for the V17C.16 research directory."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"MANIFEST.json"
EXCLUDED = {OUT.resolve()}


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.resolve() in EXCLUDED:
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        payload = path.read_bytes()
        files.append({
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    data = {
        "study": "V17C.16",
        "branch": "research/v17c-two-particle-oseen",
        "source_branch": "research/v17c-theory-validation",
        "source_sha": "9716e9a701b4ebd8fc3ae4b72a8aaeda9437289b",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "hash_algorithm": "SHA-256",
        "files": files,
    }
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n",
                   encoding="utf-8")
    print(f"{OUT} ({len(files)} files, {data['total_bytes']} bytes)")


if __name__ == "__main__":
    main()
