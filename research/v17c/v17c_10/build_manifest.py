"""Build the V17C.10 SHA-256 manifest, excluding the manifest itself."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "MANIFEST.json"


def main():
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUT or "__pycache__" in path.parts:
            continue
        data = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "study": "V17C.10 counter-rotation basin optimization",
        "created": "2026-11-19",
        "status": "P0 Sobol screening complete; P1 failure analysis pending",
        "branch": "research/v17c",
        "base_commit": "cc41be9082c62e9b907eb459f453ac51001d4267",
        "files": files,
        "manifest_note": "MANIFEST.json is intentionally excluded from its own file list.",
    }
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(OUT), "files": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
