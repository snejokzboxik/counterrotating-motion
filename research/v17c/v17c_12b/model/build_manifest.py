"""Build a SHA-256 manifest for the V17C.12B research directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", choices=("in_progress", "final"), default="in_progress")
    parser.add_argument("--exclude", action="append", default=[],
        help="Repository-relative file or directory path to omit, e.g. results/campaign")
    args = parser.parse_args()
    excluded = [Path(x) for x in args.exclude]
    items = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT:
            continue
        rel = path.relative_to(ROOT)
        if any(part in (".venv", "__pycache__", ".git") for part in rel.parts):
            continue
        if path.suffix in (".pyc", ".pyo"):
            continue
        if any(rel == ex or ex in rel.parents for ex in excluded):
            continue
        items.append({"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    manifest = {
        "study": "V17C.12B",
        "branch": "research/v17c",
        "base_commit": "a9a92bb8ca46528215729780d689f4f72bc02892",
        "status": args.status,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "excluded_paths": [x.as_posix() for x in excluded],
        "file_count": len(items),
        "total_bytes": sum(x["bytes"] for x in items),
        "files": items,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": args.status, "file_count": len(items),
        "total_bytes": manifest["total_bytes"], "excluded_paths": manifest["excluded_paths"],
        "manifest": str(OUTPUT)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
