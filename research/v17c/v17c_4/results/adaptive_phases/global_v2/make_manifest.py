"""Write SHA-256 inventory for the global_v2 screen files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "MANIFEST_SHA256.json"
entries = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == OUT:
        continue
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    entries.append({"path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": h.hexdigest()})
OUT.write_text(json.dumps({"algorithm": "SHA-256", "files": entries},
                          indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(entries)} files)")
