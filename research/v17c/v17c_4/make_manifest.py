"""Create a SHA-256 inventory for the V17C.4 deliverable tree."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.json"
EXCLUDED_PARTS = {"__pycache__", ".git"}
EXCLUDED_NAMES = {"V17C_4_results.zip", "V17C_4_results.zip.sha256"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


files = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == MANIFEST or path.name in EXCLUDED_NAMES:
        continue
    if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
        continue
    if path.suffix == ".pyc":
        continue
    files.append({"path": path.relative_to(ROOT).as_posix(),
                  "size_bytes": path.stat().st_size,
                  "sha256": sha256(path)})

doc = {
    "project": "V17C.4",
    "geometry": "12 electrodes, ADCFEBADCFEB, six equal-amplitude RF phases",
    "result": "No stable counter-rotating pair meeting the stated 10-turn criterion was found in the saved campaigns.",
    "hash_algorithm": "SHA-256",
    "manifest_excludes_itself": True,
    "archive_hash_is_in_external_sidecar": "V17C_4_results.zip.sha256",
    "file_count": len(files),
    "files": files,
}
MANIFEST.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {MANIFEST} ({len(files)} files)")
