"""Verify SHA-256 inventory and the contents of the V17C.4 ZIP."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
for name, digest in expected.items():
    h = hashlib.sha256((ROOT / Path(name)).read_bytes()).hexdigest()
    if h != digest:
        raise SystemExit(f"Manifest SHA-256 mismatch: {name}")

archive = ROOT / "V17C_4_results.zip"
sidecar = (ROOT / "V17C_4_results.zip.sha256").read_text(encoding="ascii").split()[0]
archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
if archive_hash != sidecar:
    raise SystemExit("ZIP sidecar SHA-256 mismatch")

with ZipFile(archive) as zf:
    names = set(zf.namelist())
    wanted = set(expected) | {"MANIFEST.json"}
    if names != wanted:
        raise SystemExit(f"ZIP member mismatch: extra={sorted(names-wanted)}, missing={sorted(wanted-names)}")
    for name, digest in expected.items():
        if hashlib.sha256(zf.read(name)).hexdigest() != digest:
            raise SystemExit(f"ZIP member hash mismatch: {name}")
    if zf.read("MANIFEST.json") != (ROOT / "MANIFEST.json").read_bytes():
        raise SystemExit("Embedded manifest differs from working copy")

print(f"Verified {len(expected)} payload hashes, {len(wanted)} ZIP members, SHA-256 {archive_hash}")
