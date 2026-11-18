"""Build the self-contained V17C.4 ZIP and external SHA-256 sidecar."""
from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "V17C_4_results.zip"
SIDECAR = ROOT / "V17C_4_results.zip.sha256"
SKIP_PARTS = {"__pycache__", ".git"}
SKIP_NAMES = {ARCHIVE.name, SIDECAR.name}

files = [p for p in sorted(ROOT.rglob("*")) if p.is_file()
         and p.name not in SKIP_NAMES and p.suffix != ".pyc"
         and not any(part in SKIP_PARTS for part in p.relative_to(ROOT).parts)]
with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
    for path in files:
        zf.write(path, path.relative_to(ROOT).as_posix())

h = hashlib.sha256()
with ARCHIVE.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
SIDECAR.write_text(f"{h.hexdigest()}  {ARCHIVE.name}\n", encoding="ascii")
print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size} bytes, {len(files)} files)")
print(f"SHA-256 {h.hexdigest()}")
