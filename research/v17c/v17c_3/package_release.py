"""Build a hashed, self-contained V17C.3 code and results package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
ARCHIVE = HERE / "V17C3_results.zip"
CHECKSUM = HERE / "V17C3_results.zip.sha256"
MANIFEST = HERE / "MANIFEST.json"
EXCLUDE = {"MANIFEST.json", ARCHIVE.name, CHECKSUM.name}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


files = []
for path in sorted(HERE.rglob("*")):
    if not path.is_file() or "__pycache__" in path.parts or path.name in EXCLUDE:
        continue
    rel = path.relative_to(HERE).as_posix()
    files.append({"path": rel, "size_bytes": path.stat().st_size,
                  "sha256": digest(path)})

manifest = {
    "study": "V17C.3 RF-DC localization and counter-rotation",
    "repository": "https://github.com/snejokzboxik/three-phase-12-electrode-trap",
    "branch": "research/v17c",
    "base_commit": "3a6b12d1a56c563bd5f644a67dddd9a7951c8d11",
    "scope": "research/v17c/v17c_3",
    "model": "full available transverse RF spectra, ideal harmonic 3D DC quadrupole, 3D point Coulomb force, linear dimensionless drag",
    "result": "No stable G1/G2 counter-rotating pair confirmed in the completed 2D-FEM, translation-spectrum, broad-pair, and barrier-targeted follow-up. V17C.3 remains open to further work.",
    "file_count_excluding_manifest_and_archive": len(files),
    "files": files,
}
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")

if ARCHIVE.exists():
    ARCHIVE.unlink()
with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6, allowZip64=True) as zf:
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path == ARCHIVE or path == CHECKSUM:
            continue
        zf.write(path, arcname=path.relative_to(HERE.parent).as_posix())

checksum = digest(ARCHIVE)
CHECKSUM.write_text(f"{checksum}  {ARCHIVE.name}\n", encoding="ascii")
print(json.dumps({"manifest": str(MANIFEST), "files": len(files),
                  "archive": str(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
                  "archive_sha256": checksum}, indent=2))
