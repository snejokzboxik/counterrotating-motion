"""Losslessly archive/restore the large Sobol JSONL checkpoint."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "physical_parameter_sobol_screen"
RAW = OUT / "trajectories.jsonl"
GZ = OUT / "trajectories.jsonl.gz"
MANIFEST = OUT / "archive_manifest.json"


def digest(stream) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-raw", action="store_true", help="remove the raw JSONL only after verified compression")
    parser.add_argument("--restore", action="store_true", help="restore the raw JSONL checkpoint from the gzip archive")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.restore:
        if not GZ.exists():
            raise FileNotFoundError(GZ)
        with gzip.open(GZ, "rb") as src:
            raw_hash = digest(src)
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["uncompressed_sha256"]
        if raw_hash != expected:
            raise RuntimeError("archive SHA-256 does not match its manifest")
        tmp = RAW.with_suffix(".jsonl.restore.tmp")
        with gzip.open(GZ, "rb") as src, tmp.open("wb") as dst:
            while block := src.read(1 << 20):
                dst.write(block)
        tmp.replace(RAW)
        print(json.dumps({"restored": str(RAW), "sha256": raw_hash}, indent=2))
        return

    if not RAW.exists():
        raise FileNotFoundError(RAW)
    with RAW.open("rb") as src:
        raw_hash = digest(src)
    tmp = GZ.with_suffix(".gz.tmp")
    with RAW.open("rb") as src, gzip.GzipFile(filename=str(tmp), mode="wb", compresslevel=9, mtime=0) as dst:
        while block := src.read(1 << 20):
            dst.write(block)
    with gzip.open(tmp, "rb") as src:
        archived_hash = digest(src)
    if archived_hash != raw_hash:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("gzip round-trip changed trajectory bytes")
    tmp.replace(GZ)
    with GZ.open("rb") as src:
        archive_hash = digest(src)
    info = {
        "source": "trajectories.jsonl",
        "archive": "trajectories.jsonl.gz",
        "uncompressed_bytes": RAW.stat().st_size,
        "compressed_bytes": GZ.stat().st_size,
        "uncompressed_sha256": raw_hash,
        "compressed_sha256": archive_hash,
        "compression": "gzip level 9; lossless",
    }
    MANIFEST.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    if args.remove_raw:
        RAW.unlink()
    print(json.dumps(info | {"raw_removed": args.remove_raw}, indent=2))


if __name__ == "__main__":
    main()
