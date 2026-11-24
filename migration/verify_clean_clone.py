"""Verify every archived V17C blob using bytes obtained from the published clone."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

clone = Path(sys.argv[1]).resolve()
ledger = clone / "migration" / "V17C_FILE_INTEGRITY.csv"
rows = list(csv.DictReader(ledger.open(encoding="utf-8", newline="")))
unique = {}
for row in rows:
    oid = row["git_blob_sha1"]
    if oid in unique:
        assert unique[oid]["sha256"] == row["sha256"]
    else:
        unique[oid] = row

missing_out = subprocess.run(
    ["git", "rev-list", "--objects", "--missing=print", "--all"],
    cwd=clone, check=True, capture_output=True, text=True,
).stdout
missing = sorted({x[1:41] for x in missing_out.splitlines() if x.startswith("?")} & unique.keys())
print(f"ledger_rows={len(rows)} unique_blobs={len(unique)} missing_before={len(missing)}", flush=True)
for i in range(0, len(missing), 20):
    batch = missing[i : i + 20]
    p = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"],
        cwd=clone, input="\n".join(batch) + "\n", text=True,
        capture_output=True, timeout=900,
    )
    if p.returncode or len(p.stdout.splitlines()) != len(batch) or " missing" in p.stdout:
        raise RuntimeError(f"lazy fetch batch {i//20+1} failed: {p.stderr[:1000]} {p.stdout[:1000]}")
    print(f"fetched {min(i+20,len(missing))}/{len(missing)}", flush=True)

p = subprocess.Popen(
    ["git", "cat-file", "--batch"], cwd=clone,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
assert p.stdin and p.stdout
verified_bytes = 0
for i, (oid, row) in enumerate(unique.items(), 1):
    p.stdin.write((oid + "\n").encode("ascii"))
    p.stdin.flush()
    header = p.stdout.readline().decode("ascii").strip().split()
    if len(header) != 3 or header[0] != oid or header[1] != "blob":
        raise RuntimeError(f"bad object header for {oid}: {header}")
    size = int(header[2])
    digest = hashlib.sha256()
    left = size
    while left:
        chunk = p.stdout.read(min(left, 1024 * 1024))
        if not chunk:
            raise RuntimeError(f"truncated blob {oid}")
        digest.update(chunk)
        left -= len(chunk)
    assert p.stdout.read(1) == b"\n"
    if size != int(row["bytes"]) or digest.hexdigest() != row["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch: {oid} {row['path']}")
    verified_bytes += size
    if i % 250 == 0:
        print(f"hashed {i}/{len(unique)}", flush=True)
p.stdin.close()
if p.wait() != 0:
    raise RuntimeError(p.stderr.read().decode(errors="replace"))
result = {
    "ledger_rows": len(rows),
    "unique_blobs": len(unique),
    "verified_unique_blob_bytes": verified_bytes,
    "sha256_mismatches": 0,
    "source": "fresh independent clone of published counterrotating-motion remote",
}
out = clone / "migration" / "clean_clone_file_verification.json"
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result), flush=True)
