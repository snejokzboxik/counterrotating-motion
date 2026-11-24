"""Hash every V17C file at each of the 14 frozen historical branch heads."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

MIRROR = Path(__file__).resolve().parent.parent / "counterrotating-motion-source-mirror.git"
OUTDIR = Path(__file__).resolve().parent


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(MIRROR), *args])


def branch_files(branch: str) -> list[tuple[str, str]]:
    rows = []
    for record in git("ls-tree", "-r", "-z", branch, "research/v17c").split(b"\0"):
        if not record:
            continue
        meta, path = record.split(b"\t", 1)
        mode, typ, oid = meta.decode("ascii").split()
        if typ != "blob":
            continue
        rows.append((path.decode("utf-8"), oid))
    return rows


def hash_blobs(oids: list[str]) -> dict[str, tuple[int, str]]:
    proc = subprocess.Popen(["git", "-C", str(MIRROR), "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdin and proc.stdout
    out = {}
    for i, oid in enumerate(oids, 1):
        proc.stdin.write((oid + "\n").encode("ascii"))
        proc.stdin.flush()
        header = proc.stdout.readline().decode("ascii").strip().split()
        if len(header) != 3 or header[1] != "blob":
            raise RuntimeError(f"invalid cat-file response for {oid}: {header}")
        size = int(header[2])
        remaining = size
        digest = hashlib.sha256()
        while remaining:
            chunk = proc.stdout.read(min(1 << 20, remaining))
            if not chunk:
                raise EOFError(oid)
            digest.update(chunk)
            remaining -= len(chunk)
        if proc.stdout.read(1) != b"\n":
            raise RuntimeError(f"cat-file delimiter error {oid}")
        out[oid] = (size, digest.hexdigest())
        if i % 500 == 0:
            print(f"hashed {i}/{len(oids)} unique V17C blobs", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("cat-file --batch failed")
    return out


def main() -> None:
    refs = git("for-each-ref", "--format=%(refname:short) %(objectname)", "refs/heads/research").decode().splitlines()
    branches = [(line.split()[0], line.split()[1]) for line in refs if line.startswith("research/v17c")]
    assert len(branches) == 14, len(branches)
    entries = {branch: branch_files(branch) for branch, _ in branches}
    oids = sorted({oid for files in entries.values() for _, oid in files})
    hashes = hash_blobs(oids)
    integrity = OUTDIR / "V17C_FILE_INTEGRITY.csv"
    with integrity.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_branch", "path", "git_blob_sha1", "bytes", "sha256"])
        for branch, _ in branches:
            for path, oid in entries[branch]:
                size, digest = hashes[oid]
                writer.writerow([branch, path, oid, size, digest])
    branch_map = []
    for branch, sha in branches:
        tree_sha = git("rev-parse", f"{branch}:research/v17c").decode().strip()
        branch_map.append({"source_branch": branch, "source_commit_sha": sha,
                           "archive_branch": branch, "archive_commit_sha_expected": sha,
                           "v17c_file_count": len(entries[branch]), "v17c_tree_sha": tree_sha,
                           "status": "SOURCE_HASHED_PENDING_REMOTE_VERIFICATION"})
    (OUTDIR / "branch_history_map.json").write_text(json.dumps(branch_map, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"branches": len(branch_map), "unique_v17c_blobs": len(oids),
                      "integrity_rows": sum(len(v) for v in entries.values()),
                      "integrity_csv_sha256": hashlib.sha256(integrity.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
