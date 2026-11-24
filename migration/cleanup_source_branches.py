"""Delete exactly the 14 checked source refs after verified migration.

Requires an explicit --execute flag and rechecks every source SHA immediately
before each deletion.  This script never touches master or other branches.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

here = Path(__file__).resolve().parent
source = "https://github.com/snejokzboxik/three-phase-12-electrode-trap.git"
target = "https://github.com/snejokzboxik/counterrotating-motion.git"
branches = json.loads((here / "branch_history_map.json").read_text(encoding="utf-8"))
manifest = json.loads((here.parent / "counterrotating-motion-main/migration/MIGRATION_MANIFEST.json").read_text(encoding="utf-8"))
assert manifest["status"] == "MIGRATION_VERIFIED_PRE_SOURCE_CLEANUP"
assert len(branches) == len(manifest["branches"]) == 14
assert manifest["integrity"]["sha256_mismatches"] == 0
assert manifest["backup"]["bundle_verify"] == "PASS_COMPLETE_HISTORY"

parser = argparse.ArgumentParser()
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()


def heads(url: str) -> dict[str, str]:
    out = subprocess.run(
        ["git", "ls-remote", "--heads", url],
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    return {ref.removeprefix("refs/heads/"): sha
            for sha, ref in (line.split() for line in out.splitlines())}


src, dst = heads(source), heads(target)
for b in branches:
    name, sha = b["source_branch"], b["source_commit_sha"]
    if src.get(name) != sha or dst.get(name) != sha:
        raise RuntimeError(f"preflight SHA mismatch: {name}")
extra = [name for name in src if name.startswith("research/v17c")
         and name not in {b["source_branch"] for b in branches}]
if extra:
    raise RuntimeError(f"new source V17C refs require migration: {extra}")
if not args.execute:
    print("DRY RUN: all 14 exact source/target refs verified; no deletion")
    raise SystemExit(0)

log = here / "source_branch_cleanup_log.jsonl"
for b in branches:
    name, sha = b["source_branch"], b["source_commit_sha"]
    if heads(source).get(name) != sha or heads(target).get(name) != sha:
        raise RuntimeError(f"ref changed before deletion: {name}")
    p = subprocess.run(
        ["git", "push", source, f":refs/heads/{name}"],
        capture_output=True, text=True, timeout=300,
    )
    if p.returncode:
        raise RuntimeError(f"deletion failed for {name}: {p.stderr[-1000:]}")
    if name in heads(source):
        raise RuntimeError(f"source ref still exists after deletion: {name}")
    record = {"time_utc": datetime.now(timezone.utc).isoformat(),
              "branch": name, "deleted_source_sha": sha,
              "verified_target_sha": sha, "result": "DELETED_VERIFIED"}
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"DELETED {name} {sha}", flush=True)

remaining = [name for name in heads(source) if name.startswith("research/v17c")]
if remaining:
    raise RuntimeError(f"V17C refs remain: {remaining}")
print("COMPLETE: all 14 source V17C refs deleted, target refs preserved")
