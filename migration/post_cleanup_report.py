"""Record verified source cleanup in the destination project."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

workspace = Path(__file__).resolve().parent.parent
audit = workspace / "counterrotating-motion-migration"
project = workspace / "counterrotating-motion-main"
out = project / "migration"
branches = json.loads((audit / "branch_history_map.json").read_text(encoding="utf-8"))
log = [json.loads(x) for x in (audit / "source_branch_cleanup_log.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(log) == len(branches) == 14


def heads(url: str) -> dict[str, str]:
    raw = subprocess.run(["git", "ls-remote", "--heads", url],
                         capture_output=True, text=True, check=True).stdout
    return {ref.removeprefix("refs/heads/"): sha
            for sha, ref in (line.split() for line in raw.splitlines())}


src = heads("https://github.com/snejokzboxik/three-phase-12-electrode-trap.git")
dst = heads("https://github.com/snejokzboxik/counterrotating-motion.git")
assert src == {
    "master": "e5e6b2473bf519c5670ee1c48882183fbcfc8f2d",
    "research/v17b": "9e49ae1d921841a62fbe5d30880722dc8660ed6d",
}
assert len(dst) == 15 and "main" in dst
for b in branches:
    name, sha = b["source_branch"], b["source_commit_sha"]
    assert name not in src and dst[name] == sha
mirror = workspace / "counterrotating-motion-source-mirror.git"
for branch in ("master", "research/v17b"):
    p = subprocess.run(
        ["git", "-C", str(mirror), "cat-file", "-e", f"{branch}:research/v17c"],
        capture_output=True,
    )
    assert p.returncode != 0, f"V17C catalog still in {branch}"

manifest = json.loads((out / "MIGRATION_MANIFEST.json").read_text(encoding="utf-8"))
assert manifest["status"] == "MIGRATION_VERIFIED_PRE_SOURCE_CLEANUP"
manifest["status"] = "MIGRATION_AND_SOURCE_CLEANUP_VERIFIED"
manifest["source_cleanup"] = {
    "result": "COMPLETE",
    "deleted_branch_count": 14,
    "source_remaining_heads": src,
    "target_head_count": len(dst),
    "target_main_sha_before_cleanup_report_commit": dst["main"],
    "source_master_v17c_catalog": "ABSENT_IN_TREE",
    "source_v17b_v17c_catalog": "ABSENT_IN_TREE",
    "verification_utc": datetime.now(timezone.utc).isoformat(),
    "log": "migration/source_branch_cleanup_log.jsonl",
}
(out / "MIGRATION_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
shutil.copy2(audit / "source_branch_cleanup_log.jsonl", out / "source_branch_cleanup_log.jsonl")
(out / "SOURCE_REPOSITORY_CLEANUP_REPORT.md").write_text(
    "# Source repository cleanup\n\n"
    "After migration gates A–F passed, the 14 V17C refs listed in "
    "`BRANCH_HISTORY_MAP.md` were individually rechecked and removed from "
    "`snejokzboxik/three-phase-12-electrode-trap` with ordinary Git pushes. "
    "No wildcard or force push was used. The per-ref log records the exact "
    "deleted source SHA and the preserved target SHA.\n\n"
    "A fresh remote listing after cleanup contains only:\n\n"
    "- `master` at `e5e6b2473bf519c5670ee1c48882183fbcfc8f2d`;\n"
    "- `research/v17b` at `9e49ae1d921841a62fbe5d30880722dc8660ed6d`.\n\n"
    "Neither current branch contains `research/v17c/`; therefore no "
    "master cleanup commit was needed. The source master SHA did not change. "
    "The new repository retains `main` and all 14 V17C historical refs, each "
    "at its original commit SHA. The temporary `migration/bootstrap` target "
    "ref was removed after main publication.\n\n"
    "Historical objects may remain reachable through GitHub storage until "
    "garbage collection; no history rewriting or force push was attempted.\n",
    encoding="utf-8",
)
(out / "MIGRATION_REPORT.md").write_text(
    (out / "MIGRATION_REPORT.md").read_text(encoding="utf-8")
    + "\nSource cleanup completed and independently rechecked; "
      "see `SOURCE_REPOSITORY_CLEANUP_REPORT.md`.\n",
    encoding="utf-8",
)
print("PASS: source cleanup verified, report and manifest updated", flush=True)
