"""Write machine-readable and human-readable audit records after all gates pass."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

workspace = Path(__file__).resolve().parent.parent
audit = workspace / "counterrotating-motion-migration"
project = workspace / "counterrotating-motion-main"
clean = workspace / "counterrotating-motion-clean-verify"
bundle = audit / "counterrotating-motion-full.bundle"
branches = json.loads((audit / "branch_history_map.json").read_text(encoding="utf-8"))
remote = json.loads((audit / "remote_ref_verification.json").read_text(encoding="utf-8"))
files = json.loads((clean / "migration/clean_clone_file_verification.json").read_text(encoding="utf-8"))
recovered = json.loads((project / "migration/local_recovered_manifest.json").read_text(encoding="utf-8"))
if remote["errors"] or len(remote["records"]) != 14 or files["sha256_mismatches"] != 0:
    raise RuntimeError("publication or file-integrity gate not passed")
if files["unique_blobs"] != 3095 or files["ledger_rows"] != 28577:
    raise RuntimeError("unexpected file integrity ledger size")

verify = subprocess.run(
    ["git", "-C", str(workspace / "counterrotating-motion-source-mirror.git"),
     "bundle", "verify", str(bundle)], capture_output=True, text=True,
)
if verify.returncode or "complete history" not in verify.stderr + verify.stdout:
    raise RuntimeError("bundle verification failed: " + verify.stderr[-1000:])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


out = project / "migration"
shutil.copy2(audit / "remote_ref_verification.json", out / "remote_ref_verification.json")
shutil.copy2(clean / "migration/clean_clone_file_verification.json", out / "clean_clone_file_verification.json")
shutil.copy2(audit / "verify_clean_clone.py", out / "verify_clean_clone.py")
shutil.copy2(audit / "verify_remote_refs.py", out / "verify_remote_refs.py")

for b in branches:
    b["status"] = "VERIFIED_REMOTE_SHA_AND_CLEAN_CLONE_SHA256"
(out / "branch_history_map.json").write_text(
    json.dumps(branches, indent=2) + "\n", encoding="utf-8"
)

bundle_sha = digest(bundle)
ledger_sha = digest(out / "V17C_FILE_INTEGRITY.csv")
manifest = {
    "project": "Counterrotating Motion",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "MIGRATION_VERIFIED_PRE_SOURCE_CLEANUP",
    "source_repository": "https://github.com/snejokzboxik/three-phase-12-electrode-trap",
    "target_repository": "https://github.com/snejokzboxik/counterrotating-motion",
    "source_master_sha": remote["source_master"],
    "target_main_sha_before_final_audit_commit": remote["target_main"],
    "branches": [{
        "name": b["source_branch"],
        "source_sha": b["source_commit_sha"],
        "target_sha": b["source_commit_sha"],
        "v17c_file_count": b["v17c_file_count"],
        "v17c_tree_sha": b["v17c_tree_sha"],
        "file_hash_ledger": "migration/V17C_FILE_INTEGRITY.csv",
        "git_lfs": False,
        "uncommitted_scientific_results": "8 recovered files from local v17c10-research checkout; see local_recovered_manifest.json",
        "verification": b["status"],
    } for b in branches],
    "integrity": {
        **files,
        "ledger_sha256": ledger_sha,
        "recovered_local_files": len(recovered),
        "recovered_local_bytes": sum(x["bytes"] for x in recovered),
    },
    "backup": {
        "bundle_path_local": str(bundle),
        "bundle_bytes": bundle.stat().st_size,
        "bundle_sha256": bundle_sha,
        "bundle_verify": "PASS_COMPLETE_HISTORY",
    },
    "reproducibility": {
        "fresh_clone": "PASS; core.longpaths=true required on Windows",
        "selected_tests": "30 passed (V14/V15=17, V18B19=4, V21A/B/V22=9); 6 subtests passed",
        "saved_npz_load": "PASS (V20B, V21, V22 sampled)",
        "latex_main": "Tectonic compiled V21A and V22 source; see FILE_INTEGRITY_REPORT.md for historical limitations",
    },
    "source_cleanup": "PENDING; no source refs deleted at manifest creation",
}
(out / "MIGRATION_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
(out / "FILE_INTEGRITY_REPORT.md").write_text(
    "# V17C file integrity and reproducibility\n\n"
    f"A fresh partial clone of the published target supplied all {files['unique_blobs']} "
    f"unique V17C Git blobs referenced by {files['ledger_rows']} branch/file records. "
    f"Their actual bytes ({files['verified_unique_blob_bytes']:,} total unique bytes) "
    "matched the source SHA-256 ledger with zero mismatches. "
    f"Ledger SHA-256: `{ledger_sha}`.\n\n"
    "The 14 source/target branch SHAs match exactly. The 8 recovered local files "
    "also matched their recorded SHA-256 in the fresh clone. No Git LFS objects "
    "or GitHub release assets were found.\n\n"
    "Selected published-state tests: 17 V14/V15 tests and 6 subtests, "
    "4 V18B19 tests, and 9 V21A/B/V22 tests passed. V20B, V21, and V22 NPZ "
    "arrays loaded without pickle.\n\n"
    "Tectonic compiled the standalone V21A and V22 analytical TeX documents "
    "from the fresh clone. Historical V14 English TeX fails at a literal "
    "backtick/underscore expression; V21 Russian TeX needs a T2A font not "
    "provided by this Tectonic installation; V21B is a LaTeX fragment without "
    "a preamble. These are source/environment issues, not missing migrated files. "
    "The historical originals remain unchanged on archive branches.\n\n"
    "The full-history bundle passed `git bundle verify`; its SHA-256 is "
    f"`{bundle_sha}`, size {bundle.stat().st_size:,} bytes. The bundle is "
    "outside the source repository and its worktrees.\n",
    encoding="utf-8",
)
(out / "MIGRATION_REPORT.md").write_text(
    "# V17C migration report\n\n"
    "The 14 historical V17C branch refs and their exact commits have been "
    "published at `snejokzboxik/counterrotating-motion`. The new `main` "
    "descends from V17C.22 and contains the V17C catalog, recovered local files, "
    "and migration audit. The historical branches are unchanged.\n\n"
    "A source mirror contained the full commit graph and all Git objects; "
    "`git fsck --full --strict` passed before publication. Remote branch SHA "
    "comparison, clean clone, all scientific file SHA-256 checks, selected "
    "tests/data loads, TeX availability, and independent bundle verification "
    "are documented in the manifest and file integrity report.\n\n"
    "The source master had no `research/v17c/` catalog and was not changed. "
    "The source `research/v17b` branch also lacks that catalog. Source branch "
    "deletion is recorded separately after the final pre-deletion SHA check.\n",
    encoding="utf-8",
)
print("PASS: final migration records created", bundle_sha, flush=True)
