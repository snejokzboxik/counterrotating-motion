"""Create the standalone project landing page and preserve recovered local V17C files."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
MIGRATION = WORKSPACE / "counterrotating-motion-migration"
PROJECT = WORKSPACE / "counterrotating-motion-main"
SOURCE_LOCAL = WORKSPACE / "v17c10-research"

branches = json.loads((MIGRATION / "branch_history_map.json").read_text(encoding="utf-8"))
local_inventory = json.loads((MIGRATION / "local_untracked_inventory.json").read_text(encoding="utf-8"))

out = PROJECT / "migration"
out.mkdir(exist_ok=True)
shutil.copy2(MIGRATION / "V17C_FILE_INTEGRITY.csv", out / "V17C_FILE_INTEGRITY.csv")
shutil.copy2(MIGRATION / "branch_history_map.json", out / "branch_history_map.json")
shutil.copy2(MIGRATION / "local_untracked_inventory.json", out / "local_untracked_inventory.json")

recovered = []
for entry in local_inventory["entries"]:
    if entry["generated_bytecode"] or entry["exact_blob_in_remote_history"]:
        continue
    src = SOURCE_LOCAL / entry["path"]
    dest = out / "local_recovered" / entry["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(f"hash mismatch: {src}")
    recovered.append({**entry, "project_path": dest.relative_to(PROJECT).as_posix()})
(out / "local_recovered_manifest.json").write_text(
    json.dumps(recovered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

(PROJECT / ".gitignore").write_text(
    "__pycache__/\n*.py[cod]\n.pytest_cache/\n.venv/\n*.egg-info/\n"
    "build/\ndist/\n*.bundle\n",
    encoding="utf-8",
)
(PROJECT / "README.md").write_text(
    "# Counterrotating Motion\n\n"
    "Research archive and continuing analytical study of counterrotating charged "
    "particles in a two-frequency 24-electrode RF trap.\n\n"
    "The [V17C research catalog](research/v17c/) contains the complete published "
    "scientific data, code, reports, and figures from the latest historical branch. "
    "Fourteen unchanged `research/v17c*` branches retain earlier and divergent results. "
    "The root `migration/` directory documents source commits, per-file SHA-256 "
    "checksums, verification, and recovered local files.\n\n"
    "Latest report: [V17C.22 stable counterrotation theory]"
    "(research/v17c/v17c_22/STABLE_COUNTERROTATION_THEORY_RU.md). "
    "The published long-lived counterrotation has **not** established a "
    "mathematically proven attracting torus.\n\n"
    "Historical branch commits also contain the source 12-electrode project's root "
    "files. This `main` branch keeps the V17C catalog as the independent research "
    "project; historical commits remain accessible unchanged.\n",
    encoding="utf-8",
)

lines = [
    "# Branch history map",
    "",
    "All source commits are preserved without rewriting in the new repository.",
    "The file-level SHA-256 ledger is `migration/V17C_FILE_INTEGRITY.csv`.",
    "",
    "| Historical branch | Exact commit | V17C files |",
    "|---|---|---:|",
]
for b in branches:
    lines.append(
        f"| `{b['source_branch']}` | `{b['source_commit_sha']}` | "
        f"{b['v17c_file_count']} |"
    )
lines.append("")
(out / "BRANCH_HISTORY_MAP.md").write_text("\n".join(lines), encoding="utf-8")

(out / "MIGRATION_REPORT.md").write_text(
    "# V17C migration report\n\n"
    "Source: `snejokzboxik/three-phase-12-electrode-trap`. "
    "Destination: `snejokzboxik/counterrotating-motion`.\n\n"
    "The latest historical V17C commit is "
    "`f2c0c1c38d7ce76bb55b7a3d7b84e5e321740487`. "
    "The 14 historical branch names and commits are preserved. "
    "The main branch is a new child commit of that historical state and removes "
    "unrelated root material from the 12-electrode project. "
    "The historical branches retain it.\n\n"
    "The source has no Git LFS tracked files and no release assets. "
    f"Eight previously untracked, scientifically relevant local files ({sum(x['bytes'] for x in recovered)} bytes) "
    "are stored under `migration/local_recovered/` with a provenance manifest. "
    "Bytecode and files already identical to remote history were not duplicated.\n\n"
    "Publication and independent verification are recorded in `MIGRATION_MANIFEST.json` "
    "and `FILE_INTEGRITY_REPORT.md` after completion.\n",
    encoding="utf-8",
)
print(f"Prepared project main; {len(recovered)} recovered local files.")
