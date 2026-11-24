"""Compare published archive refs against the source without trusting local refs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
branches = json.loads((root / "branch_history_map.json").read_text(encoding="utf-8"))
source_url = "https://github.com/snejokzboxik/three-phase-12-electrode-trap.git"
target_url = "https://github.com/snejokzboxik/counterrotating-motion.git"


def heads(url: str) -> dict[str, str]:
    p = subprocess.run(
        ["git", "ls-remote", "--heads", url], capture_output=True,
        text=True, check=True, timeout=120,
    )
    return {ref.removeprefix("refs/heads/"): sha for sha, ref in
            (line.split() for line in p.stdout.splitlines())}


src, dst = heads(source_url), heads(target_url)
errors = []
records = []
for b in branches:
    name, expected = b["source_branch"], b["source_commit_sha"]
    source_sha, target_sha = src.get(name), dst.get(name)
    if source_sha != expected or target_sha != expected:
        errors.append({"branch": name, "expected": expected,
                       "source": source_sha, "target": target_sha})
    records.append({"name": name, "source_sha": source_sha,
                    "target_sha": target_sha,
                    "pass": source_sha == target_sha == expected})
extra = sorted(name for name in src if name.startswith("research/v17c")
               and name not in {b["source_branch"] for b in branches})
if extra:
    errors.append({"new_source_v17c_branches": extra})
out = {"records": records, "source_master": src.get("master"),
       "target_main": dst.get("main"), "unexpected_source_v17c": extra,
       "errors": errors}
(root / "remote_ref_verification.json").write_text(
    json.dumps(out, indent=2) + "\n", encoding="utf-8"
)
print(f"checked={len(records)} pass={sum(x['pass'] for x in records)} "
      f"source_master={out['source_master']} target_main={out['target_main']}")
if errors:
    print(json.dumps(errors, indent=2), file=sys.stderr)
    raise SystemExit(1)
