"""Resume the V17C.10 local15 holdout with an explicit worker count.

The scientific runner remains unchanged; this launcher only sets the number
of independent trajectory workers and records the execution profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_local15_refinement_holdout as setup


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=min(11, os.cpu_count() or 1))
    args = parser.parse_args()
    if not 1 <= args.threads <= (os.cpu_count() or 1):
        raise ValueError(f"--threads must be in [1, {os.cpu_count() or 1}]")

    out = setup.OUT
    out.mkdir(parents=True, exist_ok=True)
    execution = {
        "study": "V17C.10 local15 refinement strict matched holdout",
        "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "resumed_worker_count": args.threads,
        "logical_processors_reported": os.cpu_count(),
        "trajectory_parallelism": "ThreadPoolExecutor over independent Numba nogil trajectories",
        "scientific_runner": str((HERE / "run_local15_refinement_holdout.py").relative_to(HERE.parents[2])).replace("/", "\\"),
        "scientific_runner_sha256": setup.sha256(HERE / "run_local15_refinement_holdout.py"),
        "parallel_launcher_sha256": sha256(Path(__file__).resolve()),
        "previous_worker_count": 4,
        "resume_policy": "continue from validated complete JSONL rows; frozen starts and phases unchanged",
    }
    (out / "parallel_execution.json").write_text(
        json.dumps(execution, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    candidates = setup.load_candidates()
    starts, phases = setup.make_inputs()
    start_paths = setup.prepare_starts(starts, phases)
    module = __import__("basin_scan")
    field_model = setup.load_model()
    args_for_runner = argparse.Namespace(threads=args.threads)
    print(json.dumps(execution, indent=2, ensure_ascii=False), flush=True)
    for candidate_id in setup.SELECTED:
        for group in setup.GROUPS:
            setup.run_group(candidate_id, candidates[candidate_id], group, args_for_runner,
                            module, field_model, starts, phases, start_paths[group])


if __name__ == "__main__":
    main()
