"""Fresh N=256 strict confirmation for the V17C.10 local refinement leaders."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
import run_local15_refinement_holdout as setup

N = 256
GROUPS = ("rest", "weak", "wide", "edge")
SEEDS = {"rest": 20261610, "weak": 20261611, "wide": 20261612, "edge": 20261613}
RUN_TAG = "refineconfirm20261610"
SELECTED = (0, 12, 18, 32, 35)
OUT = HERE / "local15_refinement_confirmation_N256"
START_DIR = OUT / "holdout_starts"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_inputs():
    starts, phases = {}, {}
    for group in GROUPS[:3]:
        starts[group], phases[group] = setup.make_initials(group, N, SEEDS[group])
    starts["edge"], phases["edge"] = setup.make_initials("wide", N, SEEDS["edge"])
    u = qmc.Sobol(d=2, scramble=True, seed=SEEDS["edge"] + 7919).random_base2(8)
    radial = np.sqrt(setup.EDGE_RMIN**2 + u[:, 0] * (setup.EDGE_RMAX**2 - setup.EDGE_RMIN**2))
    angle = 2.0 * math.pi * u[:, 1]
    for i in range(N):
        offset = 0 if i % 2 == 0 else 6
        starts["edge"][i, offset] = radial[i] * math.cos(angle[i])
        starts["edge"][i, offset + 1] = radial[i] * math.sin(angle[i])
    return starts, phases


def prepare_starts(starts, phases):
    START_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for group in GROUPS:
        path = START_DIR / f"{group}_N{N}_seed{SEEDS[group]}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (not np.array_equal(saved["initial"], starts[group])
                        or not np.array_equal(saved["phases"], phases[group])
                        or int(saved["seed"]) != SEEDS[group]):
                    raise RuntimeError(f"saved N=256 confirmation starts differ: {group}")
        else:
            np.savez_compressed(path, initial=starts[group], phases=phases[group], seed=SEEDS[group])
        paths[group] = path
    return paths


def configure_setup():
    """Reuse the audited integration code with independent N=256 inputs/outputs."""
    setup.N = N
    setup.GROUPS = GROUPS
    setup.START_SEEDS = SEEDS
    setup.RUN_TAG = RUN_TAG
    setup.SELECTED = SELECTED
    setup.OUT = OUT
    setup.START_DIR = START_DIR
    setup.make_inputs = make_inputs
    setup.prepare_starts = prepare_starts
    # run_group stores the hash of the actual confirmation protocol runner.
    setup.__file__ = str(Path(__file__).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=min(11, os.cpu_count() or 1))
    args = parser.parse_args()
    if not 1 <= args.threads <= (os.cpu_count() or 1):
        raise ValueError(f"--threads must be in [1, {os.cpu_count() or 1}]")
    configure_setup()
    OUT.mkdir(parents=True, exist_ok=True)
    execution = {
        "study": "V17C.10 local15 refinement independent N=256 confirmation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "worker_count": args.threads,
        "logical_processors_reported": os.cpu_count(),
        "trajectory_parallelism": "ThreadPoolExecutor over independent Numba nogil trajectories",
        "confirmation_runner": str(Path(__file__).resolve().relative_to(HERE.parents[2])).replace("/", "\\"),
        "confirmation_runner_sha256": sha256(Path(__file__).resolve()),
        "scientific_integrator_runner_sha256": setup.sha256(HERE / "run_local15_refinement_holdout.py"),
        "sample_size_per_candidate_and_ensemble": N,
        "fresh_start_seeds": SEEDS,
        "same_frozen_starts_and_phases_within_ensemble": True,
        "resume_policy": "continue from complete JSONL rows; frozen starts and phases must match exactly",
    }
    (OUT / "parallel_execution.json").write_text(
        json.dumps(execution, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    candidates = setup.load_candidates()
    starts, phases = make_inputs()
    start_paths = prepare_starts(starts, phases)
    module = __import__("basin_scan")
    field_model = setup.load_model()
    args_for_runner = argparse.Namespace(threads=args.threads)
    print(json.dumps(execution, indent=2, ensure_ascii=False), flush=True)
    for candidate_id in SELECTED:
        for group in GROUPS:
            setup.run_group(candidate_id, candidates[candidate_id], group, args_for_runner,
                            module, field_model, starts, phases, start_paths[group])


if __name__ == "__main__":
    main()
