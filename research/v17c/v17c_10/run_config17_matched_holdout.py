"""Run the training leader on candidate52's frozen strict holdout starts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
V17C9 = HERE.parent / "v17c_9"
STAGE4 = V17C9 / "optimization" / "stage4_candidate48"
STARTS = STAGE4 / "holdout_starts"
OUT = HERE / "candidate17_holdout"
GROUPS = ("rest", "weak", "wide")
RUN_TAG = "pairedholdout20261150"
N = 128
PERIODS = 15000
STEPS = 64
TRANSIENT = 200
SIZES = (0.0002, 0.00189)
PARAMETER_NAMES = ("A", "B", "nu", "beta", "s_m1_over_m2", "C", "lambda1", "lambda2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scan_module():
    bundled_site = HERE.parents[2] / ".venv" / "Lib" / "site-packages"
    if bundled_site.exists() and str(bundled_site) not in sys.path:
        sys.path.insert(0, str(bundled_site))
    source = V17C9 / "baseline" / "basin_scan.py"
    spec = importlib.util.spec_from_file_location("basin_scan", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parameters_from_file(path: Path):
    source = json.loads(path.read_text(encoding="utf-8"))
    values = source["parameters"]
    p = np.asarray([values[name] for name in PARAMETER_NAMES], dtype=np.float64)
    return source, p


def expected_meta(group: str, config: dict, p: np.ndarray, qratio: float, starts_path: Path):
    return {
        "config_id": 17,
        "parameters": p.tolist(),
        "qratio": qratio,
        "ensemble": group,
        "n": N,
        "seed": None,
        "periods": PERIODS,
        "steps": STEPS,
        "transient": TRANSIENT,
        "starts": str(starts_path.relative_to(HERE.parents[2])).replace("/", "\\"),
        "candidate_source": "V17C.10 short-screen config 17",
        "candidate_training_status": config["status"],
        "coupling_scale": 1.0,
        "run_tag": RUN_TAG,
        "oseen_Re_per_speed": 0.0,
        "hydro_mode": "none",
        "oseen_Re2_per_speed": 0.0,
        "a1_over_R": SIZES[0],
        "a2_over_R": SIZES[1],
    }


def output_row(i, initial, phase, result):
    (end, turns, windows, rmin, rmax, zmax, max_dtheta, dmin, completed,
     retained, escaped, singular, category, freq) = result
    return {
        "index": int(i),
        "initial": initial.tolist(), "phase": phase.tolist(),
        "final_state": end.tolist(), "turns": turns.tolist(),
        "window_turns": windows.tolist(), "frequency_Hz": freq.tolist(),
        "r_min": rmin.tolist(), "r_max": rmax.tolist(),
        "z_max": zmax.tolist(), "max_abs_inner_step_dtheta_rad": max_dtheta.tolist(),
        "minimum_separation": float(dmin), "completed_periods": int(completed),
        "retained": bool(retained), "escaped": bool(escaped),
        "singular_unresolved": bool(singular), "category": str(category),
        "hydro_model": "none", "oseen_Re_per_speed": 0.0,
        "oseen_Re2_per_speed": 0.0,
        "a1_over_R": SIZES[0], "a2_over_R": SIZES[1],
    }


def load_existing(path: Path, meta_path: Path, metadata: dict, initial: np.ndarray, phases: np.ndarray):
    if meta_path.exists():
        saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved_meta != metadata:
            if path.exists() and path.stat().st_size:
                raise RuntimeError(f"checkpoint metadata mismatch: {meta_path}")
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            i = int(row["index"])
            if i in rows or not 0 <= i < N:
                raise RuntimeError(f"duplicate or invalid saved index in {path}: {i}")
            if not np.array_equal(np.asarray(row["initial"]), initial[i]):
                raise RuntimeError(f"checkpoint initial condition mismatch at {group_name(path)} index {i}")
            if not np.array_equal(np.asarray(row["phase"]), phases[i]):
                raise RuntimeError(f"checkpoint phase mismatch at {group_name(path)} index {i}")
            rows[i] = row
    return rows


def group_name(path: Path) -> str:
    return path.name.split("_")[1]


def run_group(group: str, args, module, model, parameters, qratio, config):
    starts_path = STARTS / f"{group}_N128_seed{20261150 + GROUPS.index(group)}.npz"
    with np.load(starts_path, allow_pickle=False) as frozen:
        initial = frozen["initial"].astype(np.float64, copy=True)
        phases = frozen["phases"].astype(np.float64, copy=True)
        seed = int(frozen["seed"])
    if initial.shape != (N, 12) or phases.shape != (N, 2):
        raise ValueError(f"unexpected frozen holdout shape for {group}")
    expected_seed = 20261150 + GROUPS.index(group)
    if seed != expected_seed:
        raise ValueError(f"unexpected frozen Sobol seed for {group}: {seed}")
    manifest = json.loads((STAGE4 / "PAIRED_HOLDOUT_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_rel = f"optimization/stage4_candidate48/holdout_starts/{starts_path.name}"
    expected_start_sha = next((x["sha256"] for x in manifest["files"] if x["path"] == manifest_rel), None)
    if expected_start_sha is None or sha256(starts_path) != expected_start_sha:
        raise ValueError(f"frozen holdout hash does not match paired manifest for {group}")

    tag = f"cfg17_{group}_matched_holdout_none_chi0-0_a0.0002-0.00189_{RUN_TAG}_{N}_{PERIODS}p_{STEPS}s"
    run_path = OUT / f"{tag}.jsonl"
    meta_path = OUT / f"{tag}.meta.json"
    metadata = expected_meta(group, config, parameters, qratio, starts_path)
    metadata["frozen_start_sha256"] = sha256(starts_path)
    metadata["fixed_field_config_sha256"] = sha256(module.SOURCE_CFG)
    metadata["runner_sha256"] = sha256(Path(__file__).resolve())
    rows = load_existing(run_path, meta_path, metadata, initial, phases)
    todo = [i for i in range(N) if i not in rows]
    if not todo:
        print(f"{group}: already complete {N}/{N}", flush=True)
        return

    first = todo[0]
    module.integrate_one(
        initial[first], phases[first], parameters, *model[1:], PERIODS, STEPS, TRANSIENT,
        qratio=qratio, oseen_Re_per_speed=0.0, hydro_mode=0,
        a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0,
    )
    print(f"{group}: n={N}, already={len(rows)}, todo={len(todo)}, workers={args.threads}", flush=True)
    t0 = time.perf_counter()

    def one(i):
        result = module.integrate_one(
            initial[i], phases[i], parameters, *model[1:], PERIODS, STEPS, TRANSIENT,
            qratio=qratio, oseen_Re_per_speed=0.0, hydro_mode=0,
            a1_over_R=SIZES[0], a2_over_R=SIZES[1], oseen_Re2_per_speed=0.0,
        )
        return output_row(i, initial[i], phases[i], result)

    with run_path.open("a", encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {pool.submit(one, i): i for i in todo}
            for n, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows[row["index"]] = row
                fout.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                if n % 16 == 0 or n == len(todo):
                    print(f"{group}: checkpoint {len(rows)}/{N} ({time.perf_counter()-t0:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--group", choices=GROUPS, help="run only one ensemble; default runs all three")
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 4:
        raise ValueError("--threads must be in [1, 4] for the recorded local resource plan")
    OUT.mkdir(parents=True, exist_ok=True)
    config, p = parameters_from_file(HERE / "sobol_leader_config17.json")
    qratio = float(config["parameters"]["q2_over_q1"])
    module = load_scan_module()
    model = module.load_model()
    model = (p, *model[1:])
    selected = (args.group,) if args.group else GROUPS
    for group in selected:
        run_group(group, args, module, model, p, qratio, config)


if __name__ == "__main__":
    main()
