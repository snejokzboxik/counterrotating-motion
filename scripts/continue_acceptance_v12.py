"""Extend saved V12 trajectories to a later horizon without replaying prior RF periods.

The source run fixes the physical field, injection states, RF phases, and solver.
Provide a file with one ``sample_id`` per line so long continuations remain a
pre-registered, manageable subset instead of silently extending the full grid.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/octupole_comparison_home/dimensionless_acceptance_v12/runs"
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap.dimensionless_dynamics_v12 import DimensionlessRFModel, integrate_to_horizon


_MODELS: dict[str, dict] = {}
_PHYSICAL: dict[str, float] = {}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _decode(pairs: list[list[float]]) -> np.ndarray:
    return np.asarray([complex(float(re), float(im)) for re, im in pairs], dtype=complex)


def _init_worker(specs: list[dict], physical: dict[str, float]) -> None:
    global _MODELS, _PHYSICAL
    _MODELS = {spec["name"]: spec for spec in specs}
    _PHYSICAL = physical


def _worker(job: tuple) -> dict:
    (job_id, model_name, sample_id, state, phase_rad, source_row,
     source_period, horizon, segment, rtol, atol, max_step_rf,
     samples_per_rf, config_hash, checkpoint_path) = job
    checkpoint = Path(checkpoint_path)
    if checkpoint.exists():
        try:
            prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        except Exception:
            prior = None
        if prior and prior.get("config_hash") == config_hash:
            if prior.get("complete"):
                return prior
            start_period = float(prior.get("reached_RF", source_period))
            resume_state = np.asarray(prior.get("state_dimensionless", state), dtype=float)
            old_max = float(prior.get("max_radius_sampled_rho", source_row.get("max_radius_sampled_rho", 0.0)))
            old_nfev = int(prior.get("nfev", 0))
        else:
            start_period = source_period
            resume_state = np.asarray(state, dtype=float)
            old_max = float(source_row.get("max_radius_sampled_rho", 0.0))
            old_nfev = int(source_row.get("nfev", 0))
    else:
        start_period = source_period
        resume_state = np.asarray(state, dtype=float)
        old_max = float(source_row.get("max_radius_sampled_rho", 0.0))
        old_nfev = int(source_row.get("nfev", 0))

    # An earlier first exit is absorbing. Preserve it exactly and do not spend
    # compute extending a trajectory whose classification cannot change.
    if source_row.get("status") == "escaped":
        row = dict(source_row)
        row.update({"job_id": job_id, "model": model_name, "sample_id": sample_id,
                    "config_hash": config_hash, "horizon_RF": horizon,
                    "continued_from_RF": source_period, "complete": True})
        _atomic_json(checkpoint, row)
        return row

    spec = _MODELS[model_name]
    model = DimensionlessRFModel.from_physical(
        _decode(spec["cosine"]), _decode(spec["sine"]),
        charge_c=_PHYSICAL["charge_C"], mass_kg=_PHYSICAL["mass_kg"],
        voltage_v=spec["voltage_V"], omega_rad_s=2.0 * math.pi * _PHYSICAL["frequency_Hz"],
        length_m=spec["length_m"], reference_radius_m=spec["reference_radius_m"],
        aperture_m=spec["aperture_rho"] * spec["length_m"],
        phase_rad=phase_rad, name=model_name,
    )

    def save_segment(period: float, current: np.ndarray, maximum: float, nfev: int) -> None:
        _atomic_json(checkpoint, {
            "job_id": job_id, "model": model_name, "sample_id": sample_id,
            "config_hash": config_hash, "status": "retained", "complete": False,
            "horizon_RF": horizon, "continued_from_RF": source_period,
            "reached_RF": period, "state_dimensionless": current.tolist(),
            "escape_RF": None, "max_radius_sampled_rho": max(old_max, maximum),
            "nfev": old_nfev + nfev,
        })

    result = integrate_to_horizon(
        model, np.asarray(state, dtype=float), horizon,
        segment_periods=segment, rtol=rtol, atol=atol,
        max_step_rf=max_step_rf, samples_per_rf=samples_per_rf,
        checkpoint_callback=save_segment, resume_state=resume_state,
        resume_period=start_period,
    )
    row = {
        "job_id": job_id, "model": model_name, "sample_id": sample_id,
        "config_hash": config_hash, "kind": spec["kind"], "mesh_um": spec["mesh_um"],
        "voltage_V": spec["voltage_V"], "scale_factor": spec["scale_factor"],
        "source_config_hash": source_row.get("config_hash"),
        "horizon_RF": horizon, "continued_from_RF": source_period,
        "phase_rad": phase_rad, "status": result.status,
        "complete": result.status != "numerical_error",
        "reached_RF": result.reached_period, "escape_RF": result.escape_period,
        "state_dimensionless": result.final_state.tolist(),
        "max_radius_sampled_rho": max(old_max, result.max_radius_sampled_rho),
        "nfev": old_nfev + result.nfev, "message": result.message,
    }
    _atomic_json(checkpoint, row)
    return row


def _read_source(source: Path) -> tuple[dict, list[dict], dict[str, dict], dict[str, tuple[np.ndarray, float]]]:
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    specs = json.loads((source / "model_coefficients.json").read_text(encoding="utf-8"))
    rows_by_job: dict[str, dict] = {}
    journal = source / "completed_jobs.jsonl"
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                rows_by_job[row["job_id"]] = row
            except (json.JSONDecodeError, KeyError):
                continue
    states: dict[str, tuple[np.ndarray, float]] = {}
    for path in sorted(source.glob("initial_states_r*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                sample_id = f"r{int(row['replicate']):02d}_{int(row['sample_index']):06d}"
                length = cfg["physical"]["length_m"]
                omega = 2.0 * math.pi * cfg["physical"]["frequency_Hz"]
                state = np.array([float(row["x0_m"]) / length, float(row["y0_m"]) / length,
                                  float(row["vx0_m_s"]) / (length * omega),
                                  float(row["vy0_m_s"]) / (length * omega)])
                states[sample_id] = (state, float(row["phase_rad"]))
    return cfg, specs, rows_by_job, states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="run directory name under V12 runs/ or its absolute path")
    parser.add_argument("--sample-id-file", required=True, help="UTF-8 text file, one saved sample_id per line")
    parser.add_argument("--horizon", required=True, type=int, help="new cumulative horizon in RF periods")
    parser.add_argument("--segment", type=int, default=250)
    parser.add_argument("--models", default="", help="comma-separated exact model names; default is all models in source")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--max-step-rf", type=float, default=None)
    parser.add_argument("--samples-per-rf", type=int, default=None)
    args = parser.parse_args()

    source = Path(args.source_run)
    if not source.is_absolute():
        source = RUNS / source
    source = source.resolve()
    cfg, specs, source_rows, states = _read_source(source)
    source_period = float(cfg["horizon_RF"])
    if args.horizon <= source_period or args.segment < 1 or args.horizon % args.segment:
        parser.error("target horizon must exceed source horizon and be divisible by a positive segment")
    if args.workers < 1:
        parser.error("workers must be positive")
    ids = [line.strip() for line in Path(args.sample_id_file).read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    if not ids or len(set(ids)) != len(ids):
        parser.error("sample-id-file must contain a non-empty list of unique IDs")
    missing_ids = sorted(set(ids) - set(states))
    if missing_ids:
        parser.error(f"sample IDs not found in source run: {missing_ids[:5]}")

    by_name = {spec["name"]: spec for spec in specs}
    model_names = [name.strip() for name in args.models.split(",") if name.strip()] or list(by_name)
    missing_models = sorted(set(model_names) - set(by_name))
    if missing_models:
        parser.error(f"models not found in source run: {missing_models}")
    solver = cfg["solver"]
    rtol = args.rtol or solver["rtol"]
    atol = args.atol or solver["atol_dimensionless"]
    max_step_rf = args.max_step_rf or solver["max_step_rf_fraction"]
    samples_per_rf = args.samples_per_rf or solver["samples_per_rf_for_max_radius"]
    sample_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    run_config = {
        "study": "V12 sequential trajectory continuation",
        "source_run": source.name, "source_config_hash": cfg["config_hash"],
        "source_horizon_RF": source_period, "horizon_RF": args.horizon,
        "segment_RF": args.segment, "sample_id_sha256": sample_hash,
        "sample_count": len(ids), "models": model_names,
        "solver": {"method": "DOP853", "rtol": rtol, "atol_dimensionless": atol,
                   "max_step_rf_fraction": max_step_rf, "samples_per_rf": samples_per_rf},
        "physical": cfg["physical"],
    }
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    run_config["config_hash"] = config_hash
    out = RUNS / f"continuation_{source.name}_N{args.horizon}_{config_hash[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "config.json"
    if saved.exists() and json.loads(saved.read_text(encoding="utf-8")) != run_config:
        raise ValueError("continuation output already contains a different configuration")
    _atomic_json(saved, run_config)
    (out / "sample_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    selected_specs = [by_name[name] for name in model_names]
    _atomic_json(out / "model_coefficients.json", selected_specs)

    jobs = []
    for model_name in model_names:
        spec = by_name[model_name]
        for sample_id in ids:
            state, phase_rad = states[sample_id]
            source_job_id = hashlib.sha256(
                f"{cfg['config_hash']}|{sample_id}|{model_name}".encode()).hexdigest()[:24]
            source_row = source_rows.get(source_job_id)
            source_checkpoint = source / "checkpoints" / model_name / f"{sample_id}.json"
            if source_checkpoint.exists():
                try:
                    cp_row = json.loads(source_checkpoint.read_text(encoding="utf-8"))
                    if cp_row.get("config_hash") == cfg["config_hash"]:
                        source_row = cp_row
                except json.JSONDecodeError:
                    pass
            if source_row is None or "state_dimensionless" not in source_row:
                parser.error(f"no saved continuation state for {model_name}/{sample_id}; refusing to replay from RF=0")
            start_period = float(source_row.get("reached_RF", source_period))
            if source_row.get("status") == "retained" and source_row.get("complete"):
                start_period = max(source_period, start_period)
            source_state = source_row.get("state_dimensionless", state.tolist())
            job_id = hashlib.sha256(f"{config_hash}|{sample_id}|{model_name}".encode()).hexdigest()[:24]
            checkpoint = out / "checkpoints" / model_name / f"{sample_id}.json"
            jobs.append((job_id, model_name, sample_id, source_state, phase_rad, source_row,
                         start_period, args.horizon, args.segment, rtol, atol, max_step_rf,
                         samples_per_rf, config_hash, str(checkpoint)))

    journal = out / "completed_jobs.jsonl"
    done_ids = set()
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("config_hash") == config_hash and row.get("complete"):
                    done_ids.add(row["job_id"])
            except json.JSONDecodeError:
                continue
    jobs = [job for job in jobs if job[0] not in done_ids]
    _atomic_json(out / "run_state.json", {"status": "running", "planned_trajectories": len(model_names) * len(ids),
                                           "queued_jobs_this_invocation": len(jobs), "config_hash": config_hash})
    print(json.dumps({"out": str(out), "source_horizon_RF": source_period,
                      "target_horizon_RF": args.horizon, "new_jobs": len(jobs),
                      "samples": len(ids), "models": model_names, "workers": args.workers}, indent=2), flush=True)

    began = time.perf_counter()
    completed: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(selected_specs, cfg["physical"])) as pool, journal.open("a", encoding="utf-8") as stream:
        futures = [pool.submit(_worker, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            completed.append(row)
            if len(completed) % 25 == 0:
                stream.flush(); os.fsync(stream.fileno())
                print(f"completed {len(completed)}/{len(jobs)}; elapsed={time.perf_counter()-began:.1f}s", flush=True)
    errors = sum(row.get("status") == "numerical_error" for row in completed)
    states_count = {status: sum(row.get("status") == status for row in completed)
                    for status in ("retained", "escaped", "numerical_error")}
    final = {"status": "complete" if errors == 0 and len(done_ids) + len(completed) == len(model_names) * len(ids) else "incomplete",
             "planned_trajectories": len(model_names) * len(ids),
             "complete_trajectories": len(done_ids) + len(completed) - errors,
             "status_counts_this_invocation": states_count,
             "wall_seconds_this_invocation": time.perf_counter()-began,
             "config_hash": config_hash, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _atomic_json(out / "run_state.json", final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
