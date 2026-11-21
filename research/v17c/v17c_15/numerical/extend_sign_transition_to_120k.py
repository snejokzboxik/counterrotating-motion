"""Continue the mixed-sign Local15 panel from 50k to 120k periods."""
from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rotation_sign_scan_local15 as base  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "results/rotation_sign_scan"
SOURCE_DIR = SCAN / "extended_sign_transition_50000_checkpoints"
TARGET_DIR = SCAN / "extended_sign_transition_120000_checkpoints"
RESULT_JSON = SCAN / "local15_sign_transition_120000_mixed_node.json"
RESULT_CSV = SCAN / "local15_sign_transition_120000_mixed_node.csv"
SOURCE_PERIODS = 50_000
TARGET_PERIODS = 120_000
STEPS = 96
WINDOW = 500
HORIZONS = (50_000, 60_000, 70_000, 80_000, 90_000,
            100_000, 110_000, 120_000)
NU = 2.23125
SPECIES = "species1"
WORKERS = 8


def worker_init() -> None:
    base.PERIODS = TARGET_PERIODS
    base.STEPS = STEPS
    base.WINDOW = WINDOW


def run_one_start(task: tuple) -> dict:
    row, state, phase1, phase2, output_dir = task
    base.OUT = Path(output_dir)
    result = base.run_one((NU, SPECIES, [row], state.reshape(1, 4),
                           np.array([phase1]), np.array([phase2])))
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_checkpoints(panel_rows: list[dict]) -> tuple[Path, Path, list[tuple]]:
    name = f"nu_{NU:.3f}_{SPECIES}.npz"
    source = SOURCE_DIR / name
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as data:
        if int(data["step"]) != SOURCE_PERIODS * STEPS:
            raise ValueError("source checkpoint is not exactly at 50,000 periods")
        if float(data["nu"]) != NU or str(data["species"].item()) != SPECIES:
            raise ValueError("source checkpoint metadata does not match the requested ensemble")
        if int(data["steps_per_period"]) != STEPS:
            raise ValueError("source checkpoint uses an unexpected RK4 step scheme")
        source_data = {key: data[key].copy() for key in (
            "step", "state", "active", "escaped", "escape_period", "maxrho",
            "rho_samples", "angle_samples", "nu", "species", "periods",
            "steps_per_period")}
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    name = f"nu_{NU:.3f}_{SPECIES}.npz"
    state0 = np.array([[float(r[k]) for k in ("x0", "y0", "vx0", "vy0")]
                       for r in panel_rows])
    phase1 = np.array([float(r["phase1"]) for r in panel_rows])
    phase2 = np.array([float(r["phase2"]) for r in panel_rows])
    tasks = []
    for i, row in enumerate(panel_rows):
        outdir = TARGET_DIR / row["start_id"]
        outdir.mkdir(parents=True, exist_ok=True)
        target = outdir / name
        if not target.exists():
            np.savez_compressed(target,
                step=source_data["step"], state=source_data["state"][i:i+1],
                active=source_data["active"][i:i+1],
                escaped=source_data["escaped"][i:i+1],
                escape_period=source_data["escape_period"][i:i+1],
                maxrho=source_data["maxrho"][i:i+1],
                rho_samples=source_data["rho_samples"][:, i:i+1],
                angle_samples=source_data["angle_samples"][:, i:i+1],
                nu=NU, species=SPECIES, periods=SOURCE_PERIODS,
                steps_per_period=STEPS)
        else:
            with np.load(target, allow_pickle=False) as data:
                if (float(data["nu"]) != NU or
                        str(data["species"].item()) != SPECIES or
                        int(data["step"]) < SOURCE_PERIODS * STEPS):
                    raise ValueError(f"invalid per-start checkpoint: {target}")
        tasks.append((row, state0[i], phase1[i], phase2[i], str(outdir)))

    combined = TARGET_DIR / name
    if not combined.exists():
        shutil.copy2(source, combined)
    return source, combined, tasks


def combine_start_checkpoints(panel_rows: list[dict], combined: Path) -> None:
    names = f"nu_{NU:.3f}_{SPECIES}.npz"
    columns = {key: [] for key in ("state", "active", "escaped", "escape_period",
                                   "maxrho", "rho_samples", "angle_samples")}
    step = None
    for row in panel_rows:
        path = TARGET_DIR / row["start_id"] / names
        with np.load(path, allow_pickle=False) as data:
            if int(data["step"]) != TARGET_PERIODS * STEPS:
                raise ValueError(f"incomplete continuation checkpoint: {path}")
            step = int(data["step"])
            for key in columns:
                columns[key].append(data[key].copy())
    np.savez_compressed(combined, step=step,
        **{key: (np.concatenate(value, axis=1) if key in ("rho_samples", "angle_samples")
                 else np.concatenate(value, axis=0)) for key, value in columns.items()},
        nu=NU, species=SPECIES, periods=TARGET_PERIODS,
        steps_per_period=STEPS)


def summarize(target: Path, panel_hash: str, source_hash: str,
              result: dict, elapsed: float) -> dict:
    with np.load(target, allow_pickle=False) as data:
        step = int(data["step"])
        radii = data["rho_samples"]
        angles = data["angle_samples"]
        escape_periods = data["escape_period"].copy()
        max_rho = data["maxrho"].copy()

    panel_rows = result["rows"]
    records = []
    counts = {}
    for horizon in HORIZONS:
        counts[str(horizon)] = {"negative": 0, "positive": 0,
                                "escaped": 0, "n_starts": len(panel_rows)}

    for i, row in enumerate(panel_rows):
        escape = None if not np.isfinite(escape_periods[i]) else float(escape_periods[i])
        windows = []
        for horizon in HORIZONS:
            local_omega = float((angles[horizon, i] - angles[horizon-WINDOW, i]) /
                                (2 * np.pi * WINDOW))
            escaped = escape is not None and escape <= horizon
            sign = 0 if escaped else int(np.sign(local_omega))
            windows.append({"horizon_periods": horizon,
                            "rho_mean_last_500": float(np.mean(
                                radii[horizon-WINDOW:horizon+1, i])),
                            "omega_last_500": local_omega,
                            "escaped_by_horizon": escaped,
                            "escape_period": escape, "sign": sign})
            counts[str(horizon)]["negative"] += sign < 0
            counts[str(horizon)]["positive"] += sign > 0
            counts[str(horizon)]["escaped"] += escaped
        cumulative_50_120 = float((angles[120_000, i] - angles[50_000, i]) /
                                  (2 * np.pi * 70_000))
        cumulative_15_120 = float((angles[120_000, i] - angles[15_000, i]) /
                                  (2 * np.pi * 105_000))
        records.append({"start_id": row["start_id"], "escape_period": escape,
                        "max_rho": float(max_rho[i]), "horizons": windows,
                        "omega_mean_50k_to_120k": cumulative_50_120,
                        "omega_mean_15k_to_120k": cumulative_15_120})

    retained = [r for r in records if r["escape_period"] is None or
                r["escape_period"] > TARGET_PERIODS]

    def stats(key: str) -> dict:
        values = [r[key] for r in retained]
        return {"positive": sum(x > 0 for x in values),
                "negative": sum(x < 0 for x in values),
                "escaped": len(records) - len(retained),
                "n_starts": len(records),
                "retained_min_median_max": ([float(min(values)),
                    float(np.median(values)), float(max(values))] if values else [])}

    return {
        "status": "completed",
        "model_scope": "exact ideal k=+2,-4 Local15 RF; C=0; linear drag; no full MFS",
        "purpose": "finite-horizon continuation of the mixed-window-sign fixed panel",
        "nu": NU, "species": SPECIES,
        "source_horizon_periods": SOURCE_PERIODS,
        "target_horizon_periods": TARGET_PERIODS,
        "steps_per_primary_period": STEPS,
        "measurement_window_periods": WINDOW,
        "window_horizons": list(HORIZONS),
        "panel_seed": 15152026, "panel_sha256": panel_hash,
        "panel_is_iid_probability_sample": False, "rf_phases_fixed": True,
        "workers": WORKERS,
        "continued_trajectory_segments": len(records),
        "additional_particle_periods": len(records) * (TARGET_PERIODS - SOURCE_PERIODS),
        "wall_seconds": elapsed,
        "source_checkpoint_sha256": source_hash,
        "target_checkpoint_sha256": sha256(target),
        "checkpoint": target.relative_to(ROOT).as_posix(),
        "checkpoint_step": step,
        "leading_roots": result["leading_roots"],
        "window_sign_counts": counts,
        "cumulative_50k_to_120k": stats("omega_mean_50k_to_120k"),
        "cumulative_15k_to_120k": stats("omega_mean_15k_to_120k"),
        "trajectory_records": records,
        "limitations": [
            "one frequency/species node from a fixed deterministic panel, not a basin probability",
            "RF phases are fixed and Coulomb/MFS higher spatial harmonics are absent",
            "120,000 periods are finite horizon and do not prove asymptotic attractors",
        ],
    }


def main() -> None:
    with base.PANEL.open(newline="", encoding="utf-8") as stream:
        panel_rows = list(csv.DictReader(stream))
    source, target, tasks = prepare_checkpoints(panel_rows)
    panel_hash = sha256(base.PANEL)

    started = time.perf_counter()
    print(f"continuing 24 trajectories from 50k to 120k on {WORKERS} workers",
          flush=True)
    with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=worker_init) as pool:
        partial = list(pool.map(run_one_start, tasks))
    elapsed = time.perf_counter() - started
    result = {"rows": [row for part in partial for row in part["rows"]],
              "leading_roots": partial[0]["leading_roots"]}
    combine_start_checkpoints(panel_rows, target)
    report = summarize(target, panel_hash, sha256(source), result, elapsed)
    RESULT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
    flat = []
    for record in report["trajectory_records"]:
        for window in record["horizons"]:
            flat.append({"nu": NU, "species": SPECIES,
                         "start_id": record["start_id"],
                         "omega_mean_50k_to_120k": record["omega_mean_50k_to_120k"],
                         "omega_mean_15k_to_120k": record["omega_mean_15k_to_120k"],
                         **window})
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"status": report["status"],
                      "continued_trajectory_segments": report["continued_trajectory_segments"],
                      "additional_particle_periods": report["additional_particle_periods"],
                      "window_sign_counts_120k": report["window_sign_counts"]["120000"],
                      "cumulative_50k_to_120k": report["cumulative_50k_to_120k"],
                      "wall_seconds": elapsed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
