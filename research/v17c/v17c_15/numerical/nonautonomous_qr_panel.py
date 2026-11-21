"""Conditional QR spectra for every retained frozen-panel start.

Existing S01 trajectories are reused from the completed V17C.15 QR run. The
remaining 21 retained starts per species are integrated independently, with a
checkpoint every 500 primary RF periods. RF phases stay externally prescribed,
so these are conditional mechanical spectra, not phase-augmented exponents.
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import os
import time
from pathlib import Path

# Small 4x4 QR operations do not benefit from nested BLAS thread pools.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np

from nonautonomous_qr_local15 import OUT, PANEL, P, rhs_and_jacobian


PERIODS = 15_000
STEPS_PER_PERIOD = 96
QR_BLOCK_PERIODS = 500
WORKERS = 11
ESCAPE_IDS = {"S08", "S16"}
CHECKPOINT_DIR = OUT / "panel_qr_checkpoints"
S01_RESULT = OUT / "local15_s01_conditional_qr_15000.json"
RESULT = OUT / "local15_panel_conditional_qr_15000.json"


def load_panel() -> tuple[dict[str, dict], str]:
    with PANEL.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    starts = {
        r["start_id"]: {
            key: float(r[key])
            for key in ("x0", "y0", "vx0", "vy0", "phase1", "phase2")
        }
        for r in rows
    }
    panel_sha = hashlib.sha256(PANEL.read_bytes()).hexdigest()
    return starts, panel_sha


def checkpoint_path(species: str, start_id: str) -> Path:
    return CHECKPOINT_DIR / f"{species}_{start_id}_qr_checkpoint.npz"


def integrate_one(task: tuple) -> dict:
    species, start_id, initial, panel_sha = task[:4]
    target_periods = int(task[4]) if len(task) > 4 else PERIODS
    checkpoint_dir = Path(task[5]) if len(task) > 5 else CHECKPOINT_DIR
    alpha = P["species"][species]["alpha"]
    lam = P["species"][species]["lambda"]
    y = np.array([initial["x0"], initial["y0"],
                  initial["vx0"], initial["vy0"]], dtype=float)
    tangent = np.eye(4)
    sums = np.zeros(4)
    maxrho = float(np.hypot(*y[:2]))
    path = checkpoint_dir / f"{species}_{start_id}_qr_checkpoint.npz"
    period_grid = np.empty(0, dtype=np.int64)
    rho_grid = np.empty(0, dtype=float)
    maxrho_grid = np.empty(0, dtype=float)
    theta_grid = np.empty(0, dtype=float)
    exponent_grid = np.empty((0, 4), dtype=float)
    start_step = 0

    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved["panel_sha256"].item()) != panel_sha or
                    str(saved["species"].item()) != species or
                    str(saved["start_id"].item()) != start_id):
                raise ValueError(f"checkpoint metadata mismatch: {path}")
            start_step = int(saved["step"])
            y = saved["y"].copy()
            tangent = saved["tangent"].copy()
            sums = saved["sums"].copy()
            maxrho = float(saved["maxrho"])
            period_grid = saved["periods"].copy()
            rho_grid = saved["rhos"].copy()
            maxrho_grid = saved["maxrhos"].copy()
            theta_grid = saved["thetas"].copy()
            exponent_grid = saved["exponents"].copy()
        if start_step % STEPS_PER_PERIOD or start_step > target_periods * STEPS_PER_PERIOD:
            raise ValueError(f"invalid checkpoint step {start_step}: {path}")

    def derivative(tau: float, state: np.ndarray, matrix: np.ndarray):
        field, jac = rhs_and_jacobian(
            tau, state, alpha, lam, initial["phase1"], initial["phase2"])
        return field, jac @ matrix

    h = 2 * np.pi / STEPS_PER_PERIOD
    started = time.perf_counter()
    escaped = False
    for step in range(start_step + 1, target_periods * STEPS_PER_PERIOD + 1):
        tau = (step - 1) * h
        f1, m1 = derivative(tau, y, tangent)
        f2, m2 = derivative(tau + h/2, y + h*f1/2, tangent + h*m1/2)
        f3, m3 = derivative(tau + h/2, y + h*f2/2, tangent + h*m2/2)
        f4, m4 = derivative(tau + h, y + h*f3, tangent + h*m3)
        y += (h/6) * (f1 + 2*f2 + 2*f3 + f4)
        tangent += (h/6) * (m1 + 2*m2 + 2*m3 + m4)
        rho = float(np.hypot(*y[:2]))
        maxrho = max(maxrho, rho)

        if step % STEPS_PER_PERIOD:
            continue
        period = step // STEPS_PER_PERIOD
        q, r = np.linalg.qr(tangent)
        diagonal = np.abs(np.diag(r))
        if np.any(diagonal <= 0) or np.any(~np.isfinite(diagonal)):
            raise FloatingPointError(f"invalid QR diagonal at {species}/{start_id}/{period}")
        sums += np.log(diagonal)
        tangent = q
        if period % QR_BLOCK_PERIODS == 0:
            exponents = sums / (2 * np.pi * period)
            period_grid = np.append(period_grid, period)
            rho_grid = np.append(rho_grid, rho)
            maxrho_grid = np.append(maxrho_grid, maxrho)
            theta_grid = np.append(theta_grid, float(np.arctan2(y[1], y[0])))
            exponent_grid = np.vstack((exponent_grid, exponents))
            temp = path.with_name(path.stem + ".tmp.npz")
            with temp.open("wb") as f:
                np.savez_compressed(
                    f, step=step, y=y, tangent=tangent, sums=sums,
                    maxrho=maxrho, periods=period_grid, rhos=rho_grid,
                    maxrhos=maxrho_grid, thetas=theta_grid,
                    exponents=exponent_grid, panel_sha256=panel_sha,
                    species=species, start_id=start_id)
            os.replace(temp, path)
        if rho >= 1.0:
            escaped = True
            break

    periods = period_grid.tolist()
    exponents = exponent_grid.tolist()
    windows = {
        str(target): next(exponents[i] for i, period in enumerate(periods)
                          if period == target)
        for target in (5000, 10000, 15000)
        if target in periods
    }
    return {
        "species": species,
        "start_id": start_id,
        "initial_condition": initial,
        "alpha": alpha,
        "lambda": lam,
        "status": "escape" if escaped else ("completed" if periods and periods[-1] == target_periods else "partial"),
        "periods": target_periods,
        "steps_per_period": STEPS_PER_PERIOD,
        "computed_steps_this_run": target_periods * STEPS_PER_PERIOD - start_step,
        "elapsed_seconds": time.perf_counter() - started,
        "max_rho": maxrho,
        "windows_lyapunov_exponents_per_tau": windows,
        "periodic_diagnostics": [
            {
                "period": int(period),
                "rho": float(rho_grid[i]),
                "max_rho": float(maxrho_grid[i]),
                "theta_wrapped": float(theta_grid[i]),
                "conditional_lyapunov_per_tau": exponents[i],
            }
            for i, period in enumerate(periods)
        ],
        "checkpoint_file": path.relative_to(OUT.parent).as_posix(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    starts, panel_sha = load_panel()
    with S01_RESULT.open(encoding="utf-8") as f:
        old = json.load(f)
    if old.get("status") != "completed" or old.get("panel_sha256") != panel_sha:
        raise ValueError("S01 reference QR file is missing, incomplete, or panel-mismatched")

    by_species = {
        name: {"S01": old["species"][name]}
        for name in ("species1", "species2")
    }
    tasks = [
        (species, start_id, starts[start_id], panel_sha)
        for species in ("species1", "species2")
        for start_id in sorted(starts)
        if start_id not in ESCAPE_IDS and start_id != "S01"
    ]
    print(f"starting {len(tasks)} trajectories on {WORKERS} workers", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        future_to_task = {pool.submit(integrate_one, task): task for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(future_to_task), start=1):
            species, start_id, _, _ = future_to_task[future]
            result = future.result()
            by_species[species][start_id] = result
            print(f"{completed}/{len(tasks)} {species}/{start_id}: {result['status']} "
                  f"maxrho={result['max_rho']:.6f} wall={result['elapsed_seconds']:.1f}s",
                  flush=True)

    summaries = {}
    for species, starts_for_species in by_species.items():
        summaries[species] = {}
        for horizon in (5000, 10000, 15000):
            values = np.array([
                starts_for_species[sid]["windows_lyapunov_exponents_per_tau"][str(horizon)]
                for sid in sorted(starts_for_species)
            ])
            summaries[species][str(horizon)] = {
                "count": int(len(values)),
                "min_per_mode": values.min(axis=0).tolist(),
                "median_per_mode": np.median(values, axis=0).tolist(),
                "max_per_mode": values.max(axis=0).tolist(),
            }

    result = {
        "status": "completed",
        "method": "RK4/96 exact ideal two-mode Local15 RF plus analytic 4x4 mechanical tangent Jacobian; QR every primary period",
        "conditional_spectrum": True,
        "rf_phase_variations_included": False,
        "coulomb_and_mfs_higher_harmonics_included": False,
        "parameters": P,
        "panel_sha256": panel_sha,
        "periods": PERIODS,
        "steps_per_period": STEPS_PER_PERIOD,
        "qr_block_periods": QR_BLOCK_PERIODS,
        "worker_count": WORKERS,
        "retained_start_ids_per_species": sorted(by_species["species1"]),
        "excluded_escape_start_ids": sorted(ESCAPE_IDS),
        "new_trajectories": len(tasks),
        "new_particle_periods": len(tasks) * PERIODS,
        "panel_particle_periods_including_reused_s01": 2 * len(starts_for_species) * PERIODS,
        "species": by_species,
        "across_start_summary": summaries,
        "limitations": [
            "fixed RF phases produce conditional mechanical exponents and omit RF phase tangent directions",
            "22 retained frozen-panel starts per species are a deterministic panel, not an iid sample or basin probability estimate",
            "finite-time exponents through 15,000 periods do not prove asymptotic stability",
            "ideal k=+2,-4 field only; no Coulomb, MFS harmonics, or physical drag correction",
        ],
    }
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {RESULT.relative_to(OUT.parent)}", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
