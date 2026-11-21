"""Extend selected Local15 conditional QR trajectories from 15k to 50k."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import numpy as np

import nonautonomous_qr_panel as panel_qr
import nonautonomous_qr_local15 as s01_qr


TARGET_PERIODS = 50_000
WORKERS = 8
EXTENDED_DIR = panel_qr.OUT / "selected_qr_50000_checkpoints"
SOURCE_RESULT = panel_qr.OUT / "local15_panel_conditional_qr_15000.json"
EXTENDED_RESULT = panel_qr.OUT / "local15_selected_qr_50000.json"


def select_starts(data: dict) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for species, records in data["species"].items():
        candidates = []
        for start_id, record in records.items():
            terminal = next(row for row in record["periodic_diagnostics"]
                            if int(row["period"]) == 15_000)
            candidates.append((float(terminal["conditional_lyapunov_per_tau"][0]),
                               start_id))
        candidates.sort()
        chosen = [candidates[0][1], candidates[len(candidates)//2][1],
                  candidates[-1][1], "S01"]
        selected[species] = sorted(set(chosen))
    return selected


def seed_extended_checkpoint(species: str, start_id: str,
                             panel_sha: str) -> None:
    destination = EXTENDED_DIR / f"{species}_{start_id}_qr_checkpoint.npz"
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            if int(saved["step"]) >= TARGET_PERIODS * panel_qr.STEPS_PER_PERIOD:
                return
            if (str(saved["species"].item()) == species and
                    str(saved["start_id"].item()) == start_id and
                    str(saved["panel_sha256"].item()) == panel_sha):
                return
        raise ValueError(f"invalid extended checkpoint: {destination}")

    EXTENDED_DIR.mkdir(parents=True, exist_ok=True)
    if start_id != "S01":
        source = panel_qr.CHECKPOINT_DIR / f"{species}_{start_id}_qr_checkpoint.npz"
        with np.load(source, allow_pickle=False) as saved:
            if int(saved["step"]) != 15_000 * panel_qr.STEPS_PER_PERIOD:
                raise ValueError(f"source is not at 15k: {source}")
            values = {key: saved[key].copy() for key in saved.files}
        with destination.open("wb") as f:
            np.savez_compressed(f, **values)
        return

    # Convert the original S01 checkpoint schema into the retained-panel schema.
    source = s01_qr.OUT / f"{species}_s01_qr_checkpoint.npz"
    with np.load(source, allow_pickle=False) as saved:
        if int(saved["step"]) != 15_000 * panel_qr.STEPS_PER_PERIOD:
            raise ValueError(f"S01 source is not at 15k: {source}")
        with destination.open("wb") as f:
            np.savez_compressed(
                f,
                step=saved["step"], y=saved["y"], tangent=saved["M"],
                sums=saved["sums"], maxrho=saved["maxrho"],
                periods=saved["record_periods"], rhos=saved["record_rhos"],
                maxrhos=saved["record_max_rhos"], thetas=saved["record_thetas"],
                exponents=saved["record_exponents"], panel_sha256=panel_sha,
                species=species, start_id=start_id)


def main() -> None:
    with SOURCE_RESULT.open(encoding="utf-8") as f:
        base = json.load(f)
    starts, panel_sha = panel_qr.load_panel()
    if base["panel_sha256"] != panel_sha:
        raise ValueError("panel hash changed since the 15k QR run")
    selected = select_starts(base)
    tasks = []
    for species, start_ids in selected.items():
        for start_id in start_ids:
            seed_extended_checkpoint(species, start_id, panel_sha)
            tasks.append((species, start_id, starts[start_id], panel_sha,
                          TARGET_PERIODS, str(EXTENDED_DIR)))

    EXTENDED_DIR.mkdir(parents=True, exist_ok=True)
    results = {species: {} for species in selected}
    print(f"extending {len(tasks)} trajectories from 15k to {TARGET_PERIODS} on {WORKERS} workers",
          flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(panel_qr.integrate_one, task): task for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            species, start_id = futures[future][0:2]
            result = future.result()
            results[species][start_id] = result
            print(f"{completed}/{len(tasks)} {species}/{start_id}: {result['status']} "
                  f"wall={result['elapsed_seconds']:.1f}s", flush=True)

    report = {
        "status": "completed",
        "method": "RK4/96 exact ideal Local15 RF plus analytic 4x4 mechanical tangent Jacobian; QR every primary period",
        "conditional_spectrum": True,
        "rf_phase_variations_included": False,
        "parameters": base["parameters"],
        "panel_sha256": panel_sha,
        "start_selection_rule": "For each species include S01 plus the 15k minimum, upper median (rank index floor(n/2)), and maximum of the largest conditional exponent; duplicates removed.",
        "selected_start_ids": selected,
        "source_horizon_periods": 15_000,
        "target_horizon_periods": TARGET_PERIODS,
        "steps_per_period": panel_qr.STEPS_PER_PERIOD,
        "worker_count": WORKERS,
        "new_continuation_segments": len(tasks),
        "additional_particle_periods": len(tasks) * (TARGET_PERIODS - 15_000),
        "species": results,
        "limitations": [
            "the subset is selected from finite-time exponents and is not an iid sample",
            "fixed RF phases give conditional mechanical exponents without phase tangent directions",
            "finite-time spectra do not prove asymptotic stability",
            "ideal k=+2,-4 field only; no Coulomb, MFS harmonics, or physical drag correction",
        ],
    }
    EXTENDED_RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    print(f"wrote {EXTENDED_RESULT.relative_to(panel_qr.OUT.parent)}", flush=True)


if __name__ == "__main__":
    main()
