"""Paired charge/frequency screen for the fixed Al2O3 / 3M K1 model pair."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import physical_mixed_pair_search as base

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
OUT = HERE / "results" / "mixed_pair_charge_sweep"
OUT.mkdir(parents=True, exist_ok=True)
BASE_OUT = HERE / "results" / "mixed_alumina_k1_candidate"
WORKERS = min(11, os.cpu_count() or 1)
CHARGE_SCALES = {"q150k_f75hz": 1.5, "q220k_f110hz": 2.2}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scaled_configuration(scale: float) -> dict:
    """Scale q and omega together; retain geometry and field dimensionless ratios."""
    cfg = base.build_configuration()
    p = list(cfg["p"])
    phys = cfg["physical"]
    p[5] = p[5]  # C stays fixed because q1*q2 and omega^2 scale equally.
    p[6] /= scale
    p[7] /= scale
    cfg["p"] = p
    for i in (1, 2):
        particle = phys[f"particle{i}"]
        particle["charge_C"] *= scale
        particle["charge_e"] *= scale
        particle["specific_charge_C_kg"] *= scale
        particle["surface_charge_field_V_m"] *= scale
        particle["lambda_Stokes"] /= scale
        particle["Re_per_dimensionless_speed"] *= scale
    phys["frequency_f1_Hz"] *= scale
    phys["frequency_f2_Hz"] *= scale
    phys["omega1_rad_s"] *= scale
    phys["dimensionless_parameters"] = p.copy()
    phys["lambda_ratio_lambda2_over_lambda1"] = p[7] / p[6]
    phys["U1_peak_electrode_V"] *= scale
    phys["U2_peak_electrode_V"] *= scale
    phys["U1_plus_U2_peak_bound_V"] *= scale
    phys["V2_ideal_DC_coefficient_V_not_endcap"] *= scale
    phys["MFS_triangle_surface_field_bound_V_m"] *= scale
    phys["gravity_z_dimensionless"] /= scale**2
    phys["charge_frequency_scaling"] = {
        "common_charge_scale": scale,
        "common_frequency_scale": scale,
        "C_preserved": True,
        "A_B_beta_and_geometry_preserved": True,
        "lambda_scaled_by_inverse_frequency": True,
        "Re_per_dimensionless_speed_scaled_with_frequency": True,
    }
    return cfg


def load_starts() -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    frozen = np.load(BASE_OUT / "frozen_starts.npz", allow_pickle=True)
    initial = frozen["initial"]
    phases = frozen["phases"]
    ensembles = frozen["ensemble"].astype(str).tolist()
    seed_records = json.loads(str(frozen["seeds"].item()))
    if initial.shape != (512, 12) or phases.shape != (512, 2) or len(ensembles) != 512:
        raise RuntimeError("unexpected baseline frozen-start dimensions")
    return initial, phases, ensembles, seed_records


def main() -> None:
    starts, phases, labels, seed_records = load_starts()
    _, family, k, coeff = base.sim.load_model()
    expected_base_meta = json.loads((BASE_OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    if sha(BASE_OUT / "frozen_starts.npz") != expected_base_meta["starts_sha256"]:
        raise RuntimeError("baseline frozen-start SHA mismatch")

    configurations = {name: scaled_configuration(scale) for name, scale in CHARGE_SCALES.items()}
    meta = {
        "study": "paired common-charge/common-frequency screen for literature-constrained Al2O3 / K1 pair",
        "interpretation": "Two paired discovery points reuse identical frozen starts and phases. Not an IID basin-probability study and not an independently measured joint particle pair.",
        "parent_campaign": "results/mixed_alumina_k1_candidate",
        "parent_starts_sha256": sha(BASE_OUT / "frozen_starts.npz"),
        "configurations": configurations,
        "ensemble_counts": {g: labels.count(g) for g in base.ENSEMBLES},
        "seed_records": seed_records,
        "total_trajectories": len(starts) * len(configurations),
        "periods": base.PERIODS,
        "steps_per_rf_period": base.STEPS,
        "skip_periods": base.SKIP,
        "workers": WORKERS,
        "integrator_sha256": sha(MODEL / "basin_scan.py"),
        "field_model_sha256": sha(MODEL / "field_model.json"),
        "model_options": {"drag_mode": "Schiller-Naumann", "hydro_mode": 2,
            "gravity": True, "finite_particle_contacts": True,
            "Cunningham": "same particle-specific mean as parent campaign"},
    }
    meta_path = OUT / "campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise RuntimeError("existing charge-sweep campaign metadata differs")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    runner_copy = OUT / "campaign_runner_source.py"
    if not runner_copy.exists():
        shutil.copy2(Path(__file__), runner_copy)
    integrator_copy = OUT / "campaign_integrator_source.py"
    if not integrator_copy.exists():
        shutil.copy2(MODEL / "basin_scan.py", integrator_copy)
    code_hashes = {
        "runner_sha256": sha(Path(__file__)),
        "saved_runner_sha256": sha(runner_copy),
        "integrator_sha256": sha(MODEL / "basin_scan.py"),
        "saved_integrator_sha256": sha(integrator_copy),
        "field_model_sha256": sha(MODEL / "field_model.json"),
        "parent_starts_sha256": sha(BASE_OUT / "frozen_starts.npz"),
        "workers": WORKERS,
        "total_trajectories": meta["total_trajectories"],
    }
    (OUT / "campaign_code_sha256.json").write_text(json.dumps(code_hashes, indent=2) + "\n", encoding="utf-8")
    frozen_copy = OUT / "frozen_starts.npz"
    if not frozen_copy.exists():
        shutil.copy2(BASE_OUT / "frozen_starts.npz", frozen_copy)

    run_path = OUT / "trajectories.jsonl"
    existing = {}
    if run_path.exists():
        with run_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing[(row["configuration"], row["ensemble"], int(row["start_index"]))] = row
    tasks = [(name, group, si) for name in configurations for group in base.ENSEMBLES
             for si in range(base.N_PER_ENSEMBLE)
             if (name, group, si) not in existing]
    group_starts = {group: i * base.N_PER_ENSEMBLE for i, group in enumerate(base.ENSEMBLES)}

    if tasks:
        name, group, si = tasks[0]
        cfg = configurations[name]
        phys = cfg["physical"]
        idx = group_starts[group] + si
        base.sim.integrate_one(starts[idx], phases[idx], np.asarray(cfg["p"]), family, k, coeff,
            1, base.STEPS, 0, qratio=cfg["qratio"],
            oseen_Re_per_speed=phys["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=phys["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2, a1_over_R=phys["a1_over_R"], a2_over_R=phys["a2_over_R"],
            gravity_z=phys["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=phys["Cunningham_mean"])

    print(json.dumps({"configurations": {
        name: {"q1_e": cfg["physical"]["particle1"]["charge_e"],
               "f1_Hz": cfg["physical"]["frequency_f1_Hz"],
               "f2_Hz": cfg["physical"]["frequency_f2_Hz"],
               "lambda1": cfg["p"][6], "lambda2": cfg["p"][7],
               "U1_V": cfg["physical"]["U1_peak_electrode_V"],
               "U2_V": cfg["physical"]["U2_peak_electrode_V"]}
        for name, cfg in configurations.items()}, "todo": len(tasks),
        "total": meta["total_trajectories"], "workers": WORKERS}, indent=2), flush=True)
    started = time.perf_counter()

    def run_one(task: tuple[str, str, int]) -> dict:
        name, group, si = task
        cfg = configurations[name]
        phys = cfg["physical"]
        idx = group_starts[group] + si
        raw = base.sim.integrate_one(starts[idx], phases[idx], np.asarray(cfg["p"]),
            family, k, coeff, base.PERIODS, base.STEPS, base.SKIP,
            qratio=cfg["qratio"],
            oseen_Re_per_speed=phys["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=phys["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2, a1_over_R=phys["a1_over_R"], a2_over_R=phys["a2_over_R"],
            gravity_z=phys["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=phys["Cunningham_mean"])
        end, turns, windows, rmin, rmax, zmax, dtheta, dmin, completed, retained, escaped, singular, cat, freq, vmax = raw
        axis = bool(min(rmin) < base.sim.RMIN_CLASS)
        particle_re = np.asarray([phys["particle1"]["Re_per_dimensionless_speed"],
                                  phys["particle2"]["Re_per_dimensionless_speed"]])
        return {"configuration": name, "charge_scale": CHARGE_SCALES[name],
            "ensemble": group, "start_index": si, "seed": base.SEEDS[group],
            "initial": starts[idx].tolist(), "phase": phases[idx].tolist(),
            "final_state": end.tolist(), "turns": turns.tolist(), "window_turns": windows.tolist(),
            "raw_category": str(cat),
            "provisional_category": base.provisional(str(cat), turns, windows, bool(retained), axis),
            "retained": bool(retained), "completed_periods": int(completed),
            "r_min_over_R": rmin.tolist(), "r_max_over_R": rmax.tolist(),
            "z_max_over_R": zmax.tolist(), "distance_min_over_R": float(dmin),
            "max_Re_by_particle": (vmax * particle_re).tolist()}

    with run_path.open("a", encoding="utf-8", buffering=1) as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            key = (row["configuration"], row["ensemble"], row["start_index"])
            existing[key] = row
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if n % 32 == 0 or n == len(tasks):
                f.flush()
                os.fsync(f.fileno())
                print(f"checkpoint {len(existing)}/{meta['total_trajectories']}; {time.perf_counter()-started:.1f}s", flush=True)

    by_configuration = {}
    by_ensemble = {}
    counts = Counter(row["provisional_category"] for row in existing.values())
    for name in configurations:
        these = [row for row in existing.values() if row["configuration"] == name]
        by_configuration[name] = {
            "rows": len(these),
            "outcomes": dict(sorted(Counter(row["provisional_category"] for row in these).items())),
            "retained_by_ensemble": {group: sum(bool(row["retained"]) for row in these if row["ensemble"] == group)
                                     for group in base.ENSEMBLES},
        }
    for group in base.ENSEMBLES:
        by_ensemble[group] = {}
        for name in configurations:
            these = [row for row in existing.values() if row["configuration"] == name and row["ensemble"] == group]
            by_ensemble[group][name] = dict(sorted(Counter(row["provisional_category"] for row in these).items()))
    summary = {"study": meta["study"], "rows": len(existing), "expected_rows": meta["total_trajectories"],
        "outcome_counts": dict(sorted(counts.items())), "by_configuration": by_configuration,
        "outcomes_by_ensemble_and_configuration": by_ensemble,
        "classification": "Provisional long-horizon labels for a paired parameter screen; not IID basin probabilities.",
        "trajectory_sha256": sha(run_path), "meta_sha256": sha(meta_path),
        "wall_seconds_this_invocation": time.perf_counter() - started}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
