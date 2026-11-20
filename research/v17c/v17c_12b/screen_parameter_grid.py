"""Parallel discovery scan over physically dimensioned symmetric Al2O3 cases.

Each point is integrated for the full screening horizon. Starts are shared
across parameter points for paired comparisons; RF phases are randomized per
start. This is discovery data only and does not certify a basin probability.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
RESULTS = HERE / "results" / "symmetric_parameter_grid_corrected"
RESULTS.mkdir(parents=True, exist_ok=True)
import sys
sys.path.insert(0, str(MODEL))
import basin_scan as integrator  # noqa: E402

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
ETA = 1.81e-5
RHO_GAS = 1.177
MFP = 65.1e-9
RHO_PARTICLE = 3990.0
RADIUS = 17e-6
R_TRAP = 0.010
TEMP = 300.0
KB = 1.380649e-23
H2 = 0.870316891091623
H4 = 0.7982172624273378
N_START = 32
SEED = 20261220
PERIODS = 15000
STEPS = 64
TRANSIENT = 200
WORKERS = min(11, os.cpu_count() or 1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_configs() -> list[dict]:
    base = [0.03492413282400757, 0.15631667332941065,
            3.2196773638017477, 0.015388559140477825,
            1.0, 9.747265456858355e-6,
            0.24575362874758605, 0.24575362874758605]
    configs = []
    # A coarse, interpretable sweep around the V17C.10 shape. The full
    # independent frequency and amplitude controls are represented directly.
    for nu in (2.5, base[2], 4.0):
        for A in (0.0175, base[0], 0.0700):
            for B in (0.07816, base[1], 0.23447):
                p = [A, B, nu, base[3], 1.0, base[5], base[6], base[6]]
                cid = f"A{A:.4f}_B{B:.4f}_nu{nu:.3f}"
                configs.append({"id": cid, "p": p, "qratio": 1.0})
    return configs


def map_physics(p: list[float], qratio: float) -> dict:
    A, B, nu, beta, _, C, lam1, _ = p
    mass = 4.0 * math.pi * RHO_PARTICLE * RADIUS**3 / 3.0
    omega = 4.5 * ETA / (RHO_PARTICLE * RADIUS**2 * lam1)
    charge = math.sqrt(C * 4.0 * math.pi * EPS0 * mass * R_TRAP**3 * omega**2 / qratio)
    scale = mass * R_TRAP**2 * omega**2 / charge
    D = R_TRAP / 0.95
    return {
        "f1_Hz": omega / (2.0 * math.pi), "f2_Hz": nu * omega / (2.0 * math.pi),
        "omega1_rad_s": omega, "charge_C_each": charge, "charge_e_each": charge/E_CHARGE,
        "particle_mass_kg_each": mass,
        "isolated_surface_field_V_m": charge/(4.0*math.pi*EPS0*RADIUS**2),
        "U1_peak_electrode_V": A*scale/H2,
        "U2_peak_electrode_V": B*scale/H4,
        "DC_quadrupole_coefficient_V": beta*scale,
        "free_aperture_radius_m": R_TRAP,
        "electrode_center_radius_m": D,
        "rod_radius_m": 0.05*D,
        "adjacent_rod_gap_m": (2*math.sin(math.pi/24)-0.1)*D,
        "rod_shield_gap_m": 0.05*D,
        "gravity_z_dimensionless": -9.80665/(R_TRAP*omega**2),
        "Re_per_dimensionless_speed": 2*RHO_GAS*RADIUS*R_TRAP*omega/ETA,
        "Kn": MFP/RADIUS,
        "Cunningham": 1+(MFP/RADIUS)*(1.257+0.4*math.exp(-1.1*RADIUS/MFP)),
        "unsteady_Stokes_parameter": omega*RADIUS**2/(ETA/RHO_GAS),
        "Brownian_D_m2_s": KB*TEMP/(6*math.pi*ETA*RADIUS),
    }


def classify(raw: str, turns: np.ndarray, retained: bool, escaped: bool,
             singular: bool, axis_ambiguous: bool) -> str:
    if singular: return "UNRESOLVED"
    if raw == "finite_particle_contact": return "PARTICLE_CONTACT"
    if raw == "finite_particle_wall_contact": return "WALL_CONTACT"
    if escaped or raw == "lost_retention": return "ESCAPE"
    if axis_ambiguous or raw == "axis_ambiguous": return "AXIS_AMBIGUOUS"
    if raw == "CR_screen": return "CR_PLUS_MINUS" if turns[0] > 0 else "CR_MINUS_PLUS"
    if raw == "same_direction": return "CO_PLUS_PLUS" if turns[0] > 0 else "CO_MINUS_MINUS"
    if raw == "localized_or_slow": return "NONROTATING"
    if retained: return "UNRESOLVED"
    return "UNRESOLVED"


def main() -> None:
    if WORKERS < 1:
        raise RuntimeError("no CPU workers available")
    configs = make_configs()
    initial, phases = integrator.make_initials("rest", N_START, SEED)
    starts_path = RESULTS / "frozen_rest_starts.npz"
    np.savez_compressed(starts_path, initial=initial, phases=phases, seed=SEED,
                        ensemble="rest", n=N_START)
    family, k, coeff = integrator.load_model()[1:]
    runs = []
    for ci, cfg in enumerate(configs):
        phys = map_physics(cfg["p"], cfg["qratio"])
        runs.append({**cfg, "index": ci, "physical": phys})
    p_hash = sha256(MODEL / "field_model.json")
    meta = {"study": "symmetric Al2O3 discovery grid; rest-only", "configs": runs,
            "n_starts_per_config": N_START, "start_seed": SEED,
            "periods": PERIODS, "steps_per_RF_period": STEPS,
            "transient_periods": TRANSIENT, "workers": WORKERS,
            "logical_processors": os.cpu_count(), "field_model_sha256": p_hash,
            "frozen_starts_sha256": sha256(starts_path),
            "denominator": "all starts per configuration; unresolved/failures included"}
    meta_path = RESULTS / "campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise RuntimeError("grid campaign metadata differs from existing run")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    trajectory_path = RESULTS / "trajectories.jsonl"
    existing = {}
    if trajectory_path.exists():
        for line_no, line in enumerate(trajectory_path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line); key = (int(row["config_index"]), int(row["start_index"]))
                if key in existing: raise RuntimeError(f"duplicate row at line {line_no}: {key}")
                existing[key] = row
    tasks = [(ci, si) for ci in range(len(runs)) for si in range(N_START)
             if (ci, si) not in existing]
    # JIT compile before the pool begins. Numba kernels release the GIL; the
    # trajectory pool is the only parallel layer in this campaign.
    if tasks:
        c0, s0 = tasks[0]; cfg0 = runs[c0]; ph0 = cfg0["physical"]
        integrator.integrate_one(initial[s0], phases[s0], np.asarray(cfg0["p"]),
            family, k, coeff, 1, STEPS, 0, qratio=cfg0["qratio"], hydro_mode=2,
            a1_over_R=RADIUS/R_TRAP, a2_over_R=RADIUS/R_TRAP,
            oseen_Re_per_speed=ph0["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph0["Re_per_dimensionless_speed"],
            gravity_z=ph0["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=ph0["Cunningham"])
    print(json.dumps({"configurations": len(runs), "starts_per_config": N_START,
                      "trajectories_total": len(runs)*N_START, "todo": len(tasks),
                      "periods": PERIODS, "workers": WORKERS}, ensure_ascii=False), flush=True)

    def one(task: tuple[int, int]) -> dict:
        ci, si = task; cfg = runs[ci]; ph = cfg["physical"]
        raw = integrator.integrate_one(initial[si], phases[si], np.asarray(cfg["p"]),
            family, k, coeff, PERIODS, STEPS, TRANSIENT, qratio=cfg["qratio"],
            hydro_mode=2, a1_over_R=RADIUS/R_TRAP, a2_over_R=RADIUS/R_TRAP,
            oseen_Re_per_speed=ph["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["Re_per_dimensionless_speed"],
            gravity_z=ph["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=ph["Cunningham"])
        end, turns, windows, rmin, rmax, zmax, dtheta, dmin, completed, retained, escaped, singular, cat, freq, vmax = raw
        label = classify(cat, turns, retained, escaped, singular, bool(min(rmin)<integrator.RMIN_CLASS))
        return {"config_index": ci, "config_id": cfg["id"], "start_index": si,
            "ensemble": "rest", "seed": SEED, "initial": initial[si].tolist(),
            "phase": phases[si].tolist(), "final_state": end.tolist(),
            "turns": turns.tolist(), "window_turns": windows.tolist(),
            "category": label, "raw_category": str(cat), "retained": bool(retained),
            "completed_periods": int(completed), "r_min_over_R": rmin.tolist(),
            "r_max_over_R": rmax.tolist(), "z_max_over_R": zmax.tolist(),
            "distance_min_over_R": float(dmin), "max_speed_dimensionless": vmax.tolist(),
            "max_Re_by_particle": (vmax*ph["Re_per_dimensionless_speed"]).tolist()}

    started = time.perf_counter(); cpu0 = time.process_time()
    with trajectory_path.open("a", encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(one, t) for t in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                row = future.result(); key = (row["config_index"], row["start_index"])
                existing[key] = row
                fout.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False)+"\n")
                if count % 16 == 0 or count == len(tasks):
                    fout.flush(); os.fsync(fout.fileno())
                    print(f"checkpoint {len(existing)}/{len(runs)*N_START}; "
                          f"{time.perf_counter()-started:.1f}s", flush=True)
    counts = {}
    for row in existing.values():
        c = counts.setdefault(row["config_id"], {})
        c[row["category"]] = c.get(row["category"], 0)+1
    summary = {"config_counts": counts, "trajectory_rows": len(existing),
        "meta_sha256": sha256(meta_path), "trajectory_sha256": sha256(trajectory_path),
        "elapsed_s": time.perf_counter()-started, "process_cpu_s": time.process_time()-cpu0,
        "classification_note": "Discovery classification; N=32/config is not a probability estimate."}
    (RESULTS/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
