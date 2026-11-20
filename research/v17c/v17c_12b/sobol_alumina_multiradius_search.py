"""Dimensionally linked multi-radius Al2O3 discovery screen, V17C.12B.

The eight experimental-motivated radii are screened at fixed material density
in room-temperature air. Short classifications are discovery labels only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
OUT = HERE / "results" / "alumina_multiradius_sobol_screen"
OUT.mkdir(parents=True, exist_ok=True)
import sys
sys.path.insert(0, str(MODEL))
import basin_scan as sim  # noqa: E402

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
G = 9.80665
ETA = 1.81e-5
RHO_G = 1.177
MFP = 65.1e-9
RHO_P = 3800.0
R = 0.010
H2 = 0.870316891091623
H4 = 0.7982172624273378
RADII_UM = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 30.0, 40.0)
NCONFIG_PER_RADIUS = 64
NSTART = 8
SOBOL_CANDIDATES_PER_RADIUS = 512
SEED = 20261226
PERIODS = 5000
STEPS = 64
SKIP = 100
WORKERS = min(11, os.cpu_count() or 1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cc(a: float) -> float:
    kn = MFP / a
    return 1 + kn * (1.257 + 0.4 * math.exp(-1.1 / kn))


def paschen_screen(p_d_torr_cm: float) -> float:
    return 365 * p_d_torr_cm / (math.log(15 * p_d_torr_cm) - math.log(math.log(101)))


def physical_config(u: np.ndarray, radius_um: float, index: int) -> dict:
    A = math.exp(math.log(0.008) + u[0] * math.log(0.08 / 0.008))
    B = math.exp(math.log(0.03) + u[1] * math.log(0.30 / 0.03))
    nu = 1.5 + 3.5 * u[2]
    beta = math.exp(math.log(0.003) + u[3] * math.log(0.08 / 0.003))
    lam = math.exp(math.log(0.025) + u[4] * math.log(0.60 / 0.025))
    C = math.exp(math.log(1e-6) + u[5] * math.log(6e-5 / 1e-6))
    a = radius_um * 1e-6
    m = 4 * math.pi * RHO_P * a**3 / 3
    slip = cc(a)
    # lambda is the Stokes-defined value; the integrator multiplies by Cc once.
    omega = 4.5 * ETA / (RHO_P * a**2 * lam)
    q = math.sqrt(C * 4 * math.pi * EPS0 * m * R**3 * omega**2)
    scale = m * R**2 * omega**2 / q
    U1, U2, V2 = A * scale / H2, B * scale / H4, beta * scale
    D = R / 0.95
    gap = 0.05 * D
    vb = paschen_screen(760 * 100 * gap)
    Re_per_speed = 2 * RHO_G * a * R * omega / ETA
    phys = {
        "radius_m": a, "density_kg_m3": RHO_P, "mass_kg": m,
        "charge_C_each": q, "charge_e_each": q / E_CHARGE,
        "specific_charge_C_kg": q / m,
        "particle_isolated_surface_field_V_m": q / (4 * math.pi * EPS0 * a**2),
        "f1_Hz": omega / (2 * math.pi), "f2_Hz": nu * omega / (2 * math.pi),
        "omega1_rad_s": omega, "U1_peak_electrode_V": U1,
        "U2_peak_electrode_V": U2,
        "V2_ideal_DC_coefficient_V_not_endcap": V2,
        "RF_rod_shield_voltage_bound_V": U1 + U2,
        "flat_air_Paschen_rod_shield_V_screen_only": vb,
        "RF_to_flat_Paschen_ratio_screen_only": (U1 + U2) / vb,
        "gravity_z_dimensionless": -G / (R * omega**2),
        "gravity_equilibrium_z_over_R_if_vertical": -G / (2 * beta * R * omega**2),
        "lambda_stokes_dimensionless": lam,
        "lambda_with_cunningham_dimensionless": lam * slip,
        "Kn": MFP / a, "Cunningham": slip,
        "Re_per_dimensionless_speed": Re_per_speed,
        "Omega_a2_over_nu_gas": omega * a**2 / (ETA / RHO_G),
    }
    return {"id": f"a{radius_um:g}um_phys{index:04d}", "sobol_index": index,
        "radius_um": radius_um, "p": [A, B, nu, beta, 1.0, C, lam, lam],
        "qratio": 1.0, "physical": phys,
        "empirical_charge_window": 1e4 <= phys["charge_e_each"] <= 1e6,
        "preferred_charge_window": 1e5 <= phys["charge_e_each"] <= 5e5,
        "reported_voltage_envelope_4p4kV": U1 + U2 <= 4400.0,
        "flat_Paschen_screen_under_unity": (U1 + U2) / vb <= 1.0}


def provisional(raw: str, turns: np.ndarray, windows: np.ndarray,
                retained: bool, axis: bool) -> str:
    if raw == "finite_particle_contact": return "PARTICLE_CONTACT"
    if raw == "finite_particle_wall_contact": return "WALL_OR_AXIAL_ESCAPE"
    if raw == "lost_retention": return "ESCAPE"
    if raw == "point_singularity_unresolved": return "UNRESOLVED"
    if axis: return "AXIS_AMBIGUOUS"
    signs = np.zeros((3, 2), dtype=np.int8)
    for w in range(3):
        for i in range(2):
            if abs(windows[w, i]) >= 0.05:
                signs[w, i] = 1 if windows[w, i] > 0 else -1
    if retained and np.all(signs != 0) and np.all(np.abs(turns) >= 1.0):
        if np.all(signs[:, 0] * signs[:, 1] < 0):
            return "PROVISIONAL_CR_PLUS_MINUS" if turns[0] > 0 else "PROVISIONAL_CR_MINUS_PLUS"
        if np.all(signs[:, 0] * signs[:, 1] > 0):
            return "PROVISIONAL_CO_PLUS_PLUS" if turns[0] > 0 else "PROVISIONAL_CO_MINUS_MINUS"
    if retained and np.all(np.abs(turns) < 0.25): return "NONROTATING"
    return "UNRESOLVED_SHORT"


def build_designs() -> list[dict]:
    selected = []
    diagnostics = {}
    for ridx, radius_um in enumerate(RADII_UM):
        seed = SEED + ridx
        raw = qmc.Sobol(d=6, scramble=True, seed=seed).random_base2(
            int(math.log2(SOBOL_CANDIDATES_PER_RADIUS)))
        proposals = [physical_config(u, radius_um, i) for i, u in enumerate(raw)]
        valid = [c for c in proposals if c["empirical_charge_window"]
            and c["reported_voltage_envelope_4p4kV"]]
        diagnostics[str(radius_um)] = {"proposed": len(proposals), "passing_charge_voltage_screen": len(valid)}
        if len(valid) < NCONFIG_PER_RADIUS:
            raise RuntimeError(f"only {len(valid)} valid designs for radius {radius_um} um")
        for j, c in enumerate(valid[:NCONFIG_PER_RADIUS]):
            c["screen_index"] = len(selected)
            c["sobol_seed"] = seed
            selected.append(c)
    return selected, diagnostics


def main() -> None:
    configs, physical_diagnostics = build_designs()
    starts, phases = sim.make_initials("rest", NSTART, SEED + 100)
    starts_path = OUT / "frozen_rest_starts.npz"
    if not starts_path.exists():
        np.savez_compressed(starts_path, initial=starts, phases=phases,
            seed=SEED + 100, ensemble="rest", n=NSTART)
    else:
        frozen = np.load(starts_path)
        if not np.array_equal(frozen["initial"], starts) or not np.array_equal(frozen["phases"], phases):
            raise RuntimeError("existing frozen starts differ")
    _, family, k, coeff = sim.load_model()
    meta = {"study": "dimensionally linked multi-radius Al2O3 discovery; air at 300K",
        "design": "8 physical radii, 64 configs per radius; 512 scrambled Sobol proposals/radius; select first 64 within broad charge and voltage envelope",
        "physical_screen_diagnostics": physical_diagnostics,
        "source_sobol_seed_base": SEED, "screen_seed": SEED + 100,
        "configurations": configs, "n_configurations": len(configs),
        "n_starts_per_config": NSTART, "periods": PERIODS, "steps_per_rf_period": STEPS,
        "skip_periods": SKIP, "workers": WORKERS,
        "particle": {"material": "Al2O3", "radii_um": list(RADII_UM),
                     "density_kg_m3": RHO_P},
        "medium": {"gas": "air", "T_K": 300, "eta_Pa_s": ETA,
                   "rho_kg_m3": RHO_G, "mfp_m": MFP},
        "trap": {"R_m": R, "geometry": "24 rods, V17C MFS field, shield 1.10D"},
        "integrator_sha256": sha(MODEL / "basin_scan.py"),
        "field_model_sha256": sha(MODEL / "field_model.json"),
        "starts_sha256": sha(starts_path),
        "voltage_screen_caveat": "4.4 kV is a published reference-apparatus envelope, not a universal safety/breakdown limit; Paschen value is diagnostic only.",
        "denominator": "all eight frozen rest starts/configuration; discovery screen only"}
    meta_path = OUT / "campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise RuntimeError("existing campaign metadata differs")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    src_copy = OUT / "campaign_runner_source.py"
    if not src_copy.exists(): shutil.copy2(Path(__file__), src_copy)
    hash_path = OUT / "campaign_code_sha256.json"
    code = {"runner_sha256": sha(Path(__file__)), "saved_runner_sha256": sha(src_copy),
        "integrator_sha256": sha(MODEL / "basin_scan.py"),
        "field_model_sha256": sha(MODEL / "field_model.json"),
        "campaign": OUT.name, "seed": SEED, "configs": len(configs),
        "starts_per_config": NSTART, "periods": PERIODS, "steps": STEPS,
        "workers": WORKERS}
    hash_path.write_text(json.dumps(code, indent=2) + "\n", encoding="utf-8")

    run_path = OUT / "trajectories.jsonl"
    existing = {}
    if run_path.exists():
        with run_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing[(int(row["screen_index"]), int(row["start_index"]))] = row
    tasks = [(ci, si) for ci in range(len(configs)) for si in range(NSTART)
             if (ci, si) not in existing]

    def run_one(task):
        ci, si = task
        c, ph = configs[ci], configs[ci]["physical"]
        raw = sim.integrate_one(starts[si], phases[si], np.asarray(c["p"]), family, k, coeff,
            PERIODS, STEPS, SKIP, qratio=1.0,
            oseen_Re_per_speed=ph["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["Re_per_dimensionless_speed"], hydro_mode=2,
            a1_over_R=ph["radius_m"] / R, a2_over_R=ph["radius_m"] / R,
            gravity_z=ph["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=ph["Cunningham"])
        end, turns, windows, rmin, rmax, zmax, dtheta, dmin, completed, retained, escaped, singular, cat, freq, vmax = raw
        axis = bool(min(rmin) < sim.RMIN_CLASS)
        return {"screen_index": ci, "config_id": c["id"], "radius_um": c["radius_um"],
            "start_index": si, "seed": SEED + 100, "initial": starts[si].tolist(),
            "phase": phases[si].tolist(), "final_state": end.tolist(),
            "turns": turns.tolist(), "window_turns": windows.tolist(),
            "raw_category": str(cat), "provisional_category": provisional(
                str(cat), turns, windows, bool(retained), axis),
            "retained": bool(retained), "completed_periods": int(completed),
            "r_min_over_R": rmin.tolist(), "r_max_over_R": rmax.tolist(),
            "z_max_over_R": zmax.tolist(), "distance_min_over_R": float(dmin),
            "max_speed_dimensionless": vmax.tolist(),
            "max_Re_by_particle": (vmax * ph["Re_per_dimensionless_speed"]).tolist()}

    if tasks:
        ci, si = tasks[0]; c = configs[ci]; ph = c["physical"]
        sim.integrate_one(starts[si], phases[si], np.asarray(c["p"]), family, k, coeff,
            1, STEPS, 0, qratio=1.0, oseen_Re_per_speed=ph["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["Re_per_dimensionless_speed"], hydro_mode=2,
            a1_over_R=ph["radius_m"] / R, a2_over_R=ph["radius_m"] / R,
            gravity_z=ph["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=ph["Cunningham"])

    print(json.dumps({"configs": len(configs), "trajectories": len(configs) * NSTART,
        "todo": len(tasks), "workers": WORKERS,
        "physical_screen_diagnostics": physical_diagnostics}, indent=2), flush=True)
    started = time.perf_counter()
    with run_path.open("a", encoding="utf-8", buffering=1) as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            existing[(row["screen_index"], row["start_index"])] = row
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if n % 64 == 0 or n == len(tasks):
                f.flush(); os.fsync(f.fileno())
                print(f"checkpoint {len(existing)}/{len(configs) * NSTART}; {time.perf_counter()-started:.1f}s", flush=True)

    outcome_counts = Counter(r["provisional_category"] for r in existing.values())
    by_radius = {}
    for radius_um in RADII_UM:
        z = [r for r in existing.values() if r["radius_um"] == radius_um]
        by_radius[str(radius_um)] = dict(Counter(r["provisional_category"] for r in z))
    summary = {"study": meta["study"], "rows": len(existing),
        "expected_rows": len(configs) * NSTART,
        "outcome_counts": dict(sorted(outcome_counts.items())), "outcomes_by_radius": by_radius,
        "classification": "Short discovery labels only; no probabilities or attractor claims.",
        "trajectory_sha256": sha(run_path), "meta_sha256": sha(meta_path),
        "wall_seconds": time.perf_counter() - started}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": summary["rows"], "outcomes": summary["outcome_counts"],
        "outcomes_by_radius": summary["outcomes_by_radius"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
