"""Experiment-anchored Al2O3 screen at 50 Hz, with rest and injection ensembles."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
OUT = HERE / "results" / "alumina_experimental_50hz_screen"
OUT.mkdir(parents=True, exist_ok=True)
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
D = R / 0.95
H2 = 0.870316891091623
H4 = 0.7982172624273378
E1_UNIT = 26.3631  # MFS surface maximum per (V/D) for RF1 basis
E2_UNIT = 26.4430  # MFS surface maximum per (V/D) for RF2 basis
RADII_UM = (10.0, 12.5, 15.0)
NCONFIG_PER_RADIUS = 128
N_PER_ENSEMBLE = 8
NENSEMBLES = ("rest", "experimental_injection")
SEED = 20261228
F1_HZ = 50.0
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


def build_configs() -> list[dict]:
    configs = []
    for radius_index, radius_um in enumerate(RADII_UM):
        design = qmc.Sobol(d=5, scramble=True, seed=SEED + radius_index).random_base2(7)
        for local_index, u in enumerate(design):
            a = radius_um * 1e-6
            m = 4 * math.pi * RHO_P * a**3 / 3
            q_e = math.exp(math.log(1e5) + u[0] * math.log(5.0))
            q = q_e * E_CHARGE
            total_voltage = math.exp(math.log(100.0) + u[1] * math.log(4400.0 / 100.0))
            voltage_ratio = math.exp(math.log(0.1) + u[2] * math.log(100.0))
            U1 = total_voltage / (1 + voltage_ratio)
            U2 = total_voltage * voltage_ratio / (1 + voltage_ratio)
            nu = 1.0 + 4.0 * u[3]
            beta = math.exp(math.log(0.003) + u[4] * math.log(0.08 / 0.003))
            omega = 2 * math.pi * F1_HZ
            lam_stokes = 6 * math.pi * ETA * a / (m * omega)
            C = q*q / (4 * math.pi * EPS0 * m * R**3 * omega**2)
            A = q * U1 * H2 / (m * R**2 * omega**2)
            B = q * U2 * H4 / (m * R**2 * omega**2)
            V2 = beta * m * R**2 * omega**2 / q
            slip = cc(a)
            Re_per_speed = 2 * RHO_G * a * R * omega / ETA
            flat_breakdown = paschen_screen(760 * 100 * 0.05 * D)
            physical = {
                "material": "Al2O3", "radius_m": a, "density_kg_m3": RHO_P,
                "mass_kg": m, "charge_C_each": q, "charge_e_each": q_e,
                "specific_charge_C_kg": q / m,
                "particle_isolated_surface_field_V_m": q / (4 * math.pi * EPS0 * a*a),
                "f1_Hz": F1_HZ, "f2_Hz": nu * F1_HZ,
                "omega1_rad_s": omega, "lambda_stokes_dimensionless": lam_stokes,
                "lambda_with_cunningham_dimensionless": lam_stokes * slip,
                "C_dimensionless": C,
                "U1_peak_electrode_V": U1, "U2_peak_electrode_V": U2,
                "U_sum_peak_bound_V": U1 + U2,
                "V2_ideal_DC_coefficient_V_not_endcap": V2,
                "MFS_separate_basis_triangle_bound_V_m": (E1_UNIT * U1 + E2_UNIT * U2) / D,
                "flat_air_Paschen_rod_shield_V_screen_only": flat_breakdown,
                "RF_to_flat_Paschen_ratio_voltage_only_screen": (U1 + U2) / flat_breakdown,
                "gravity_z_dimensionless": -G / (R * omega**2),
                "gravity_equilibrium_z_over_R_if_vertical": -G / (2 * beta * R * omega**2),
                "Kn": MFP / a, "Cunningham": slip,
                "Re_per_dimensionless_speed": Re_per_speed,
                "Omega_a2_over_nu_gas": omega * a*a / (ETA / RHO_G),
            }
            configs.append({"id": f"a{radius_um:g}um_exp50_{local_index:04d}",
                "screen_index": len(configs), "sobol_index": local_index,
                "radius_um": radius_um, "p": [A, B, nu, beta, 1.0, C, lam_stokes, lam_stokes],
                "qratio": 1.0, "U_voltage_ratio_U2_over_U1": voltage_ratio,
                "physical": physical})
    return configs


def build_frozen_starts() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = N_PER_ENSEMBLE
    rest, phase_rest = sim.make_initials("rest", n, SEED + 100)
    injection, phase_injection = sim.make_initials("rest", n, SEED + 101)
    velocity_u = qmc.Sobol(d=6, scramble=True, seed=SEED + 102).random_base2(3)
    physical_speed = np.zeros((n, 2))
    omega = 2 * math.pi * F1_HZ
    velocity_scale = R * omega
    for k in range(n):
        for i, (v0, v1, v2) in enumerate(((3, 4, 5), (9, 10, 11))):
            u0, u1, u2 = velocity_u[k, 3*i:3*i+3]
            speed = 0.5 + 1.5 * u0  # published particle injection range, m/s
            cos_theta = 2 * u1 - 1
            sin_theta = math.sqrt(max(0.0, 1 - cos_theta*cos_theta))
            phi = 2 * math.pi * u2
            injection[k, v0] = speed * sin_theta * math.cos(phi) / velocity_scale
            injection[k, v1] = speed * sin_theta * math.sin(phi) / velocity_scale
            injection[k, v2] = speed * cos_theta / velocity_scale
            physical_speed[k, i] = speed
    initial = np.vstack((rest, injection))
    phases = np.vstack((phase_rest, phase_injection))
    ensemble = np.array(["rest"] * n + ["experimental_injection"] * n)
    speed_data = np.vstack((np.zeros((n, 2)), physical_speed))
    return initial, phases, ensemble, speed_data


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


def main() -> None:
    configs = build_configs()
    starts, phases, ensemble, speed_data = build_frozen_starts()
    starts_path = OUT / "frozen_starts.npz"
    if not starts_path.exists():
        np.savez_compressed(starts_path, initial=starts, phases=phases,
            ensemble=ensemble, injection_speed_m_s=speed_data,
            sobol_seed=SEED + 100, velocity_seed=SEED + 102)
    else:
        frozen = np.load(starts_path)
        if not np.array_equal(frozen["initial"], starts) or not np.array_equal(frozen["phases"], phases):
            raise RuntimeError("existing frozen starts differ")
    _, family, k, coeff = sim.load_model()
    meta = {"study": "experiment-anchored Al2O3 50 Hz physical discovery",
        "design": "128 physical Sobol designs at each of three measured Al2O3 radii; q and electrode voltages sampled directly, then dimensionless coefficients derived",
        "sobol_seed_base": SEED, "n_configurations": len(configs),
        "configurations": configs, "n_per_ensemble_per_config": N_PER_ENSEMBLE,
        "ensembles": list(NENSEMBLES), "periods": PERIODS, "steps_per_rf_period": STEPS,
        "skip_periods": SKIP, "workers": WORKERS,
        "particle_source": "Lapitsky et al. 2013, DOI 10.1002/ctpp.201300011: Al2O3 radius 10-15 um, rho~3800 kg/m3, reported q=1e5-5e5 e",
        "injection_source": "Vasilyak et al. 2013, DOI 10.1088/1367-2630/15/4/043047: reported charged-particle injection speed 0.5-2 m/s",
        "voltage_source_caveat": "The 4.4 kV total peak-electrode-amplitude envelope is a separate four-rod apparatus reference, not a universal breakdown limit; voltage notation transfer is uncertain.",
        "physical_model": {"medium": "air, 300 K, 760 Torr", "eta_Pa_s": ETA,
            "rho_gas_kg_m3": RHO_G, "mean_free_path_m": MFP,
            "R_m": R, "geometry": "24 rods, V17C MFS transverse field, shield 1.10D"},
        "starts_seed": SEED + 100, "velocity_seed": SEED + 102,
        "integrator_sha256": sha(MODEL / "basin_scan.py"),
        "field_model_sha256": sha(MODEL / "field_model.json"),
        "starts_sha256": sha(starts_path),
        "denominator": "all 16 rest or experimental-injection starts per configuration; short discovery screen only"}
    meta_path = OUT / "campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise RuntimeError("existing campaign metadata differs")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    source_copy = OUT / "campaign_runner_source.py"
    if not source_copy.exists(): shutil.copy2(Path(__file__), source_copy)
    (OUT / "campaign_code_sha256.json").write_text(json.dumps({
        "runner_sha256": sha(Path(__file__)), "saved_runner_sha256": sha(source_copy),
        "integrator_sha256": sha(MODEL / "basin_scan.py"),
        "field_model_sha256": sha(MODEL / "field_model.json"),
        "campaign": OUT.name, "seed": SEED, "configs": len(configs),
        "starts_per_config": len(starts), "periods": PERIODS, "steps": STEPS,
        "workers": WORKERS}, indent=2) + "\n", encoding="utf-8")

    run_path = OUT / "trajectories.jsonl"
    existing = {}
    if run_path.exists():
        with run_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing[(int(row["screen_index"]), int(row["start_index"]))] = row
    tasks = [(ci, si) for ci in range(len(configs)) for si in range(len(starts))
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
            "start_index": si, "ensemble": str(ensemble[si]),
            "injection_speed_m_s": speed_data[si].tolist(),
            "seed": SEED + 100, "initial": starts[si].tolist(), "phase": phases[si].tolist(),
            "final_state": end.tolist(), "turns": turns.tolist(), "window_turns": windows.tolist(),
            "raw_category": str(cat), "provisional_category": provisional(str(cat), turns, windows, bool(retained), axis),
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
            gravity_z=ph["gravity_z_dimensionless"], drag_mode=2, slip_correction=ph["Cunningham"])

    print(json.dumps({"configs": len(configs), "starts_per_config": len(starts),
        "trajectories": len(configs) * len(starts), "todo": len(tasks), "workers": WORKERS}, indent=2), flush=True)
    started = time.perf_counter()
    with run_path.open("a", encoding="utf-8", buffering=1) as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            existing[(row["screen_index"], row["start_index"])] = row
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if n % 64 == 0 or n == len(tasks):
                f.flush(); os.fsync(f.fileno())
                print(f"checkpoint {len(existing)}/{len(configs)*len(starts)}; {time.perf_counter()-started:.1f}s", flush=True)

    by_ensemble = defaultdict(Counter)
    by_radius = defaultdict(Counter)
    for r in existing.values():
        by_ensemble[r["ensemble"]][r["provisional_category"]] += 1
        by_radius[str(r["radius_um"])][r["provisional_category"]] += 1
    summary = {"study": meta["study"], "rows": len(existing),
        "expected_rows": len(configs) * len(starts),
        "outcome_counts": dict(sorted(Counter(r["provisional_category"] for r in existing.values()).items())),
        "outcomes_by_ensemble": {k: dict(sorted(v.items())) for k, v in sorted(by_ensemble.items())},
        "outcomes_by_radius": {k: dict(sorted(v.items())) for k, v in sorted(by_radius.items(), key=lambda x: float(x[0]))},
        "classification": "Short discovery labels only; not final probabilities or attractor verification.",
        "trajectory_sha256": sha(run_path), "meta_sha256": sha(meta_path),
        "wall_seconds": time.perf_counter() - started}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": summary["rows"], "outcome_counts": summary["outcome_counts"],
        "outcomes_by_ensemble": summary["outcomes_by_ensemble"],
        "outcomes_by_radius": summary["outcomes_by_radius"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
