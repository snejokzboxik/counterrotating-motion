"""Four-ensemble physical screen for an Al2O3 / hollow-glass K1 pair scenario."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
OUT = HERE / "results" / "mixed_alumina_k1_candidate"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MODEL))
import basin_scan as sim  # noqa: E402

EPS0 = 8.8541878128e-12
ECHARGE = 1.602176634e-19
G = 9.80665
ETA = 1.81e-5
RHO_GAS = 1.177
MFP = 65.1e-9
R = 0.010
D = R / 0.95
RHO_ALUMINA = 3800.0
RHO_K1_TYPICAL = 125.0
A2 = 50e-6  # 100 um outer diameter; within the K1 upper size range
F1 = 50.0
PERIODS = 15000
STEPS = 64
SKIP = 100
N_PER_ENSEMBLE = 128
ENSEMBLES = ("rest", "weak", "wide", "edge")
SEEDS = {"rest": 20261230, "weak": 20261231, "wide": 20261232, "edge": 20261233}
WORKERS = min(11, os.cpu_count() or 1)

# V17C.10 refine12_sobol17 field/ratio coordinates, treated as a starting
# hypothesis only. Dimensional masses, charges, drag, C and voltages are linked
# below and are not scanned independently.
A_FIELD = 0.03492413282400757
B_FIELD = 0.15631667332941065
NU = 3.2196773638017477
BETA = 0.015388559140477825
MASS_RATIO = 0.10963763906505944  # m1/m2
Q_RATIO = 3.4490536225028334  # q2/q1
LAMBDA_RATIO_TARGET = 0.17750147818833362 / 0.24575362874758605
H2 = 0.870316891091623
H4 = 0.7982172624273378
E1_UNIT = 26.3631  # MFS surface maximum per (V/D), RF1 basis
E2_UNIT = 26.4430  # MFS surface maximum per (V/D), RF2 basis


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cunningham(a: float) -> float:
    kn = MFP / a
    return 1.0 + kn * (1.257 + 0.4 * math.exp(-1.1 / kn))


def build_configuration() -> dict:
    # Select a fixed experimental-range charge pair; q2/q1 follows the old
    # candidate ratio. This is a literature-constrained scenario, not a
    # measured joint size/mass/charge record for one particular pair.
    q1 = 100000.0 * ECHARGE
    q2 = Q_RATIO * q1
    a1 = A2 * MASS_RATIO / LAMBDA_RATIO_TARGET
    m1 = 4.0 * math.pi * RHO_ALUMINA * a1**3 / 3.0
    m2 = m1 / MASS_RATIO
    rho2 = m2 / (4.0 * math.pi * A2**3 / 3.0)
    if not 100.0 <= rho2 <= 140.0:
        raise RuntimeError(f"derived K1 density {rho2} kg/m3 outside 100-140 range")
    if not 1.0e5 <= q1 / ECHARGE <= 1.0e6 or not 1.0e5 <= q2 / ECHARGE <= 1.0e6:
        raise RuntimeError("charge pair lies outside 2013 experimental range")

    omega = 2.0 * math.pi * F1
    lam1 = 6.0 * math.pi * ETA * a1 / (m1 * omega)
    lam2 = 6.0 * math.pi * ETA * A2 / (m2 * omega)
    coupling = q1 * q2 / (4.0 * math.pi * EPS0 * m1 * R**3 * omega**2)
    u1 = A_FIELD * m1 * R**2 * omega**2 / (q1 * H2)
    u2 = B_FIELD * m1 * R**2 * omega**2 / (q1 * H4)
    v2 = BETA * m1 * R**2 * omega**2 / q1
    a1_R, a2_R = a1 / R, A2 / R
    cc1, cc2 = cunningham(a1), cunningham(A2)
    re1 = 2.0 * RHO_GAS * a1 * R * omega / ETA
    re2 = 2.0 * RHO_GAS * A2 * R * omega / ETA
    physical = {
        "scenario": "Al2O3 + 3M K1-like hollow borosilicate sphere; literature-constrained, not a measured paired record",
        "medium": "air, 300 K, 1 bar; Stokes lambda with Schiller-Naumann drag correction",
        "trap_radius_m": R,
        "frequency_f1_Hz": F1,
        "frequency_f2_Hz": NU * F1,
        "omega1_rad_s": omega,
        "particle1": {
            "material": "Al2O3", "radius_m": a1, "diameter_um": 2e6 * a1,
            "density_kg_m3": RHO_ALUMINA, "mass_kg": m1, "charge_C": q1,
            "charge_e": q1 / ECHARGE, "specific_charge_C_kg": q1 / m1,
            "surface_charge_field_V_m": q1 / (4.0 * math.pi * EPS0 * a1**2),
            "Kn": MFP / a1, "Cunningham": cc1, "lambda_Stokes": lam1,
            "Re_per_dimensionless_speed": re1,
        },
        "particle2": {
            "material": "hollow soda-lime-borosilicate glass, 3M K1 density branch",
            "radius_m": A2, "diameter_um": 2e6 * A2,
            "density_kg_m3": rho2, "density_spec_typical_kg_m3": RHO_K1_TYPICAL,
            "density_spec_range_kg_m3": [100.0, 140.0], "mass_kg": m2,
            "charge_C": q2, "charge_e": q2 / ECHARGE,
            "specific_charge_C_kg": q2 / m2,
            "surface_charge_field_V_m": q2 / (4.0 * math.pi * EPS0 * A2**2),
            "Kn": MFP / A2, "Cunningham": cc2, "lambda_Stokes": lam2,
            "Re_per_dimensionless_speed": re2,
            "manufacturer_size_d10_d50_d90_um": [30.0, 65.0, 115.0],
            "manufacturer_effective_top_size_um": 120.0,
        },
        "dimensionless_parameters": [A_FIELD, B_FIELD, NU, BETA,
            MASS_RATIO, coupling, lam1, lam2],
        "mass_ratio_m1_over_m2": m1 / m2,
        "charge_ratio_q2_over_q1": q2 / q1,
        "lambda_ratio_lambda2_over_lambda1": lam2 / lam1,
        "U1_peak_electrode_V": u1,
        "U2_peak_electrode_V": u2,
        "U1_plus_U2_peak_bound_V": u1 + u2,
        "V2_ideal_DC_coefficient_V_not_endcap": v2,
        "MFS_triangle_surface_field_bound_V_m": (E1_UNIT*u1 + E2_UNIT*u2) / D,
        "a1_over_R": a1_R, "a2_over_R": a2_R,
        "gravity_z_dimensionless": -G / (R * omega**2),
        "Cunningham_factors_applied": "mean of two particle-specific Cc values in the current scalar-drag integrator",
        "Cunningham_mean": 0.5 * (cc1 + cc2),
        "maximum_relative_Cunningham_approximation": max(
            abs(0.5*(cc1+cc2)-cc1)/cc1, abs(0.5*(cc1+cc2)-cc2)/cc2),
        "boundary_field_note": "MFS triangle sum of separate basis maxima; not a phase-torus maximum or breakdown verdict.",
    }
    return {"p": [A_FIELD, B_FIELD, NU, BETA, MASS_RATIO,
            coupling, lam1, lam2], "qratio": q2/q1, "physical": physical}


def edge_initials(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    u = qmc.Sobol(d=14, scramble=True, seed=seed).random_base2(int(math.log2(count)))
    out = np.zeros((count, 12), dtype=np.float64)
    for particle, rcol, tcol, zcol in ((0, 0, 1, 4), (1, 2, 3, 5)):
        radius = np.sqrt(0.72**2 + u[:, rcol] * (0.90**2 - 0.72**2))
        theta = 2.0 * math.pi * u[:, tcol]
        out[:, 6*particle] = radius * np.cos(theta)
        out[:, 6*particle+1] = radius * np.sin(theta)
        out[:, 6*particle+2] = 0.8 * (2.0*u[:, zcol] - 1.0)
    velocity_columns = np.clip(u[:, 6:12], 1e-7, 1.0-1e-7)
    out[:, [3, 4, 5, 9, 10, 11]] = 0.05 * ndtri(velocity_columns)
    phases = 2.0 * math.pi * u[:, 12:14]
    return out, phases


def frozen_starts() -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    all_y, all_phase, ensemble, seeds = [], [], [], []
    for group in ENSEMBLES:
        if group == "edge":
            y, phase = edge_initials(N_PER_ENSEMBLE, SEEDS[group])
        else:
            y, phase = sim.make_initials(group, N_PER_ENSEMBLE, SEEDS[group])
        all_y.append(y); all_phase.append(phase)
        ensemble.extend([group] * N_PER_ENSEMBLE)
        seeds.append({"ensemble": group, "seed": SEEDS[group], "n": N_PER_ENSEMBLE})
    return np.vstack(all_y), np.vstack(all_phase), ensemble, seeds


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
    cfg = build_configuration()
    starts, phases, labels, seeds = frozen_starts()
    starts_path = OUT / "frozen_starts.npz"
    if not starts_path.exists():
        np.savez_compressed(starts_path, initial=starts, phases=phases,
            ensemble=np.asarray(labels), seeds=np.asarray(json.dumps(seeds)),
            start_seed_json=np.asarray(json.dumps(SEEDS, sort_keys=True)))
    else:
        frozen = np.load(starts_path, allow_pickle=True)
        if not np.array_equal(frozen["initial"], starts) or not np.array_equal(frozen["phases"], phases):
            raise RuntimeError("existing frozen physical-pair starts differ")
    _, family, k, coeff = sim.load_model()
    meta = {
        "study": "four-ensemble discovery for a literature-constrained Al2O3 / hollow K1 glass pair",
        "interpretation": "Fixed physical scenario derived from separately published ranges; not a directly measured joint particle pair and not a final probability campaign.",
        "source_basis": [
            "Vasilyak et al., NJP 15 043047 (2013), DOI 10.1088/1367-2630/15/4/043047: positive dust-particle charge range 1e5-1e6 e, Al2O3 sizes 10-80 um and hollow glass 30-100 um; individual joint records not reported.",
            "3M K1 manufacturer data sheet: true density 0.125 g/cm3 (0.10-0.14), d10/d50/d90 30/65/115 um, effective top size 120 um.",
            "V17C.10 refine12_sobol17 is used only as a starting hypothesis for field and dimensionless ratios; it is not treated as a physical data source."
        ],
        "configuration": cfg,
        "ensembles": list(ENSEMBLES), "starts_per_ensemble": N_PER_ENSEMBLE,
        "seed_records": seeds, "total_trajectories": len(starts),
        "periods": PERIODS, "steps_per_rf_period": STEPS, "skip_periods": SKIP,
        "workers": WORKERS, "integrator_sha256": sha(MODEL / "basin_scan.py"),
        "field_model_sha256": sha(MODEL / "field_model.json"),
        "starts_sha256": sha(starts_path),
        "model_options": {"drag_mode": "Schiller-Naumann", "hydro_mode": 2,
            "gravity": True, "finite_particle_contacts": True,
            "Cunningham": "mean Cc applied to both particles; maximum relative approximation recorded in physical metadata"},
    }
    meta_path = OUT / "campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise RuntimeError("existing physical-pair campaign metadata differs")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    source_copy = OUT / "campaign_runner_source.py"
    if not source_copy.exists(): shutil.copy2(Path(__file__), source_copy)
    integrator_copy = OUT / "campaign_integrator_source.py"
    if not integrator_copy.exists(): shutil.copy2(MODEL / "basin_scan.py", integrator_copy)
    code = {"runner_sha256": sha(Path(__file__)), "saved_runner_sha256": sha(source_copy),
        "integrator_sha256": sha(MODEL / "basin_scan.py"),
        "saved_integrator_sha256": sha(integrator_copy),
        "field_model_sha256": sha(MODEL / "field_model.json"),
        "campaign": OUT.name, "workers": WORKERS, "seed_records": seeds,
        "trajectories": len(starts), "periods": PERIODS, "steps": STEPS}
    (OUT / "campaign_code_sha256.json").write_text(json.dumps(code, indent=2) + "\n", encoding="utf-8")

    runs_path = OUT / "trajectories.jsonl"
    existing = {}
    if runs_path.exists():
        with runs_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line); existing[(r["ensemble"], int(r["start_index"]))] = r
    tasks = [(group, si) for group in ENSEMBLES for si in range(N_PER_ENSEMBLE)
        if (group, si) not in existing]
    group_starts = {group: i*N_PER_ENSEMBLE for i, group in enumerate(ENSEMBLES)}
    phys = cfg["physical"]
    a1_R, a2_R = phys["a1_over_R"], phys["a2_over_R"]
    cc_mean = phys["Cunningham_mean"]
    if tasks:
        group, si = tasks[0]; idx = group_starts[group] + si
        sim.integrate_one(starts[idx], phases[idx], np.asarray(cfg["p"]), family, k, coeff,
            1, STEPS, 0, qratio=cfg["qratio"],
            oseen_Re_per_speed=phys["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=phys["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2, a1_over_R=a1_R, a2_over_R=a2_R,
            gravity_z=phys["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=cc_mean)

    print(json.dumps({"configurations": 1, "ensembles": len(ENSEMBLES),
        "starts_per_ensemble": N_PER_ENSEMBLE, "trajectories": len(starts),
        "todo": len(tasks), "workers": WORKERS,
        "dimensionless_parameters": cfg["p"], "physical": phys}, indent=2), flush=True)
    started = time.perf_counter()

    def run_one(task):
        group, si = task
        idx = group_starts[group] + si
        raw = sim.integrate_one(starts[idx], phases[idx], np.asarray(cfg["p"]),
            family, k, coeff, PERIODS, STEPS, SKIP, qratio=cfg["qratio"],
            oseen_Re_per_speed=phys["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=phys["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2, a1_over_R=a1_R, a2_over_R=a2_R,
            gravity_z=phys["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=cc_mean)
        end, turns, windows, rmin, rmax, zmax, dtheta, dmin, completed, retained, escaped, singular, cat, freq, vmax = raw
        axis = bool(min(rmin) < sim.RMIN_CLASS)
        return {"ensemble": group, "start_index": si, "seed": SEEDS[group],
            "initial": starts[idx].tolist(), "phase": phases[idx].tolist(),
            "final_state": end.tolist(), "turns": turns.tolist(),
            "window_turns": windows.tolist(), "raw_category": str(cat),
            "provisional_category": provisional(str(cat), turns, windows, bool(retained), axis),
            "retained": bool(retained), "completed_periods": int(completed),
            "r_min_over_R": rmin.tolist(), "r_max_over_R": rmax.tolist(),
            "z_max_over_R": zmax.tolist(), "distance_min_over_R": float(dmin),
            "max_Re_by_particle": (vmax*np.asarray([
                phys["particle1"]["Re_per_dimensionless_speed"],
                phys["particle2"]["Re_per_dimensionless_speed"]])).tolist()}

    with runs_path.open("a", encoding="utf-8", buffering=1) as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result(); existing[(row["ensemble"], row["start_index"])] = row
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if n % 32 == 0 or n == len(tasks):
                f.flush(); os.fsync(f.fileno())
                print(f"checkpoint {len(existing)}/{len(starts)}; {time.perf_counter()-started:.1f}s", flush=True)

    outcomes = Counter(r["provisional_category"] for r in existing.values())
    by_group = defaultdict(Counter)
    for r in existing.values(): by_group[r["ensemble"]][r["provisional_category"]] += 1
    summary = {"study": meta["study"], "rows": len(existing), "expected_rows": len(starts),
        "outcome_counts": dict(sorted(outcomes.items())),
        "outcomes_by_ensemble": {k: dict(sorted(v.items())) for k,v in sorted(by_group.items())},
        "pair_retained_by_ensemble": {k: sum(bool(r["retained"]) for r in existing.values() if r["ensemble"]==k) for k in ENSEMBLES},
        "classification": "Provisional long-horizon labels for one fixed literature-constrained scenario; not final basin probabilities.",
        "trajectory_sha256": sha(runs_path), "meta_sha256": sha(meta_path),
        "wall_seconds": time.perf_counter()-started}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
