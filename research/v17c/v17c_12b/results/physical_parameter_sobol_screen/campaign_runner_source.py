"""Broad physically linked Sobol screen for an Al2O3 / hollow K1 pair."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
from scipy.stats import qmc

import physical_mixed_pair_search as base

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
OUT = HERE / "results" / "physical_parameter_sobol_screen"
OUT.mkdir(parents=True, exist_ok=True)

EPS0 = 8.8541878128e-12
ECHARGE = 1.602176634e-19
G = 9.80665
ETA = 1.81e-5
RHO_GAS = 1.177
MFP = 65.1e-9
R = 0.010
D = R / 0.95
RHO_ALUMINA = 3800.0
RHO_K1_RANGE = (100.0, 140.0)
H2 = 0.870316891091623
H4 = 0.7982172624273378
E1_UNIT = 26.3631
E2_UNIT = 26.4430
PROPOSAL_COUNT = 2048
CONFIG_COUNT = 1024
CONFIG_SEED = 20261250
N_PER_ENSEMBLE = 32
ENSEMBLES = ("rest", "weak", "wide", "edge")
START_SEEDS = {"rest": 20261330, "weak": 20261331, "wide": 20261332, "edge": 20261333}
PERIODS = 3000
STEPS = 128
SKIP = 100
CHUNK_SIZE = 256
WORKERS = min(11, os.cpu_count() or 1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cunningham(a: float) -> float:
    kn = MFP / a
    return 1.0 + kn * (1.257 + 0.4 * math.exp(-1.1 / kn))


def build_config(index: int, u: np.ndarray) -> dict:
    """Map physical Sobol coordinates to a mutually consistent parameter set."""
    a1 = (5.0 + 10.0 * u[0]) * 1e-6
    a2 = (15.0 + 35.0 * u[1]) * 1e-6
    rho2 = 100.0 + 40.0 * u[2]
    q1e = math.exp(math.log(1e5) + u[3] * (math.log(5e5) - math.log(1e5)))
    q2e = math.exp(math.log(1e5) + u[4] * (math.log(1e6) - math.log(1e5)))
    f1 = 50.0 + 70.0 * u[5]
    f2 = 30.0 + 150.0 * u[6]
    u1 = 100.0 + 2100.0 * u[7]
    u2 = 100.0 + 2100.0 * u[8]
    v2 = 100.0 + 800.0 * u[9]
    m1 = 4.0 * math.pi * RHO_ALUMINA * a1**3 / 3.0
    m2 = 4.0 * math.pi * rho2 * a2**3 / 3.0
    q1, q2 = q1e * ECHARGE, q2e * ECHARGE
    om1 = 2.0 * math.pi * f1
    om2 = 2.0 * math.pi * f2
    s = m1 / m2
    qratio = q2 / q1
    nu = om2 / om1
    A = q1 * u1 * H2 / (m1 * R**2 * om1**2)
    B = q1 * u2 * H4 / (m1 * R**2 * om1**2)
    beta = q1 * v2 / (m1 * R**2 * om1**2)
    C = q1 * q2 / (4.0 * math.pi * EPS0 * m1 * R**3 * om1**2)
    lam1 = 6.0 * math.pi * ETA * a1 / (m1 * om1)
    lam2 = 6.0 * math.pi * ETA * a2 / (m2 * om1)
    cc1, cc2 = cunningham(a1), cunningham(a2)
    re1 = 2.0 * RHO_GAS * a1 * R * om1 / ETA
    re2 = 2.0 * RHO_GAS * a2 * R * om1 / ETA
    gravity_z = -G / (R * om1**2)
    beta_cm = beta * (1.0 + qratio) / (1.0 + 1.0 / s)
    zcm_over_R = -gravity_z / (2.0 * beta_cm)
    mfs_sum = (E1_UNIT * u1 + E2_UNIT * u2) / D
    physical = {
        "trap_radius_m": R,
        "medium": "air, 300 K, 1 bar; eta=1.81e-5 Pa s; rho_gas=1.177 kg/m3",
        "frequency_f1_Hz": f1, "frequency_f2_Hz": f2,
        "omega1_rad_s": om1, "omega2_rad_s": om2,
        "particle1": {
            "material": "Al2O3", "radius_m": a1, "diameter_um": 2e6*a1,
            "density_kg_m3": RHO_ALUMINA, "mass_kg": m1, "charge_C": q1,
            "charge_e": q1e, "specific_charge_C_kg": q1/m1,
            "surface_charge_field_V_m": q1/(4.0*math.pi*EPS0*a1**2),
            "Kn": MFP/a1, "Cunningham": cc1, "lambda_Stokes": lam1,
            "Re_per_dimensionless_speed": re1,
        },
        "particle2": {
            "material": "hollow soda-lime-borosilicate glass, 3M K1 density branch",
            "radius_m": a2, "diameter_um": 2e6*a2,
            "density_kg_m3": rho2, "density_spec_range_kg_m3": list(RHO_K1_RANGE),
            "mass_kg": m2, "charge_C": q2, "charge_e": q2e,
            "specific_charge_C_kg": q2/m2,
            "surface_charge_field_V_m": q2/(4.0*math.pi*EPS0*a2**2),
            "Kn": MFP/a2, "Cunningham": cc2, "lambda_Stokes": lam2,
            "Re_per_dimensionless_speed": re2,
        },
        "dimensionless_parameters": [A, B, nu, beta, s, C, lam1, lam2],
        "mass_ratio_m1_over_m2": s, "charge_ratio_q2_over_q1": qratio,
        "A": A, "B": B, "nu": nu, "beta": beta, "C": C,
        "lambda1": lam1, "lambda2": lam2,
        "U1_peak_electrode_V": u1, "U2_peak_electrode_V": u2,
        "U1_plus_U2_peak_bound_V": u1 + u2,
        "V2_ideal_DC_coefficient_V_not_endcap": v2,
        "MFS_triangle_surface_field_bound_V_m": mfs_sum,
        "a1_over_R": a1/R, "a2_over_R": a2/R,
        "gravity_z_dimensionless": gravity_z,
        "center_of_mass_equilibrium_z_over_R_analytic": zcm_over_R,
        "beta_center_of_mass": beta_cm,
        "Cunningham_mean": 0.5*(cc1+cc2),
        "max_relative_mean_Cunningham_error": max(abs(0.5*(cc1+cc2)-cc1)/cc1, abs(0.5*(cc1+cc2)-cc2)/cc2),
        "MFS_note": "Triangle sum of separate RF-basis surface maxima; not phase maximum, FEM, or breakdown result.",
        "DC_note": "V2 is an ideal quadrupole coefficient, not an end-cap terminal voltage or validated segmented-electrode field.",
    }
    return {"id": f"phys{index:04d}", "sobol_index": index,
        "sobol_coordinates": u.tolist(), "p": [A, B, nu, beta, s, C, lam1, lam2],
        "qratio": qratio, "physical": physical}


def configs_from_sobol() -> tuple[list[dict], dict]:
    points = qmc.Sobol(d=10, scramble=True, seed=CONFIG_SEED).random_base2(int(math.log2(PROPOSAL_COUNT)))
    proposals = [build_config(i, x) for i, x in enumerate(points)]
    screened = []
    rejected = Counter()
    for cfg in proposals:
        ph = cfg["physical"]
        if ph["U1_plus_U2_peak_bound_V"] > 4400.0:
            rejected["RF_voltage_sum"] += 1
            continue
        if not (30.0 <= ph["frequency_f2_Hz"] <= 180.0 and 50.0 <= ph["frequency_f1_Hz"] <= 120.0):
            rejected["frequency_range"] += 1
            continue
        if not (1e5 <= ph["particle1"]["charge_e"] <= 5e5 and 1e5 <= ph["particle2"]["charge_e"] <= 1e6):
            rejected["charge_range"] += 1
            continue
        if not (100.0 <= ph["particle2"]["density_kg_m3"] <= 140.0):
            rejected["K1_density_range"] += 1
            continue
        if abs(ph["center_of_mass_equilibrium_z_over_R_analytic"]) > 1.5:
            rejected["gravity_center_of_mass_screen"] += 1
            continue
        screened.append(cfg)
    if len(screened) < CONFIG_COUNT:
        raise RuntimeError(f"only {len(screened)} configurations passed analytic screens; need {CONFIG_COUNT}")
    selected_idx = np.linspace(0, len(screened)-1, CONFIG_COUNT, dtype=int)
    selected = []
    for new_index, old_index in enumerate(selected_idx):
        cfg = dict(screened[int(old_index)])
        cfg["id"] = f"phys{new_index:04d}"
        selected.append(cfg)
    screen = {"sobol_proposals": PROPOSAL_COUNT, "physical_candidates_after_analytic_filters": len(screened),
        "selected_configurations": CONFIG_COUNT, "rejected_proposals": dict(sorted(rejected.items())),
        "filters": {"Al2O3_radius_um": [5.0, 15.0], "K1_diameter_um": [30.0, 100.0],
            "K1_density_kg_m3": [100.0, 140.0], "q1_e": [1e5, 5e5], "q2_e": [1e5, 1e6],
            "f1_Hz": [50.0, 120.0], "f2_Hz": [30.0, 180.0],
            "U1_plus_U2_V_max": 4400.0, "abs_center_of_mass_equilibrium_z_over_R_max": 1.5}}
    return selected, screen


def frozen_starts() -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    ys, ps, labels, seed_rows = [], [], [], []
    for group in ENSEMBLES:
        if group == "edge":
            y, phase = base.edge_initials(N_PER_ENSEMBLE, START_SEEDS[group])
        else:
            y, phase = base.sim.make_initials(group, N_PER_ENSEMBLE, START_SEEDS[group])
        ys.append(y); ps.append(phase); labels.extend([group]*N_PER_ENSEMBLE)
        seed_rows.append({"ensemble": group, "seed": START_SEEDS[group], "n": N_PER_ENSEMBLE})
    return np.vstack(ys), np.vstack(ps), labels, seed_rows


def main() -> None:
    configs, analytic_screen = configs_from_sobol()
    starts, phases, labels, seed_rows = frozen_starts()
    starts_path = OUT / "frozen_starts.npz"
    starts_seed_dict = dict(START_SEEDS)
    if starts_path.exists():
        frozen = np.load(starts_path, allow_pickle=True)
        if not np.array_equal(frozen["initial"], starts) or not np.array_equal(frozen["phases"], phases):
            raise RuntimeError("existing Sobol-screen frozen starts differ")
    else:
        np.savez_compressed(starts_path, initial=starts, phases=phases,
            ensemble=np.asarray(labels), seeds=np.asarray(json.dumps(seed_rows)),
            start_seed_json=np.asarray(json.dumps(starts_seed_dict, sort_keys=True)))

    _, family, k, coeff = base.sim.load_model()
    meta = {"study": "physically linked Sobol screen: Al2O3 + hollow 3M K1 in air",
        "interpretation": "Discovery screen over literature-constrained physical variables; not a measured joint pair or IID basin study.",
        "sobol_seed": CONFIG_SEED, "sobol_proposals": PROPOSAL_COUNT,
        "sobol_sampling_bounds": {"Al2O3_radius_um": [5.0, 15.0], "K1_outer_diameter_um": [30.0, 100.0],
            "K1_density_kg_m3": [100.0, 140.0], "q1_e": [1e5, 5e5], "q2_e": [1e5, 1e6],
            "f1_Hz": [50.0, 120.0], "f2_Hz": [30.0, 180.0],
            "U1_peak_V": [100.0, 2200.0], "U2_peak_V": [100.0, 2200.0], "V2_ideal_DC_coefficient_V": [100.0, 900.0]},
        "analytic_screen": analytic_screen, "configurations": configs,
        "ensembles": list(ENSEMBLES), "starts_per_ensemble": N_PER_ENSEMBLE,
        "start_seed_records": seed_rows, "total_trajectories": len(configs)*len(starts),
        "periods": PERIODS, "steps_per_rf_period": STEPS, "skip_periods": SKIP,
        "checkpoint_chunk_size": CHUNK_SIZE, "workers": WORKERS,
        "frozen_starts_sha256": sha(starts_path),
        "integrator_sha256": sha(MODEL/"basin_scan.py"),
        "field_model_sha256": sha(MODEL/"field_model.json"),
        "model_options": {"drag": "Schiller-Naumann with Cunningham mean", "hydrodynamics": 2,
            "gravity": True, "finite_particle_contacts": True,
            "note": "same legacy integrator and field basis as parent Al2O3/K1 screens"}}
    meta_path = OUT / "campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise RuntimeError("existing physical Sobol campaign metadata differs")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    runner_copy = OUT / "campaign_runner_source.py"
    if not runner_copy.exists(): shutil.copy2(Path(__file__), runner_copy)
    integrator_copy = OUT / "campaign_integrator_source.py"
    if not integrator_copy.exists(): shutil.copy2(MODEL/"basin_scan.py", integrator_copy)
    code = {"runner_sha256": sha(Path(__file__)), "saved_runner_sha256": sha(runner_copy),
        "integrator_sha256": sha(MODEL/"basin_scan.py"), "saved_integrator_sha256": sha(integrator_copy),
        "field_model_sha256": sha(MODEL/"field_model.json"), "frozen_starts_sha256": sha(starts_path),
        "workers": WORKERS, "configs": len(configs), "trajectories": meta["total_trajectories"],
        "periods": PERIODS, "steps_per_period": STEPS}
    (OUT/"campaign_code_sha256.json").write_text(json.dumps(code, indent=2)+"\n", encoding="utf-8")

    run_path = OUT / "trajectories.jsonl"
    existing = {}
    if run_path.exists():
        with run_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row=json.loads(line)
                    existing[(row["configuration"],row["ensemble"],int(row["start_index"]))]=row
    config_by_id = {c["id"]: c for c in configs}
    group_start = {g: i*N_PER_ENSEMBLE for i,g in enumerate(ENSEMBLES)}
    all_tasks = [(c["id"],g,i) for c in configs for g in ENSEMBLES for i in range(N_PER_ENSEMBLE)
        if (c["id"],g,i) not in existing]

    def run_one(task: tuple[str,str,int]) -> dict:
        config_id, group, si = task
        cfg = config_by_id[config_id]
        ph = cfg["physical"]
        idx = group_start[group] + si
        raw=base.sim.integrate_one(starts[idx], phases[idx], np.asarray(cfg["p"]), family, k, coeff,
            PERIODS, STEPS, SKIP, qratio=cfg["qratio"],
            oseen_Re_per_speed=ph["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2, a1_over_R=ph["a1_over_R"], a2_over_R=ph["a2_over_R"],
            gravity_z=ph["gravity_z_dimensionless"], drag_mode=2,
            slip_correction=ph["Cunningham_mean"])
        end, turns, windows, rmin, rmax, zmax, dtheta, dmin, completed, retained, escaped, singular, cat, freq, vmax = raw
        axis=bool(min(rmin)<base.sim.RMIN_CLASS)
        re=np.asarray([ph["particle1"]["Re_per_dimensionless_speed"],ph["particle2"]["Re_per_dimensionless_speed"]])
        return {"configuration":config_id,"ensemble":group,"start_index":si,
            "seed":START_SEEDS[group],"initial":starts[idx].tolist(),"phase":phases[idx].tolist(),
            "final_state":end.tolist(),"turns":turns.tolist(),"window_turns":windows.tolist(),
            "raw_category":str(cat),"provisional_category":base.provisional(str(cat),turns,windows,bool(retained),axis),
            "retained":bool(retained),"completed_periods":int(completed),
            "r_min_over_R":rmin.tolist(),"r_max_over_R":rmax.tolist(),"z_max_over_R":zmax.tolist(),
            "distance_min_over_R":float(dmin),"max_Re_by_particle":(vmax*re).tolist()}

    # A short warm-up compiles the Numba kernel before launching the full pool.
    if all_tasks:
        c0,g0,i0=all_tasks[0]; cfg0=config_by_id[c0]; ph0=cfg0["physical"]; idx0=group_start[g0]+i0
        base.sim.integrate_one(starts[idx0],phases[idx0],np.asarray(cfg0["p"]),family,k,coeff,
            1,STEPS,0,qratio=cfg0["qratio"],
            oseen_Re_per_speed=ph0["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph0["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2,a1_over_R=ph0["a1_over_R"],a2_over_R=ph0["a2_over_R"],
            gravity_z=ph0["gravity_z_dimensionless"],drag_mode=2,slip_correction=ph0["Cunningham_mean"])

    print(json.dumps({"configs":len(configs),"trajectories":meta["total_trajectories"],
        "todo":len(all_tasks),"workers":WORKERS,"periods":PERIODS,"steps_per_period":STEPS,
        "analytic_screen":analytic_screen},indent=2),flush=True)
    started=time.perf_counter()
    with run_path.open("a",encoding="utf-8",buffering=1) as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for offset in range(0,len(all_tasks),CHUNK_SIZE):
            batch=all_tasks[offset:offset+CHUNK_SIZE]
            future_map={pool.submit(run_one,t):t for t in batch}
            for future in as_completed(future_map):
                row=future.result(); key=(row["configuration"],row["ensemble"],row["start_index"])
                existing[key]=row
                f.write(json.dumps(row,separators=(",",":"))+"\n")
            f.flush(); os.fsync(f.fileno())
            progress={"completed_rows":len(existing),"expected_rows":meta["total_trajectories"],
                "last_checkpoint_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "elapsed_seconds_this_invocation":time.perf_counter()-started}
            (OUT/"progress.json").write_text(json.dumps(progress,indent=2)+"\n",encoding="utf-8")
            print(f"checkpoint {len(existing)}/{meta['total_trajectories']}; {time.perf_counter()-started:.1f}s",flush=True)

    per_config={}
    for cfg in configs:
        local=[r for r in existing.values() if r["configuration"]==cfg["id"]]
        counts=Counter(r["provisional_category"] for r in local)
        ret={g:sum(bool(r["retained"]) for r in local if r["ensemble"]==g) for g in ENSEMBLES}
        per_config[cfg["id"]]={"rows":len(local),"outcomes":dict(sorted(counts.items())),
            "retained_by_ensemble":ret,"retained_total":sum(ret.values()),
            "cr_total":counts["PROVISIONAL_CR_PLUS_MINUS"]+counts["PROVISIONAL_CR_MINUS_PLUS"],
            "co_total":counts["PROVISIONAL_CO_PLUS_PLUS"]+counts["PROVISIONAL_CO_MINUS_MINUS"]}
    candidate_ids=[cid for cid,x in per_config.items() if x["outcomes"].get("PROVISIONAL_CR_PLUS_MINUS",0)>0 and x["outcomes"].get("PROVISIONAL_CR_MINUS_PLUS",0)>0]
    all_counts=Counter(r["provisional_category"] for r in existing.values())
    summary={"study":meta["study"],"rows":len(existing),"expected_rows":meta["total_trajectories"],
        "outcome_counts":dict(sorted(all_counts.items())),"configuration_results":per_config,
        "configurations_with_both_counterrotation_labels":candidate_ids,
        "classification":"3000-period RK4 discovery labels, not strict long-horizon outcomes or IID probabilities.",
        "trajectory_sha256":sha(run_path),"meta_sha256":sha(meta_path),
        "wall_seconds_this_invocation":time.perf_counter()-started}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="configuration_results"},indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
