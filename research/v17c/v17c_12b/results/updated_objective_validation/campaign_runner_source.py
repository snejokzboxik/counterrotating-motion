"""Fresh-start 15,000-period validation for updated-objective candidates.

Candidates: the two best physically linked Sobol discoveries and literal
10-mm-air dimensional mappings of candidate52, local15, and refine12_sobol17.
The output distinguishes the parent screen's provisional label from a stricter
late-window label requiring at least ten net turns per particle.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
RESULTS = HERE / "results"
OUT = RESULTS / "updated_objective_validation"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MODEL))
import physical_mixed_pair_search as base  # noqa: E402

R = 0.010
ETA = 1.81e-5
RHO_GAS = 1.177
MFP = 65.1e-9
EPS0 = 8.8541878128e-12
ECHARGE = 1.602176634e-19
H2 = 0.870316891091623
H4 = 0.7982172624273378
G = 9.80665
GROUPS = ("rest", "weak", "wide", "edge")
N_PER_GROUP = 128
SEEDS = {"rest": 20261750, "weak": 20261751, "wide": 20261752, "edge": 20261753}
PERIODS = 15000
STEPS = 64
SKIP = 200
CHUNK_SIZE = 128
WORKERS = min(11, os.cpu_count() or 1)
STRICT_TURN_THRESHOLD = 10.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cunningham(a: float) -> float:
    kn = MFP / a
    return 1.0 + kn * (1.257 + 0.4 * math.exp(-1.1 / kn))


def get_candidates() -> list[dict]:
    screen_meta = json.loads((RESULTS / "physical_parameter_sobol_screen" / "campaign_meta.json").read_text(encoding="utf-8"))
    ranking = json.loads((RESULTS / "physical_parameter_sobol_screen" / "updated_objective_ranking.json").read_text(encoding="utf-8"))
    screen_by_id = {c["id"]: c for c in screen_meta["configurations"]}
    selected = []
    for row in ranking["top_50"][:2]:
        cfg = screen_by_id[row["configuration"]]
        selected.append({**cfg, "source": "physical Sobol discovery leader", "screen_rank": row["rank_updated_objective"],
                         "screen_CR_count_of_128": row["preliminary_CR_count_of_128"]})

    legacy = json.loads((RESULTS / "legacy_candidate_physical_map_10mm.json").read_text(encoding="utf-8"))
    legacy_by_name = {row["candidate"]: row for row in legacy["rows"]}
    for name in ("candidate52", "local15", "refine12_sobol17"):
        row = legacy_by_name[name]
        p = row["source_dimensionless"]
        d = row["dimensioned"]
        a1, a2 = d["radius1_m"], d["radius2_m"]
        m1, m2 = d["mass1_kg"], d["mass2_kg"]
        rho1, rho2 = 3990.0, d["rho2_kg_m3"]
        q1, q2 = d["q1_C"], d["q2_C"]
        f1, f2 = d["f1_Hz"], d["f2_Hz"]
        om1, om2 = 2*math.pi*f1, 2*math.pi*f2
        u1, u2, v2 = d["U1_peak_electrode_V"], d["U2_peak_electrode_V"], d["V2_ideal_DC_coefficient_V_not_endcap"]
        re1 = 2*RHO_GAS*a1*R*om1/ETA
        re2 = 2*RHO_GAS*a2*R*om1/ETA
        cc1, cc2 = cunningham(a1), cunningham(a2)
        physical = {
            "trap_radius_m": R,
            "medium": "dry air, 300 K, 1 atm; eta=1.81e-5 Pa s; rho_gas=1.177 kg/m3",
            "frequency_f1_Hz": f1, "frequency_f2_Hz": f2,
            "omega1_rad_s": om1, "omega2_rad_s": om2,
            "particle1": {"material": "solid spherical Al2O3 (legacy map assumption)", "radius_m": a1,
                          "diameter_um": 2e6*a1, "density_kg_m3": rho1, "mass_kg": m1,
                          "charge_C": q1, "charge_e": d["q1_e"], "specific_charge_C_kg": q1/m1,
                          "surface_charge_field_V_m": q1/(4*math.pi*EPS0*a1*a1), "Kn": MFP/a1,
                          "Cunningham": cc1, "lambda_Stokes": 6*math.pi*ETA*a1/(m1*om1),
                          "Re_per_dimensionless_speed": re1},
            "particle2": {"material": "unidentified low-density hollow/glass sphere (legacy target density)",
                          "radius_m": a2, "diameter_um": 2e6*a2, "density_kg_m3": rho2,
                          "mass_kg": m2, "charge_C": q2, "charge_e": d["q2_e"],
                          "specific_charge_C_kg": q2/m2, "surface_charge_field_V_m": q2/(4*math.pi*EPS0*a2*a2),
                          "Kn": MFP/a2, "Cunningham": cc2,
                          "lambda_Stokes": 6*math.pi*ETA*a2/(m2*om1),
                          "Re_per_dimensionless_speed": re2},
            "dimensionless_parameters": [p["A"], p["B"], p["nu"], p["beta"], p["m1_over_m2"], p["C"],
                                         p["lambda1_target_effective"], p["lambda2_target_effective"]],
            "mass_ratio_m1_over_m2": p["m1_over_m2"], "charge_ratio_q2_over_q1": p["q2_over_q1"],
            "A": p["A"], "B": p["B"], "nu": p["nu"], "beta": p["beta"], "C": p["C"],
            "lambda1": p["lambda1_target_effective"], "lambda2": p["lambda2_target_effective"],
            "U1_peak_electrode_V": u1, "U2_peak_electrode_V": u2,
            "U1_plus_U2_peak_bound_V": u1+u2,
            "V2_ideal_DC_coefficient_V_not_endcap": v2,
            "a1_over_R": a1/R, "a2_over_R": a2/R,
            "gravity_z_dimensionless": -G/(R*om1**2),
            "Cunningham_mean": 0.5*(cc1+cc2),
            "max_relative_mean_Cunningham_error": max(abs((0.5*(cc1+cc2))/cc1-1), abs((0.5*(cc1+cc2))/cc2-1)),
            "MFS_triangle_surface_field_bound_V_m": d.get("MFS_triangle_surface_field_bound_V_m"),
            "physical_mapping_status": "dimensional mapping of old dimensionless candidate; not an identified joint particle pair",
        }
        selected.append({"id": f"legacy_{name}", "source": "V17C.10 literal 10-mm air mapping",
                         "p": physical["dimensionless_parameters"], "qratio": p["q2_over_q1"],
                         "physical": physical, "legacy_candidate": name,
                         "legacy_source_dimensionless": p, "legacy_dimensioned": d})
    return selected


def frozen_starts() -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    ys, phases, labels, seed_rows = [], [], [], []
    for group in GROUPS:
        if group == "edge":
            y, phase = base.edge_initials(N_PER_GROUP, SEEDS[group])
        else:
            y, phase = base.sim.make_initials(group, N_PER_GROUP, SEEDS[group])
        ys.append(y)
        phases.append(phase)
        labels.extend([group] * N_PER_GROUP)
        seed_rows.append({"ensemble": group, "seed": SEEDS[group], "n": N_PER_GROUP})
    return np.vstack(ys), np.vstack(phases), labels, seed_rows


def strict_late_label(raw: str, turns: np.ndarray, windows: np.ndarray, retained: bool, axis: bool) -> str:
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
    if retained and np.all(signs != 0) and np.all(np.abs(turns) >= STRICT_TURN_THRESHOLD):
        if np.all(signs[:, 0] * signs[:, 1] < 0):
            return "STRICT_CR_PLUS_MINUS" if turns[0] > 0 else "STRICT_CR_MINUS_PLUS"
        if np.all(signs[:, 0] * signs[:, 1] > 0):
            return "STRICT_CO_PLUS_PLUS" if turns[0] > 0 else "STRICT_CO_MINUS_MINUS"
    if retained and np.all(np.abs(turns) < 0.25): return "NONROTATING"
    return "UNRESOLVED_LATE_WINDOW"


def main() -> None:
    configs = get_candidates()
    starts, phases, labels, seed_rows = frozen_starts()
    starts_path = OUT / "frozen_starts.npz"
    if starts_path.exists():
        frozen = np.load(starts_path, allow_pickle=True)
        if not np.array_equal(frozen["initial"], starts) or not np.array_equal(frozen["phases"], phases):
            raise RuntimeError("existing frozen starts differ from requested validation seeds")
    else:
        np.savez_compressed(starts_path, initial=starts, phases=phases, ensemble=np.asarray(labels),
                            seed_records=np.asarray(json.dumps(seed_rows, sort_keys=True)))
    _, family, k, coeff = base.sim.load_model()
    meta = {
        "study": "V17C.12B fresh-start validation under updated any-direction counterrotation objective",
        "objective": "count either stable counterrotation label as success; no balance requirement between (+-) and (-+); co-rotation and escape remain separate failures",
        "candidates": configs, "ensembles": list(GROUPS), "starts_per_ensemble": N_PER_GROUP,
        "seed_records": seed_rows, "total_trajectories": len(configs)*len(GROUPS)*N_PER_GROUP,
        "periods": PERIODS, "steps_per_rf_period": STEPS, "transient_periods": SKIP,
        "strict_late_turn_threshold_per_particle": STRICT_TURN_THRESHOLD,
        "workers": WORKERS, "checkpoint_chunk_size": CHUNK_SIZE,
        "frozen_starts_sha256": sha(starts_path),
        "integrator_sha256": sha(MODEL/"basin_scan.py"),
        "field_model_sha256": sha(MODEL/"field_model.json"),
        "classification_note": "Pooled CR counts are screening frequencies on four separate frozen ensembles. The stricter label requires 10 net turns per particle and consistent signs in three late windows; neither label alone proves an attractor or an IID probability.",
        "limitations": [
            "The 2 selected Sobol discovery points inherit their model assumptions and screening bounds.",
            "The 3 legacy candidate mappings use 3990 kg/m3 Al2O3, 50 um outer-radius hollow particle, and solved low-density particle-2 densities; these are not identified measured particle pairs.",
            "Legacy literal mappings are diagnostic even when voltage, frequency, or charge lies outside the earlier reference screen.",
            "No new Sobol sequence or local optimizer is run by this validation script.",
        ],
    }
    meta_path = OUT / "campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise RuntimeError("existing updated-objective campaign metadata differs")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    runner_copy = OUT / "campaign_runner_source.py"
    if not runner_copy.exists(): shutil.copy2(Path(__file__), runner_copy)
    integrator_copy = OUT / "campaign_integrator_source.py"
    if not integrator_copy.exists(): shutil.copy2(MODEL/"basin_scan.py", integrator_copy)
    code_hashes = {"runner_sha256": sha(Path(__file__)), "saved_runner_sha256": sha(runner_copy),
                   "integrator_sha256": sha(MODEL/"basin_scan.py"), "saved_integrator_sha256": sha(integrator_copy),
                   "field_model_sha256": sha(MODEL/"field_model.json"), "frozen_starts_sha256": sha(starts_path)}
    (OUT/"campaign_code_sha256.json").write_text(json.dumps(code_hashes,indent=2)+"\n",encoding="utf-8")

    run_path = OUT / "trajectories.jsonl"
    existing = {}
    if run_path.exists():
        with run_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row=json.loads(line)
                    existing[(row["configuration"],row["ensemble"],int(row["start_index"]))]=row
    cfg_by_id = {c["id"]: c for c in configs}
    group_offset = {g: i*N_PER_GROUP for i,g in enumerate(GROUPS)}
    tasks = [(c["id"],g,i) for c in configs for g in GROUPS for i in range(N_PER_GROUP)
             if (c["id"],g,i) not in existing]

    def integrate(task: tuple[str,str,int]) -> dict:
        cid, group, si = task
        cfg = cfg_by_id[cid]
        ph = cfg["physical"]
        idx = group_offset[group]+si
        raw = base.sim.integrate_one(starts[idx],phases[idx],np.asarray(cfg["p"],dtype=np.float64),
            family,k,coeff,PERIODS,STEPS,SKIP,qratio=cfg["qratio"],
            oseen_Re_per_speed=ph["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2,a1_over_R=ph["a1_over_R"],a2_over_R=ph["a2_over_R"],
            gravity_z=ph["gravity_z_dimensionless"],drag_mode=2,slip_correction=ph["Cunningham_mean"])
        end,turns,windows,rmin,rmax,zmax,dtheta,dmin,completed,retained,escaped,singular,rawcat,freq,vmax=raw
        axis=bool(min(rmin)<base.sim.RMIN_CLASS)
        provisional=base.provisional(str(rawcat),turns,windows,bool(retained),axis)
        strict=strict_late_label(str(rawcat),turns,windows,bool(retained),axis)
        re=np.asarray([ph["particle1"]["Re_per_dimensionless_speed"],ph["particle2"]["Re_per_dimensionless_speed"]])
        return {"configuration":cid,"ensemble":group,"start_index":si,"seed":SEEDS[group],
                "initial":starts[idx].tolist(),"phase":phases[idx].tolist(),"final_state":end.tolist(),
                "raw_category":str(rawcat),"provisional_category":provisional,"strict_late_category":strict,
                "turns":turns.tolist(),"window_turns":windows.tolist(),"frequency_over_rf_Hz":freq.tolist(),
                "r_min_over_R":rmin.tolist(),"r_max_over_R":rmax.tolist(),"z_max_over_R":zmax.tolist(),
                "distance_min_over_R":float(dmin),"max_Re_by_particle":(vmax*re).tolist(),
                "retained":bool(retained),"escaped":bool(escaped),"singular_unresolved":bool(singular),
                "completed_periods":int(completed)}

    if tasks:
        c0,g0,i0=tasks[0]; cfg0=cfg_by_id[c0]; ph0=cfg0["physical"]; idx0=group_offset[g0]+i0
        base.sim.integrate_one(starts[idx0],phases[idx0],np.asarray(cfg0["p"],dtype=np.float64),
            family,k,coeff,1,STEPS,0,qratio=cfg0["qratio"],
            oseen_Re_per_speed=ph0["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph0["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2,a1_over_R=ph0["a1_over_R"],a2_over_R=ph0["a2_over_R"],
            gravity_z=ph0["gravity_z_dimensionless"],drag_mode=2,slip_correction=ph0["Cunningham_mean"])

    print(json.dumps({"configs":len(configs),"todo":len(tasks),"workers":WORKERS,
                      "periods":PERIODS,"steps_per_rf_period":STEPS},indent=2),flush=True)
    started=time.perf_counter()
    with run_path.open("a",encoding="utf-8",buffering=1) as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for offset in range(0,len(tasks),CHUNK_SIZE):
            batch=tasks[offset:offset+CHUNK_SIZE]
            future_map={pool.submit(integrate,t):t for t in batch}
            for future in as_completed(future_map):
                row=future.result()
                existing[(row["configuration"],row["ensemble"],row["start_index"])]=row
                f.write(json.dumps(row,separators=(",",":"))+"\n")
            f.flush(); os.fsync(f.fileno())
            progress={"completed_rows":len(existing),"expected_rows":meta["total_trajectories"],
                      "last_checkpoint_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "elapsed_seconds_this_invocation":time.perf_counter()-started}
            (OUT/"progress.json").write_text(json.dumps(progress,indent=2)+"\n",encoding="utf-8")
            print(f"checkpoint {len(existing)}/{meta['total_trajectories']}; {time.perf_counter()-started:.1f}s",flush=True)

    results={}
    for cfg in configs:
        local=[r for r in existing.values() if r["configuration"]==cfg["id"]]
        by_group={}
        for g in GROUPS:
            rows=[r for r in local if r["ensemble"]==g]
            by_group[g]={"n":len(rows),
                "provisional":dict(sorted(Counter(r["provisional_category"] for r in rows).items())),
                "strict_late":dict(sorted(Counter(r["strict_late_category"] for r in rows).items())),
                "strict_CR_any_direction":sum(r["strict_late_category"] in ("STRICT_CR_PLUS_MINUS","STRICT_CR_MINUS_PLUS") for r in rows),
                "strict_CR_plus_minus":sum(r["strict_late_category"]=="STRICT_CR_PLUS_MINUS" for r in rows),
                "strict_CR_minus_plus":sum(r["strict_late_category"]=="STRICT_CR_MINUS_PLUS" for r in rows),
                "co_rotation":sum(r["strict_late_category"] in ("STRICT_CO_PLUS_PLUS","STRICT_CO_MINUS_MINUS") for r in rows),
                "escape":sum(r["strict_late_category"] in ("ESCAPE","WALL_OR_AXIAL_ESCAPE") for r in rows)}
        cr=sum(v["strict_CR_any_direction"] for v in by_group.values())
        results[cfg["id"]]={"rows":len(local),"strict_CR_any_direction_of_512":cr,
            "strict_CR_fraction_of_512":cr/512,"co_rotation_total":sum(v["co_rotation"] for v in by_group.values()),
            "escape_total":sum(v["escape"] for v in by_group.values()),"per_ensemble":by_group}
    summary={"study":meta["study"],"rows":len(existing),"expected_rows":meta["total_trajectories"],
        "trajectory_sha256":sha(run_path),"meta_sha256":sha(meta_path),"configuration_results":results,
        "classification":"Fresh-start 15,000-period RK4/64 screen. Strict late label requires consistent signs in three late windows and at least ten net turns per particle; discovery frequencies only.",
        "wall_seconds_this_invocation":time.perf_counter()-started}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="configuration_results"},indent=2,ensure_ascii=True),flush=True)
    for cid,value in results.items():
        print(json.dumps({"configuration":cid,**value},ensure_ascii=True),flush=True)


if __name__=="__main__":
    main()
