"""Re-map legacy candidates to lower, experiment-scale RF frequencies.

The transformation preserves A, B, nu, beta, mass and charge ratios, C, and
Stokes lambdas analytically while deriving new sizes, masses, charges, voltages,
drag corrections, and gravity from the changed f1. It is a fixed three-point
physical replay, not a Sobol sequence or local optimizer.
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
OUT = RESULTS / "rescaled_legacy_candidate_validation"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MODEL))
import physical_mixed_pair_search as base  # noqa: E402
import updated_objective_validation as original  # noqa: E402

R = 0.010
ETA = 1.81e-5
RHO_GAS = 1.177
MFP = 65.1e-9
H2 = 0.870316891091623
H4 = 0.7982172624273378
G = 9.80665
EPS0 = 8.8541878128e-12
GROUPS = ("rest", "weak", "wide", "edge")
N_PER_GROUP = 128
SEEDS = {"rest": 20261850, "weak": 20261851, "wide": 20261852, "edge": 20261853}
TARGET_F1_HZ = {"candidate52": 45.0, "local15": 50.0, "refine12_sobol17": 50.0}
PERIODS = 15000
STEPS = 64
SKIP = 200
CHUNK_SIZE = 128
WORKERS = min(11, os.cpu_count() or 1)
E1_UNIT = 26.3631
E2_UNIT = 26.4430


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def cunningham(a: float) -> float:
    kn=MFP/a
    return 1+kn*(1.257+0.4*math.exp(-1.1/kn))


def rescale(cfg: dict) -> dict:
    old=cfg["physical"]
    name=cfg["legacy_candidate"]
    f1_new=TARGET_F1_HZ[name]
    f1_old=old["frequency_f1_Hz"]
    ratio=f1_new/f1_old
    size_scale=1/math.sqrt(ratio)
    charge_voltage_scale=ratio**0.25
    p1=old["particle1"].copy(); p2=old["particle2"].copy()
    for particle in (p1,p2):
        particle["radius_m"]*=size_scale
        particle["diameter_um"]*=size_scale
        particle["mass_kg"]=4*math.pi*particle["density_kg_m3"]*particle["radius_m"]**3/3
        particle["charge_C"]*=charge_voltage_scale
        particle["charge_e"]*=charge_voltage_scale
        particle["specific_charge_C_kg"]=particle["charge_C"]/particle["mass_kg"]
        particle["surface_charge_field_V_m"]=particle["charge_C"]/(4*math.pi*EPS0*particle["radius_m"]**2)
        particle["Kn"]=MFP/particle["radius_m"]
        particle["Cunningham"]=cunningham(particle["radius_m"])
    omega1=2*math.pi*f1_new
    f2_new=cfg["p"][2]*f1_new
    omega2=2*math.pi*f2_new
    p1["lambda_Stokes"]=6*math.pi*ETA*p1["radius_m"]/(p1["mass_kg"]*omega1)
    p2["lambda_Stokes"]=6*math.pi*ETA*p2["radius_m"]/(p2["mass_kg"]*omega1)
    p1["Re_per_dimensionless_speed"]=2*RHO_GAS*p1["radius_m"]*R*omega1/ETA
    p2["Re_per_dimensionless_speed"]=2*RHO_GAS*p2["radius_m"]*R*omega1/ETA
    u1=old["U1_peak_electrode_V"]*charge_voltage_scale
    u2=old["U2_peak_electrode_V"]*charge_voltage_scale
    v2=old["V2_ideal_DC_coefficient_V_not_endcap"]*charge_voltage_scale
    ph={**old,"frequency_f1_Hz":f1_new,"frequency_f2_Hz":f2_new,
        "omega1_rad_s":omega1,"omega2_rad_s":omega2,"particle1":p1,"particle2":p2,
        "U1_peak_electrode_V":u1,"U2_peak_electrode_V":u2,
        "U1_plus_U2_peak_bound_V":u1+u2,
        "V2_ideal_DC_coefficient_V_not_endcap":v2,
        "a1_over_R":p1["radius_m"]/R,"a2_over_R":p2["radius_m"]/R,
        "gravity_z_dimensionless":-G/(R*omega1**2),
        "Cunningham_mean":0.5*(p1["Cunningham"]+p2["Cunningham"]),
        "max_relative_mean_Cunningham_error":max(
            abs(0.5*(p1["Cunningham"]+p2["Cunningham"])/p1["Cunningham"]-1),
            abs(0.5*(p1["Cunningham"]+p2["Cunningham"])/p2["Cunningham"]-1)),
        "MFS_triangle_surface_field_bound_V_m":(E1_UNIT*u1+E2_UNIT*u2)/(R/0.95),
        "frequency_scaling": {"f1_old_Hz":f1_old,"f1_new_Hz":f1_new,
            "size_scale":size_scale,"charge_and_voltage_scale":charge_voltage_scale,
            "A_B_nu_beta_s_qratio_C_lambda_Stokes_preserved":True}}
    return {"id":f"scaled_{name}_f{int(f1_new)}hz","source":"analytic frequency rescaling of literal V17C.10 physical map",
        "legacy_candidate":name,"p":cfg["p"],"qratio":cfg["qratio"],"physical":ph,
        "legacy_source_dimensionless":cfg["legacy_source_dimensionless"],
        "legacy_dimensioned_at_original_f1":cfg["legacy_dimensioned"]}


def candidates() -> list[dict]:
    wanted=original.get_candidates()
    by_name={c.get("legacy_candidate"):c for c in wanted if c.get("legacy_candidate")}
    return [rescale(by_name[name]) for name in ("candidate52","local15","refine12_sobol17")]


def frozen_starts() -> tuple[np.ndarray,np.ndarray,list[str],list[dict]]:
    ys,phases,labels,seed_rows=[],[],[],[]
    for group in GROUPS:
        if group=="edge": y,phase=base.edge_initials(N_PER_GROUP,SEEDS[group])
        else: y,phase=base.sim.make_initials(group,N_PER_GROUP,SEEDS[group])
        ys.append(y);phases.append(phase);labels.extend([group]*N_PER_GROUP)
        seed_rows.append({"ensemble":group,"seed":SEEDS[group],"n":N_PER_GROUP})
    return np.vstack(ys),np.vstack(phases),labels,seed_rows


def main() -> None:
    cfgs=candidates()
    starts,phases,labels,seeds=frozen_starts()
    starts_path=OUT/"frozen_starts.npz"
    if starts_path.exists():
        frozen=np.load(starts_path,allow_pickle=True)
        if not np.array_equal(frozen["initial"],starts) or not np.array_equal(frozen["phases"],phases):
            raise RuntimeError("existing rescaled validation starts differ")
    else:
        np.savez_compressed(starts_path,initial=starts,phases=phases,ensemble=np.asarray(labels),
            seed_records=np.asarray(json.dumps(seeds,sort_keys=True)))
    _,family,k,coeff=base.sim.load_model()
    meta={"study":"V17C.12B frequency-rescaled legacy candidate validation",
        "objective":"any stable CR direction counts; no balance condition; retain CO and losses as separate outcomes",
        "scaling_rule":"for each old candidate and target f1, set a_i proportional to f1^-1/2, m_i from spherical density, q_i and electrode/DC voltages proportional to f1^1/4; preserve A/B/nu/beta/s/qratio/C and Stokes lambdas analytically",
        "target_f1_Hz":TARGET_F1_HZ,"candidates":cfgs,"ensembles":list(GROUPS),
        "starts_per_ensemble":N_PER_GROUP,"seed_records":seeds,
        "total_trajectories":len(cfgs)*len(GROUPS)*N_PER_GROUP,
        "periods":PERIODS,"steps_per_rf_period":STEPS,"transient_periods":SKIP,
        "strict_late_turn_threshold_per_particle":10.0,"workers":WORKERS,
        "checkpoint_chunk_size":CHUNK_SIZE,"frozen_starts_sha256":sha(starts_path),
        "integrator_sha256":sha(MODEL/"basin_scan.py"),"field_model_sha256":sha(MODEL/"field_model.json"),
        "limitations":["Analytic scaling preserves point-particle dimensionless coefficients but changes finite radii/R, gravity, Cunningham correction, Brownian forcing, and actual Reynolds numbers; each scaled configuration must be integrated independently.",
            "Particle 2 density is inherited from the old solved-density mapping and is not proof of an identified commercial grade or jointly measured pair.",
            "This fixed three-point replay does not validate breakdown, surface charging, or 100-orbit stability."]}
    meta_path=OUT/"campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8"))!=meta:
        raise RuntimeError("existing frequency-rescaled campaign metadata differs")
    meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    runner_copy=OUT/"campaign_runner_source.py"
    if not runner_copy.exists():shutil.copy2(Path(__file__),runner_copy)
    integrator_copy=OUT/"campaign_integrator_source.py"
    if not integrator_copy.exists():shutil.copy2(MODEL/"basin_scan.py",integrator_copy)
    code={"runner_sha256":sha(Path(__file__)),"saved_runner_sha256":sha(runner_copy),
        "integrator_sha256":sha(MODEL/"basin_scan.py"),"saved_integrator_sha256":sha(integrator_copy),
        "field_model_sha256":sha(MODEL/"field_model.json"),"frozen_starts_sha256":sha(starts_path)}
    (OUT/"campaign_code_sha256.json").write_text(json.dumps(code,indent=2)+"\n",encoding="utf-8")
    rows_path=OUT/"trajectories.jsonl";existing={}
    if rows_path.exists():
        with rows_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    x=json.loads(line);existing[(x["configuration"],x["ensemble"],int(x["start_index"]))]=x
    cfg={x["id"]:x for x in cfgs}; offsets={g:i*N_PER_GROUP for i,g in enumerate(GROUPS)}
    tasks=[(c["id"],g,i) for c in cfgs for g in GROUPS for i in range(N_PER_GROUP)
        if (c["id"],g,i) not in existing]
    def one(task):
        cid,g,si=task; x=cfg[cid]; ph=x["physical"]; idx=offsets[g]+si
        raw=base.sim.integrate_one(starts[idx],phases[idx],np.asarray(x["p"],dtype=np.float64),
            family,k,coeff,PERIODS,STEPS,SKIP,qratio=x["qratio"],
            oseen_Re_per_speed=ph["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2,a1_over_R=ph["a1_over_R"],a2_over_R=ph["a2_over_R"],
            gravity_z=ph["gravity_z_dimensionless"],drag_mode=2,slip_correction=ph["Cunningham_mean"])
        end,turns,windows,rmin,rmax,zmax,dtheta,dmin,completed,retained,escaped,singular,rawcat,freq,vmax=raw
        axis=bool(min(rmin)<base.sim.RMIN_CLASS)
        pro=base.provisional(str(rawcat),turns,windows,bool(retained),axis)
        strict=original.strict_late_label(str(rawcat),turns,windows,bool(retained),axis)
        re=np.asarray([ph["particle1"]["Re_per_dimensionless_speed"],ph["particle2"]["Re_per_dimensionless_speed"]])
        return {"configuration":cid,"ensemble":g,"start_index":si,"seed":SEEDS[g],
            "initial":starts[idx].tolist(),"phase":phases[idx].tolist(),"final_state":end.tolist(),
            "raw_category":str(rawcat),"provisional_category":pro,"strict_late_category":strict,
            "turns":turns.tolist(),"window_turns":windows.tolist(),"frequency_over_rf_Hz":freq.tolist(),
            "r_min_over_R":rmin.tolist(),"r_max_over_R":rmax.tolist(),"z_max_over_R":zmax.tolist(),
            "distance_min_over_R":float(dmin),"max_Re_by_particle":(vmax*re).tolist(),
            "retained":bool(retained),"escaped":bool(escaped),"singular_unresolved":bool(singular),
            "completed_periods":int(completed)}
    if tasks:
        c0,g0,i0=tasks[0];ph0=cfg[c0]["physical"];idx0=offsets[g0]+i0
        base.sim.integrate_one(starts[idx0],phases[idx0],np.asarray(cfg[c0]["p"],dtype=np.float64),
            family,k,coeff,1,STEPS,0,qratio=cfg[c0]["qratio"],
            oseen_Re_per_speed=ph0["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph0["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2,a1_over_R=ph0["a1_over_R"],a2_over_R=ph0["a2_over_R"],
            gravity_z=ph0["gravity_z_dimensionless"],drag_mode=2,slip_correction=ph0["Cunningham_mean"])
    print(json.dumps({"configs":len(cfgs),"todo":len(tasks),"workers":WORKERS,"periods":PERIODS,"steps":STEPS},indent=2),flush=True)
    started=time.perf_counter()
    with rows_path.open("a",encoding="utf-8",buffering=1) as f,ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for offset in range(0,len(tasks),CHUNK_SIZE):
            batch=tasks[offset:offset+CHUNK_SIZE];fmap={pool.submit(one,t):t for t in batch}
            for fut in as_completed(fmap):
                row=fut.result();existing[(row["configuration"],row["ensemble"],row["start_index"])]=row
                f.write(json.dumps(row,separators=(",",":"))+"\n")
            f.flush();os.fsync(f.fileno())
            progress={"completed_rows":len(existing),"expected_rows":meta["total_trajectories"],
                "last_checkpoint_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "elapsed_seconds_this_invocation":time.perf_counter()-started}
            (OUT/"progress.json").write_text(json.dumps(progress,indent=2)+"\n",encoding="utf-8")
            print(f"checkpoint {len(existing)}/{meta['total_trajectories']}; {time.perf_counter()-started:.1f}s",flush=True)
    results={}
    for x in cfgs:
        local=[r for r in existing.values() if r["configuration"]==x["id"]];by={}
        for g in GROUPS:
            rr=[r for r in local if r["ensemble"]==g];cc=Counter(r["strict_late_category"] for r in rr)
            by[g]={"n":len(rr),"strict_late":dict(sorted(cc.items())),
                "CR_any_direction":cc["STRICT_CR_PLUS_MINUS"]+cc["STRICT_CR_MINUS_PLUS"],
                "CR_plus_minus":cc["STRICT_CR_PLUS_MINUS"],"CR_minus_plus":cc["STRICT_CR_MINUS_PLUS"],
                "co_rotation":cc["STRICT_CO_PLUS_PLUS"]+cc["STRICT_CO_MINUS_MINUS"],
                "escape":cc["ESCAPE"]+cc["WALL_OR_AXIAL_ESCAPE"]}
        cr=sum(g["CR_any_direction"] for g in by.values())
        results[x["id"]]={"rows":len(local),"CR_any_direction_of_512":cr,"CR_fraction_of_512":cr/512,
            "co_rotation_total":sum(v["co_rotation"] for v in by.values()),
            "escape_total":sum(v["escape"] for v in by.values()),"per_ensemble":by}
    summary={"study":meta["study"],"rows":len(existing),"expected_rows":meta["total_trajectories"],
        "trajectory_sha256":sha(rows_path),"meta_sha256":sha(meta_path),"configuration_results":results,
        "classification":"Fresh-start 15,000-period RK4/64 screen, strict late-window label and >=10 net turns per particle; discovery frequency only.",
        "wall_seconds_this_invocation":time.perf_counter()-started}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="configuration_results"},indent=2,ensure_ascii=True),flush=True)
    for cid,v in results.items():print(json.dumps({"configuration":cid,**v},ensure_ascii=True),flush=True)


if __name__=="__main__":main()
