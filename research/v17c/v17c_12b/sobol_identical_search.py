"""Physically linked identical-Al2O3 discovery scan, with short-horizon labels.

Controls A/B/nu/beta are electrode/frequency controls. lambda and C are used
only to reconstruct the actual drive frequency and common particle charge;
the radius, mass, medium, q/m, drag and Coulomb terms are then dimensionally
linked. Short classifications are discovery labels only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
MODEL = HERE/"model"
OUT = HERE/"results"/"identical_physical_sobol_screen_corrected"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MODEL))
import basin_scan as sim  # noqa: E402

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
ETA = 1.81e-5
RHO_G = 1.177
MFP = 65.1e-9
RHO_P = 3990.0
A_P = 17e-6
R = 0.010
H2 = 0.870316891091623
H4 = 0.7982172624273378
NCONFIG = 512
NSTART = 8
SEED = 20261225
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
    kn = MFP/a
    return 1+kn*(1.257+0.4*math.exp(-1.1/kn))


def paschen(p_d_torr_cm: float) -> float:
    return 365*p_d_torr_cm/(math.log(15*p_d_torr_cm)-math.log(math.log(101)))


def build_config(u: np.ndarray, index: int) -> dict:
    A = math.exp(math.log(0.008)+u[0]*math.log(0.08/0.008))
    B = math.exp(math.log(0.03)+u[1]*math.log(0.30/0.03))
    nu = 1.5+3.5*u[2]
    beta = math.exp(math.log(0.003)+u[3]*math.log(0.08/0.003))
    lam = math.exp(math.log(0.08)+u[4]*math.log(0.60/0.08))
    C = math.exp(math.log(1e-6)+u[5]*math.log(6e-5/1e-6))
    m = 4*math.pi*RHO_P*A_P**3/3
    slip = cc(A_P)
    # p[6] is the code's Stokes lambda=gamma_Stokes/(m*Omega). The actual
    # dimensionless drag in _rhs is lambda*Cc, so do not also put Cc into this
    # frequency inversion.
    omega = 4.5*ETA/(RHO_P*A_P**2*lam)
    q = math.sqrt(C*4*math.pi*EPS0*m*R**3*omega**2)
    scale = m*R**2*omega**2/q
    U1, U2, V2 = A*scale/H2, B*scale/H4, beta*scale
    D = R/0.95
    gap = 0.05*D
    vbreak = paschen(760*100*gap)
    recoef = 2*RHO_G*A_P*R*omega/ETA
    p = [A,B,nu,beta,1.0,C,lam,lam]
    phys = {"f1_Hz":omega/(2*math.pi), "f2_Hz":nu*omega/(2*math.pi),
        "omega1_rad_s":omega, "mass_kg":m, "radius_m":A_P,
        "charge_C_each":q, "charge_e_each":q/E_CHARGE,
        "specific_charge_C_kg":q/m,
        "particle_isolated_surface_field_V_m":q/(4*math.pi*EPS0*A_P**2),
        "U1_peak_electrode_V":U1, "U2_peak_electrode_V":U2,
        "V2_ideal_DC_coefficient_V_not_endcap":V2,
        "RF_rod_shield_voltage_bound_V":U1+U2,
        "flat_air_Paschen_rod_shield_V_screen_only":vbreak,
        "RF_to_flat_Paschen_ratio_screen_only":(U1+U2)/vbreak,
        "gravity_z_dimensionless":-9.80665/(R*omega**2),
        "gravity_equilibrium_z_over_R_if_vertical":-9.80665/(2*beta*R*omega**2),
        "lambda1_stokes_dimensionless":lam,
        "lambda1_with_cunningham_dimensionless":lam*slip,
        "Kn":MFP/A_P, "Cunningham":slip,
        "Re_per_dimensionless_speed":recoef,
        "Omega_a2_over_nu_gas":omega*A_P**2/(ETA/RHO_G)}
    return {"id":f"phys{index:04d}", "index":index, "p":p,
        "qratio":1.0, "physical":phys,
        "empirical_charge_window":1e4 <= phys["charge_e_each"] <= 1e6,
        "preferred_charge_window":1e5 <= phys["charge_e_each"] <= 5e5,
        "reported_voltage_envelope_4p4kV":U1+U2 <= 4400.0,
        "flat_Paschen_screen_under_unity":(U1+U2)/vbreak <= 1.0}


def provisional(raw: str, turns: np.ndarray, windows: np.ndarray,
                retained: bool, axis: bool) -> str:
    if raw == "finite_particle_contact": return "PARTICLE_CONTACT"
    if raw == "finite_particle_wall_contact": return "WALL_CONTACT"
    if raw == "lost_retention": return "ESCAPE"
    if raw == "point_singularity_unresolved": return "UNRESOLVED"
    if axis: return "AXIS_AMBIGUOUS"
    signs = np.zeros((3,2), dtype=np.int8)
    for w in range(3):
        for i in range(2):
            if abs(windows[w,i]) >= 0.05:
                signs[w,i] = 1 if windows[w,i] > 0 else -1
    if retained and np.all(signs != 0) and np.all(np.abs(turns) >= 1.0):
        if np.all(signs[:,0]*signs[:,1] < 0):
            return "PROVISIONAL_CR_PLUS_MINUS" if turns[0] > 0 else "PROVISIONAL_CR_MINUS_PLUS"
        if np.all(signs[:,0]*signs[:,1] > 0):
            return "PROVISIONAL_CO_PLUS_PLUS" if turns[0] > 0 else "PROVISIONAL_CO_MINUS_MINUS"
    if retained and np.all(np.abs(turns) < 0.25): return "NONROTATING"
    return "UNRESOLVED_SHORT"


def main() -> None:
    # Oversample the physical design, retain configurations inside broad
    # measured charge and published apparatus-voltage envelopes.
    raw_design = qmc.Sobol(d=6, scramble=True, seed=SEED).random_base2(11)
    all_configs = [build_config(u, i) for i,u in enumerate(raw_design)]
    valid = [c for c in all_configs if c["empirical_charge_window"]
             and c["reported_voltage_envelope_4p4kV"]]
    if len(valid) < NCONFIG:
        raise RuntimeError(f"only {len(valid)} designs pass broad charge/voltage screens")
    configs = valid[:NCONFIG]
    for i, c in enumerate(configs): c["screen_index"] = i

    starts, phases = sim.make_initials("rest", NSTART, SEED+100)
    starts_path = OUT/"frozen_rest_starts.npz"
    np.savez_compressed(starts_path, initial=starts, phases=phases,
        seed=SEED+100, ensemble="rest", n=NSTART)
    _, family, k, coeff = sim.load_model()
    meta = {"study":"dimensionally linked Sobol discovery; identical Al2O3 in air",
        "design":"2048 Sobol candidates in A,B,nu,beta,lambda,C; first 512 passing broad q and voltage screens",
        "source_sobol_seed":SEED,"screen_seed":SEED+100,
        "n_physical_designs_before_filter":len(all_configs),
        "n_designs_passing_q_and_voltage_screen":len(valid),
        "configurations":configs,"n_starts_per_config":NSTART,"periods":PERIODS,
        "steps_per_rf_period":STEPS,"skip_periods":SKIP,"workers":WORKERS,
        "particle":{"material":"Al2O3","radius_m":A_P,"density_kg_m3":RHO_P},
        "medium":{"gas":"air","T_K":300,"eta_Pa_s":ETA,"rho_kg_m3":RHO_G,"mfp_m":MFP},
        "trap":{"R_m":R,"geometry":"24 rods, V17C MFS field, shield 1.10D"},
        "integrator_sha256":sha(MODEL/"basin_scan.py"),
        "field_model_sha256":sha(MODEL/"field_model.json"),
        "starts_sha256":sha(starts_path),
        "denominator":"all eight rest starts; short-horizon screen, not final probability"}
    meta_path=OUT/"campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise RuntimeError("existing Sobol-screen metadata differs")
    meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    run_path=OUT/"trajectories.jsonl"
    existing={}
    if run_path.exists():
        for line in run_path.read_text(encoding="utf-8").splitlines():
            row=json.loads(line); existing[(row["screen_index"],row["start_index"])]=row
    tasks=[(ci,si) for ci in range(NCONFIG) for si in range(NSTART) if (ci,si) not in existing]

    def run_one(task:tuple[int,int])->dict:
        ci,si=task; c=configs[ci]; ph=c["physical"]
        raw=sim.integrate_one(starts[si],phases[si],np.asarray(c["p"]),family,k,coeff,
            PERIODS,STEPS,SKIP,qratio=1.0,oseen_Re_per_speed=ph["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["Re_per_dimensionless_speed"],hydro_mode=2,
            a1_over_R=A_P/R,a2_over_R=A_P/R,gravity_z=ph["gravity_z_dimensionless"],
            drag_mode=2,slip_correction=ph["Cunningham"])
        end,turns,windows,rmin,rmax,zmax,dtheta,dmin,completed,retained,escaped,singular,cat,freq,vmax=raw
        axis=bool(min(rmin)<sim.RMIN_CLASS)
        return {"screen_index":ci,"config_id":c["id"],"start_index":si,
            "seed":SEED+100,"initial":starts[si].tolist(),"phase":phases[si].tolist(),
            "final_state":end.tolist(),"turns":turns.tolist(),"window_turns":windows.tolist(),
            "raw_category":str(cat),"provisional_category":provisional(str(cat),turns,windows,bool(retained),axis),
            "retained":bool(retained),"completed_periods":int(completed),
            "r_min_over_R":rmin.tolist(),"r_max_over_R":rmax.tolist(),"z_max_over_R":zmax.tolist(),
            "distance_min_over_R":float(dmin),"max_speed_dimensionless":vmax.tolist(),
            "max_Re_by_particle":(vmax*ph["Re_per_dimensionless_speed"]).tolist()}

    if tasks:
        ci,si=tasks[0]; c=configs[ci]; ph=c["physical"]
        sim.integrate_one(starts[si],phases[si],np.asarray(c["p"]),family,k,coeff,1,STEPS,0,
            qratio=1.0,oseen_Re_per_speed=ph["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["Re_per_dimensionless_speed"],hydro_mode=2,
            a1_over_R=A_P/R,a2_over_R=A_P/R,gravity_z=ph["gravity_z_dimensionless"],
            drag_mode=2,slip_correction=ph["Cunningham"])
    print(json.dumps({"selected_configs":len(configs),"passing_designs":len(valid),
        "trajectories":len(configs)*NSTART,"todo":len(tasks),"workers":WORKERS}),flush=True)
    started=time.perf_counter()
    with run_path.open("a",encoding="utf-8",buffering=1) as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures=[pool.submit(run_one,t) for t in tasks]
        for n,fut in enumerate(as_completed(futures),1):
            row=fut.result(); key=(row["screen_index"],row["start_index"])
            existing[key]=row; f.write(json.dumps(row,separators=(",",":"))+"\n")
            if n%64==0 or n==len(tasks):
                f.flush(); os.fsync(f.fileno())
                print(f"checkpoint {len(existing)}/{NCONFIG*NSTART}; {time.perf_counter()-started:.1f}s",flush=True)
    by=defaultdict(Counter)
    for row in existing.values(): by[row["config_id"]][row["provisional_category"]]+=1
    ranking=[]
    for c in configs:
        z=by[c["id"]]; n=sum(z.values())
        pplus=z.get("PROVISIONAL_CR_PLUS_MINUS",0)/NSTART
        pminus=z.get("PROVISIONAL_CR_MINUS_PLUS",0)/NSTART
        pco=(z.get("PROVISIONAL_CO_PLUS_PLUS",0)+z.get("PROVISIONAL_CO_MINUS_MINUS",0))/NSTART
        pcr=pplus+pminus; b=min(pplus,pminus)/pcr if pcr else 0.0
        J=pcr-2*pco-max(0.0,0.35-b)
        ranking.append({"screen_index":c["screen_index"],"config_id":c["id"],
            "p":c["p"],"physical":c["physical"],"provisional_counts":dict(z),
            "P_CR":pcr,"P_CO":pco,"balance_b":b,"J_discovery":J,
            "empirical_charge_window":c["empirical_charge_window"],
            "flat_Paschen_screen_under_unity":c["flat_Paschen_screen_under_unity"]})
    ranking.sort(key=lambda x:(x["J_discovery"],x["P_CR"],-x["P_CO"]),reverse=True)
    summary={"study":meta["study"],"rows":len(existing),"expected_rows":NCONFIG*NSTART,
        "provisional_outcomes":dict(Counter(r["provisional_category"] for r in existing.values())),
        "classification":"Short discovery labels only; every retained finalist requires 15,000-cycle verification.",
        "top_50":ranking[:50],"trajectory_sha256":sha(run_path),"meta_sha256":sha(meta_path),
        "wall_seconds":time.perf_counter()-started}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"rows":summary["rows"],"provisional_outcomes":summary["provisional_outcomes"],
        "top_config":ranking[0]["config_id"],"top_config_provisional_counts":ranking[0]["provisional_counts"]},indent=2))


if __name__=="__main__":
    main()
