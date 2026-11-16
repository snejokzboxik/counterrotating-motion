"""Checkpointed fixed-voltage mass-ray retention pilot for V13.3."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
V13_1 = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1"
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap.mass_filter_v13_3 import acceleration

PHASES = (0., math.pi/2)
RTOL, ATOL = 2e-10, 2e-12
MAX_STEP = 2*math.pi/64
N_RF = 100


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def wilson(k, n):
    if not n:
        return math.nan, math.nan
    z=1.959963984540054; p=k/n; den=1+z*z/n
    c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,c-h),min(1.,c+h)


def task_key(r):
    return "|".join((str(r["ray_id"]),r["model"],str(r["mu"]),str(r["phase_i"]),str(r["state_i"]),str(r["N_RF"])))


def integrate(task):
    point, model, mu, phase_i, phase, state_i, initial = task
    alpha0,beta0,kappa0=map(float,(point["alpha"],point["beta"],point["kappa"]))
    alpha,beta=alpha0/mu,beta0/mu
    kappa=kappa0/mu if model=="RO" else 0.
    x0,y0,vx0,vy0=map(float,initial)
    def rhs(t,z):
        ax,ay=acceleration(z[0],z[1],t,kappa,beta,alpha,delta=0.,tau0=phase)
        return np.array([z[2],z[3],ax,ay])
    def event(t,z):return math.hypot(z[0],z[1])-1.
    event.terminal=True;event.direction=1.
    row={"ray_id":point["candidate_id"],"base_alpha":alpha0,"base_beta":beta0,
         "base_kappa":kappa0,"eta":kappa0/beta0,"mu":mu,"M_over_Q_Th_per_e":3200.04652261404*mu,
         "alpha":alpha,"beta":beta,"kappa":kappa,"model":model,"phase_i":phase_i,
         "tau0_rad":phase,"tau0_deg":math.degrees(phase),"state_i":state_i,
         "x0_rho":x0,"y0_rho":y0,"vx0_rho_tau":vx0,"vy0_rho_tau":vy0,
         "N_RF":N_RF,"aperture_rho":1.,"integrator":"DOP853","rtol":RTOL,
         "atol":ATOL,"max_step_tau":MAX_STEP,
         "protocol":"fixed dimensionless transverse start ensemble; finite-time transverse retention only"}
    try:
        sol=solve_ivp(rhs,(0.,2*math.pi*N_RF),(x0,y0,vx0,vy0),method="DOP853",
                      rtol=RTOL,atol=ATOL,max_step=MAX_STEP,events=event,dense_output=True)
        if not sol.success:status="unresolved";exitp=math.nan
        elif len(sol.t_events[0]):status="escaped";exitp=float(sol.t_events[0][0]/(2*math.pi))
        else:status="retained";exitp=math.nan
        rmax=float(np.max(np.hypot(sol.y[0],sol.y[1])))
        if len(sol.t)>1:
            mid=sol.sol((sol.t[:-1]+sol.t[1:])/2)
            rmax=max(rmax,float(np.max(np.hypot(mid[0],mid[1]))))
        if status=="escaped":rmax=max(1.,rmax)
        row.update({"status":status,"exit_periods":exitp,"rho_max":rmax,
                    "final_state":json.dumps([float(v) for v in sol.y[:,-1]],separators=(",",":")),
                    "nfev":int(sol.nfev)})
    except Exception as exc:
        row.update({"status":"unresolved","exit_periods":math.nan,"rho_max":math.nan,
                    "final_state":"","nfev":math.nan,"error":f"{type(exc).__name__}:{exc}"})
    row["key"]=task_key(row)
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--mu-min",type=float,default=.80)
    p.add_argument("--mu-max",type=float,default=1.20)
    p.add_argument("--mu-step",type=float,default=.025)
    p.add_argument("--states",type=int,default=8)
    args=p.parse_args()
    if not (1<=args.workers<=8):p.error("workers must be 1..8")
    points=read_csv(BASE/"fem_control_points.csv")
    initials=read_csv(V13_1/"ideal_pilot"/"initial_conditions.csv")[:args.states]
    starts=[[float(r[k]) for k in ("x0_rho","y0_rho","vx0_rho_per_tau","vy0_rho_per_tau")] for r in initials]
    n=int(round((args.mu_max-args.mu_min)/args.mu_step))+1
    mus=[round(args.mu_min+i*args.mu_step,12) for i in range(n)]
    initial_path=V13_1/"ideal_pilot"/"initial_conditions.csv"
    voltage_summary={r["candidate_id"]:float(r["max_electrode_peak_abs_V"])
                     for r in read_csv(BASE/"fem_controls/fem_control_summary.csv") if r["model"]=="Q" and r["horizon_RF"]=="100"}
    atlas=read_csv(BASE/"atlas_trajectory_results.csv")
    completed={}
    ck=BASE/"mass_ray_checkpoint.jsonl"
    if ck.exists():
        for line in ck.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r=json.loads(line)
                if r["key"] in completed:raise RuntimeError(f"duplicate mass-ray checkpoint key {r['key']}")
                completed[r["key"]]=r
    tasks=[];reused=0
    for point in points:
      for mu in mus:
       for phase_i,phase in enumerate(PHASES):
        for model in ("Q","RO"):
         k=0. if model=="Q" else float(point["kappa"])
         for state_i,state in enumerate(starts):
          task={"ray_id":point["candidate_id"],"model":model,"mu":mu,"phase_i":phase_i,
                "state_i":state_i,"N_RF":N_RF}
          key=task_key(task)
          if key in completed:continue
          if abs(mu-1.)<1e-12:
              row=next((r for r in atlas if r["model"]==model and int(r["state_i"])==state_i
                        and int(r["phase_i"])==phase_i and abs(float(r["alpha"])-float(point["alpha"]))<1e-12
                        and abs(float(r["beta"])-float(point["beta"]))<1e-12
                        and abs(float(r["kappa"])-k)<1e-12),None)
              if row is not None:
                  out={"ray_id":point["candidate_id"],"base_alpha":float(point["alpha"]),
                       "base_beta":float(point["beta"]),"base_kappa":float(point["kappa"]),
                       "eta":float(point["kappa"])/float(point["beta"]),"mu":mu,
                       "M_over_Q_Th_per_e":3200.04652261404*mu,"alpha":float(point["alpha"]),
                       "beta":float(point["beta"]),"kappa":k,"model":model,"phase_i":phase_i,
                       "tau0_rad":phase,"tau0_deg":math.degrees(phase),"state_i":state_i,
                       "x0_rho":float(state[0]),"y0_rho":float(state[1]),"vx0_rho_tau":float(state[2]),"vy0_rho_tau":float(state[3]),
                       "N_RF":N_RF,"aperture_rho":1.,"integrator":row["integrator"],"rtol":float(row["rtol"]),
                       "atol":float(row["atol"]),"max_step_tau":float(row["max_step_tau"]),
                       "protocol":"reused exact V13.3 atlas trajectory at mu=1; same seed-131 start",
                       "status":row["status"],"exit_periods":float(row["exit_periods"]),"rho_max":float(row["rho_max"]),
                       "final_state":row["final_state"],"nfev":int(row["nfev"]),"source":"atlas_checkpoint exact node"}
              out["key"]=key;completed[key]=out;reused+=1
              continue
          tasks.append((point,model,mu,phase_i,phase,state_i,state))
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest={"script_sha256":script_sha,"fixed_voltage_rays":points,"mu_grid":mus,"phases_rad":PHASES,
              "initial_conditions_file":"outputs/octupole_comparison_home/mass_filter_v13/v13_1/ideal_pilot/initial_conditions.csv",
              "initial_conditions_sha256":hashlib.sha256(initial_path.read_bytes()).hexdigest(),
              "initial_state_count":len(starts),"horizon_RF":N_RF,"aperture_rho":1.,"delta_rad":0.,
              "physical_constants":{"Q_C":1.602176634e-19,"M0_Th":3200.04652261404,"R_m":.008,"f_Hz":100000.},
              "FEM_basis_coefficients":{"c2":.7718255558274212,"d2":.7718255558274212,"c4":.5413395699439091},
              "fixed_voltage_max_electrode_peak_V":voltage_summary,
              "parameter_scaling":"alpha,beta,kappa all divide by mu; Q always sets kappa=0",
              "solver":{"method":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP},
              "unresolved_policy":"preserved separately","interpretation":"finite-time transverse retention, not finite-length mass-filter transmission"}
    (BASE/"mass_ray_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    if tasks:
      n_done=0
      with ProcessPoolExecutor(max_workers=args.workers) as pool,ck.open("a",encoding="utf-8") as stream:
        futures=[pool.submit(integrate,t) for t in tasks]
        for future in as_completed(futures):
          row=future.result()
          if row["key"] in completed:raise RuntimeError(f"duplicate mass-ray key {row['key']}")
          stream.write(json.dumps(row,separators=(",",":"))+"\n")
          completed[row["key"]]=row;n_done+=1
          if n_done%20==0:stream.flush();os.fsync(stream.fileno())
        stream.flush();os.fsync(stream.fileno())
    rows=list(completed.values())
    write_csv(BASE/"mass_ray_trajectory_results.csv",rows)
    print(json.dumps({"points":len(points),"mu_count":len(mus),"tasks_new":len(tasks),"reused_atlas_rows":reused,
                      "total_rows":len(rows),"checkpoint":str(ck)},indent=2))


if __name__=="__main__":main()
