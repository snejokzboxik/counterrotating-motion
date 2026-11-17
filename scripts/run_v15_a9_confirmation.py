"""Independent, paired A9 confirmation with continuous 100 -> 300 RF extension."""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v15/confirmation_a9_uniform4d"
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT))
from three_phase_trap.v15_dynamics import integrate_rk4_batch, dimensionless_angular_momentum
from scripts.run_v15_chiral_pilot import (POINTS, E_CHARGE, M0_KG, R_M, OMEGA, C2, D2, C4,
                                           field_voltage_peaks)

SEEDS=(616161,717171,818181,919191,12344321,56788765,80801234,24681357)
ALPHA,BETA,KAPPA=POINTS["A9"]
M0=(ALPHA,BETA,KAPPA)
PHASES=(0.0,math.pi/2)
DELTAS=(0.0,math.pi/2)
CONFIGS=(
    ("Q","Q",0.0),
    ("SO_equal_harmonic","SO",1.0),
    ("SO_equal_rms","SO",math.sqrt(2.0)),
    ("RO_PLUS","RO_PLUS",1.0),
    ("RO_MINUS","RO_MINUS",1.0),
)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def starts_for_seed(seed,n):
    u=qmc.Sobol(4,scramble=True,seed=seed).random_base2(int(math.log2(n)))
    radius=.0075*np.sqrt(u[:,0])/R_M
    theta=2*np.pi*u[:,1]
    energy=5*u[:,2]
    v=np.sqrt(2*energy*E_CHARGE/M0_KG)/(R_M*OMEGA)
    # Conditional Lz>0 half of the uniform velocity disk: the angle relative
    # to the outward radial direction is uniform on (0, pi). Mirror gives the
    # matching Lz<0 half with exactly the same radial/tangential speeds.
    phi=np.pi*u[:,3]
    vr=v*np.cos(phi); vt=v*np.sin(phi)
    plus=np.column_stack((radius*np.cos(theta),radius*np.sin(theta),
      vr*np.cos(theta)-vt*np.sin(theta),vr*np.sin(theta)+vt*np.cos(theta)))
    minus=plus.copy(); minus[:,1]*=-1; minus[:,3]*=-1
    return np.stack((plus,minus),axis=1)

def build_tasks(args):
    tasks=[]
    for delta in DELTAS:
      for tau0 in PHASES:
       for cid,model,mult in CONFIGS:
        peaks=field_voltage_peaks(ALPHA,BETA,KAPPA*mult,cid,delta)
        if peaks.max()>250: continue
        tasks.append((delta,tau0,cid,model,mult,args.pairs_per_replica,args.steps_per_rf))
    return tasks

def run_task(task):
    delta,tau0,cid,model,mult,npairs,steps=task
    starts=[]; meta=[]
    for rep,seed in enumerate(SEEDS):
      pairs=starts_for_seed(seed,npairs)
      for i in range(npairs):
       for signidx in (0,1):
        starts.append(pairs[i,signidx]); meta.append((rep,seed,i,1 if signidx==0 else -1))
    initial=np.asarray(starts)
    kap=KAPPA*mult
    first=integrate_rk4_batch(initial,ALPHA,BETA,kap,model,delta,tau0,100,steps,1.0,0.0)
    state300=first["final_state"].copy(); status300=first["status"].copy()
    exit300=first["exit_rf_step_estimate"].copy(); rmax300=first["rho_max_stage_sampled"].copy()
    torque300=first["torque_integral"].copy()
    alive=np.flatnonzero(first["status"]=="retained")
    if len(alive):
      second=integrate_rk4_batch(state300[alive],ALPHA,BETA,kap,model,delta,tau0,200,steps,1.0,100.0)
      state300[alive]=second["final_state"]
      status300[alive]=second["status"]
      e=second["exit_rf_step_estimate"]
      exit300[alive]=e
      rmax300[alive]=np.maximum(rmax300[alive],second["rho_max_stage_sampled"])
      torque300[alive]+=second["torque_integral"]
    ell0=dimensionless_angular_momentum(initial)
    ell100=first["ell_end"]
    ell300=dimensionless_angular_momentum(state300)
    resid100=(ell100-ell0)-first["torque_integral"]
    resid300=(ell300-ell0)-torque300
    volt=float(field_voltage_peaks(ALPHA,BETA,kap,cid,delta).max())
    out=[]
    for n_horizon,state,status,exit_rf,rmax,ell,torque,resid in (
      (100,first["final_state"],first["status"],first["exit_rf_step_estimate"],first["rho_max_stage_sampled"],ell100,first["torque_integral"],resid100),
      (300,state300,status300,exit300,rmax300,ell300,torque300,resid300)):
      for idx,(rep,seed,pair,sign) in enumerate(meta):
       out.append({"key":f"{delta:.9f}|{tau0:.9f}|{cid}|{steps}|{n_horizon}|{rep}|{pair}|{sign}",
         "delta_rad":delta,"tau0_rad":tau0,"config":cid,"model":model,"replica":rep,"seed":seed,
         "pair_index":pair,"Lz_sign":sign,"horizon_rf":n_horizon,"steps_per_rf":steps,
         "alpha":ALPHA,"beta":BETA,"kappa_applied":kap,"mu":1.0,"peak_electrode_V":volt,
         "x0":float(initial[idx,0]),"y0":float(initial[idx,1]),"vx0":float(initial[idx,2]),"vy0":float(initial[idx,3]),
         "rho0":float(np.linalg.norm(initial[idx,:2])),"Eperp0_eV":float(.5*M0_KG*(R_M*OMEGA)**2*np.sum(initial[idx,2:]**2)/E_CHARGE),
         "status":str(status[idx]),"exit_rf":None if not np.isfinite(exit_rf[idx]) else float(exit_rf[idx]),
         "rho_max":float(rmax[idx]),"Lz0":float(ell0[idx]),"Lz_end":float(ell[idx]),
         "Delta_Lz":float(ell[idx]-ell0[idx]),"torque_integral":float(torque[idx]),"torque_residual":float(resid[idx]),
         "xend":float(state[idx,0]),"yend":float(state[idx,1]),"vxend":float(state[idx,2]),"vyend":float(state[idx,3])})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs-per-replica",type=int,default=128)
    p.add_argument("--steps-per-rf",type=int,default=128)
    p.add_argument("--workers",type=int,default=6)
    a=p.parse_args()
    if a.pairs_per_replica<16 or 2**round(math.log2(a.pairs_per_replica))!=a.pairs_per_replica:
      raise ValueError("pairs-per-replica must be a power of two >= 16")
    OUT.mkdir(parents=True,exist_ok=True)
    modelpath=ROOT/"src/three_phase_trap/v15_dynamics.py"
    manifest={"study":"independent V15 A9 confirmation","candidate":"A9","alpha":ALPHA,"beta":BETA,"kappa":KAPPA,
      "mu":1,"physical_domain":"uniform four-dimensional domain: r0<=7.5 mm by area; velocity disk Eperp<=5 eV by area; each Lz sign has one half-disk; exact mirrored pairs match r,E,vr,|vt|",
      "sobol_seeds":SEEDS,"pairs_per_replica":a.pairs_per_replica,"independent_pairs_per_group":len(SEEDS)*a.pairs_per_replica,
      "tau0_rad":PHASES,"delta_rad":DELTAS,"models":CONFIGS,"horizons_rf":[100,300],"steps_per_rf":a.steps_per_rf,
      "method":"fixed-step RK4; 300 RF obtained by continuous extension of states surviving 100 RF, tau_start_cycles=100",
      "hardware_limit_V":250,"source_hashes":{"script":sha(Path(__file__)),"field_model":sha(modelpath)}}
    manifest["protocol_hash"]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=OUT/"manifest.json"; cp=OUT/"checkpoint.jsonl"
    if mp.exists():
      old=json.loads(mp.read_text(encoding="utf8"))
      if old!=manifest: raise RuntimeError("confirmation manifest mismatch; use a new output directory for protocol changes")
    else: mp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    done={}
    if cp.exists():
      for line in cp.read_text(encoding="utf8").splitlines():
       if line:
        r=json.loads(line)
        if r["protocol_hash"]!=manifest["protocol_hash"]: raise RuntimeError("checkpoint protocol hash mismatch")
        if r["key"] in done: raise RuntimeError(f"duplicate checkpoint {r['key']}")
        done[r["key"]]=r
    tasks=build_tasks(a); pending=[]
    for t in tasks:
      delta,tau0,cid,model,mult,npairs,steps=t
      expected=2*len(SEEDS)*npairs*2
      found=sum(1 for r in done.values() if r["delta_rad"]==delta and r["tau0_rad"]==tau0 and r["config"]==cid and r["steps_per_rf"]==steps)
      if found < expected: pending.append(t)
      elif found > expected: raise RuntimeError("checkpoint row count exceeds expected task size")
    print(f"A9 confirmation groups={len(tasks)} pending={len(pending)} pairs/group={len(SEEDS)*a.pairs_per_replica} workers={a.workers}",flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool, cp.open("a",encoding="utf8") as stream:
      futures={pool.submit(run_task,t):t for t in pending}
      for i,f in enumerate(as_completed(futures),1):
       rows=f.result()
       for r in rows:
        r["protocol_hash"]=manifest["protocol_hash"]
        if r["key"] in done: raise RuntimeError(f"duplicate write {r['key']}")
        stream.write(json.dumps(r,separators=(",",":"))+"\n"); done[r["key"]]=r
       stream.flush(); os.fsync(stream.fileno())
       print(f"finished {i}/{len(pending)} groups rows={len(done)}",flush=True)
    print(f"done rows={len(done)} max torque residual={max(abs(r['torque_residual']) for r in done.values()):.3e}")

if __name__=="__main__": main()
