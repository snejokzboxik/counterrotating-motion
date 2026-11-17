"""Continue 300-RF A9 confirmation states to 1000 RF without phase reset."""
from __future__ import annotations
import json, math, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from three_phase_trap.v15_dynamics import integrate_rk4_batch, dimensionless_angular_momentum
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v15/confirmation_a9_uniform4d"
SRC=OUT/"checkpoint.jsonl"
DST=OUT/"checkpoint_1000_extension.jsonl"

def task(rows,steps):
    rows=sorted(rows,key=lambda r:(r["replica"],r["pair_index"],-r["Lz_sign"]))
    alive=[i for i,r in enumerate(rows) if r["status"]=="retained"]
    result=[]
    if alive:
      initial=np.array([[rows[i][k] for k in ("xend","yend","vxend","vyend")] for i in alive])
      first=rows[alive[0]]
      ext=integrate_rk4_batch(initial,first["alpha"],first["beta"],first["kappa_applied"],first["model"],
        delta=first["delta_rad"],tau0=first["tau0_rad"],horizon_rf=700,steps_per_rf=steps,aperture=1.0,tau_start_cycles=300.0)
    else: ext=None
    extmap={oldi:j for j,oldi in enumerate(alive)}
    for i,r0 in enumerate(rows):
      r=dict(r0); r["horizon_rf"]=1000; r["continuation_from_rf"]=300; r["continuation_to_rf"]=1000
      keyparts=r["key"].split("|"); keyparts[4]="1000"; r["key"]="|".join(keyparts)
      if i in extmap:
        j=extmap[i]
        r["status"]=str(ext["status"][j])
        if r["status"]=="escaped": r["exit_rf"]=float(ext["exit_rf_step_estimate"][j])
        r["rho_max"]=float(max(r["rho_max"],ext["rho_max_stage_sampled"][j]))
        end=ext["final_state"][j]
        r["xend"],r["yend"],r["vxend"],r["vyend"]=map(float,end)
        r["Lz_end"]=float(dimensionless_angular_momentum(end))
        r["torque_integral"]+=float(ext["torque_integral"][j])
      r["Delta_Lz"]=float(r["Lz_end"]-r["Lz0"])
      r["torque_residual"]=float(r["Delta_Lz"]-r["torque_integral"])
      r["extension_protocol"]="continuous 300-to-1000 RF; h=2pi/steps; tau_start_cycles=300"
      result.append(r)
    return result

def main():
    rows=[json.loads(line) for line in SRC.read_text(encoding="utf8").splitlines() if line]
    groups={}
    for r in rows:
      if r["horizon_rf"]==300:
        groups.setdefault((r["delta_rad"],r["tau0_rad"],r["config"],r["steps_per_rf"]),[]).append(r)
    existing={}
    if DST.exists():
      for line in DST.read_text(encoding="utf8").splitlines():
       if line:
        r=json.loads(line); existing[r["key"]]=r
    pending=[(rr,steps) for (_,_,_,steps),rr in groups.items()
      if sum(1 for r in existing.values() if r["delta_rad"]==rr[0]["delta_rad"] and r["tau0_rad"]==rr[0]["tau0_rad"] and r["config"]==rr[0]["config"] and r["steps_per_rf"]==steps)<len(rr)]
    print(f"A9 1000-RF extension groups={len(groups)} pending={len(pending)}",flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool, DST.open("a",encoding="utf8") as f:
      fs=[pool.submit(task,rr,steps) for rr,steps in pending]
      for n,future in enumerate(as_completed(fs),1):
        out=future.result()
        for r in out:
          if r["key"] not in existing:
            f.write(json.dumps(r,separators=(",",":"))+"\n"); existing[r["key"]]=r
        f.flush(); os.fsync(f.fileno()); print(f"extended {n}/{len(pending)} groups rows={len(existing)}",flush=True)
    print("rows",len(existing),"max_abs_residual",max(abs(r["torque_residual"]) for r in existing.values()))

if __name__=="__main__": main()
