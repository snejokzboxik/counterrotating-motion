"""Matched h=128 vs h=256 A9 classification and torque-closure check."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from three_phase_trap.v15_dynamics import integrate_rk4_batch,dimensionless_angular_momentum
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v15/confirmation_a9_uniform4d"
SRC=OUT/"checkpoint.jsonl"
ALPHA,BETA,KAPPA=.00625,.075,-.015
CONFIGS=(("Q","Q",0.),("SO_equal_harmonic","SO",1.),("SO_equal_rms","SO",math.sqrt(2)),("RO_PLUS","RO_PLUS",1.),("RO_MINUS","RO_MINUS",1.))

def main():
    rows=[json.loads(x) for x in SRC.read_text(encoding="utf8").splitlines() if x]
    initrows=[r for r in rows if r["horizon_rf"]==100 and r["delta_rad"]==0 and r["tau0_rad"]==0 and r["config"]=="Q"]
    initrows.sort(key=lambda r:(r["replica"],r["pair_index"],-r["Lz_sign"]))
    initial=np.array([[r[k] for k in ("x0","y0","vx0","vy0")] for r in initrows])
    meta=[(r["replica"],r["seed"],r["pair_index"],r["Lz_sign"]) for r in initrows]
    ref={(r["config"],r["horizon_rf"],r["replica"],r["pair_index"],r["Lz_sign"]):r for r in rows if r["delta_rad"]==0 and r["tau0_rad"]==0}
    outputs=[]; summary=[]
    for cid,model,mult in CONFIGS:
      r1=integrate_rk4_batch(initial,ALPHA,BETA,KAPPA*mult,model,0.,0.,100,256,1.,0.)
      st300=r1["final_state"].copy(); status300=r1["status"].copy(); exits300=r1["exit_rf_step_estimate"].copy(); maxr300=r1["rho_max_stage_sampled"].copy(); tq300=r1["torque_integral"].copy()
      alive=np.flatnonzero(status300=="retained")
      if len(alive):
       r2=integrate_rk4_batch(st300[alive],ALPHA,BETA,KAPPA*mult,model,0.,0.,200,256,1.,100.)
       st300[alive]=r2["final_state"]; status300[alive]=r2["status"]; exits300[alive]=r2["exit_rf_step_estimate"]
       maxr300[alive]=np.maximum(maxr300[alive],r2["rho_max_stage_sampled"]); tq300[alive]+=r2["torque_integral"]
      for horizon,state,status,exits,maxr,ell,tq in (
        (100,r1["final_state"],r1["status"],r1["exit_rf_step_estimate"],r1["rho_max_stage_sampled"],r1["ell_end"],r1["torque_integral"]),
        (300,st300,status300,exits300,maxr300,dimensionless_angular_momentum(st300),tq300)):
       for i,(rep,seed,pair,sign) in enumerate(meta):
        baseline=ref[(cid,horizon,rep,pair,sign)]
        r={"config":cid,"model":model,"delta_rad":0.,"tau0_rad":0.,"horizon_rf":horizon,"steps_per_rf":256,
           "replica":rep,"seed":seed,"pair_index":pair,"Lz_sign":sign,"status":str(status[i]),
           "status_h128":baseline["status"],"status_match_h128":str(status[i])==baseline["status"],
           "exit_rf":None if not np.isfinite(exits[i]) else float(exits[i]),"rho_max":float(maxr[i]),
           "Lz0":float(dimensionless_angular_momentum(initial[i])),"Lz_end":float(ell[i]),
           "Delta_Lz":float(ell[i]-dimensionless_angular_momentum(initial[i])),
           "torque_integral":float(tq[i]),"torque_residual":float((ell[i]-dimensionless_angular_momentum(initial[i]))-tq[i])}
        outputs.append(r)
      for horizon in (100,300):
       for rep in range(8):
        group=[r for r in outputs if r["config"]==cid and r["horizon_rf"]==horizon and r["replica"]==rep]
        plus=[r for r in group if r["Lz_sign"]>0]; minus=[r for r in group if r["Lz_sign"]<0]
        pp=np.mean([r["status"]=="retained" for r in plus]); pm=np.mean([r["status"]=="retained" for r in minus])
        summary.append({"config":cid,"horizon_rf":horizon,"replica":rep,"P_plus":pp,"P_minus":pm,"P_total":(pp+pm)/2,
          "delta_chi":pp-pm,"Delta_Lz_mean":np.mean([r["Delta_Lz"] for r in group]),
          "n_status_mismatch_vs_h128":sum(not r["status_match_h128"] for r in group),"max_abs_torque_residual":max(abs(r["torque_residual"]) for r in group)})
    for name,data in (("h256_individual.csv",outputs),("h256_summary_by_scramble.csv",summary)):
      with (OUT/name).open("w",newline="",encoding="utf8") as f:
       w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print("h256 rows",len(outputs),"mismatches",sum(not r["status_match_h128"] for r in outputs),"max torque residual",max(abs(r["torque_residual"]) for r in outputs))

if __name__=="__main__":main()
