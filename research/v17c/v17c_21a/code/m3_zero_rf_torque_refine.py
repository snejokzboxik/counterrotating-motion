"""Refine zero mean RF-torque brackets in a fixed omega=0 M3 slice."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE.parents[0]/"v17c_21"/"code"))
sys.path.insert(0,str(BASE/"code"))
import analytical_model as model
from m3_harmonic_balance import HBConfig,solve_harmonic_balance

def main():
 d=model.load_inputs()
 cfg=HBConfig(ntheta=96,nphase=12,cutoff=.05,oversample=1,method="newton_krylov",
              tolerance=1e-7,equation_tolerance=5e-5,max_iter=30)
 A,B,nu=d["exact_p"][:3]
 output={"study":"V17C.21.1 refined M3 zero-RF-torque slice",
  "scope":"Single-particle diagnostic with omega=0 and Coulomb off. This is not the zero of total torque, a full orbital branch, or a stability boundary.",
  "grid":[96,12,12],"cutoff":.05,"leading_estimate_rcr_over_R":float((A**2*nu**3/(12*B**2))**.25),"particles":[]}
 for i in range(2):
  cache=[]
  def evaluate(r,seed=None):
   hb=solve_harmonic_balance(r,0.0,d["alpha"][i],d["lambda"][i],d["Re_per_speed"][i],
       d["beta"],d["modes"],cfg,xi0=seed,require_convergence=True)
   row={"r_R":float(r),"omega_over_Omega1":0.0,"mean_RF_torque":hb["mean_total_RF_torque"],
        "mean_drag_torque":hb["mean_total_drag_torque"],
        "relative_fast_residual":hb["relative_original_fast_equation_residual"],
        "inner_map_residual":hb["relative_nonlinear_residual"],"micromotion_rms":hb["rms_micromotion"]}
   cache.append((float(r),hb["xi_grid"],row))
   return row,hb["xi_grid"]
  try:
   lo,xi_lo=evaluate(.64)
   hi,xi_hi=evaluate(.70,cache[-1][1])
   if lo["mean_RF_torque"]*hi["mean_RF_torque"]>=0:
    output["particles"].append({"particle":i+1,"status":"NO_BRACKET_AT_INITIAL_INTERVAL",
      "samples":[lo,hi],"zero_torque_bracket_R":None,"iterations":0})
    continue
   history=[lo,hi]
   for iteration in range(1,9):
    # Bisection keeps the bracket valid even when the torque curve is nonlinear.
    midr=.5*(lo["r_R"]+hi["r_R"])
    seed=xi_lo if midr-lo["r_R"] <= hi["r_R"]-midr else xi_hi
    mid,xi_mid=evaluate(midr,seed)
    history.append(mid)
    if lo["mean_RF_torque"]*mid["mean_RF_torque"]<=0:
     hi,xi_hi=mid,xi_mid
    else:
     lo,xi_lo=mid,xi_mid
    if hi["r_R"]-lo["r_R"]<2e-4: break
   output["particles"].append({"particle":i+1,"status":"BRACKET_REFINED",
      "samples":history,"zero_torque_bracket_R":[lo["r_R"],hi["r_R"]],
      "linear_interpolant_r_R":float(lo["r_R"]-lo["mean_RF_torque"]*(hi["r_R"]-lo["r_R"])/(hi["mean_RF_torque"]-lo["mean_RF_torque"])),
      "iterations":iteration})
  except Exception as exc:
   output["particles"].append({"particle":i+1,"status":"UNRESOLVED_INNER_HB",
      "successful_samples":[x[2] for x in cache],"error":str(exc),"iterations_completed":len(cache)})
 output["parameterization"]={"alpha":d["alpha"],"lambda":d["lambda"],"Re_per_speed":d["Re_per_speed"],
  "beta":d["beta"],"A":A,"B":B,"nu":nu,"mode_count":len(d["modes"])}
 path=BASE/"results"/"m3_zero_rf_torque_refine.json"
 path.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 print(json.dumps(output,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
