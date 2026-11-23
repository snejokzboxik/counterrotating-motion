"""Finite-difference stationary M3 balance Jacobian at zero-spin events."""
import json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/"research"/"v17c"/"v17c_21a"/"code"));sys.path.insert(0,str(ROOT/"research"/"v17c"/"v17c_21"/"code"))
import analytical_model as legacy
import m3_continuation as cont
from m3_harmonic_balance import HBConfig
from m3_system import M3PairSystem
SCALE=np.array([.55,.0011,.75,.001,.33,.86])
STEPS=np.array([2e-4,2e-6,2e-4,2e-6,2e-4,2e-4])
def run(name):
 event=json.loads((BASE/"results"/f"event_omega1_zero_{name}.json").read_text(encoding="utf-8")); aug=event["augmented_event_solve"]
 data=cont.change_field_parameter(legacy.load_inputs(),name,float(aug["parameter_value"]))
 root=aug["root"];x=np.array([root["radii_R"][0],root["omega_over_Omega1"][0],root["radii_R"][1],root["omega_over_Omega1"][1],root["mean_z_R"][0],root["mean_z_R"][1]])
 with np.load(BASE/"results"/f"event_omega1_zero_{name}_xi.npz") as f: xi=[f["xi1"].copy(),f["xi2"].copy()]
 cfg=HBConfig(96,12,cutoff=.05,oversample=1,method="newton_krylov",tolerance=1e-7,equation_tolerance=5e-5,max_iter=30,relaxation=.5,history=6)
 variants=[]
 for factor in ((1.,.5) if name=="A" else (1.,)):
  system=M3PairSystem(data,cfg,xi_seed=xi);steps=STEPS*factor;J=np.zeros((6,6));f0,_=system.balance(x)
  for j,h in enumerate(steps):
   xp=x.copy();xm=x.copy();xp[j]+=h;xm[j]-=h;fp,_=system.balance(xp);fm,_=system.balance(xm);J[:,j]=(fp-fm)/(2*h)*SCALE[j]
  sv=np.linalg.svd(J,compute_uv=False)
  variants.append({"fd_step_factor":factor,"jacobian_scaled":J.tolist(),"singular_values":sv.tolist(),"sigma_min":float(sv[-1]),"condition_number":float(sv[0]/sv[-1]),"balance_at_root_inf":float(np.max(np.abs(f0))),"balance_evaluations":system.evaluations})
  print(f"stationary J {name} h={factor:g} sigma_min={sv[-1]:.6g}",flush=True)
 return {"parameter":name,"factor":aug["factor"],"parameter_value":aug["parameter_value"],"root":root,"grid":[96,12,12],"variants":variants,"status":"COMPLETE"}
def main():
 rows=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  fs=[pool.submit(run,n) for n in ("A","nu","beta")]
  for f in as_completed(fs): rows.append(f.result())
 rows.sort(key=lambda x:("A","nu","beta").index(x["parameter"]))
 out={"study":"V17C.21.2 stationary M3 Jacobian at zero-spin boundaries","source_commit":"94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8","purpose":"Distinguish simple zero-spin events from possible singular stationary folds; this is not the 10D slow-flow stability spectrum.","rows":rows,"status":"COMPLETE"}
 (BASE/"results"/"event_stationary_jacobian.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
if __name__=="__main__":main()
