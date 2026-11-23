"""Re-solve the three M3 zero-spin roots on independently refined grids."""
from __future__ import annotations
import json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/"research"/"v17c"/"v17c_21a"/"code"));sys.path.insert(0,str(ROOT/"research"/"v17c"/"v17c_21"/"code"))
import analytical_model as legacy
import m3_continuation as cont
from m3_harmonic_balance import HBConfig
from m3_system import solve_m3_root

def safe(obj,path="$",bad=None):
 if bad is None:bad=[]
 if isinstance(obj,dict):return {str(k):safe(v,f"{path}.{k}",bad) for k,v in obj.items()}
 if isinstance(obj,(list,tuple)):return [safe(v,f"{path}[{i}]",bad) for i,v in enumerate(obj)]
 if isinstance(obj,np.ndarray):return safe(obj.tolist(),path,bad)
 if isinstance(obj,np.integer):return int(obj)
 if isinstance(obj,(np.floating,float)):
  value=float(obj)
  if not np.isfinite(value):bad.append(path);return None
  return value
 return obj

def write(path,obj):
 bad=[];clean=safe(obj,bad=bad)
 if bad and isinstance(clean,dict):clean["non_finite_diagnostics_replaced_with_null"]=bad
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(clean,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8");tmp.replace(path)

def run(name):
 event=json.loads((BASE/"results"/f"event_omega1_zero_{name}.json").read_text(encoding="utf-8")); aug=event["augmented_event_solve"]
 data=cont.change_field_parameter(legacy.load_inputs(),name,float(aug["parameter_value"]))
 x=np.array([aug["root"]["radii_R"][0],aug["root"]["omega_over_Omega1"][0],aug["root"]["radii_R"][1],aug["root"]["omega_over_Omega1"][1],aug["root"]["mean_z_R"][0],aug["root"]["mean_z_R"][1]])
 seedpath=BASE/"results"/f"event_omega1_zero_{name}_xi.npz"
 with np.load(seedpath) as f: xi=[f["xi1"].copy(),f["xi2"].copy()]
 rows=[]
 outpath=BASE/"results"/f"event_grid_convergence_{name}.json"
 out={"parameter":name,"factor":aug["factor"],"fixed_parameter_value_no_retargeting":aug["parameter_value"],"starting_root_grid":[96,12,12],"grid_results":rows,"status":"IN_PROGRESS","interpretation":"The control parameter is held fixed at the 96x12x12 zero-spin estimate; any refined-grid nonzero omega1 is a discretization shift, not a retuned boundary."}
 for nt,np_ in ((128,12),(192,16)):
  cfg=HBConfig(nt,np_,cutoff=.05,oversample=1,method="newton_krylov",tolerance=1e-7,equation_tolerance=5e-5,max_iter=35,relaxation=.5,history=6)
  t=time.perf_counter()
  try:
   res=solve_m3_root(data,x,cfg,max_nfev=160,scaled_balance_tolerance=2e-6,require_convergence=True,xi_seed=xi)
   xi=[a.copy() for a in res.pop("_xi_grids")];res.pop("_xi_fourier",None)
   row={"grid":[nt,np_,np_],"status":res["status"],"root":res["root"],"scaled_balance_residual":res["scaled_balance_residual"],"scaled_balance_max_norm":res["scaled_balance_max_norm"],"inner":res["inner"],"outer_nfev":res["outer_nfev"],"elapsed_s":time.perf_counter()-t}
   x=np.array([res["root"]["radii_R"][0],res["root"]["omega_over_Omega1"][0],res["root"]["radii_R"][1],res["root"]["omega_over_Omega1"][1],res["root"]["mean_z_R"][0],res["root"]["mean_z_R"][1]])
  except Exception as e: row={"grid":[nt,np_,np_],"status":"NUMERICALLY_UNRESOLVED","failure_type":type(e).__name__,"message":str(e),"elapsed_s":time.perf_counter()-t}
  rows.append(row)
  write(outpath,out)
  print(f"event grid {name} {row['grid']} {row['status']} w={row.get('root',{}).get('omega_over_Omega1')}",flush=True)
 out["status"]="COMPLETE" if all(r["status"]=="VERIFIED_ROOT" for r in rows) else "PARTIAL_UNRESOLVED"
 write(outpath,out)
 return name,out["status"]

def main():
 with ProcessPoolExecutor(max_workers=3) as pool:
  fs=[pool.submit(run,n) for n in ("A","nu","beta")]
  results=[]
  for f in as_completed(fs): results.append(f.result())
 summary={"study":"V17C.21.2 M3 zero-spin event grid convergence","source_commit":"94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8","rows":[json.loads((BASE/"results"/f"event_grid_convergence_{n}.json").read_text(encoding="utf-8")) for n in ("A","nu","beta")],"status":"COMPLETE" if all(s=="COMPLETE" for _,s in results) else "PARTIAL_UNRESOLVED"}
 write(BASE/"results"/"event_grid_convergence.json",summary)
if __name__=="__main__":main()
