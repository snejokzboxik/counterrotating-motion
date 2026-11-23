"""Small multi-seed M3 equilibrium check at the nominal parameter vector."""
import json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/"research"/"v17c"/"v17c_21a"/"code"));sys.path.insert(0,str(ROOT/"research"/"v17c"/"v17c_21"/"code"))
import analytical_model as legacy
from m3_harmonic_balance import HBConfig
from m3_system import solve_m3_root

def safe_json(obj):
 if isinstance(obj,dict):return {str(k):safe_json(v) for k,v in obj.items()}
 if isinstance(obj,(list,tuple)):return [safe_json(v) for v in obj]
 if isinstance(obj,np.ndarray):return safe_json(obj.tolist())
 if isinstance(obj,np.integer):return int(obj)
 if isinstance(obj,(np.floating,float)):
  v=float(obj)
  return v if np.isfinite(v) else None
 return obj

def run(label,seed):
 data=legacy.load_inputs()
 with np.load(ROOT/"research"/"v17c"/"v17c_21a"/"results"/"xi_seed_192_16.npz") as f:xi=[f["xi1"].copy(),f["xi2"].copy()]
 cfg=HBConfig(96,12,cutoff=.05,oversample=1,method="newton_krylov",tolerance=1e-7,equation_tolerance=5e-5,max_iter=30,relaxation=.5,history=6)
 t=time.perf_counter()
 try:
  row=solve_m3_root(data,seed,cfg,max_nfev=160,scaled_balance_tolerance=2e-6,require_convergence=True,xi_seed=xi)
  row.pop("_xi_grids",None);row.pop("_xi_fourier",None)
  return {"seed_class":label,"status":row["status"],"root":row["root"],"scaled_balance_max_norm":row["scaled_balance_max_norm"],"inner":row["inner"],"outer_nfev":row["outer_nfev"],"elapsed_s":time.perf_counter()-t}
 except Exception as e:return {"seed_class":label,"status":"NUMERICALLY_UNRESOLVED","failure_type":type(e).__name__,"message":str(e),"elapsed_s":time.perf_counter()-t}

def main():
 d=legacy.load_inputs(); p=ROOT/"research"/"v17c"/"v17c_21a"/"results"/"full_grid_campaign.json";j=json.loads(p.read_text(encoding="utf-8"));base=next(x for x in j["stages"] if x.get("grid")==[192,16,16] and x.get("status")=="VERIFIED_ROOT")["root"]
 r,w,z=base["radii_R"],base["omega_over_Omega1"],base["mean_z_R"]
 seeds={"seed_CR_minus_plus":np.array([r[0],-w[0],r[1],-w[1],z[0],z[1]]),"seed_CO_plus_plus":np.array([r[0],abs(w[0]),r[1],abs(w[1]),z[0],z[1]]),"seed_CO_minus_minus":np.array([r[0],-abs(w[0]),r[1],-abs(w[1]),z[0],z[1]])}
 rows=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  fs=[pool.submit(run,k,v) for k,v in seeds.items()]
  for f in as_completed(fs):rows.append(f.result())
 out={"study":"V17C.21.2 alternative nominal M3 roots","source_commit":"94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8","grid":[96,12,12],"initial_seeds":{k:v.tolist() for k,v in seeds.items()},"rows":rows,"interpretation":"Finite multi-seed corrector outcomes only; convergence to the same root does not prove uniqueness, and distinct M3 equilibria are not by themselves distinct RF attractors.","status":"COMPLETE"}
 (BASE/"results"/"alternative_roots.json").write_text(json.dumps(safe_json(out),indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
if __name__=="__main__":main()
