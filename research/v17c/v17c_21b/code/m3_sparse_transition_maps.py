"""Sparse continuation maps concentrated on the observed 1-D sign transitions."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; BASE=HERE.parent; ROOT=HERE.parents[3]
V21A=ROOT/"research"/"v17c"/"v17c_21a"/"code"
sys.path.insert(0,str(HERE));sys.path.insert(0,str(V21A));sys.path.insert(0,str(ROOT/"research"/"v17c"/"v17c_21"/"code"))
import analytical_model as legacy
import m3_continuation as one_d
from m3_harmonic_balance import HBConfig
from m3_system import solve_m3_root

MAPS={
 "A_B":{"x":"A","y":"B","x_factors":[0.90,0.95,1.00],"y_factors":[0.70,1.00,1.30]},
 "A_beta":{"x":"A","y":"beta","x_factors":[0.90,0.95,1.00],"y_factors":[1.00,1.20,1.30]},
 "B_nu":{"x":"B","y":"nu","x_factors":[0.80,1.00,1.20],"y_factors":[0.70,0.80,1.00]},
}

def save(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8");tmp.replace(path)

def xvec(row):
 r,w,z=row["root"]["radii_R"],row["root"]["omega_over_Omega1"],row["root"]["mean_z_R"]
 return np.array([r[0],w[0],r[1],w[1],z[0],z[1]])

def run_row(map_id,yf):
 spec=MAPS[map_id];base,root,xi=one_d._baseline();nx=spec["x_factors"];xnom=one_d._parameter_value(base,spec["x"]);ynom=one_d._parameter_value(base,spec["y"])
 cfg=HBConfig(96,12,cutoff=.05,oversample=1,method="newton_krylov",tolerance=1e-7,equation_tolerance=5e-5,max_iter=30,relaxation=.5,history=6)
 path=BASE/"results"/f"map_{map_id}_y{yf:.3f}.json"
 if path.exists():
  old=json.loads(path.read_text(encoding="utf-8"))
  if old.get("status")=="ROW_COMPLETE":
   return map_id,yf,"ROW_ALREADY_COMPLETE"
 out={"study":"V17C.21.2 sparse transition-focused M3 map","source_commit":"94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8","map_id":map_id,"x_parameter":spec["x"],"y_parameter":spec["y"],"y_factor":yf,"y_value":ynom*yf,"x_factors":nx,"fixed_particle_parameters":True,"grid":[96,12,12],"cutoff":.05,"method":"newton_krylov","points":[],"status":"IN_PROGRESS","scope":"Sparse branch-tracked local equilibria near selected 1-D sign transitions. No global completeness or capture-probability claim."}
 data_y=one_d.change_field_parameter(base,spec["y"],ynom*yf)
 # A matching 1-D continuation root is the correct row anchor.  This avoids
 # asking a distant nominal guess to rediscover the same branch at y != 1.
 one_d_payload=json.loads((BASE/"results"/f"continuation_{spec['y']}.json").read_text(encoding="utf-8"))
 if abs(yf-1.0)<1e-12:
  yrow=one_d_payload["baseline"]
 else:
  yrow=next(row for seq in one_d_payload["directions"].values() for row in seq
            if row.get("status")=="VERIFIED_ROOT" and abs(row["factor_from_nominal"]-yf)<1e-10)
 anchor=xvec(yrow)
 # Solve x=1 as row anchor, then continue in each x direction.
 ordered=[1.0]+sorted([x for x in nx if x>1])+sorted([x for x in nx if x<1],reverse=True)
 prev=anchor.copy();xip=[q.copy() for q in xi]
 last_factor=1.0
 for xf in ordered:
  if last_factor>1.0 and xf<1.0:
   prev=anchor.copy();xip=[q.copy() for q in xi]
  last_factor=xf
  if abs(yf-1.0)<1e-12:
   xp=json.loads((BASE/"results"/f"continuation_{spec['x']}.json").read_text(encoding="utf-8"))
   xrow=xp["baseline"] if abs(xf-1.0)<1e-12 else next(row for seq in xp["directions"].values() for row in seq if row.get("status")=="VERIFIED_ROOT" and abs(row["factor_from_nominal"]-xf)<1e-10)
   row={"x_factor":xf,"x_value":xnom*xf,"status":xrow["status"],"classification":xrow["classification"],"root":xrow["root"],"scaled_balance_residual":xrow["scaled_balance_residual"],"scaled_balance_max_norm":xrow["scaled_balance_max_norm"],"outer_nfev":xrow["outer_nfev"],"balance_evaluations":xrow["balance_evaluations"],"wall_time_s":0.0,"geometry":xrow.get("geometry"),"seed_source":f"continuation_{spec['x']} factor {xf:.6f}","inner":xrow.get("inner",[])}
   out["points"].append(row);save(path,out);continue
  if abs(xf-1.0)<1e-12:
   row={"x_factor":xf,"x_value":xnom,"status":"VERIFIED_ROOT","classification":yrow["classification"],"root":yrow["root"],"scaled_balance_residual":yrow["scaled_balance_residual"],"scaled_balance_max_norm":yrow["scaled_balance_max_norm"],"outer_nfev":yrow["outer_nfev"],"balance_evaluations":yrow["balance_evaluations"],"wall_time_s":0.0,"geometry":yrow.get("geometry"),"seed_source":f"continuation_{spec['y']} factor {yf:.6f}","inner":yrow.get("inner",[])}
   out["points"].append(row);save(path,out);continue
  data=one_d.change_field_parameter(data_y,spec["x"],xnom*xf); t=time.perf_counter()
  try:
   res=solve_m3_root(data,prev,cfg,max_nfev=140,scaled_balance_tolerance=2e-6,require_convergence=True,xi_seed=xip)
   xigrid=[a.copy() for a in res.pop("_xi_grids")];res.pop("_xi_fourier",None);label,geom=one_d._classify(res,data)
   row={"x_factor":xf,"x_value":xnom*xf,"status":res["status"],"classification":label,"root":res["root"],"scaled_balance_residual":res["scaled_balance_residual"],"scaled_balance_max_norm":res["scaled_balance_max_norm"],"outer_nfev":res["outer_nfev"],"balance_evaluations":res["balance_evaluations"],"wall_time_s":time.perf_counter()-t,"geometry":geom,"inner":[{k:v for k,v in q.items() if k in {"status","converged","relative_nonlinear_residual","rms_unresolved_slow_force","rms_unresolved_fast_force","minimum_fast_mode_denominator","number_of_excluded_rf_combination_modes"}} for q in res["inner"]]}
   prev=np.array([res["root"]["radii_R"][0],res["root"]["omega_over_Omega1"][0],res["root"]["radii_R"][1],res["root"]["omega_over_Omega1"][1],res["root"]["mean_z_R"][0],res["root"]["mean_z_R"][1]]);xip=xigrid
  except Exception as e:
   row={"x_factor":xf,"x_value":xnom*xf,"status":"NUMERICALLY_UNRESOLVED","failure_type":type(e).__name__,"message":str(e),"wall_time_s":time.perf_counter()-t}
  out["points"].append(row);save(path,out)
  print(f"map {map_id} y={yf:.3f} x={xf:.3f} {row['status']} {row.get('classification','')}",flush=True)
 out["status"]="ROW_COMPLETE" if all(p["status"]=="VERIFIED_ROOT" for p in out["points"]) else "ROW_PARTIAL_UNRESOLVED"
 save(path,out);return map_id,yf,out["status"]

def main():
 jobs=[(mid,y) for mid,s in MAPS.items() for y in s["y_factors"]]
 with ProcessPoolExecutor(max_workers=min(4,os.cpu_count() or 1)) as pool:
  fs={pool.submit(run_row,mid,y):(mid,y) for mid,y in jobs}
  for f in as_completed(fs):
   mid,y=fs[f]
   try:print(f"complete {f.result()}",flush=True)
   except Exception as e:print(f"failed {mid} {y}: {type(e).__name__}: {e}",flush=True)
 for mid,s in MAPS.items():
  rows=[]
  for y in s["y_factors"]:
   p=BASE/"results"/f"map_{mid}_y{y:.3f}.json"
   if p.exists():rows.append(json.loads(p.read_text(encoding="utf-8")))
  out={"study":"V17C.21.2 sparse transition-focused M3 map","source_commit":"94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8","map_id":mid,"x_parameter":s["x"],"y_parameter":s["y"],"x_factors":s["x_factors"],"y_factors":s["y_factors"],"rows":rows,"status":"COMPLETE" if len(rows)==len(s["y_factors"]) else "PARTIAL","interpretation":"Discrete branch-tracked mesh only; stability spectra are reported separately at selected roots; no interpolation through unresolved cells."}
  save(BASE/"results"/f"sparse_map_{mid}.json",out)

if __name__=="__main__":main()
