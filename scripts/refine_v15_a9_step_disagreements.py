"""High-accuracy DOP853 refinement for A9 h128/h256 status disagreements."""
from __future__ import annotations
import csv,json,math,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from three_phase_trap.v15_dynamics import rhs
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v15/confirmation_a9_uniform4d"
ALPHA,BETA,KAPPA=.00625,.075,-.015

def trajectory(row):
    model=row["model"]; delta=float(row["delta_rad"]);tau0=float(row["tau0_rad"])
    mult=math.sqrt(2) if row["config"]=="SO_equal_rms" else 1.0
    kap=KAPPA*mult
    y0=np.array([float(row[k]) for k in ("x0","y0","vx0","vy0")])
    def fun(t,y):return rhs(t,y,ALPHA,BETA,kap,model,delta,tau0)
    def event(t,y):return math.hypot(y[0],y[1])-1
    event.terminal=True;event.direction=1
    max_step=2*math.pi/256
    a=solve_ivp(fun,(0,2*math.pi*100),y0,method="DOP853",rtol=2e-11,atol=2e-13,max_step=max_step,events=event,dense_output=True)
    if not a.success:status100="unresolved";exit_rf=float('nan');state100=a.y[:,-1]
    elif len(a.t_events[0]):status100="escaped";exit_rf=float(a.t_events[0][0]/(2*math.pi));state100=a.y[:,-1]
    else:status100="retained";exit_rf=float('nan');state100=a.y[:,-1]
    if status100=="retained":
      b=solve_ivp(fun,(2*math.pi*100,2*math.pi*300),state100,method="DOP853",rtol=2e-11,atol=2e-13,max_step=max_step,events=event,dense_output=True)
      if not b.success:status300="unresolved"
      elif len(b.t_events[0]):status300="escaped";exit_rf=float(b.t_events[0][0]/(2*math.pi))
      else:status300="retained"
    else: status300=status100
    return {"config":row["config"],"replica":row["replica"],"pair_index":row["pair_index"],"Lz_sign":row["Lz_sign"],
      "status_h128_100":row["status_h128_100"],"status_h256_100":row["status_h256_100"],
      "status_dop853_100":status100,"status_h128_300":row["status_h128_300"],"status_h256_300":row["status_h256_300"],
      "status_dop853_300":status300,"exit_rf_dop853":None if not np.isfinite(exit_rf) else exit_rf,
      "rtol":2e-11,"atol":2e-13,"max_step_tau":max_step,"method":"DOP853; 0-100 then continuous 100-300 RF"}

def main():
    ref=list(csv.DictReader((OUT/"h256_individual.csv").open(newline="",encoding="utf8")))
    bad=[r for r in ref if r["status_match_h128"]=="False"]
    ids={(r["config"],r["replica"],r["pair_index"],r["Lz_sign"]) for r in bad}
    base=[json.loads(x) for x in (OUT/"checkpoint.jsonl").read_text(encoding="utf8").splitlines() if x]
    lo={}
    for r in base:
      if r["delta_rad"]==0 and r["tau0_rad"]==0 and r["horizon_rf"]==100:
        lo[(r["config"],str(r["replica"]),str(r["pair_index"]),str(r["Lz_sign"]))]=r
    tasks=[]
    for c,rep,pair,sign in sorted(ids):
      q=lo[(c,rep,pair,sign)]
      q=dict(q)
      for h in (100,300):
        z=next(r for r in ref if r["config"]==c and r["replica"]==rep and r["pair_index"]==pair and r["Lz_sign"]==sign and int(r["horizon_rf"])==h)
        q[f"status_h128_{h}"]=z["status_h128"];q[f"status_h256_{h}"]=z["status"]
      tasks.append(q)
    out=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
      for r in as_completed([pool.submit(trajectory,x) for x in tasks]):out.append(r.result())
    out.sort(key=lambda r:(r['config'],int(r['replica']),int(r['pair_index']),-int(r['Lz_sign'])))
    with (OUT/"dop853_refinement_disagreements.csv").open('w',newline='',encoding='utf8') as f:
      w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print(json.dumps({"n_disputed_trajectories":len(out),
      "h128_h256_dop853_100_counts":{},
      "h128_h256_dop853_300_counts":{}},indent=2))
    for h in (100,300):
      print('H',h)
      for col in (f'status_h128_{h}',f'status_h256_{h}',f'status_dop853_{h}'):
       from collections import Counter
       print(col,dict(Counter(r[col] for r in out)))
if __name__=="__main__":main()
