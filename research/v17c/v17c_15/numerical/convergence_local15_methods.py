"""Independent long-horizon Local15 method comparison on frozen start S01."""
from __future__ import annotations
import concurrent.futures as cf
import csv, json, math, sys, time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"numerical"))
from local15_exact_validation import acceleration, PARAM, load_panel

PERIODS=15000
OMEGA=2*math.pi
ROOTS={
    "species1":{"rho":0.5503691530067869,"omega":0.001440054163088106},
    "species2":{"rho":0.7663818561742767,"omega":-0.0010579232360122394},
}

def one_case(task):
    species,method,nsteps,initial=task
    alpha=float(PARAM["species"][species]["alpha"])
    lam=float(PARAM["species"][species]["lambda"])
    phase1=np.array([initial["phase1"]]); phase2=np.array([initial["phase2"]])
    y0=np.array([[initial["x0"],initial["y0"],initial["vx0"],initial["vy0"]]],float)
    t0=time.perf_counter()
    if method=="RK4":
        h=OMEGA/nsteps
        sample=np.empty((PERIODS+1,4),float); sample[0]=y0[0]
        y=y0
        for step in range(1,PERIODS*nsteps+1):
            t=(step-1)*h
            y=local_rk4(t,y,h,phase1,phase2,alpha,lam)
            if step%nsteps==0: sample[step//nsteps]=y[0]
    else:
        h=OMEGA/96
        t_eval=OMEGA*np.arange(PERIODS+1,dtype=float)
        def fun(t,y):
            return acceleration(t,y[None,:],phase1,phase2,alpha,lam)[0]
        sol=solve_ivp(fun,(0.0,OMEGA*PERIODS),y0[0],method="DOP853",
            t_eval=t_eval,rtol=2e-11,atol=2e-13,max_step=h)
        if not sol.success: raise RuntimeError(sol.message)
        sample=sol.y.T
        nsteps=96
    rho=np.hypot(sample[:,0],sample[:,1])
    raw=np.arctan2(sample[:,1],sample[:,0])
    theta=np.unwrap(raw)
    sl=slice(PERIODS-500,PERIODS+1)
    omega=(theta[PERIODS]-theta[PERIODS-500])/(OMEGA*500)
    result={"species":species,"method":method,"steps_per_primary_period":nsteps,
        "periods":PERIODS,"n_samples":int(len(sample)),
        "rho_mean_last_500":float(np.mean(rho[sl])),
        "rho_sd_last_500":float(np.std(rho[sl])),
        "omega_mean_last_500":float(omega),"max_rho":float(np.max(rho)),
        "relative_rho_error_vs_DOP853_percent":None,
        "relative_abs_omega_error_vs_DOP853_percent":None,
        "relative_rho_error_vs_leading_percent":float(100*(np.mean(rho[sl])-ROOTS[species]["rho"])/ROOTS[species]["rho"]),
        "relative_abs_omega_error_vs_leading_percent":float(100*(abs(omega)-abs(ROOTS[species]["omega"]))/abs(ROOTS[species]["omega"])),
        "sign":int(np.sign(omega)),"elapsed_seconds":time.perf_counter()-t0}
    return result

def local_rk4(t,y,h,p1,p2,alpha,lam):
    f=lambda tt,yy: acceleration(tt,yy,p1,p2,alpha,lam)
    k1=f(t,y); k2=f(t+h/2,y+h*k1/2)
    k3=f(t+h/2,y+h*k2/2); k4=f(t+h,y+h*k3)
    return y+h*(k1+2*k2+2*k3+k4)/6

def main():
    rows,arr=load_panel()
    index=next(i for i,r in enumerate(rows) if r["start_id"]=="S01")
    initial={k:float(v[index]) for k,v in arr.items()}
    tasks=[]
    for name in ("species1","species2"):
        for n in (64,96,128): tasks.append((name,"RK4",n,initial))
        tasks.append((name,"DOP853",96,initial))
    started=time.perf_counter()
    out=ROOT/"results/local15_exact_rf/numerical_convergence_S01.json"
    results=[]
    if out.exists():
        try:
            old=json.loads(out.read_text(encoding="utf-8"))
            if old.get("start_id")=="S01" and old.get("periods")==PERIODS:
                results=old.get("results",[])
        except (ValueError,OSError):
            results=[]

    def refresh_errors():
        for name in ("species1","species2"):
            ref=next((r for r in results if r["species"]==name and r["method"]=="DOP853"),None)
            if ref is None: continue
            ref["relative_rho_error_vs_DOP853_percent"]=0.0
            ref["relative_abs_omega_error_vs_DOP853_percent"]=0.0
            for r in results:
                if r["species"]!=name or r["method"]!="RK4": continue
                r["relative_rho_error_vs_DOP853_percent"]=float(100*(r["rho_mean_last_500"]-ref["rho_mean_last_500"])/ref["rho_mean_last_500"])
                r["relative_abs_omega_error_vs_DOP853_percent"]=float(100*(abs(r["omega_mean_last_500"])-abs(ref["omega_mean_last_500"]))/abs(ref["omega_mean_last_500"]))

    def save_checkpoint(status):
        refresh_errors()
        results.sort(key=lambda r:(r["species"],r["method"],r["steps_per_primary_period"]))
        doc={"status":status,"panel_seed":15152026,"start_id":"S01",
          "initial_conditions":initial,"parameters":PARAM,"periods":PERIODS,
          "DOP853":{"rtol":2e-11,"atol":2e-13,"max_step_primary_period_fraction":1/96},
          "comparison_window":"last 500 primary RF periods sampled once per primary phase",
          "workers":8,"wall_seconds":time.perf_counter()-started,"results":results}
        out.write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
        if results:
            with (ROOT/"results/local15_exact_rf/numerical_convergence_S01.csv").open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)

    known={(r["species"],r["method"],r["steps_per_primary_period"]) for r in results}
    pending=[task for task in tasks if (task[0],task[1],task[2]) not in known]
    with cf.ProcessPoolExecutor(max_workers=min(8,len(pending))) as pool:
        futures={pool.submit(one_case,task):task for task in pending}
        for future in cf.as_completed(futures):
            task=futures[future]
            r=future.result()
            results.append(r); known.add((task[0],task[1],task[2]))
            save_checkpoint("partial")
            print(json.dumps({"completed":(task[0],task[1],task[2]),
                "rho":r["rho_mean_last_500"],"omega":r["omega_mean_last_500"],
                "seconds":r["elapsed_seconds"],"finished":len(results)}),flush=True)
    save_checkpoint("completed independent frozen-S01 convergence comparison")
    print(json.dumps({"out":str(out),"wall_seconds":time.perf_counter()-started,
        "results":results},indent=2),flush=True)

if __name__=="__main__": main()
