"""Paired ideal-field V13.1 mass pilot with 100/300-cycle continuation."""
from __future__ import annotations
import argparse, csv, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1/ideal_pilot"
MU=np.array([.85,.90,.95,.975,1.,1.025,1.05,1.10,1.15])
ETAS=[-.10,-.05,.05,.10]
PHASES=[0.,np.pi/2]
N=32
ALPHA0=.027000013157894725
BETA0=.175

def wilson(k,n,z=1.96):
    if n==0:return (None,None)
    p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,float(c-h)),min(1.,float(c+h))

def advance(task):
    (mu,phase,model,eta,index,state,hmax)=task
    from three_phase_trap.mass_filter_v13 import acceleration,linear_status
    kappa=0. if model in ("Q","Qopt") else eta*BETA0/mu
    alpha,beta=(ALPHA0/mu,BETA0/mu) if model!="Qopt" else (ALPHA0*30./29.3143/mu,BETA0/mu)
    standing=model=="SO"
    y=np.asarray(state,float); maxr=float(np.hypot(*y[:2])); exit_tau=None
    def rhs(t,z):
        return np.r_[z[2:],acceleration(z[0],z[1],t+phase,kappa,beta,alpha,standing_octupole=standing)]
    def event(t,z): return np.hypot(z[0],z[1])-1.
    event.terminal=True; event.direction=1
    s1=solve_ivp(rhs,(0,2*np.pi*100),y,method="DOP853",rtol=2e-10,atol=2e-12,
                 max_step=2*np.pi/64,events=event)
    if not s1.success: return {"mu":mu,"phase_rad":phase,"model":model,"eta":eta,"index":index,"status":"numerical_error","max_rho_100":np.nan,"exit_periods":np.nan,"linear_status":linear_status(alpha,beta)["status"]}
    maxr=max(maxr,float(np.hypot(s1.y[0],s1.y[1]).max()))
    if len(s1.t_events[0]):
        exit_tau=float(s1.t_events[0][0]); maxr=max(maxr,1.); status="escaped"
    else:
        exit_tau=None; status="retained"
    t300=exit_tau; maxr300=maxr; status300=status
    if exit_tau is None:
        s2=solve_ivp(rhs,(2*np.pi*100,2*np.pi*300),s1.y[:,-1],method="DOP853",
                     rtol=2e-10,atol=2e-12,max_step=2*np.pi/64,events=event)
        if not s2.success: status300="numerical_error"
        else:
            maxr300=max(maxr300,float(np.hypot(s2.y[0],s2.y[1]).max()))
            if len(s2.t_events[0]):
                t300=float(s2.t_events[0][0]); maxr300=max(maxr300,1.); status300="escaped"
    return {"mu":mu,"phase_rad":phase,"phase_deg":phase*180/np.pi,"model":model,"eta":eta,
            "index":index,"alpha":alpha,"beta":beta,"kappa":kappa,
            "linear_status":linear_status(alpha,beta)["status"],
            "status_100":"escaped" if (exit_tau is not None and exit_tau<=2*np.pi*100) else ("numerical_error" if not s1.success else "retained"),
            "exit_periods_100":exit_tau/(2*np.pi) if exit_tau is not None and exit_tau<=2*np.pi*100 else None,
            "max_rho_100":maxr,"status_300":status300,
            "exit_periods_300":t300/(2*np.pi) if t300 is not None else None,
            "max_rho_300":maxr300,"x0":float(state[0]),"y0":float(state[1]),
            "vx0":float(state[2]),"vy0":float(state[3])}

def main():
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=6);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    initials=(qmc.Sobol(4,scramble=True,seed=131).random_base2(5)-.5)*np.array([.5,.5,.08,.08])
    with (OUT/"initial_conditions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["index","x0_rho","y0_rho","vx0_rho_per_tau","vy0_rho_per_tau"])
        w.writerows([[i,*x] for i,x in enumerate(initials)])
    models=[("Q",0.),("Qopt",0.)]+[(m,e) for e in ETAS for m in ("SO","RO")]
    conf={"field":"ideal_leading_harmonic_V13.1","mu":MU.tolist(),"etas":ETAS,"models":[[m,e] for m,e in models],
          "phase_rad":PHASES,"n_common_sobol":N,"sobol_seed":131,"initial_domain":{"x_rho":[-.25,.25],"y_rho":[-.25,.25],"vx_rho_per_tau":[-.04,.04],"vy_rho_per_tau":[-.04,.04]},
          "RF_phase_applied_as":"tau+phase at injection; identical initial states across models/phases",
          "Qopt_drive_V":{"U2":190.0,"Vdc":30.0,"peak":220.0,"selection":"coarse Floquet voltage scan; narrowest stable interval with min nominal trace margin >=0.03"},
          "aperture_rho":1.,"horizons_rf_periods":[100,300],"long_horizon":"continued from state at 100 periods without resetting RF phase",
          "integrator":"DOP853 rtol=2e-10 atol=2e-12 max_step=2pi/64", "not_physical_transmission":True}
    (OUT/"parameters.json").write_text(json.dumps(conf,indent=2)+"\n",encoding="utf-8")
    ck=OUT/"checkpoint.jsonl";done=[]
    if ck.exists(): done=[json.loads(x) for x in ck.read_text(encoding="utf-8").splitlines() if x]
    keys={(r["mu"],r["phase_rad"],r["model"],r["eta"],r["index"]) for r in done}
    tasks=[]
    for mu in MU:
      for phase in PHASES:
       for model,eta in models:
        for i,x in enumerate(initials):
         key=(float(mu),float(phase),model,float(eta),i)
         if key not in keys: tasks.append((*key[:4],i,x,2*np.pi/64))
    # de-duplicate Q across eta by using one Q task per unique mass/phase/initial
    tasks_by_key={}
    for t in tasks: tasks_by_key[(t[0],t[1],t[2],t[3],t[4])]=t
    tasks=list(tasks_by_key.values())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
      futs=[pool.submit(advance,t) for t in tasks]
      with ck.open("a",encoding="utf-8") as f:
       for fut in as_completed(futs):
        row=fut.result();done.append(row);f.write(json.dumps(row)+"\n");f.flush()
    with (OUT/"trajectory_results.csv").open("w",newline="",encoding="utf-8") as f:
      w=csv.DictWriter(f,fieldnames=list(done[0]));w.writeheader();w.writerows(done)
    # Per-mass finite-time fractions and paired differences, only resolved outcomes in denominators.
    rows=[]
    for mu in MU:
     for phase in PHASES:
      for horizon in (100,300):
       col=f"status_{horizon}"
       base=[r for r in done if r["mu"]==mu and r["phase_rad"]==phase and r["model"]=="Q"]
       for model,eta in models:
        rs=[r for r in done if r["mu"]==mu and r["phase_rad"]==phase and r["model"]==model and r["eta"]==eta]
        good=[r for r in rs if r[col] in ("retained","escaped")]
        vals=np.array([r[col]=="retained" for r in good],float)
        qgood=[r for r in base if r[col] in ("retained","escaped")]
        qv=np.array([r[col]=="retained" for r in qgood],float)
        # Common Sobol indices ensure paired binomial differences.
        lookup={r["index"]:(r[col]=="retained") for r in qgood}
        paired=[(r[col]=="retained")-lookup[r["index"]] for r in good if r["index"] in lookup]
        ng=sum(v==1 for v in paired);nl=sum(v==-1 for v in paired)
        pd=float(np.mean(paired)) if paired else None
        se=float(np.std(paired,ddof=1)/np.sqrt(len(paired))) if len(paired)>1 else None
        wlo,whi=wilson(int(vals.sum()),len(vals))
        rows.append(dict(mu=mu,phase_deg=phase*180/np.pi,horizon_rf= horizon,model=model,eta=eta,
                         n_resolved=len(good),n_unresolved=len(rs)-len(good),n_retained=int(vals.sum()),
                         retention=float(vals.mean()) if len(vals) else None,
                         retention_wilson95_low=wlo,retention_wilson95_high=whi,
                         paired_diff_vs_Q=pd,paired_diff_vs_Q_se=se,
                         paired_diff_vs_Q_ci95_low=max(-1.,pd-1.96*se) if pd is not None and se is not None else None,
                         paired_diff_vs_Q_ci95_high=min(1.,pd+1.96*se) if pd is not None and se is not None else None,
                         paired_gains_vs_Q=ng,paired_losses_vs_Q=nl,paired_diff_n=len(paired),
                         mean_exit_periods=float(np.mean([r["exit_periods_300"] for r in good if r[col]=="escaped" and r["exit_periods_300"] is not None])) if any(r[col]=="escaped" and r["exit_periods_300"] is not None for r in good) else None))
    with (OUT/"retention_summary.csv").open("w",newline="",encoding="utf-8") as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    paired_rows=[]
    for mu in MU:
     for phase in PHASES:
      for horizon in (100,300):
       col=f"status_{horizon}"
       for eta in ETAS:
        lookup={r["index"]:(r[col]=="retained") for r in done if r["mu"]==mu and r["phase_rad"]==phase and r["model"]=="SO" and r["eta"]==eta and r[col] in ("retained","escaped")}
        ro=[r for r in done if r["mu"]==mu and r["phase_rad"]==phase and r["model"]=="RO" and r["eta"]==eta and r[col] in ("retained","escaped") and r["index"] in lookup]
        diffs=np.array([(r[col]=="retained")-lookup[r["index"]] for r in ro],float)
        n=len(diffs);d=float(diffs.mean()) if n else None;se=float(diffs.std(ddof=1)/np.sqrt(n)) if n>1 else None
        paired_rows.append(dict(mu=mu,phase_deg=phase*180/np.pi,horizon_rf=horizon,eta=eta,n_paired=n,
           n_RO_gains=int(np.sum(diffs==1)),n_RO_losses=int(np.sum(diffs==-1)),
           retention_diff_RO_minus_SO=d,se=se,
           ci95_low=max(-1.,d-1.96*se) if d is not None and se is not None else None,
           ci95_high=min(1.,d+1.96*se) if d is not None and se is not None else None))
    with (OUT/"paired_RO_minus_SO.csv").open("w",newline="",encoding="utf-8") as f:
      w=csv.DictWriter(f,fieldnames=list(paired_rows[0]));w.writeheader();w.writerows(paired_rows)

if __name__=="__main__": main()
