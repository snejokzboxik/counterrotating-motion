"""Coarse nonlinear Floquet-like (alpha,beta) maps for V13.2."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability"
V13_1 = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1"
ALPHAS = np.linspace(.018, .038, 15)
BETAS = np.linspace(.145, .200, 19)
ETAS = (-.05, .05)
PHASES = (0.0, math.pi/2)
N_INITIAL = 8
ALPHA0 = .027000013157894725
BETA0 = .175
R, FREQUENCY = .008, 100e3
OMEGA = 2*math.pi*FREQUENCY
E_CHARGE = 1.602176634e-19
M0 = 3200.0465226 * 1.66053906660e-27
C4, C2 = .5413395699439091, .7718255558274
RTOL, ATOL = 2e-10, 2e-12
MAX_STEP = 2*math.pi/64


def task_key(row: dict) -> tuple:
    return (round(float(row["alpha"]),9),round(float(row["beta"]),9),round(float(row["phase_rad"]),12),
            row["model"],round(float(row["eta"]),8),int(row["index"]))


def advance(task: tuple) -> dict:
    alpha,beta,phase,model,eta,index,state=task
    from three_phase_trap.mass_filter_v13 import acceleration
    kappa=0. if model=="Q" else eta*beta
    standing=model=="SO"
    y0=np.asarray(state,float)
    def rhs(t,y):
        return np.r_[y[2:],acceleration(y[0],y[1],t+phase,kappa,beta,alpha,standing_octupole=standing)]
    def aperture(t,y): return float(np.hypot(y[0],y[1])-1.)
    aperture.terminal=True;aperture.direction=1
    try:
        sol=solve_ivp(rhs,(0,2*math.pi*100),y0,method="DOP853",rtol=RTOL,atol=ATOL,
                      max_step=MAX_STEP,events=aperture,dense_output=True)
        if not sol.success: status="numerical_error";exitp=None
        elif len(sol.t_events[0]): status="escaped";exitp=float(sol.t_events[0][0]/(2*math.pi))
        else: status="retained";exitp=None
        radius=np.hypot(sol.y[0],sol.y[1]);rmax=float(np.max(radius))
        if len(sol.t)>1:
            mid=sol.sol((sol.t[:-1]+sol.t[1:])/2)
            rmax=float(max(rmax,np.max(np.hypot(mid[0],mid[1]))))
        if status=="escaped":rmax=max(1.,rmax)
        return {"alpha":float(alpha),"beta":float(beta),"kappa":float(kappa),"eta":float(eta),
                "phase_rad":float(phase),"phase_deg":float(np.degrees(phase)),"model":model,"index":int(index),
                "status_100":status,"exit_periods_100":exitp,"max_rho_100":rmax,
                "state_100":[float(v) for v in sol.y[:,-1]],"x0":float(y0[0]),"y0":float(y0[1]),
                "vx0":float(y0[2]),"vy0":float(y0[3]),"nfev":int(sol.nfev),
                "integrator":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP}
    except Exception as exc:
        return {"alpha":float(alpha),"beta":float(beta),"kappa":float(kappa),"eta":float(eta),
                "phase_rad":float(phase),"phase_deg":float(np.degrees(phase)),"model":model,"index":int(index),
                "status_100":"numerical_error","exit_periods_100":None,"max_rho_100":float("nan"),
                "state_100":None,"x0":float(y0[0]),"y0":float(y0[1]),"vx0":float(y0[2]),"vy0":float(y0[3]),
                "nfev":None,"integrator":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP,
                "error":f"{type(exc).__name__}:{exc}"}


def initials() -> np.ndarray:
    data=np.genfromtxt(V13_1/"ideal_pilot"/"initial_conditions.csv",delimiter=",",names=True)
    return np.stack([data[n] for n in ("x0_rho","y0_rho","vx0_rho_per_tau","vy0_rho_per_tau")],axis=1)[:N_INITIAL]


def wilson(k:int,n:int,z:float=1.959963984540054)->tuple[float,float]:
    if n==0:return (float("nan"),float("nan"))
    p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,c-h),min(1.,c+h)


def write_csv(path:Path,rows:list[dict])->None:
    if not rows:return
    fields=[]
    for row in rows:fields.extend(k for k in row if k not in fields)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,separators=(",",":")) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def load_electrodes()->list[dict]:
    with (V13_1/"electrode_voltage_peaks.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    return [{k:float(r[k]) for k in ("theta_deg","C4","S4","Q2")} for r in rows if r["scheme"]=="Q"]


def electrode_peak(alpha:float,beta:float,model:str,eta:float,electrodes:list[dict])->tuple[float,float,float]:
    u2=190.*beta/BETA0
    vdc=29.3143*alpha/ALPHA0
    u4=0. if model=="Q" else eta*u2*C2/C4
    peaks=[]
    for e in electrodes:
        rf_cos=u2*e["Q2"]+u4*e["C4"]
        rf_sin=0. if model=="SO" else u4*e["S4"]
        dc=vdc*e["Q2"]
        peaks.append(abs(dc)+math.hypot(rf_cos,rf_sin))
    return u2,vdc,u4,max(peaks)


def floquet(alpha:float,beta:float)->dict:
    from three_phase_trap.mass_filter_v13 import linear_status
    return linear_status(alpha,beta,tolerance=1e-8,rtol=2e-11,atol=2e-13)


def find_boundary(beta:float,sign:int)->dict:
    from three_phase_trap.mass_filter_v13 import monodromy
    def margin(mag:float)->float:
        alpha=sign*mag
        tx=float(np.trace(monodromy(alpha,beta,"x",rtol=2e-11,atol=2e-13)))
        ty=float(np.trace(monodromy(alpha,beta,"y",rtol=2e-11,atol=2e-13)))
        return min(2-abs(tx),2-abs(ty))
    if margin(0)<=0:
        return {"beta":beta,"side":sign,"alpha_boundary":float("nan"),"status":"no_strict_core_at_alpha0"}
    bracket=None;last=0.;lastv=margin(0)
    for mag in np.arange(.002,.2001,.002):
        val=margin(float(mag))
        if lastv>0 and val<=0:
            bracket=(last,float(mag));break
        last,lastv=float(mag),val
    if bracket is None:
        return {"beta":beta,"side":sign,"alpha_boundary":float("nan"),"status":"no_crossing_in_scan"}
    root=brentq(margin,*bracket,xtol=1e-12,rtol=1e-12)
    alpha=sign*root
    mx=monodromy(alpha,beta,"x",rtol=2e-11,atol=2e-13)
    my=monodromy(alpha,beta,"y",rtol=2e-11,atol=2e-13)
    tx,ty=float(np.trace(mx)),float(np.trace(my))
    axis="x" if 2-abs(tx)<=2-abs(ty) else "y"
    tr=tx if axis=="x" else ty
    return {"beta":beta,"side":sign,"alpha_boundary":alpha,"status":"root_found","active_axis":axis,
            "active_trace":tr,"nearest_trace_limit":2. if tr>=0 else -2.,
            "det_x":float(np.linalg.det(mx)),"det_y":float(np.linalg.det(my))}


def summary_rows(rows:list[dict],physical:dict,floquet_map:dict)->list[dict]:
    out=[]
    for alpha in ALPHAS:
      for beta in BETAS:
       for phase in PHASES:
        group=[r for r in rows if abs(float(r["alpha"])-alpha)<1e-10 and abs(float(r["beta"])-beta)<1e-10
               and abs(float(r["phase_rad"])-phase)<1e-10]
        by={(r["model"],round(float(r["eta"]),8),int(r["index"])):r for r in group}
        for model,eta in [("Q",0.),("SO",-.05),("RO",-.05),("SO",.05),("RO",.05)]:
         rr=[r for r in group if r["model"]==model and round(float(r["eta"]),8)==eta]
         resolved=[r for r in rr if r["status_100"] in ("retained","escaped")]
         k=sum(r["status_100"]=="retained" for r in resolved);lo,hi=wilson(k,len(resolved))
         q={int(r["index"]):r for r in group if r["model"]=="Q"}
         dq=[int(r["status_100"]=="retained")-int(q[int(r["index"])]["status_100"]=="retained")
             for r in resolved if int(r["index"]) in q and q[int(r["index"])]["status_100"] in ("retained","escaped")]
         pd=float(np.mean(dq)) if dq else float("nan");se=float(np.std(dq,ddof=1)/math.sqrt(len(dq))) if len(dq)>1 else float("nan")
         state=floquet_map[(round(float(alpha),10),round(float(beta),10))]
         voltage=physical[(round(float(alpha),10),round(float(beta),10),model,round(float(eta),8))]
         row={"alpha":alpha,"beta":beta,"phase_deg":float(np.degrees(phase)),"model":model,"eta":eta,
              "n_total":len(rr),"n_resolved":len(resolved),"n_unresolved":len(rr)-len(resolved),"n_retained":k,
              "P_retention":k/len(resolved) if resolved else float("nan"),"Wilson95_low":lo,"Wilson95_high":hi,
              "retention_bounds_low_allN":k/len(rr) if rr else float("nan"),
              "retention_bounds_high_allN":(k+len(rr)-len(resolved))/len(rr) if rr else float("nan"),
              "paired_n_vs_Q":len(dq),"paired_diff_vs_Q":pd,
              "paired_diff_vs_Q_ci95_low":max(-1,pd-1.96*se) if np.isfinite(se) else float("nan"),
              "paired_diff_vs_Q_ci95_high":min(1,pd+1.96*se) if np.isfinite(se) else float("nan"),
              "paired_gains_vs_Q":int(np.sum(np.asarray(dq)==1)),"paired_losses_vs_Q":int(np.sum(np.asarray(dq)==-1)),
              "median_exit_periods":float(np.median([float(r["exit_periods_100"]) for r in rr if r["status_100"]=="escaped"]))
                    if any(r["status_100"]=="escaped" for r in rr) else float("nan"),
              "median_max_rho":float(np.median([float(r["max_rho_100"]) for r in rr if np.isfinite(float(r["max_rho_100"]))]))
                    if rr else float("nan"),
              "floquet_status":state["status"],"trace_x":state["trace_x"],"trace_y":state["trace_y"],
              "margin_x":state["margin_x"],"margin_y":state["margin_y"],
              "U2_V":voltage["U2_V"],"VDC_V":voltage["VDC_V"],"U4_V":voltage["U4_V"],
              "max_electrode_peak_V":voltage["max_electrode_peak_V"],"within_250V":voltage["within_250V"]}
         if model.startswith("RO") or model=="RO":
          so={int(r["index"]):r for r in group if r["model"]=="SO" and round(float(r["eta"]),8)==eta}
          ds=[int(r["status_100"]=="retained")-int(so[int(r["index"])]["status_100"]=="retained")
              for r in rr if int(r["index"]) in so and r["status_100"] in ("retained","escaped")
              and so[int(r["index"])]["status_100"] in ("retained","escaped")]
          d=np.asarray(ds,float);dmu=float(d.mean()) if len(d) else float("nan")
          dse=float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else float("nan")
          row.update({"paired_n_RO_minus_SO":len(d),"paired_diff_RO_minus_SO":dmu,
                      "paired_diff_RO_minus_SO_ci95_low":max(-1,dmu-1.96*dse) if np.isfinite(dse) else float("nan"),
                      "paired_diff_RO_minus_SO_ci95_high":min(1,dmu+1.96*dse) if np.isfinite(dse) else float("nan"),
                      "RO_SO_gains":int(np.sum(d==1)),"RO_SO_losses":int(np.sum(d==-1))})
         out.append(row)
    return out


def boundary_overlay(ax, boundaries:list[dict])->None:
    for side in (-1,1):
        rr=sorted([r for r in boundaries if r["side"]==side and r["status"]=="root_found"],key=lambda r:r["beta"])
        if rr:ax.plot([r["beta"] for r in rr],[r["alpha_boundary"] for r in rr],"k--",lw=1.2,zorder=4)


def make_maps(summary:list[dict],boundaries:list[dict],phase:float,eta:float)->None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig,axes=plt.subplots(1,5,figsize=(17,4.4),sharex=True,sharey=True)
    fields=[("Q",0.,"Q retention"),("SO",eta,"Q + SO retention"),("RO",eta,"Q + RO retention")]
    lookup={(round(float(r["alpha"]),10),round(float(r["beta"]),10),r["model"],round(float(r["eta"]),8)):r
            for r in summary if abs(float(r["phase_deg"])-np.degrees(phase))<1e-8}
    ref=[]
    for a in ALPHAS:
      for b in BETAS:
       ref.append((a,b,lookup[(round(float(a),10),round(float(b),10),"Q",0.)]))
    for ax,(model,modeta,title) in zip(axes[:3],fields):
        pts=[(a,b,lookup[(round(float(a),10),round(float(b),10),model,round(float(modeta),8))]) for a,b,_ in ref]
        sc=ax.scatter([p[1] for p in pts],[p[0] for p in pts],c=[p[2]["P_retention"] for p in pts],
                      cmap="viridis",vmin=0,vmax=1,marker="s",s=54,linewidths=0,zorder=2)
        bad=[p for p in pts if not bool(p[2]["within_250V"])]
        if bad:ax.scatter([p[1] for p in bad],[p[0] for p in bad],marker="x",s=16,c="#e11d48",lw=.8,zorder=3)
        fig.colorbar(sc,ax=ax,pad=.02,fraction=.046,label="P(N=8)")
        ax.set_title(title,fontsize=9)
    # Paired probability differences use common initial-state indices.
    for ax,which,title in zip(axes[3:],("Q","SO"),("P_RO − P_Q","P_RO − P_SO")):
        z=[]
        for a,b,qrow in ref:
            ro=lookup[(round(float(a),10),round(float(b),10),"RO",round(float(eta),8))]
            target=qrow if which=="Q" else lookup[(round(float(a),10),round(float(b),10),"SO",round(float(eta),8))]
            z.append((a,b,ro,target))
        values=[r[2]["paired_diff_vs_Q"] if which=="Q" else r[2]["paired_diff_RO_minus_SO"] for r in z]
        sc=ax.scatter([r[1] for r in z],[r[0] for r in z],c=values,cmap="coolwarm",vmin=-1,vmax=1,
                      marker="s",s=54,linewidths=0,zorder=2)
        for a,b,ro,target in z:
            if not bool(ro["within_250V"]) or not bool(target["within_250V"]):
                ax.scatter([b],[a],marker="x",s=16,c="#e11d48",lw=.8,zorder=3)
        fig.colorbar(sc,ax=ax,pad=.02,fraction=.046,label="парная ΔP")
        ax.set_title(title,fontsize=9)
    # Fixed-voltage M/Q path, evaluated for the requested eta at the nominal mass.
    mus=np.linspace(.97,1.16,70)
    for ax in axes:
        boundary_overlay(ax,boundaries)
        ax.plot(BETA0/mus,ALPHA0/mus,color="white",lw=2.8,zorder=5)
        ax.plot(BETA0/mus,ALPHA0/mus,color="#111827",lw=1.15,ls=":",zorder=6)
        ax.set_xlabel("β")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("α")
    axes[0].set_xlim(BETAS.min()-.002,BETAS.max()+.002)
    axes[0].set_ylim(ALPHAS.min()-.001,ALPHAS.max()+.001)
    handles=[Line2D([0],[0],color="k",ls="--",lw=1.2,label="Floquet граница Q"),
             Line2D([0],[0],color="#111827",ls=":",lw=1.2,label="линия μ при фиксированных напряжениях"),
             Line2D([0],[0],color="#e11d48",marker="x",ls="",label="пик >250 В")]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,-.02),fontsize=8)
    fig.suptitle(f"Конечновременное удержание: η={eta:+.2f}, τ₀={np.degrees(phase):.0f}°, 100 RF периодов; 15×19 точек, N=8",y=.985)
    fig.tight_layout(rect=(0,.07,1,.93))
    name=f"stage_b_map_eta_{eta:+.2f}_phase_{np.degrees(phase):.0f}deg"
    for ext in ("png","pdf","svg"):fig.savefig(OUT/f"{name}.{ext}",dpi=220 if ext=="png" else None)
    plt.close(fig)


def make_exit_maps(summary:list[dict],phase:float,eta:float)->None:
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.0),sharex=True,sharey=True)
    for ax,(model,modeta) in zip(axes,(("Q",0.),("SO",eta),("RO",eta))):
        rr=[r for r in summary if r["model"]==model and round(float(r["eta"]),8)==round(float(modeta),8)
            and abs(float(r["phase_deg"])-np.degrees(phase))<1e-8 and np.isfinite(float(r["median_exit_periods"]))]
        if rr:
            sc=ax.scatter([float(r["beta"]) for r in rr],[float(r["alpha"]) for r in rr],
                          c=[float(r["median_exit_periods"]) for r in rr],cmap="magma_r",vmin=0,vmax=100,
                          marker="s",s=62,linewidths=0)
            fig.colorbar(sc,ax=ax,pad=.02,fraction=.046,label="RF-периоды")
        ax.set_title(f"{model}: медиана выхода",fontsize=9);ax.set_xlabel("β");ax.grid(alpha=.2)
    axes[0].set_ylabel("α")
    fig.suptitle(f"Медианное время первого выхода, только точки с выходом: η={eta:+.2f}, τ₀={np.degrees(phase):.0f}°")
    fig.tight_layout()
    name=f"stage_b_exit_time_eta_{eta:+.2f}_phase_{np.degrees(phase):.0f}deg"
    for ext in ("png","pdf","svg"):fig.savefig(OUT/f"{name}.{ext}",dpi=220 if ext=="png" else None)
    plt.close(fig)


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=8);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    states=initials()
    with (OUT/"stage_b_initial_conditions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["index","x0_rho","y0_rho","vx0_rho_per_tau","vy0_rho_per_tau"])
        w.writerows([[i,*x] for i,x in enumerate(states)])
    tasks=[]
    for alpha in ALPHAS:
      for beta in BETAS:
       for phase in PHASES:
        for model,eta in (("Q",0.),("SO",-.05),("RO",-.05),("SO",.05),("RO",.05)):
         for i,state in enumerate(states):tasks.append((float(alpha),float(beta),float(phase),model,eta,i,state.tolist()))
    ck=OUT/"stage_b_checkpoint.jsonl";done=[json.loads(x) for x in ck.read_text(encoding="utf-8").splitlines() if x.strip()] if ck.exists() else []
    completed={task_key(r) for r in done};pending=[t for t in tasks if (round(t[0],9),round(t[1],9),round(t[2],12),t[3],round(t[4],8),t[5]) not in completed]
    manifest={"stage":"V13.2-B-nonlinear-map","alpha_grid":ALPHAS.tolist(),"beta_grid":BETAS.tolist(),
        "etas":[0.,-.05,.05],"models":"Q; SO and RO for eta +/-0.05; eta=0 is identical Q",
        "initial_conditions":"first 8 of V13.1 seed131 scrambled Sobol sample, common across every map point/model/phase",
        "n_common":N_INITIAL,"phases_rad":list(PHASES),"aperture_rho":1.,"horizon_rf_periods":100,
        "integrator":{"method":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP},
        "mass_ray":"(alpha,beta,kappa)=(alpha0,beta0,eta*beta0)/mu at fixed voltages; mu=.97..1.16",
        "voltage_limit_V":250,"geometry_coefficients":{"c4":C4,"c2":C2,"d2":C2,"R_m":R,"f_Hz":FREQUENCY,"M0_Th":3200.0465226},
        "interpretation":"finite-time 2D transverse retention, not asymptotic stability or finite-length transmission",
        "unknown_outcomes":"kept unresolved and bounded; not counted as escaped","workers":args.workers}
    (OUT/"stage_b_parameters.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Stage B: {len(pending)} trajectories remaining of {len(tasks)}; workers={args.workers}",flush=True)
    if pending:
      with ProcessPoolExecutor(max_workers=args.workers) as pool,ck.open("a",encoding="utf-8") as f:
        futures=[pool.submit(advance,t) for t in pending]
        for n,fu in enumerate(as_completed(futures),1):
          row=fu.result();done.append(row);f.write(json.dumps(row,separators=(",",":"))+"\n");f.flush()
          if n%500==0 or n==len(futures):print(f"Stage B completed {n}/{len(futures)}",flush=True)
    raw=sorted(done,key=lambda r:(float(r["alpha"]),float(r["beta"]),float(r["phase_rad"]),r["model"],float(r["eta"]),int(r["index"])))
    write_csv(OUT/"stage_b_trajectory_results.csv",raw)

    # Compute strict centre Floquet status on every displayed parameter point.
    floquet_map={};fr=[]
    for alpha in ALPHAS:
      for beta in BETAS:
        result=floquet(float(alpha),float(beta));floquet_map[(round(float(alpha),10),round(float(beta),10))]=result
        fr.append({"alpha":alpha,"beta":beta,**result})
    write_csv(OUT/"stage_b_floquet_grid.csv",fr)
    boundaries=[find_boundary(float(beta),side) for beta in np.linspace(BETAS.min(),BETAS.max(),45) for side in (-1,1)]
    write_csv(OUT/"stage_b_floquet_boundaries.csv",boundaries)

    electrodes=load_electrodes();physical={};vrows=[]
    for alpha in ALPHAS:
      for beta in BETAS:
       for model,eta in (("Q",0.),("SO",-.05),("RO",-.05),("SO",.05),("RO",.05)):
        u2,vdc,u4,peak=electrode_peak(float(alpha),float(beta),model,eta,electrodes)
        entry={"U2_V":u2,"VDC_V":vdc,"U4_V":u4,"max_electrode_peak_V":peak,"within_250V":peak<=250.0+1e-9}
        physical[(round(float(alpha),10),round(float(beta),10),model,round(float(eta),8))]=entry
        vrows.append({"alpha":alpha,"beta":beta,"model":model,"eta":eta,**entry})
    write_csv(OUT/"stage_b_voltage_feasibility.csv",vrows)
    summary=summary_rows(raw,physical,floquet_map)
    write_csv(OUT/"stage_b_nonlinear_map_summary.csv",summary)
    for eta in ETAS:
      for phase in PHASES:make_maps(summary,boundaries,phase,eta)
      make_exit_maps(summary,0.,eta);make_exit_maps(summary,math.pi/2,eta)


if __name__=="__main__":main()
