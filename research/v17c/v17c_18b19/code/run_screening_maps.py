"""Deterministic preliminary stability maps for the leading-harmonic model.

Screen only: the maps use the 2-D k=+2/-4 field, ideal quadrupole DC, no
finite-length correction, and RK4.  They do not certify an attractor or a
physical segmented-electrode device.  Outputs are exact grid points; no
interpolation or smoothing is used.
"""
from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from model import Params, rk4_single_fast, rk4_pair_fast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
FIG = ROOT / "figures"
P = Params()
PERIODS = 1000
STEPS_PER_PERIOD = 64
MAX_WORKERS = min(10, max(1, (os.cpu_count() or 2)-1))
CLASS = {0:"UNRESOLVED", 1:"ROT_POS_SCREEN", 2:"ROT_NEG_SCREEN",
         3:"BOUND_NONROTATING", 4:"ELECTRODE_CONTACT", 5:"NUMERICAL_FAILURE"}
PAIR_CLASS = {0:"UNRESOLVED", 1:"CR_PLUS_MINUS_SCREEN", 2:"CR_MINUS_PLUS_SCREEN",
              3:"CO_PLUS_PLUS_SCREEN", 4:"CO_MINUS_MINUS_SCREEN",
              5:"BOUND_NONROTATING", 6:"ONE_LOST", 7:"BOTH_LOST",
              8:"COLLISION", 9:"ELECTRODE_CONTACT", 10:"NUMERICAL_FAILURE"}


def _classify(turns_late: float, retained: bool, enough_turns: float = 0.10) -> int:
    if not retained:
        return 4
    if abs(turns_late) < enough_turns:
        return 3 if abs(turns_late) < 0.10 else 0
    return 1 if turns_late > 0 else 2


def _one(payload: tuple) -> dict:
    alpha, lam, A, B, nu, beta, pol, label = payload
    s = (0.55*math.cos(0.71), 0.55*math.sin(0.71), 0.0, 0.0)
    nsp = max(STEPS_PER_PERIOD, int(math.ceil(STEPS_PER_PERIOD*max(1.0,nu)/P.nu)))
    dt = 2.0*math.pi/nsp
    theta = 0.0
    th_by_period = [0.0]
    r_by_period = []
    max_r = 0.0
    min_r = float("inf")
    alive = True
    early_status = 0
    for k in range(PERIODS*nsp):
        try:
            sn = rk4_single_fast(s, k*dt, dt,
                                 (alpha,lam,A,B,nu,beta,pol,0.0,0.0))
        except (FloatingPointError, OverflowError, ValueError):
            alive = False
            early_status = 5
            break
        if not all(math.isfinite(v) for v in sn):
            alive = False
            early_status = 5
            break
        cross = s[0]*sn[1]-s[1]*sn[0]
        dot = s[0]*sn[0]+s[1]*sn[1]
        theta += math.atan2(cross, dot)
        s = sn
        r = math.hypot(s[0], s[1])
        max_r, min_r = max(max_r, r), min(min_r, r)
        if r >= 1.0-0.007:
            alive = False
            early_status = 4
            break
        if (k+1) % nsp == 0:
            th_by_period.append(theta)
            r_by_period.append(r)
    done = len(r_by_period)
    if done >= max(2, int(0.8*PERIODS)):
        i0 = int(0.8*done)
        late_turns = (th_by_period[done]-th_by_period[i0])/(2.0*math.pi)
        status = _classify(late_turns, alive)
    else:
        late_turns = 0.0
        status = early_status if done < 2 and early_status else 0
    return {"map":label,"alpha":alpha,"lambda":lam,"A":A,"B":B,"nu":nu,
            "beta":beta,"pol_coeff":pol,"status_code":status,
            "status":CLASS[status],"turns_total":theta/(2.0*math.pi),
            "turns_late_window":late_turns,"periods_completed":done,
            "b_over_a":round(B/A,12) if A else None,
            "pol_multiplier":round(pol/P.pol_coeff[1],12) if label=="alpha_pol_PS2" else None,
            "r_mean_late":float(np.mean(r_by_period[int(0.8*done):])) if done else None,
            "r_min":min_r if min_r < float("inf") else None,"r_max":max_r if done else None,
            "initial_r_over_R":0.55,"initial_theta_rad":0.71,
            "horizon_RF1_periods":PERIODS,"steps_per_RF1_period":nsp}


def _pair_one(payload: tuple) -> dict:
    a1, a2, r1, r2, th1, th2, vscale, ph1, ph2, run_id = payload
    s=(r1*math.cos(th1),r1*math.sin(th1),
       -vscale[0]*r1*math.sin(th1),vscale[0]*r1*math.cos(th1),
       r2*math.cos(th2),r2*math.sin(th2),
       -vscale[1]*r2*math.sin(th2),vscale[1]*r2*math.cos(th2))
    dt=2.0*math.pi/STEPS_PER_PERIOD
    angle=[0.0,0.0]
    theta_period=[(0.0,0.0)]
    maxr=np.zeros(2); minr=np.full(2,np.inf)
    mind=float("inf")
    status=None
    done=0
    args=(a1,a2,P,)
    for k in range(PERIODS*STEPS_PER_PERIOD):
        try:
            sn=rk4_pair_fast(s,k*dt,dt,args+(ph1,ph2))
        except FloatingPointError:
            status=8; break
        except (OverflowError,ValueError,ZeroDivisionError):
            status=10; break
        if not all(math.isfinite(v) for v in sn):
            status=10; break
        for i,off in enumerate((0,4)):
            ox,oy=s[off],s[off+1]; nx,ny=sn[off],sn[off+1]
            angle[i]+=math.atan2(ox*ny-oy*nx,ox*nx+oy*ny)
            rad=math.hypot(nx,ny)
            maxr[i]=max(maxr[i],rad); minr[i]=min(minr[i],rad)
        d=math.hypot(sn[0]-sn[4],sn[1]-sn[5])
        mind=min(mind,d)
        if d <= sum(P.particle_radii_R):
            status=8; break
        if (math.hypot(sn[0],sn[1]) >= 1.0-P.particle_radii_R[0] or
                math.hypot(sn[4],sn[5]) >= 1.0-P.particle_radii_R[1]):
            status=9; break
        s=sn
        if (k+1)%STEPS_PER_PERIOD==0:
            theta_period.append(tuple(angle)); done+=1
    if status is None and done >= int(0.8*PERIODS):
        i0=int(0.8*done)
        turns=np.array(theta_period[-1])-np.array(theta_period[i0])
        turns=turns/(2.0*math.pi)
        sig=np.where(np.abs(turns)>=0.10,np.sign(turns),0).astype(int)
        if np.all(sig==[1,-1]): status=1
        elif np.all(sig==[-1,1]): status=2
        elif np.all(sig==[1,1]): status=3
        elif np.all(sig==[-1,-1]): status=4
        elif np.all(sig==0): status=5
        else: status=0
    elif status is None:
        status=0
    return {"alpha1":a1,"alpha2":a2,"start_id":run_id,"phase1":ph1,"phase2":ph2,
            "outcome_code":int(status),"outcome":PAIR_CLASS[int(status)],
            "turns1_total":angle[0]/(2*math.pi),"turns2_total":angle[1]/(2*math.pi),
            "turns1_late":float(turns[0]) if 'turns' in locals() else None,
            "turns2_late":float(turns[1]) if 'turns' in locals() else None,
            "r1_max":float(maxr[0]),"r2_max":float(maxr[1]),
            "min_pair_separation_R":mind if mind<float("inf") else None,
            "periods_completed":done,"horizon_RF1_periods":PERIODS,
            "steps_per_RF1_period":STEPS_PER_PERIOD}


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _plot_categorical(rows: list[dict], key_x: str, key_y: str, key_z: str,
                      xs: np.ndarray, ys: np.ndarray, title: str, outfile: str):
    matrix=np.full((len(ys),len(xs)),-1,dtype=int)
    xmap={float(v):i for i,v in enumerate(xs)}; ymap={float(v):i for i,v in enumerate(ys)}
    for r in rows:
        matrix[ymap[float(r[key_y])],xmap[float(r[key_x])]]=int(r[key_z])
    colors=["#f3f4f6","#2a9d8f","#457b9d","#e9c46a","#d1495b","#6c757d"]
    cmap=ListedColormap(colors); norm=BoundaryNorm(np.arange(-0.5,6.5,1),cmap.N)
    fig,ax=plt.subplots(figsize=(6.5,5.2),layout="constrained")
    ax.imshow(matrix,origin="lower",aspect="auto",interpolation="nearest",cmap=cmap,norm=norm,
              extent=[xs[0]-(xs[1]-xs[0])/2,xs[-1]+(xs[1]-xs[0])/2,
                      ys[0]-(ys[1]-ys[0])/2,ys[-1]+(ys[1]-ys[0])/2])
    ax.set_xlabel(r"$\alpha=(q/m)/(q/m)_*$")
    ylabels={"lambda":r"$\lambda=\gamma/(m\Omega_1)$",
             "nu":r"$\nu=\Omega_2/\Omega_1$",
             "b_over_a":r"$b/a$",
             "pol_multiplier":r"$\chi/\chi_{PS,2}$"}
    ax.set_ylabel(ylabels.get(key_y,key_y))
    ax.set_title(title)
    from matplotlib.patches import Patch
    handles=[Patch(facecolor=colors[i],label=CLASS[i]) for i in range(6)]
    ax.legend(handles=handles,fontsize=7,loc="best",frameon=False)
    ax.grid(False)
    for ext in ("png","pdf","svg"):
        fig.savefig(FIG/f"{outfile}.{ext}",dpi=600 if ext=="png" else None)
    plt.close(fig)


def wilson(k: int, n: int, z: float=1.959963984540054) -> tuple[float,float]:
    if n==0: return (float("nan"),float("nan"))
    p=k/n; den=1+z*z/n
    ctr=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return ctr-half,ctr+half


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    alpha=np.linspace(0.25,2.0,13)
    lam=np.linspace(0.04,0.80,13)
    nu=np.linspace(1.25,4.50,13)
    br=np.linspace(0.50,6.50,13)
    # Four deliberately separate two-dimensional mathematical sections.
    maps=[("alpha_lambda",[(float(a),float(l),P.A,P.B,P.nu,P.beta,0.0,"alpha_lambda")
                               for l in lam for a in alpha],alpha,lam,"lambda",
          r"(a) Mathematical screening: $(\alpha,\lambda)$"),
          ("alpha_nu",[(float(a),P.lam1,P.A,P.B,float(n),P.beta,0.0,"alpha_nu")
                            for n in nu for a in alpha],alpha,nu,"nu",
          r"(b) Mathematical screening: $(\alpha,\nu)$"),
          ("alpha_b_over_a",[(float(a),P.lam1,P.A,P.A*float(r),P.nu,P.beta,0.0,"alpha_b_over_a")
                                  for r in br for a in alpha],alpha,br,"b_over_a",
          r"(c) Mathematical screening: $(\alpha,b/a)$"),
          ("alpha_pol_PS2",[(float(a),P.lam2,P.A,P.B,P.nu,P.beta,
                              P.pol_coeff[1]*float(d),"alpha_pol_PS2")
                             for d in np.linspace(0,3,13) for a in alpha],alpha,
          np.linspace(0,3,13),"pol_multiplier",
          r"(d) Polarization sensitivity: $\alpha$ and $\chi/\chi_{\rm PS,2}$")]
    metadata=[]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for name,payload,xs,ys,ykey,title in maps:
            print(f"running {name}: {len(payload)} trajectories on {MAX_WORKERS} workers",flush=True)
            rows=list(pool.map(_one,payload,chunksize=4))
            _write_csv(OUT/f"{name}.csv",rows)
            _plot_categorical(rows,"alpha",ykey,"status_code",xs,ys,title,name)
            counts={CLASS[i]:sum(r["status_code"]==i for r in rows) for i in CLASS}
            metadata.append({"map":name,"n_grid":len(rows),"x_values":xs.tolist(),
                             "y_values":ys.tolist(),"outcome_counts":counts,
                             "horizon_RF1_periods":PERIODS,"steps_per_RF1_period":STEPS_PER_PERIOD,
                             "workers":MAX_WORKERS,"starting_state":"r/R=0.55; theta=0.71 rad; v=0",
                             "field":"2-D leading k=+2,-4; ideal DC quadrupole; no Coulomb; no 3-D end correction",
                             "classification":"bounded if center radius stays below 1-a/R; sign if >=0.10 turn in final 20%; short/ambiguous is UNRESOLVED; screening only"})

    # Main pair scan: charges vary at fixed PS masses/radii; Coulomb follows q1*q2.
    pair_axis=np.linspace(0.25,1.50,9)
    starts=[(0.55,0.766,0.0,math.pi/2,(0.0,0.0),"zero_v_90deg"),
            (0.50,0.82,math.pi/3,-math.pi/4,(0.0,0.0),"zero_v_phase2"),
            (0.58,0.78,-math.pi/5,math.pi/2,(0.00116,-0.000983),"branch_velocity"),
            (0.62,0.70,0.25,2.10,(0.0,0.0),"radial_offset")]
    pp=[]
    for a1 in pair_axis:
        for a2 in pair_axis:
            for r1,r2,t1,t2,v,tag in starts:
                pp.append((float(a1),float(a2),r1,r2,t1,t2,v,0.0,0.0,tag))
    print(f"running alpha-pair: {len(pp)} trajectories on {MAX_WORKERS} workers",flush=True)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        prows=list(pool.map(_pair_one,pp,chunksize=2))
    _write_csv(OUT/"alpha_pair_starts.csv",prows)
    cells=[]
    for a1 in pair_axis:
        for a2 in pair_axis:
            rr=[x for x in prows if x["alpha1"]==a1 and x["alpha2"]==a2]
            counts={PAIR_CLASS[i]:sum(x["outcome_code"]==i for x in rr) for i in PAIR_CLASS}
            k=counts["CR_PLUS_MINUS_SCREEN"]+counts["CR_MINUS_PLUS_SCREEN"]
            lo,hi=wilson(k,len(rr))
            ranked=[(sum(x["outcome_code"]==code for x in rr),code)
                    for code in range(1,10)]
            top=max(n for n,_ in ranked)
            winners=[code for n,code in ranked if n==top and n>0]
            dom=10 if len(winners)>1 else (winners[0] if winners else 0)
            cells.append({"alpha1":float(a1),"alpha2":float(a2),"n_starts":len(rr),
                          "cr_count":k,"cr_fraction_panel":k/len(rr),
                          "wilson95_low":lo,"wilson95_high":hi,
                          "dominant_outcome_code":dom,
                          "dominant_outcome":("MIXED" if dom==10 else PAIR_CLASS.get(dom,"UNRESOLVED")),
                          **counts})
    _write_csv(OUT/"alpha_pair_summary.csv",cells)
    cr=np.array([r["cr_fraction_panel"] for r in cells]).reshape(len(pair_axis),len(pair_axis))
    fig,axs=plt.subplots(1,3,figsize=(15.0,5.1),layout="constrained",
                         gridspec_kw={"width_ratios":[1,1,1.18]})
    ax=axs[0]
    im=ax.imshow(cr,origin="lower",aspect="auto",interpolation="nearest",vmin=0,vmax=1,
                 extent=[pair_axis[0]-.078,pair_axis[-1]+.078,pair_axis[0]-.078,pair_axis[-1]+.078],
                 cmap="viridis")
    ax.set_xlabel(r"$\alpha_1=(q_1/m_1)/(q_*/m_*)$")
    ax.set_ylabel(r"$\alpha_2=(q_2/m_2)/(q_*/m_*)$")
    ax.set_title("(a) Counterrotation fraction")
    cb=fig.colorbar(im,ax=ax); cb.set_label("CR starts / 4 (finite panel)")
    ax=axs[1]
    ci_width=np.array([r["wilson95_high"]-r["wilson95_low"] for r in cells]).reshape(len(pair_axis),len(pair_axis))
    im_ci=ax.imshow(ci_width,origin="lower",aspect="auto",interpolation="nearest",vmin=0,vmax=1,
                    extent=[pair_axis[0]-.078,pair_axis[-1]+.078,pair_axis[0]-.078,pair_axis[-1]+.078],
                    cmap="magma")
    ax.set_xlabel(r"$\alpha_1=(q_1/m_1)/(q_*/m_*)$")
    ax.set_ylabel(r"$\alpha_2=(q_2/m_2)/(q_*/m_*)$")
    ax.set_title("(b) Wilson 95% interval width")
    cb_ci=fig.colorbar(im_ci,ax=ax); cb_ci.set_label("upper − lower bound")
    ax=axs[2]
    dom=np.array([r["dominant_outcome_code"] for r in cells],dtype=int).reshape(len(pair_axis),len(pair_axis))
    pcolors=["#808080","#2a9d8f","#457b9d","#e9c46a","#8f5aa2",
             "#f4a261","#d1495b","#7f0000","#222222","#9c6644","#c7c7c7"]
    from matplotlib.colors import ListedColormap,BoundaryNorm
    pcmap=ListedColormap(pcolors); pnorm=BoundaryNorm(np.arange(-.5,11.5,1),pcmap.N)
    im2=ax.imshow(dom,origin="lower",aspect="auto",interpolation="nearest",cmap=pcmap,norm=pnorm,
                  extent=[pair_axis[0]-.078,pair_axis[-1]+.078,pair_axis[0]-.078,pair_axis[-1]+.078])
    ax.set_xlabel(r"$\alpha_1=(q_1/m_1)/(q_*/m_*)$")
    ax.set_ylabel(r"$\alpha_2=(q_2/m_2)/(q_*/m_*)$")
    ax.set_title("(c) Plurality outcome in the same 4 starts")
    cb2=fig.colorbar(im2,ax=ax,ticks=np.arange(11));
    cb2.ax.set_yticklabels(["unresolved","CR +−","CR −+","CO ++","CO −−","bound/nonrot.",
                            "one lost","both lost","collision","electrode","mixed/tie"])
    fig.suptitle("Two-particle finite-horizon screening; n=4 starts per grid point",fontsize=12)
    for ext in ("png","pdf","svg"):
        fig.savefig(FIG/f"alpha_pair_panel_fraction.{ext}",dpi=600 if ext=="png" else None)
    plt.close(fig)
    metadata.append({"map":"alpha_pair_panel_fraction","n_grid":len(cells),
                     "starts_per_cell":len(starts),"total_trajectories":len(prows),
                     "x_values":pair_axis.tolist(),"y_values":pair_axis.tolist(),
                     "fixed_particle_masses_kg":[1.8857409903172726e-10,4.007886827817179e-10],
                     "charge_scan":"q_i=alpha_i*M_i*q_ref; masses/radii/material/drag fixed; Coulomb recomputed from q1*q2",
                     "initial_conditions":[x[-1] for x in starts],"RF_phases_rad":[[0,0]],
                     "horizon_RF1_periods":PERIODS,"steps_per_RF1_period":STEPS_PER_PERIOD,
                     "outcome_counts":{PAIR_CLASS[i]:sum(x["outcome_code"]==i for x in prows) for i in PAIR_CLASS},
                     "interval":"Wilson 95% interval reported for finite four-start panel; not a calibrated basin probability",
                     "field":"planar leading-harmonic model; gravity and axial force omitted; ideal DC quadrupole transverse component retained"})
    summary={"study":"V17C.18B-19 preliminary mathematical screening",
             "base_sha":"718d9eadf5873cdf1a5b602de31ec20647092f0f",
             "physical_benchmark_sha":"6708ec62acec19f11ade7b8bdc3d24c0b6985a33",
             "maps":metadata,"workers":MAX_WORKERS,"cpu_logical":os.cpu_count(),
             "interpretation":"Short finite-horizon screening only. No cell is called an attractor or a 3-D physical operating point.",
             "total_trajectories":sum(x["n_grid"] for x in metadata[:-1])+len(prows)}
    (ROOT/"results"/"screening_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"trajectories":summary["total_trajectories"],
                      "workers":MAX_WORKERS,
                      "maps":{x["map"]:x.get("outcome_counts") for x in metadata}},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
