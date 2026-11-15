"""Selected full h50 P4 FEM validation around the V13.2 retention trough."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_v13_1_fem_pilot import advance, initialize  # noqa: E402

OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability"
V13_1 = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1"
MUS = (1.04,)
ETA = -0.05
PHASES = (0.0, math.pi/2)
MODELS = (("Q", 0.0), ("SO", ETA), ("RO", ETA))


def key(r: dict) -> tuple:
    return (round(float(r["mu"]),8),round(float(r["phase_rad"]),12),r["model"],
            round(float(r["eta"]),8),int(r["index"]),round(float(r.get("max_step_tau",0.)),12))


def read_initials() -> np.ndarray:
    data=np.genfromtxt(V13_1/"ideal_pilot"/"initial_conditions.csv",delimiter=",",names=True)
    return np.stack([data[n] for n in ("x0_rho","y0_rho","vx0_rho_per_tau","vy0_rho_per_tau")],axis=1)


def wilson(k:int,n:int,z:float=1.959963984540054)->tuple[float,float]:
    if not n:return (float("nan"),float("nan"))
    p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,c-h),min(1.,c+h)


def write_csv(path:Path,rows:list[dict])->None:
    if not rows:return
    fields=[]
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)


def summarize(rows:list[dict])->list[dict]:
    out=[]
    for mu in MUS:
      for phase in PHASES:
       group=[r for r in rows if abs(float(r["mu"])-mu)<1e-10 and abs(float(r["phase_rad"])-phase)<1e-10]
       q={int(r["index"]):r for r in group if r["model"]=="Q"}
       so={int(r["index"]):r for r in group if r["model"]=="SO"}
       for model,eta in MODELS:
        rr=[r for r in group if r["model"]==model]
        for horizon in (100,300):
         col=f"status_{horizon}"
         resolved=[r for r in rr if r.get(col) in ("retained","escaped")]
         k=sum(r[col]=="retained" for r in resolved);lo,hi=wilson(k,len(resolved))
         base=q if model!="Q" else q
         d=[int(r[col]=="retained")-int(base[int(r["index"])][col]=="retained") for r in resolved
            if int(r["index"]) in base and base[int(r["index"])].get(col) in ("retained","escaped")]
         mean=float(np.mean(d)) if d else float("nan")
         se=float(np.std(d,ddof=1)/math.sqrt(len(d))) if len(d)>1 else float("nan")
         row={"mu":mu,"phase_deg":float(np.degrees(phase)),"horizon_rf":horizon,"model":model,"eta":eta,
              "n_total":len(rr),"n_resolved":len(resolved),"n_unresolved":len(rr)-len(resolved),"n_retained":k,
              "retention":k/len(resolved) if resolved else float("nan"),"wilson95_low":lo,"wilson95_high":hi,
              "paired_n_vs_Q":len(d),"paired_diff_vs_Q":mean,
              "paired_diff_vs_Q_ci95_low":mean-1.96*se if np.isfinite(se) else float("nan"),
              "paired_diff_vs_Q_ci95_high":mean+1.96*se if np.isfinite(se) else float("nan"),
              "paired_gains_vs_Q":sum(v==1 for v in d),"paired_losses_vs_Q":sum(v==-1 for v in d)}
         if model=="RO":
          ds=[int(r[col]=="retained")-int(so[int(r["index"])][col]=="retained") for r in resolved
              if int(r["index"]) in so and so[int(r["index"])].get(col) in ("retained","escaped")]
          dm=float(np.mean(ds)) if ds else float("nan")
          de=float(np.std(ds,ddof=1)/math.sqrt(len(ds))) if len(ds)>1 else float("nan")
          row.update({"paired_n_RO_minus_SO":len(ds),"paired_diff_RO_minus_SO":dm,
              "paired_diff_RO_minus_SO_ci95_low":dm-1.96*de if np.isfinite(de) else float("nan"),
              "paired_diff_RO_minus_SO_ci95_high":dm+1.96*de if np.isfinite(de) else float("nan"),
              "RO_SO_gains":sum(v==1 for v in ds),"RO_SO_losses":sum(v==-1 for v in ds)})
         out.append(row)
    return out


def combine_and_plot_fem(new_rows:list[dict])->None:
    sources=[V13_1/"fem_h50_p4_mu100_105"/"retention_summary.csv",
             V13_1/"fem_h50_p4_mu110"/"retention_summary.csv"]
    selected=[]
    for path in sources:
        with path.open(newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if int(float(r["horizon_rf"]))!=300 or r["model"] not in ("Q","SO","RO"):continue
                if r["model"]!="Q" and abs(float(r["eta"])-ETA)>1e-8:continue
                selected.append({"mu":float(r["mu"]),"phase_deg":float(r["phase_deg"]),"horizon_rf":300,
                    "model":r["model"],"eta":float(r["eta"]),"n_total":int(r["n_total"]),
                    "n_resolved":int(r["n_resolved"]),"n_unresolved":int(r["n_unresolved"]),
                    "n_retained":int(r["n_retained"]),"retention":float(r["retention"]),
                    "wilson95_low":float(r["wilson95_low"]),"wilson95_high":float(r["wilson95_high"]),
                    "source":"reused_V13.1_h50_P4"})
    for r in new_rows:
        if int(r["horizon_rf"])==300:
            row=dict(r);row["source"]="new_V13.2_h50_P4";selected.append(row)
    selected.sort(key=lambda r:(r["mu"],r["phase_deg"],r["model"]))
    write_csv(OUT/"stage_a_full_fem_mass_retention.csv",selected)

    import matplotlib.pyplot as plt
    colors={"Q":"#202020","SO":"#d97706","RO":"#2563eb"}
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.1),sharey=True)
    for ax,phase in zip(axes,(0.,90.)):
        for model in ("Q","SO","RO"):
            rr=sorted([r for r in selected if r["phase_deg"]==phase and r["model"]==model],key=lambda r:r["mu"])
            x=np.asarray([r["mu"] for r in rr]);p=np.asarray([r["retention"] for r in rr])
            lo=np.asarray([r["wilson95_low"] for r in rr]);hi=np.asarray([r["wilson95_high"] for r in rr])
            ax.errorbar(x,p,yerr=np.vstack((p-lo,hi-p)),fmt="o",ms=5,capsize=3,lw=1,
                        color=colors[model],label=model)
        ax.set_title(f"Фаза {phase:.0f}°");ax.set_xlabel("μ=(M/Q)/(M₀/Q₀)")
        ax.set_ylim(-.04,1.04);ax.grid(alpha=.25)
    axes[0].set_ylabel("Удержание за 300 RF периодов")
    axes[1].legend(frameon=False)
    fig.suptitle("Полное h50 P4 FEM поле; точки μ V13.1/V13.2 рассчитаны или переиспользованы без интерполяции")
    fig.tight_layout()
    for ext in ("png","pdf","svg"):fig.savefig(OUT/f"stage_a_full_fem_mass_scan.{ext}",dpi=220 if ext=="png" else None)
    plt.close(fig)


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--step-rf",type=float,default=1/64);args=parser.parse_args()
    dest=OUT/"stage_a_full_fem_h50_p4";dest.mkdir(parents=True,exist_ok=True)
    states=read_initials()
    (dest/"initial_conditions.csv").write_text((V13_1/"ideal_pilot"/"initial_conditions.csv").read_text(encoding="utf-8"),encoding="utf-8")
    ck=dest/"checkpoint.jsonl";old=[json.loads(x) for x in ck.read_text(encoding="utf-8").splitlines() if x.strip()] if ck.exists() else []
    # A restart after interruption may leave duplicate completed keys; preserve one result per task.
    old=list({key(r):r for r in old}.values())
    done={key(r) for r in old}
    tasks=[]
    for mu in MUS:
      for phase in PHASES:
       for model,eta in MODELS:
        for i,state in enumerate(states):
         task=(mu,phase,model,eta,i,state,args.step_rf)
         if (round(mu,8),round(phase,12),model,round(eta,8),i,round(2*math.pi*args.step_rf,12)) not in done:
          tasks.append(task)
    conf={"field":"full native h50 P4 FEM; saved m=4 and m=2 unit solutions on shared mesh",
          "mu":list(MUS),"eta":ETA,"models":[list(x) for x in MODELS],"phases_rad":list(PHASES),
          "n_initial":len(states),"initial_source":"V13.1 common seed131 Sobol N=32",
          "horizons_rf":[100,300],"aperture_rho":1.0,"continuation":"same state and RF phase from 100 to 300",
          "integrator":"DOP853 rtol=3e-9 atol=3e-11","max_step_rf_fraction":args.step_rf,
          "mesh_sha256":"8ea670e6bf5bada2c86756bc98718063fe38cbfd1692c872c4a4e77f59b1a284",
          "solution_files":{"m4":"outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz",
                            "m2":"outputs/octupole_comparison_home/mass_filter_v13/v13_1/fem/quadrupole_h50_P4_sector_solution.npz"},
          "max_voltage_V":"existing V13.1 drive: Q+RO/SO eta=-0.05 has peak 232.8591 V; <=250 V",
          "interpretation":"full P4 transverse finite-time retention, not finite-length mass-filter transmission"}
    (dest/"parameters.json").write_text(json.dumps(conf,indent=2)+"\n",encoding="utf-8")
    print(f"full FEM h50/P4 tasks={len(tasks)}, workers={args.workers}",flush=True)
    if tasks:
      with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool,ck.open("a",encoding="utf-8") as f:
       fs=[pool.submit(advance,t) for t in tasks]
       for n,fu in enumerate(as_completed(fs),1):
        r=fu.result();old.append(r);f.write(json.dumps(r,separators=(",", ":"))+"\n");f.flush()
        if n%32==0 or n==len(fs):print(f"FEM completed {n}/{len(fs)}",flush=True)
    old.sort(key=lambda r:(float(r["mu"]),float(r["phase_rad"]),r["model"],int(r["index"])))
    with ck.open("w",encoding="utf-8") as f:
        for row in old:f.write(json.dumps(row,separators=(",", ":"))+"\n")
    write_csv(dest/"trajectory_results.csv",old)
    summary=summarize(old)
    write_csv(dest/"retention_summary.csv",summary)
    combine_and_plot_fem(summary)
    # Store hashes of the actual immutable source FEM solutions for provenance.
    files=[ROOT/"outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz",
           V13_1/"fem"/"quadrupole_h50_P4_sector_solution.npz"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (dest/"fem_source_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main()
