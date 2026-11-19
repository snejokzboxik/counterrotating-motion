"""Summarize fresh ensemble holdouts for a frozen P1 configuration."""
import json,collections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; RUNS=HERE/"stage2"/"runs"

def wilson(k,n,z=1.959963984540054):
    if not n:return [None,None]
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

def main():
    out={}; data={}
    for g in ("rest","weak","wide"):
        path=RUNS/f"cfg81_{g}_holdout_128_15000p_64s.jsonl"
        rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        data[g]=rows; n=len(rows); cr=sum(x["category"]=="CR_screen" for x in rows)
        ret=sum(bool(x["retained"]) for x in rows); c=collections.Counter(x["category"] for x in rows)
        good=[x for x in rows if x["category"]=="CR_screen"]
        out[g]={"N_total":n,"counts":dict(c),"N_CR":cr,"P_CR":cr/n,"P_CR_Wilson95":wilson(cr,n),
          "N_retained":ret,"P_retention":ret/n,"P_retention_Wilson95":wilson(ret,n),
          "P_CR_given_retention":cr/ret if ret else None,"P_CR_given_retention_Wilson95":wilson(cr,ret),
          "median_frequency_Hz":np.median([x["frequency_Hz"] for x in good],axis=0).tolist() if good else None,
          "median_r_min":np.median([x["r_min"] for x in good],axis=0).tolist() if good else None,
          "median_r_max":np.median([x["r_max"] for x in good],axis=0).tolist() if good else None,
          "min_particle_separation":min((x["minimum_separation"] for x in good),default=None)}
    dop=RUNS/"dop_cfg81_results.json"
    if dop.exists():
        ds=json.loads(dop.read_text(encoding="utf-8")); checks=[]
        for d in ds:
            _,g,i=d["key"].split(":"); rk=next(x for x in data[g] if x["index"]==int(i))
            checks.append({"key":d["key"],"delta_frequency_Hz":(np.asarray(d["frequency_Hz"])-rk["frequency_Hz"]).tolist(),
              "DOP853_frequency_Hz":d["frequency_Hz"],"RK4_64_frequency_Hz":rk["frequency_Hz"],
              "DOP853_turns":d["turns_after_transient"],"energy_residual_max":d["energy_residual_max"],
              "energy_residual_over_RF_work":d["energy_residual_over_RF_work"],
              "angle_error_turns":d["max_angle_integral_difference_turns"],"wall_s":d["wall_s"]})
        out["DOP853_checks"]=checks
    (HERE/"stage2"/"holdout_summary.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    fig,axs=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
    cmap={"CR_screen":"#198754","lost_retention":"#b91c1c","axis_ambiguous":"#9333ea","inconclusive":"#6b7280","point_singularity_unresolved":"#111827"}
    for ax,(g,rows) in zip(axs,data.items()):
        for cat in sorted(set(x["category"] for x in rows)):
            sub=[x for x in rows if x["category"]==cat]
            r1=np.hypot([x["initial"][0] for x in sub],[x["initial"][1] for x in sub])
            r2=np.hypot([x["initial"][6] for x in sub],[x["initial"][7] for x in sub])
            ax.scatter(r1,r2,s=16,alpha=.8,c=cmap.get(cat,"#6b7280"),edgecolors="none",label=f"{cat} (n={len(sub)})")
        ax.set(xlim=(0,1),ylim=(0,1),xlabel=r"initial $r_1/R$",ylabel=r"initial $r_2/R$",title=f"{g} holdout, N={len(rows)}")
        ax.grid(alpha=.2); ax.legend(fontsize=7,loc="best")
    fig.suptitle("P1 candidate 81 — fresh validation ensembles\n15000 primary RF cycles, RK4 64 steps/cycle; all outcomes in denominator")
    fig.savefig(HERE/"holdout_map.png",dpi=160)
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
