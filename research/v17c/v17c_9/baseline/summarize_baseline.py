"""Summarize the fixed P0 ensembles with unconditional denominators and Wilson CIs."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RUNS=HERE/"runs"

def wilson(k,n,z=1.959963984540054):
    if n==0: return [None,None]
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-h),min(1.,c+h)]

def main():
    result={}
    rows={}; rows32={}
    for name in ("rest","weak","wide"):
        path=RUNS/f"{name}_5000p_64s.jsonl"
        data=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        rows[name]=data; n=len(data); c=Counter(x["category"] for x in data)
        path32=RUNS/f"{name}_5000p_32s.jsonl"
        data32=[json.loads(x) for x in path32.read_text(encoding="utf-8").splitlines() if x.strip()]
        rows32[name]=data32
        cat32={x["index"]:x["category"] for x in data32}
        mismatches=[x for x in data if cat32.get(x["index"])!=x["category"]]
        hits=c["CR_screen"]; retained=sum(x["retained"] for x in data)
        result[name]={"N_total":n,"counts":dict(c),"N_CR_screen":hits,
            "N_retained":retained,"P_CR_lower_bound":hits/n if n else None,
            "P_CR_Wilson95":wilson(hits,n),"P_retention":retained/n if n else None,
            "P_retention_Wilson95":wilson(retained,n),
            "P_CR_given_retention":hits/retained if retained else None,
            "P_CR_given_retention_Wilson95":wilson(hits,retained),
            "screen_horizon_periods":5000,"initial_steps_per_primary_RF_period":32,
            "refinement_steps_per_primary_RF_period":64,
            "classification_mismatches_vs_32step":len(mismatches),
            "mismatch_examples":[{"index":x["index"],"32step":cat32.get(x["index"]),"64step":x["category"]} for x in mismatches[:12]],
            "transient_discard_periods":200,
            "uncertain_count":sum(v for k,v in c.items() if k in ("inconclusive","localized_or_slow","axis_ambiguous","point_singularity_unresolved"))}
    out={"parameter_source":"../v17c_5/input/config.json","ensembles":result,
         "classification":"Only CR_screen counts as a confirmed screen success; escaped, slow, axis ambiguous, singular and inconclusive outcomes remain failures in the unconditional denominator.",
         "sampling":"independent scrambled Sobol designs; separate seeds and start distributions; both RF phases uniform and independent. The 64-step rerun reuses exactly the frozen 32-step initial conditions.",
         "caveat":"P0 includes full ensemble scans at 32 and 64 RK4 steps/cycle over 5000 cycles. Any CR_screen remains a screening result requiring independent high-accuracy verification. Finite-size/contact physics is not yet applied."}
    (HERE/"baseline_summary.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    fig,axs=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
    colors={"CR_screen":"#198754","same_direction":"#d97706","lost_retention":"#b91c1c","localized_or_slow":"#2563eb","axis_ambiguous":"#9333ea","inconclusive":"#6b7280","point_singularity_unresolved":"#111827"}
    for ax,(name,data) in zip(axs,rows.items()):
        for cat in sorted(set(x["category"] for x in data)):
            sub=[x for x in data if x["category"]==cat]
            r1=np.hypot([x["initial"][0] for x in sub],[x["initial"][1] for x in sub])
            r2=np.hypot([x["initial"][6] for x in sub],[x["initial"][7] for x in sub])
            ax.scatter(r1,r2,s=12,alpha=.75,c=colors.get(cat,"#6b7280"),label=f"{cat} (n={len(sub)})",edgecolors="none")
        ax.set(xlim=(0,1),ylim=(0,1),xlabel=r"initial $r_1/R$",ylabel=r"initial $r_2/R$",title=f"{name}: N={len(data)}")
        ax.grid(alpha=.2); ax.legend(fontsize=7,loc="best")
    fig.suptitle("V17C.9 P0 screening — initial radii colored by outcome\n24-electrode V17C.5 baseline; 5000 primary RF cycles, 64 RK4 steps/cycle (same starts screened at 32)")
    fig.savefig(HERE/"baseline_basin_map.png",dpi=160)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
