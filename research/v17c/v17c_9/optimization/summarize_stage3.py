"""Summarize candidate holdouts without pooling the three launch distributions."""
import argparse, collections, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RUNS=HERE/"stage3"/"runs"

def wilson(k,n,z=1.959963984540054):
    if not n: return [None,None]
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

def strict_cr(r):
    turns=np.asarray(r["turns"],float)
    windows=np.asarray(r["window_turns"],float)
    if not r["retained"] or min(r["r_min"])<.02: return False
    signs=np.sign(windows)
    return bool(np.all(np.abs(turns)>=10.) and np.all(np.abs(windows)>.05)
                and np.all(signs[:,0]*signs[:,1]<0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config-id",type=int,required=True)
    ap.add_argument("--count",type=int,default=128)
    ap.add_argument("--periods",type=int,default=15000)
    a=ap.parse_args()
    result={"config_id":a.config_id,"N_total_is_denominator":True,"ensembles":{}}
    fig,axs=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    for ax,g in zip(axs,("rest","weak","wide")):
        path=RUNS/f"cfg{a.config_id}_{g}_holdout_{a.count}_{a.periods}p_64s.jsonl"
        rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        cats=collections.Counter(r["category"] for r in rows)
        strict=[r for r in rows if strict_cr(r)]
        qualitative=[r for r in rows if r["retained"] and min(r["r_min"])>=.02 and
            np.all(np.abs(np.asarray(r["window_turns"]))>.05) and
            np.all(np.sign(np.asarray(r["window_turns"])[:,0])*np.sign(np.asarray(r["window_turns"])[:,1])<0)]
        n=len(rows); k=len(strict)
        result["ensembles"][g]={"N_total":n,"category_counts":dict(cats),
            "strict_counter_rotation_N":k,"strict_counter_rotation_fraction":k/n if n else None,
            "strict_counter_rotation_Wilson95":wilson(k,n),
            "retained_N":sum(bool(r["retained"]) for r in rows),
            "qualitative_opposite_sign_N_without_10_turn_cutoff":len(qualitative),
            "median_frequency_Hz_all":np.median([r["frequency_Hz"] for r in rows],axis=0).tolist() if rows else None,
            "median_frequency_Hz_qualitative":np.median([r["frequency_Hz"] for r in qualitative],axis=0).tolist() if qualitative else None,
            "median_turns_qualitative":np.median([r["turns"] for r in qualitative],axis=0).tolist() if qualitative else None,
            "minimum_particle_separation_qualitative":min((r["minimum_separation"] for r in qualitative),default=None)}
        palette={"CR_screen":"#198754","inconclusive":"#6b7280","lost_retention":"#b91c1c",
                 "axis_ambiguous":"#9333ea","point_singularity_unresolved":"#111827",
                 "same_direction":"#d97706","localized_or_slow":"#0891b2"}
        for cat in sorted(cats):
            sub=[r for r in rows if r["category"]==cat]
            ax.scatter([np.hypot(*r["initial"][:2]) for r in sub],
                       [np.hypot(*r["initial"][6:8]) for r in sub],
                       s=19,alpha=.8,c=palette.get(cat,"#6b7280"),edgecolors="none",
                       label=f"{cat}: {len(sub)}")
        ax.set(xlim=(0,1),ylim=(0,1),xlabel=r"initial $r_1/R$",ylabel=r"initial $r_2/R$",
               title=f"{g}: strict {k}/{n}")
        ax.grid(alpha=.22); ax.legend(fontsize=7,loc="best")
    fig.suptitle(f"P1 local candidate {a.config_id}: {a.periods} RF cycles, RK4-64\nAll failures and slow trajectories remain in N_total")
    fig.savefig(HERE/f"stage3_cfg{a.config_id}_basins.png",dpi=160)
    (HERE/f"stage3_cfg{a.config_id}_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
