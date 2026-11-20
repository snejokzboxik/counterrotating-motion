"""Strict CR outcome and confidence-interval summary for 15k-period runs."""
from collections import Counter,defaultdict
import csv,json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE/"strict_long_validation"
GROUPS=("rest","weak","wide","edge")

def wilson(k,n):
    z=1.96;p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(m-h),float(m+h)]

def main():
    rows=[json.loads(s) for s in (DATA/"strict_runs.jsonl").read_text().splitlines() if s.strip()]
    keys=[(r["protocol_id"],r["ensemble"],r["start_id"]) for r in rows]
    if len(keys)!=len(set(keys)):raise RuntimeError("duplicate strict keys")
    meta=json.loads((DATA/"strict_metadata.json").read_text())
    if len(rows)!=len(meta["protocol_ids"])*meta["N_per_ensemble"]*4:raise RuntimeError("strict run list incomplete")
    prot=json.loads((HERE/"loading_screen"/"protocols.json").read_text())
    by=defaultdict(list)
    for r in rows:by[(r["protocol_id"],r["ensemble"])].append(r)
    summary={"status":"strict_15k_period_validation_complete; numeric-method_validation_pending",
             "N_per_ensemble":meta["N_per_ensemble"],"periods":meta["periods"],
             "steps_per_period":meta["steps_per_period"],
             "transient_discard":"transition completion plus 200 primary RF cycles",
             "strict_success":"category CR_screen only; every other state counts as failure",
             "unique_run_keys":len(set(keys)),"by_protocol":{}}
    csvrows=[]
    for pid in meta["protocol_ids"]:
        pdata={"family":prot[pid]["family"],"transition_cycles":prot[pid].get("duration_cycles",0),"by_ensemble":{}}
        for g in GROUPS:
            a=by[(pid,g)];cats=Counter(r["category"] for r in a);k=cats["CR_screen"];n=len(a)
            turns=np.asarray([r["turns_after_relaxation"] for r in a],float)
            freqs=np.asarray([r["frequency_Hz"] for r in a],float)
            forw=np.asarray([r["window_turns"] for r in a],float)
            success_indices=[i for i,r in enumerate(a) if r["category"]=="CR_screen"]
            srturns=turns[success_indices] if success_indices else np.empty((0,2))
            srfreqs=freqs[success_indices] if success_indices else np.empty((0,2))
            entry={"CR_count":k,"N":n,"P_CR":k/n,"wilson95":wilson(k,n),
                   "category_counts":dict(cats),"contact_count":cats.get("finite_particle_contact",0)+cats.get("finite_particle_wall_contact",0),
                   "axis_ambiguous_count":cats.get("axis_ambiguous",0),
                   "mean_turns_successes":np.mean(srturns,axis=0).tolist() if len(srturns) else None,
                   "mean_frequency_Hz_successes":np.mean(srfreqs,axis=0).tolist() if len(srfreqs) else None,
                   "mean_min_radius_all_starts":np.mean([r["r_min"] for r in a],axis=0).tolist(),
                   "mean_min_separation_all_starts":float(np.mean([r["minimum_separation"] for r in a]))}
            pdata["by_ensemble"][g]=entry
            csvrows.append({"protocol_id":pid,"family":prot[pid]["family"],"ensemble":g,
                "CR_count":k,"N":n,"P_CR":k/n,"wilson95_low":entry["wilson95"][0],"wilson95_high":entry["wilson95"][1],
                "contacts":entry["contact_count"],"axis_ambiguous":entry["axis_ambiguous_count"],
                "mean_turn1_CR":entry["mean_turns_successes"][0] if entry["mean_turns_successes"] else None,
                "mean_turn2_CR":entry["mean_turns_successes"][1] if entry["mean_turns_successes"] else None})
        pdata["J_standard_min_rest_weak_wide"]=min(pdata["by_ensemble"][g]["P_CR"] for g in GROUPS[:3])
        pdata["J_robust_min_four"]=min(pdata["by_ensemble"][g]["P_CR"] for g in GROUPS)
        summary["by_protocol"][str(pid)]=pdata
    (DATA/"strict_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    with (DATA/"strict_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    fig,ax=plt.subplots(figsize=(10,5.5));x=np.arange(len(meta["protocol_ids"]));width=.18
    colors=["#3465a4","#4e9a06","#c17d11","#a40000"]
    for j,g in enumerate(GROUPS):
        vals=[summary["by_protocol"][str(pid)]["by_ensemble"][g]["P_CR"] for pid in meta["protocol_ids"]]
        ax.bar(x+(j-1.5)*width,vals,width,label=g,color=colors[j])
    ax.axhline(.95,color="black",ls="--",lw=1,label="95% target")
    ax.set_xticks(x,[f"#{pid}\n{prot[pid]['family']}" for pid in meta["protocol_ids"]]);ax.set_ylim(0,1.04)
    ax.set_ylabel("Strict CR outcomes / all starts")
    ax.set_title("15,000-period independent validation; N=64 per group")
    ax.grid(axis="y",alpha=.25);ax.legend(ncol=3);fig.tight_layout()
    fig.savefig(DATA/"strict_counterrotation_rates.png",dpi=180);plt.close(fig)
    for pid in meta["protocol_ids"]:
        s=summary["by_protocol"][str(pid)]
        print(pid,prot[pid]["family"],{g:(s["by_ensemble"][g]["CR_count"],s["by_ensemble"][g]["N"],s["by_ensemble"][g]["category_counts"]) for g in GROUPS})

if __name__=="__main__":main()
