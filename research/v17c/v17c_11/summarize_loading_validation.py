"""Audit and summarize independent 3000-period validation results."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv, json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "loading_validation"
GROUPS = ("rest", "weak", "wide", "edge")


def ci(k, n):
    z = 1.96; p = k/n; d=1+z*z/n
    mid=(p+z*z/(2*n))/d
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(mid-half), float(mid+half)]


def main():
    rows=[json.loads(s) for s in (DATA/"validation_runs.jsonl").read_text().splitlines() if s.strip()]
    keys=[(r["protocol_id"],r["ensemble"],r["start_id"]) for r in rows]
    if len(keys)!=len(set(keys)): raise RuntimeError("duplicate validation keys")
    meta=json.loads((DATA/"validation_metadata.json").read_text())
    if len(rows)!=len(meta["protocol_ids"])*64*4: raise RuntimeError("validation is incomplete")
    by=defaultdict(list)
    for r in rows:by[(r["protocol_id"],r["ensemble"])].append(r)
    protocols=json.loads((HERE/"loading_screen"/"protocols.json").read_text())
    summary={"status":"complete_independent_3000_cycle_screen_strict_CR_not_reached",
             "N_per_ensemble":64,"periods":3000,"steps_per_period":96,
             "strict_CR_definition":"retained, no contact, no axis ambiguity, >=10 turns per particle after the ramp, and opposite signs in all three late windows",
             "by_protocol":{}}
    records=[]
    for pid in meta["protocol_ids"]:
        pdata={"family":protocols[pid]["family"],"transition_cycles":protocols[pid].get("duration_cycles",0),"by_ensemble":{}}
        for g in GROUPS:
            a=by[(pid,g)]; cats=Counter(r["category"] for r in a)
            retained=sum(r["retained"] for r in a)
            cr=cats["CR_screen"]
            opp=sum(all(w[0]*w[1]<0 for w in r["window_turns"]) for r in a)
            avgturn=np.mean([r["turns_after_transition"] for r in a if r["retained"]],axis=0).tolist()
            freq=np.mean([r["frequency_Hz"] for r in a if r["retained"]],axis=0).tolist()
            pdata["by_ensemble"][g]={"retained":retained,"N":64,"retained_rate":retained/64,
                "retained_wilson95":ci(retained,64),"strict_CR":cr,"strict_CR_rate":cr/64,
                "strict_CR_wilson95":ci(cr,64),"opposite_late_window_proxy":opp,
                "category_counts":dict(cats),"mean_turns_if_retained":avgturn,
                "mean_frequency_Hz_if_retained":freq}
            records.append({"protocol_id":pid,"family":protocols[pid]["family"],"ensemble":g,
                "retained":retained,"N":64,"retained_rate":retained/64,"strict_CR":cr,
                "opposite_late_window_proxy":opp,"mean_turn1_if_retained":avgturn[0],
                "mean_turn2_if_retained":avgturn[1],"mean_frequency1_Hz":freq[0],"mean_frequency2_Hz":freq[1],
                "category_counts":json.dumps(dict(cats),separators=(",",":"))})
        summary["by_protocol"][str(pid)]=pdata
    (DATA/"validation_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    with (DATA/"validation_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

    pids=meta["protocol_ids"]
    fig,ax=plt.subplots(figsize=(11,5.5)); x=np.arange(len(pids)); width=.18
    colors=["#3465a4","#4e9a06","#c17d11","#a40000"]
    for i,g in enumerate(GROUPS):
        ax.bar(x+(i-1.5)*width,[summary["by_protocol"][str(pid)]["by_ensemble"][g]["retained_rate"] for pid in pids],
               width,label=g,color=colors[i])
    ax.set_xticks(x,[f"#{pid}\n{protocols[pid]['family']}" for pid in pids]);ax.set_ylim(0,1.04)
    ax.set_ylabel("Retained through 3000 primary RF periods")
    ax.set_title("Independent validation, N=64/group; retention is not strict P_CR")
    ax.grid(axis="y",alpha=.25);ax.legend(ncol=4);fig.tight_layout()
    fig.savefig(DATA/"validation_retention.png",dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5.5))
    for idx,pid in enumerate(pids):
        vals=[]
        for g in GROUPS:
            a=by[(pid,g)]
            selected=[r["frequency_Hz"] for r in a if r["retained"]]
            vals.append(np.mean(selected,axis=0))
        v=np.mean(vals,axis=0)
        ax.scatter(idx,v[0],color="#3465a4",marker="o",label="particle 1 mean" if idx==0 else None)
        ax.scatter(idx,v[1],color="#a40000",marker="s",label="particle 2 mean" if idx==0 else None)
    ax.axhline(0,color="black",lw=.8);ax.set_xticks(np.arange(len(pids)),[f"#{pid}" for pid in pids])
    ax.set_ylabel("Mean orbital frequency among retained starts (Hz)")
    ax.set_title("Late-direction diagnostic across independent groups; strict 10-turn test not met")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(DATA/"validation_frequencies.png",dpi=180);plt.close(fig)
    print(f"audited {len(rows)} unique validation trajectories; strict CR rows={sum(r['category']=='CR_screen' for r in rows)}")


if __name__=="__main__":main()
