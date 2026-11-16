"""Replicate-level uncertainty summary for the V13.3 local mass scan."""
from __future__ import annotations
import argparse,csv, math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
OUT=BASE/"fine_mass_scan"

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write(path,rows):
    if not rows:return
    names=[]
    for r in rows:names.extend(k for k in r if k not in names)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
def mean_ci(values):
    vals=[float(x) for x in values]
    if not vals:return math.nan,math.nan,math.nan
    m=sum(vals)/len(vals)
    if len(vals)<2:return m,math.nan,math.nan
    h=float(t.ppf(.975,len(vals)-1))*math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1)/len(vals))
    return m,m-h,m+h
def main():
    global OUT
    p=argparse.ArgumentParser()
    p.add_argument("--directory",default="fine_mass_scan",help="V13.3 subdirectory containing trajectory_results.csv")
    args=p.parse_args()
    OUT=BASE/args.directory
    rows=read(OUT/"trajectory_results.csv")
    grouped=defaultdict(dict)
    for r in rows:
        k=(float(r["mu"]),int(r["phase_i"]),int(r["replica"]),int(r["state_i"]))
        grouped[(k[0],k[1],r["model"],k[2])][k[3]]=r
    # Replicate-level estimates, with unresolved outcomes left as explicit bounds.
    rep_stats=[]
    for (mu,phase,model,rep),states in sorted(grouped.items()):
        total=len(states);ret=sum(r["status"]=="retained" for r in states.values())
        unres=sum(r["status"]=="unresolved" for r in states.values())
        exits=[float(r["exit_periods"]) for r in states.values() if r["status"]=="escaped" and r["exit_periods"] not in ("", "nan")]
        rep_stats.append({"mu":mu,"phase_deg":90*phase,"model":model,"replica":rep,"seed":next(iter(states.values()))["sobol_seed"],
                          "n_total":total,"n_resolved":total-unres,"n_retained":ret,"n_unresolved":unres,
                          "P_lower":ret/total if total else math.nan,"P_upper":(ret+unres)/total if total else math.nan,
                          "median_exit_RF":sorted(exits)[len(exits)//2] if exits else math.nan,
                          "max_exit_RF":max(exits) if exits else math.nan,
                          "max_rho_max":max(float(r["rho_max"]) for r in states.values() if r["rho_max"] not in ("", "nan")) if states else math.nan})
    write(OUT/"replicate_retention.csv",rep_stats)
    summary=[]
    index={(float(r["mu"]),int(float(r["phase_deg"])/90),r["model"],int(r["replica"])):r for r in rep_stats}
    mus=sorted({float(r["mu"]) for r in rows})
    for mu in mus:
      for phase in (0,1,2):
       for model in ("Q","RO"):
        rr=[]
        for rep in range(4):
            a=index.get((mu,0,model,rep));b=index.get((mu,1,model,rep))
            if phase<2:r=a if phase==0 else b
            else:
                if a is None or b is None:continue
                r={"P_lower":(float(a["P_lower"])+float(b["P_lower"]))*.5,
                   "P_upper":(float(a["P_upper"])+float(b["P_upper"]))*.5,
                   "n_unresolved":int(a["n_unresolved"])+int(b["n_unresolved"]),
                   "n_total":int(a["n_total"])+int(b["n_total"])}
            if r is not None:rr.append(r)
        if not rr:continue
        lowmean,lcl,lch=mean_ci([r["P_lower"] for r in rr]);upmean,ucl,uch=mean_ci([r["P_upper"] for r in rr])
        summary.append({"mu":mu,"M_over_Q_Th_per_e":3200.04652261404*mu,
                        "phase_deg":(0,90,"0+90 pooled equally")[phase],"model":model,
                        "replicas":len(rr),"n_total":sum(int(r["n_total"]) for r in rr),
                        "n_unresolved":sum(int(r["n_unresolved"]) for r in rr),
                        "P_ret_lower_mean":lowmean,"P_ret_upper_mean":upmean,
                        "P_ret_mean":(lowmean+upmean)/2,
                        "replicate_t95_low":lcl,"replicate_t95_high":uch,
                        "unresolved_upper_t95_low":ucl,"unresolved_upper_t95_high":uch,
                        "uncertainty":"Student-t 95% interval over four independent scrambled-Sobol replicate means; QMC points are not pooled as iid Bernoulli"})
    write(OUT/"retention_summary.csv",summary)
    paired=[]
    for mu in mus:
      for phase in (0,1):
       q_by={(int(r["replica"]),int(r["state_i"])):r for r in rows if float(r["mu"])==mu and int(r["phase_i"])==phase and r["model"]=="Q"}
       for ro in (r for r in rows if float(r["mu"])==mu and int(r["phase_i"])==phase and r["model"]=="RO"):
        q=q_by[(int(ro["replica"]),int(ro["state_i"]))]
        def bounds(status):return (1,1) if status=="retained" else (0,0) if status=="escaped" else (0,1)
        rl,rh=bounds(ro["status"]);ql,qh=bounds(q["status"])
        paired.append({"mu":mu,"phase_deg":90*phase,"replica":int(ro["replica"]),"state_i":int(ro["state_i"]),
                       "Q_status":q["status"],"RO_status":ro["status"],
                       "RO_minus_Q_lower":rl-qh,"RO_minus_Q_upper":rh-ql,
                       "RO_minus_Q_exact":rl-qh if rl==rh and ql==qh else math.nan,
                       "unresolved_pair":int("unresolved" in (q["status"],ro["status"]))})
    write(OUT/"paired_state_differences.csv",paired)
    pgroup=defaultdict(list)
    for r in paired:pgroup[(float(r["mu"]),int(r["phase_deg"]),int(r["replica"]))].append(r)
    pd=[]
    for (mu,phase,rep),rs in sorted(pgroup.items()):
        pd.append({"mu":mu,"phase_deg":phase,"replica":rep,"n_total":len(rs),
                   "n_resolved_pairs":sum(r["unresolved_pair"]==0 for r in rs),
                   "n_unresolved_pairs":sum(r["unresolved_pair"]==1 for r in rs),
                   "mean_difference_lower":sum(float(r["RO_minus_Q_lower"]) for r in rs)/len(rs),
                   "mean_difference_upper":sum(float(r["RO_minus_Q_upper"]) for r in rs)/len(rs),
                   "mean_difference_exact":sum(float(r["RO_minus_Q_exact"]) for r in rs if r["unresolved_pair"]==0)/sum(r["unresolved_pair"]==0 for r in rs) if all(r["unresolved_pair"]==0 for r in rs) else math.nan})
    write(OUT/"paired_replica_differences.csv",pd)
    ps=[]
    for mu in mus:
      for phase in (0,90):
       rr=[r for r in pd if r["mu"]==mu and r["phase_deg"]==phase]
       if not rr:continue
       m,lo,hi=mean_ci([r["mean_difference_lower"] for r in rr]);u,ulo,uhi=mean_ci([r["mean_difference_upper"] for r in rr])
       ps.append({"mu":mu,"phase_deg":phase,"replicas":len(rr),"mean_difference_lower":m,
                  "mean_difference_upper":u,"replicate_t95_lower_bound":lo,"replicate_t95_upper_bound":uhi,
                  "mean_difference_exact":sum(r["mean_difference_exact"] for r in rr)/len(rr) if all(math.isfinite(r["mean_difference_exact"]) for r in rr) else math.nan,
                  "exact_t95_low":mean_ci([r["mean_difference_exact"] for r in rr])[1] if all(math.isfinite(r["mean_difference_exact"]) for r in rr) else math.nan,
                  "exact_t95_high":mean_ci([r["mean_difference_exact"] for r in rr])[2] if all(math.isfinite(r["mean_difference_exact"]) for r in rr) else math.nan})
    write(OUT/"paired_difference_summary.csv",ps)
    for phase,label in ((0,"0deg"),(90,"90deg"),("0+90 pooled equally","phase_average")):
        fig,ax=plt.subplots(figsize=(8.5,5.2))
        for model,color,name in (("Q","#355c9a","Q"),("RO","#bd4b37","Q+RO")):
            ss=sorted((r for r in summary if r["phase_deg"]==phase and r["model"]==model),key=lambda x:x["mu"])
            if not ss:continue
            x=[float(r["mu"]) for r in ss];y=[float(r["P_ret_mean"]) for r in ss]
            low=[max(0.,yy-float(r["replicate_t95_low"])) for yy,r in zip(y,ss)]
            high=[max(0.,float(r["unresolved_upper_t95_high"])-yy) for yy,r in zip(y,ss)]
            ax.errorbar(x,y,yerr=[low,high],marker="o",ms=3,lw=1,capsize=2,label=name,color=color)
        ax.set(xlabel="mass ratio μ",ylabel="finite-time transverse retention, N_RF=100",ylim=(-.04,1.04))
        ax.grid(alpha=.25);ax.legend(frameon=False);ax.set_title(f"Independent scrambled-Sobol mass scan, τ₀={phase}°" if isinstance(phase,int) else "Mass scan, equal average over τ₀=0° and 90°")
        fig.tight_layout();stem=OUT/f"retention_{label}"
        for ext in ("png","pdf","svg"):fig.savefig(stem.with_suffix("."+ext),dpi=190,bbox_inches="tight")
        plt.close(fig)
    print({"trajectories":len(rows),"mass_points":len(mus),"unresolved":sum(r["status"]=="unresolved" for r in rows),
           "summary_rows":len(summary),"paired_rows":len(paired),"plots":3})
if __name__=="__main__":main()
