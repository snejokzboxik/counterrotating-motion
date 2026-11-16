"""Summarize paired coarse fixed-voltage mass-ray retention scans."""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write(path,rows):
    if not rows:return
    names=[]
    for row in rows:names.extend(k for k in row if k not in names)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
def wilson(k,n):
    if n==0:return math.nan,math.nan
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)

def main():
    rows=read(BASE/"mass_ray_trajectory_results.csv")
    grouped=defaultdict(list)
    for r in rows:grouped[(r["ray_id"],r["model"],int(r["phase_i"]),float(r["mu"]))].append(r)
    summary=[]
    for (ray,model,phase,mu),rs in sorted(grouped.items()):
        resolved=[r for r in rs if r["status"] in ("retained","escaped")]
        nret=sum(r["status"]=="retained" for r in resolved);lo,hi=wilson(nret,len(resolved))
        exits=[float(r["exit_periods"]) for r in resolved if r["status"]=="escaped"]
        summary.append({"ray_id":ray,"model":model,"phase_deg":90*phase,"mu":mu,
                        "M_over_Q_Th_per_e":3200.04652261404*mu,"n_total":len(rs),
                        "n_resolved":len(resolved),"n_retained":nret,"n_unresolved":len(rs)-len(resolved),
                        "P_ret":nret/len(resolved) if resolved else math.nan,
                        "wilson95_low":lo,"wilson95_high":hi,
                        "median_exit_RF":sorted(exits)[len(exits)//2] if exits else math.nan,
                        "max_exit_RF":max(exits) if exits else math.nan,
                        "max_rho_max":max(float(r["rho_max"]) for r in rs),
                        "estimate_kind":"single seed-131 pilot ensemble; Wilson interval is descriptive, not replicate-QMC uncertainty"})
    write(BASE/"mass_ray_retention_summary.csv",summary)
    idx={(r["ray_id"],int(r["phase_i"]),float(r["mu"]),int(r["state_i"])):r for r in rows if r["model"]=="Q"}
    paired=[]
    for r in rows:
        if r["model"]!="RO":continue
        q=idx.get((r["ray_id"],int(r["phase_i"]),float(r["mu"]),int(r["state_i"])))
        if q is None:continue
        qr=int(q["status"]=="retained");rr=int(r["status"]=="retained")
        paired.append({"ray_id":r["ray_id"],"phase_deg":90*int(r["phase_i"]),"mu":float(r["mu"]),
                       "state_i":int(r["state_i"]),"Q_status":q["status"],"RO_status":r["status"],
                       "RO_minus_Q_retention":rr-qr,
                       "RO_retained_Q_escaped":int(rr==1 and qr==0),
                       "RO_escaped_Q_retained":int(rr==0 and qr==1),
                       "unresolved_pair":int(q["status"]=="unresolved" or r["status"]=="unresolved")})
    write(BASE/"mass_ray_paired_differences.csv",paired)
    pidx=defaultdict(list)
    for r in paired:pidx[(r["ray_id"],r["phase_deg"],r["mu"])].append(r)
    diff=[]
    for key,rs in sorted(pidx.items()):
        resolved=[r for r in rs if r["unresolved_pair"]==0]
        diff.append({"ray_id":key[0],"phase_deg":key[1],"mu":key[2],"n_pairs":len(resolved),
                     "paired_RO_minus_Q_mean":sum(r["RO_minus_Q_retention"] for r in resolved)/len(resolved) if resolved else math.nan,
                     "RO_retained_Q_escaped":sum(r["RO_retained_Q_escaped"] for r in resolved),
                     "RO_escaped_Q_retained":sum(r["RO_escaped_Q_retained"] for r in resolved),
                     "unresolved_pairs":len(rs)-len(resolved),"note":"descriptive paired difference from one common small ensemble"})
    write(BASE/"mass_ray_paired_summary.csv",diff)
    figure_dir=BASE
    for phase in (0.,90.):
        rays=sorted({r["ray_id"] for r in summary})
        fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
        for ax,ray in zip(axs.flat,rays):
            for model,color,label in (("Q","#355c9a","Q"),("RO","#bd4b37","Q+RO")):
                rs=sorted((r for r in summary if r["ray_id"]==ray and r["model"]==model and r["phase_deg"]==phase),key=lambda r:r["mu"])
                if not rs:continue
                x=[r["mu"] for r in rs];y=[r["P_ret"] for r in rs]
                low=[r["P_ret"]-r["wilson95_low"] for r in rs];high=[r["wilson95_high"]-r["P_ret"] for r in rs]
                ax.errorbar(x,y,yerr=[low,high],marker="o",ms=3.3,lw=1.1,capsize=1.7,color=color,label=label)
            ax.set_title(ray.replace("_"," "));ax.grid(alpha=.25);ax.set_ylim(-.05,1.05)
        axs[0,0].set_ylabel("finite-time retained fraction");axs[1,0].set_ylabel("finite-time retained fraction")
        axs[1,0].set_xlabel("mass ratio μ");axs[1,1].set_xlabel("mass ratio μ")
        axs[0,0].legend(frameon=False)
        fig.suptitle(f"V13.3 fixed-voltage mass rays, τ₀={phase:.0f}°; sampled grid points with Wilson intervals (n≤8)")
        fig.tight_layout()
        stem=figure_dir/f"mass_ray_pilot_phase_{int(phase)}deg"
        for ext in ("png","pdf","svg"):fig.savefig(stem.with_suffix("."+ext),dpi=190,bbox_inches="tight")
        plt.close(fig)
    # Explicit peak scan audit: only flag interior local sample maxima/minima, never infer widths.
    peak_rows=[]
    for ray in sorted({r["ray_id"] for r in summary}):
      for model in ("Q","RO"):
       for phase in (0.,90.):
        rs=sorted((r for r in summary if r["ray_id"]==ray and r["model"]==model and r["phase_deg"]==phase),key=lambda r:r["mu"])
        for i in range(1,len(rs)-1):
         if rs[i]["P_ret"]>rs[i-1]["P_ret"] and rs[i]["P_ret"]>rs[i+1]["P_ret"]:
          peak_rows.append({"ray_id":ray,"model":model,"phase_deg":phase,"kind":"interior sampled local maximum",
                            "mu":rs[i]["mu"],"P_ret":rs[i]["P_ret"],"left_mu":rs[i-1]["mu"],
                            "left_P":rs[i-1]["P_ret"],"right_mu":rs[i+1]["mu"],"right_P":rs[i+1]["P_ret"],
                            "FWHM_status":"not estimated from this sparse pilot"})
    write(BASE/"mass_ray_sampled_extrema.csv",peak_rows)
    print(json.dumps({"mass_trajectory_rows":len(rows),"summary_rows":len(summary),"paired_rows":len(paired),
                      "sampled_extrema":len(peak_rows),"figures":[f"mass_ray_pilot_phase_{int(p)}deg" for p in (0.,90.)]},indent=2))

if __name__=="__main__":main()
