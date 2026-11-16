"""Plot only directly simulated adaptive V13.3 refinement nodes and intervals."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument("--no-long",action="store_true");args=p.parse_args()
    summary=read(BASE/"adaptive_refinement_summary.csv")
    candidates=read(BASE/"adaptive_candidate_points.csv")
    cidx={(round(float(r["alpha_i"]),8),round(float(r["beta_i"]),8),int(r["eta_i"])):r for r in candidates}
    for phase in (0.,90.):
      rs=[r for r in summary if float(r["phase_deg"])==phase and int(r["horizon_RF"])==100]
      fig,axs=plt.subplots(3,3,figsize=(13.4,10.1),sharex=True,sharey=True)
      vals=[float(r["paired_RO_minus_Q_rep_mean"]) for r in rs]
      norm=TwoSlopeNorm(vmin=-1.,vcenter=0.,vmax=1.)
      sc=None
      for ax,eta_i in zip(axs.flat,range(9)):
        group=[r for r in rs if int(r["eta_i"])==eta_i]
        if not group:continue
        for r in group:
          key=(round(float(r["alpha_i"]),8),round(float(r["beta_i"]),8),eta_i)
          point=cidx.get(key,{});half=str(point.get("selection_reason","")).startswith("actual half-cell")
          d=float(r["paired_RO_minus_Q_rep_mean"]);lo=float(r["paired_QMC_t95_low"]);hi=float(r["paired_QMC_t95_high"])
          excludes=math.isfinite(lo) and math.isfinite(hi) and (lo>0 or hi<0)
          sc=ax.scatter(float(r["beta"]),float(r["alpha"]),c=[d],cmap="coolwarm",norm=norm,
                        marker="^" if half else "o",s=54 if excludes else 34,
                        edgecolors="black" if excludes else "#4b5563",linewidths=.8 if excludes else .35,
                        alpha=.95 if excludes else .68)
        eta=float(group[0]["eta"])
        ax.set_title(f"η={eta:+.3f}; κ=ηβ")
        ax.grid(alpha=.2)
      axs[1,0].set_ylabel("α");axs[2,0].set_ylabel("α")
      for ax in axs[2,:]:ax.set_xlabel("β")
      fig.colorbar(sc,ax=list(axs.flat),label="paired Q+RO − Q retention; t 95% across four scrambles",shrink=.82)
      fig.suptitle(f"V13.3 adaptive refinement, N=100 RF, τ₀={phase:.0f}°; directly simulated nodes only\nfilled outlines: replicate t interval excludes zero; triangles: half-cell refinement")
      fig.tight_layout()
      stem=BASE/f"adaptive_refinement_difference_phase_{int(phase)}deg_N100"
      for ext in ("png","pdf","svg"):fig.savefig(stem.with_suffix("."+ext),dpi=190,bbox_inches="tight")
      plt.close(fig)
    if not args.no_long:
      rs=[r for r in summary if int(r["horizon_RF"]) in (100,300,1000) and float(r["phase_deg"]) in (0.,90.)]
      ids=sorted(f"a{r['alpha_i']}_b{r['beta_i']}_e{r['eta_i']}"
                 for r in candidates if r.get("long_horizon_selected","").lower()=="true")
      fig,axs=plt.subplots(2,2,figsize=(13,8),sharey=True)
      for ax,cid in zip(axs.flat,ids[:4]):
        for model,mean_key,lo_key,hi_key,color,model_label in (
          ("RO","RO_P_ret_rep_mean","RO_QMC_t95_low","RO_QMC_t95_high","#bd4b37","Q+RO"),
          ("Q","Q_P_ret_rep_mean","Q_QMC_t95_low","Q_QMC_t95_high","#355c9a","Q")):
          for phase,linestyle,marker in ((0.,"-","o"),(90.,"--","s")):
            data=[r for r in rs if r["candidate_id"]==cid and float(r["phase_deg"])==phase]
            data=sorted(data,key=lambda r:int(r["horizon_RF"]))
            if data:
              x=[int(r["horizon_RF"]) for r in data];y=[float(r[mean_key]) for r in data]
              low=[max(0.,y[i]-float(r[lo_key])) if math.isfinite(float(r[lo_key])) else 0. for i,r in enumerate(data)]
              high=[max(0.,float(r[hi_key])-y[i]) if math.isfinite(float(r[hi_key])) else 0. for i,r in enumerate(data)]
              label=f"{model_label}, τ₀={phase:.0f}°"
              ax.errorbar(x,y,yerr=[low,high],marker=marker,ms=4,lw=1,ls=linestyle,
                          capsize=2,color=color,label=label)
        ax.set_xscale("log");ax.set_xticks([100,300,1000],["100","300","1000"])
        ax.set_ylim(-.05,1.05);ax.grid(alpha=.25);ax.set_title(cid.replace("_"," "))
      axs[0,0].set_ylabel("finite-time retention");axs[1,0].set_ylabel("finite-time retention")
      axs[1,0].set_xlabel("RF periods");axs[1,1].set_xlabel("RF periods");axs[0,0].legend(frameon=False)
      fig.suptitle("V13.3 adaptive long horizons at τ₀=0° and 90°; independent scramble t intervals")
      fig.tight_layout()
      for ext in ("png","pdf","svg"):fig.savefig(BASE/f"adaptive_long_horizon_N100_300_1000.{ext}",dpi=190,bbox_inches="tight")
      plt.close(fig)
    metadata={"source":"adaptive_refinement_summary.csv","nodes":"directly integrated only; no spatial interpolation",
              "uncertainty":"Student-t 95% interval across four independent scrambled-Sobol replicate means",
              "marker_key":"triangles are half-cell points; filled outlines mean difference interval excludes zero",
              "phases_deg":[0,90],"long_horizons_included":not args.no_long}
    (BASE/"adaptive_refinement_plot_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metadata,indent=2))
if __name__=="__main__":main()
