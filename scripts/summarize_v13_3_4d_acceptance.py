"""Replicate-level randomized-QMC estimates and plots for V13.3 A4."""
from __future__ import annotations
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
OUT=BASE/"acceptance_4d"
R=.008;OMEGA=2*math.pi*100_000.

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write(path,rows):
    if not rows:return
    fields=[]
    for r in rows:fields.extend(k for k in r if k not in fields)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def t_ci(v):
    v=np.asarray(v,float);m=float(np.mean(v))
    if len(v)<2:return m,math.nan,math.nan
    half=float(student_t.ppf(.975,len(v)-1)*np.std(v,ddof=1)/math.sqrt(len(v)))
    return m,m-half,m+half

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument("--horizon",type=int,default=100);p.add_argument("--directory",default="acceptance_4d");args=p.parse_args()
    if Path(args.directory).name!=args.directory:raise SystemExit("directory must be one folder below the V13.3 atlas")
    OUT=BASE/args.directory
    raw=read(OUT/"trajectory_results.csv")
    rows=[r for r in raw if int(r["horizon"])==args.horizon]
    if not rows:raise SystemExit(f"no trajectories at N={args.horizon}")
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    field_mode=manifest.get("field_mode","legacy_ideal_harmonic")
    volumes={float(k):float(v["D_volume_m4_s2"]) for k,v in manifest["domain_volumes"].items()}
    by=defaultdict(list)
    for r in rows:by[(r["point_id"],r["model"],float(r["mu"]),int(r["phase_i"]),int(r["replica"]))].append(r)
    per_rep=[]
    for (point,model,mu,phase,rep),rs in sorted(by.items()):
        n=len(rs);ret=sum(x["status"]=="retained" for x in rs);unr=sum(x["status"]=="unresolved" for x in rs)
        lo=ret/n;hi=(ret+unr)/n;v=volumes[mu]
        per_rep.append({"point_id":point,"model":model,"mu":mu,"phase_deg":90*phase,"replica":rep,
                        "sobol_seed":manifest["replica_seeds"][rep],"n_total":n,"n_retained":ret,"n_escaped":sum(x["status"]=="escaped" for x in rs),
                        "n_unresolved":unr,"P_ret_lower":lo,"P_ret_upper":hi,
                        "A4_lower_m4_s2":lo*v,"A4_upper_m4_s2":hi*v,
                        "a4_lower":lo*v/(R**4*OMEGA**2),"a4_upper":hi*v/(R**4*OMEGA**2)})
    write(OUT/f"replicate_acceptance_N{args.horizon}.csv",per_rep)
    groups=defaultdict(list)
    for r in per_rep:groups[(r["point_id"],r["model"],r["mu"],r["phase_deg"])].append(r)
    phase_summary=[]
    for (point,model,mu,phase),rs in sorted(groups.items()):
        if len(rs)!=4:raise RuntimeError(f"incomplete QMC replicas at {(point,model,mu,phase)}: {len(rs)}")
        for key in ("P_ret_lower","P_ret_upper","A4_lower_m4_s2","A4_upper_m4_s2","a4_lower","a4_upper"):
            mean,lo,hi=t_ci([r[key] for r in rs]);phase_summary.append({"point_id":point,"model":model,"mu":mu,"phase_deg":phase,"metric":key,"mean":mean,"t95_low":lo,"t95_high":hi,"replicates":4})
    write(OUT/f"phase_acceptance_N{args.horizon}.csv",phase_summary)

    # Uniform RF phase is approximated by the requested four-phase quadrature. Each
    # replicate is first averaged over its four matched phase slices, then t intervals
    # are computed across the four independent scrambles.
    comb=defaultdict(lambda:defaultdict(list))
    for r in per_rep:comb[(r["point_id"],r["model"],r["mu"])][r["replica"]].append(r)
    rep_phase=[]
    for (point,model,mu),repdict in sorted(comb.items()):
        for rep,rs in sorted(repdict.items()):
            if len(rs)!=4:raise RuntimeError(f"missing one of four phases at {(point,model,mu,rep)}")
            rep_phase.append({"point_id":point,"model":model,"mu":mu,"replica":rep,
                              "phase_average_P_lower":float(np.mean([r["P_ret_lower"] for r in rs])),
                              "phase_average_P_upper":float(np.mean([r["P_ret_upper"] for r in rs])),
                              "phase_average_A4_lower_m4_s2":float(np.mean([r["A4_lower_m4_s2"] for r in rs])),
                              "phase_average_A4_upper_m4_s2":float(np.mean([r["A4_upper_m4_s2"] for r in rs])),
                              "phase_average_a4_lower":float(np.mean([r["a4_lower"] for r in rs])),
                              "phase_average_a4_upper":float(np.mean([r["a4_upper"] for r in rs]))})
    write(OUT/f"phase_averaged_replica_acceptance_N{args.horizon}.csv",rep_phase)
    avg=defaultdict(list)
    for r in rep_phase:avg[(r["point_id"],r["model"],r["mu"])].append(r)
    mean_rows=[]
    for key,rs in sorted(avg.items()):
        if len(rs)!=4:raise RuntimeError(f"incomplete replicate set at {key}")
        out={"field_mode":field_mode,"point_id":key[0],"model":key[1],"mu":key[2],"horizon_RF":args.horizon,"replicates":4,
             "phase_quadrature_deg":[0,90,180,270],"domain_volume_m4_s2":volumes[key[2]],
             "uncertainty":"95% Student-t interval across independent scrambled Sobol replicate means; phase average within each replicate"}
        for metric in ("phase_average_P_lower","phase_average_P_upper","phase_average_A4_lower_m4_s2","phase_average_A4_upper_m4_s2","phase_average_a4_lower","phase_average_a4_upper"):
            m,l,h=t_ci([r[metric] for r in rs]);out[metric+"_mean"]=m;out[metric+"_t95_low"]=l;out[metric+"_t95_high"]=h
        mean_rows.append(out)
    write(OUT/f"acceptance_summary_N{args.horizon}.csv",mean_rows)

    # Pair Q and RO within each point, mass, phase, replicate, and launch.
    pair_idx={(r["point_id"],float(r["mu"]),int(r["phase_i"]),int(r["replica"]),int(r["state_i"])):r for r in rows if r["model"]=="Q"}
    pair_rep=defaultdict(list)
    for r in rows:
        if r["model"]!="RO":continue
        q=pair_idx[(r["point_id"],float(r["mu"]),int(r["phase_i"]),int(r["replica"]),int(r["state_i"]))]
        qlo,qhi=(int(q["status"]=="retained"),int(q["status"] in ("retained","unresolved")))
        rlo,rhi=(int(r["status"]=="retained"),int(r["status"] in ("retained","unresolved")))
        pair_rep[(r["point_id"],float(r["mu"]),int(r["replica"]))].append((rlo-qhi,rhi-qlo))
    pair_summary=[]
    for (point,mu,rep),vals in sorted(pair_rep.items()):
        pair_summary.append({"point_id":point,"mu":mu,"replica":rep,
                             "paired_delta_P_lower":float(np.mean([x[0] for x in vals])),
                             "paired_delta_P_upper":float(np.mean([x[1] for x in vals])),
                             "paired_delta_A4_lower_m4_s2":float(np.mean([x[0] for x in vals]))*volumes[mu],
                             "paired_delta_A4_upper_m4_s2":float(np.mean([x[1] for x in vals]))*volumes[mu],
                             "phase_slices_and_common_pairs":len(vals)})
    write(OUT/f"paired_acceptance_replica_N{args.horizon}.csv",pair_summary)
    pair_group=defaultdict(list)
    for r in pair_summary:pair_group[(r["point_id"],r["mu"])].append(r)
    paired_out=[]
    for (point,mu),rs in sorted(pair_group.items()):
        if len(rs)!=4:raise RuntimeError(f"incomplete paired replicas at {(point,mu)}")
        out={"field_mode":field_mode,"point_id":point,"mu":mu,"horizon_RF":args.horizon,"replicates":4,
             "method":"paired RO−Q on same Sobol starts/phases; unresolved statuses widen per-pair bounds"}
        for metric in ("paired_delta_P_lower","paired_delta_P_upper","paired_delta_A4_lower_m4_s2","paired_delta_A4_upper_m4_s2"):
            m,l,h=t_ci([r[metric] for r in rs]);out[metric+"_mean"]=m;out[metric+"_t95_low"]=l;out[metric+"_t95_high"]=h
        paired_out.append(out)
    write(OUT/f"paired_acceptance_summary_N{args.horizon}.csv",paired_out)

    # Joint selectivity/acceptance metrics. Ratios use ratios of replicate means,
    # with a paired bootstrap over the four independent scrambles.
    qmeans={(r["point_id"],float(r["mu"])):r for r in mean_rows if r["model"]=="Q"}
    romeans={(r["point_id"],float(r["mu"])):r for r in mean_rows if r["model"]=="RO"}
    pidx={(r["point_id"],float(r["mu"])):r for r in paired_out}
    qrep=defaultdict(dict);rorep=defaultdict(dict);drep=defaultdict(dict)
    for r in rep_phase:
        key=(r["point_id"],float(r["mu"]))
        target=qrep if r["model"]=="Q" else rorep
        target[key][int(r["replica"])]=r
    for r in pair_summary:drep[(r["point_id"],float(r["mu"]))][int(r["replica"])]=r
    rng=np.random.default_rng(20261116)
    trade=[]
    for key in sorted(qmeans):
        q=qmeans[key];ro=romeans[key];pair=pidx[key]
        qlo=float(q["phase_average_A4_lower_m4_s2_mean"]);qhi=float(q["phase_average_A4_upper_m4_s2_mean"])
        rlo=float(ro["phase_average_A4_lower_m4_s2_mean"]);rhi=float(ro["phase_average_A4_upper_m4_s2_mean"])
        ratio_lo=rlo/qhi if qhi>0 else math.nan
        ratio_hi=rhi/qlo if qlo>0 else math.nan
        bootstrap=[]
        if all(int(r["n_unresolved"])==0 for r in per_rep if r["point_id"]==key[0] and float(r["mu"])==key[1]):
            qv=np.array([float(qrep[key][i]["phase_average_A4_lower_m4_s2"]) for i in range(4)])
            rv=np.array([float(rorep[key][i]["phase_average_A4_lower_m4_s2"]) for i in range(4)])
            for _ in range(20000):
                ix=rng.integers(0,4,4);den=float(np.mean(qv[ix]))
                if den>0:bootstrap.append(float(np.mean(rv[ix])/den))
        ratio_boot_low,ratio_boot_high=(np.quantile(bootstrap,[.025,.975]).tolist() if bootstrap else [math.nan,math.nan])
        trade.append({"field_mode":field_mode,"point_id":key[0],"mu":key[1],"horizon_RF":args.horizon,
                      "Q_P_ret":float(q["phase_average_P_lower_mean"]),
                      "RO_P_ret":float(ro["phase_average_P_lower_mean"]),
                      "paired_RO_minus_Q_P_ret_lower_mean":float(pair["paired_delta_P_lower_mean"]),
                      "paired_RO_minus_Q_P_ret_upper_mean":float(pair["paired_delta_P_upper_mean"]),
                      "Q_A4_lower_mean_m4_s2":qlo,"Q_A4_upper_mean_m4_s2":qhi,
                      "RO_A4_lower_mean_m4_s2":rlo,"RO_A4_upper_mean_m4_s2":rhi,
                      "A4_ratio_lower":ratio_lo,"A4_ratio_upper":ratio_hi,
                      "A4_ratio_paired_bootstrap_t95_low":ratio_boot_low,
                      "A4_ratio_paired_bootstrap_t95_high":ratio_boot_high,
                      "paired_delta_A4_lower_mean_m4_s2":float(pair["paired_delta_A4_lower_m4_s2_mean"]),
                      "paired_delta_A4_upper_mean_m4_s2":float(pair["paired_delta_A4_upper_m4_s2_mean"]),
                      "ratio_method":"ratio of QMC replicate means; paired bootstrap resamples independent Sobol scrambles",
                      "domain":"uniform-area position disk r<=7.5 mm; uniform-area velocity disk E_trans<=5 eV at each mass",
                      "status":"unresolved bounds included" if ratio_lo!=ratio_hi else "resolved"})
    write(OUT/f"selectivity_acceptance_tradeoff_N{args.horizon}.csv",trade)

    # Plot physical 4D acceptance. Lines join sampled mass nodes only as visual guides.
    point_ids=sorted({r["point_id"] for r in mean_rows})
    ncols=min(3,max(1,len(point_ids)));nrows=math.ceil(len(point_ids)/ncols)
    fig,axs=plt.subplots(nrows,ncols,figsize=(5.6*ncols,4.1*nrows),sharex=True,squeeze=False)
    flat_axes=axs.ravel()
    for ax,point in zip(flat_axes,point_ids):
        for model,color,label in (("Q","#355c9a","Q"),("RO","#bd4b37","Q+RO")):
            z=sorted((r for r in mean_rows if r["point_id"]==point and r["model"]==model),key=lambda r:r["mu"])
            x=[r["mu"] for r in z];y=[r["phase_average_A4_lower_m4_s2_mean"] for r in z]
            low=[max(0.,y[i]-z[i]["phase_average_A4_lower_m4_s2_t95_low"]) for i in range(len(z))]
            high=[max(0.,z[i]["phase_average_A4_upper_m4_s2_t95_high"]-y[i]) for i in range(len(z))]
            ax.errorbar(x,y,yerr=[low,high],marker="o",ms=3.2,lw=1.,capsize=2,color=color,label=label)
        ax.set_title(point.replace("_"," "));ax.grid(alpha=.25)
    for ax in flat_axes[len(point_ids):]:ax.set_visible(False)
    for i,ax in enumerate(flat_axes[:len(point_ids)]):
        if i%ncols==0:ax.set_ylabel("A₄, m⁴ s⁻² (phase quadrature)")
        if i//ncols==nrows-1:ax.set_xlabel("μ")
    if point_ids:flat_axes[0].legend(frameon=False)
    fig.suptitle(f"V13.3 4D acceptance ({field_mode}), N={args.horizon} RF; replicate-level scrambled-Sobol 95% t intervals")
    fig.tight_layout()
    for ext in ("png","pdf","svg"):fig.savefig(OUT/f"four_dimensional_acceptance_N{args.horizon}.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,5.5))
    plot_rows=[r for r in trade if math.isfinite(r["A4_ratio_lower"])]
    if plot_rows:
        pts=ax.scatter([r["paired_RO_minus_Q_P_ret_lower_mean"] for r in plot_rows],
                       [0.5*(r["A4_ratio_lower"]+r["A4_ratio_upper"]) for r in plot_rows],
                       c=[r["mu"] for r in plot_rows],cmap="viridis",s=55,edgecolor="black",linewidth=.4)
        fig.colorbar(pts,ax=ax,label="mass ratio μ")
        for r in plot_rows:ax.annotate(r["point_id"].replace("ro_",""),
                                       (r["paired_RO_minus_Q_P_ret_lower_mean"],
                                        .5*(r["A4_ratio_lower"]+r["A4_ratio_upper"])),
                                       fontsize=6,xytext=(3,3),textcoords="offset points")
    ax.axvline(0,color="black",lw=.8,alpha=.6);ax.axhline(1,color="black",lw=.8,alpha=.6)
    ax.set(xlabel="paired change in finite-time retention, Q+RO − Q",
           ylabel="4D acceptance ratio, A₄(Q+RO) / A₄(Q)",
           title=f"V13.3 retention–acceptance tradeoff ({field_mode}), N={args.horizon} RF")
    ax.grid(alpha=.25);fig.tight_layout()
    for ext in ("png","pdf","svg"):fig.savefig(OUT/f"selectivity_acceptance_tradeoff_N{args.horizon}.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"trajectories":len(rows),"replicate_rows":len(per_rep),"mass_summary_rows":len(mean_rows),
                      "paired_summary_rows":len(paired_out),"horizon":args.horizon},indent=2))

if __name__=="__main__":main()
