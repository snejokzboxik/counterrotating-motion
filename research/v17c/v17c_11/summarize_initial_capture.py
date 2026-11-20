"""Create P1 early-loss tables and descriptive maps from frozen N=256 starts."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "p1_early_diagnostics"
GROUPS = ("rest", "weak", "wide", "edge")
CANDIDATES = ("local15", "candidate52")
RADIAL_EDGES = np.asarray([0.0,0.50,0.60,0.70,0.75,0.80,0.85,0.90,1.01])
PHASE_EDGES = np.linspace(0.0,2.0*math.pi,5)


def main():
    cases=[json.loads(line) for line in (DATA/"early_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    samples=np.load(DATA/"early_samples.npz",allow_pickle=False)
    key_to_index={str(key):i for i,key in enumerate(samples["keys"])}
    outdir=HERE/"p1_early_diagnostics"; outdir.mkdir(exist_ok=True)
    radial_rows=[]; phase_rows=[]; diagnostics={}
    for candidate in CANDIDATES:
        diagnostics[candidate]={}
        for group in GROUPS:
            rows=[r for r in cases if r["candidate"]==candidate and r["group"]==group]
            if len(rows)!=256 or len({r["index"] for r in rows})!=256:
                raise RuntimeError(f"expected 256 unique early rows: {candidate}/{group}")
            wall=np.asarray([r["published_category"]=="finite_particle_wall_contact" for r in rows])
            radius=np.asarray([r["initial_max_radius_R"] for r in rows])
            phase=np.asarray([r["rf_phase_rad"] for r in rows])
            rbin=np.clip(np.digitize(radius,RADIAL_EDGES)-1,0,len(RADIAL_EDGES)-2)
            for b in range(len(RADIAL_EDGES)-1):
                m=rbin==b
                radial_rows.append({"candidate":candidate,"group":group,
                    "rmax_lo_R":RADIAL_EDGES[b],"rmax_hi_R":RADIAL_EDGES[b+1],
                    "n":int(m.sum()),"wall":int(wall[m].sum()),
                    "wall_fraction":float(wall[m].mean()) if m.any() else None})
            nphase=len(PHASE_EDGES)-1
            pbins=np.column_stack([np.clip(np.digitize(phase[:,i],PHASE_EDGES)-1,0,nphase-1) for i in range(2)])
            for a in range(nphase):
                for b in range(nphase):
                    m=(pbins[:,0]==a)&(pbins[:,1]==b)
                    phase_rows.append({"candidate":candidate,"group":group,
                        "phase1_bin":a,"phase2_bin":b,
                        "phase1_center_rad":float((PHASE_EDGES[a]+PHASE_EDGES[a+1])/2),
                        "phase2_center_rad":float((PHASE_EDGES[b]+PHASE_EDGES[b+1])/2),
                        "n":int(m.sum()),"wall":int(wall[m].sum()),
                        "wall_fraction":float(wall[m].mean()) if m.any() else None})
            # Outer particle initial radial acceleration and initial radial speed.
            init_summary={"n":256,"wall":int(wall.sum()),"wall_fraction":float(wall.mean()),
                          "wall_event_cycles":{},"radial_bins":[]}
            walls=[r for r in rows if r["event_cycle"] is not None]
            times=np.asarray([r["event_cycle"] for r in walls],float)
            if len(times):
                init_summary["wall_event_cycles"]={"median":float(np.median(times)),
                    "q90":float(np.quantile(times,.9)),"max":float(times.max()),
                    "fraction_lt_0_25":float(np.mean(times<.25)),
                    "fraction_lt_1":float(np.mean(times<1)),"fraction_lt_2":float(np.mean(times<2)),
                    "first_boundary_particle_counts":dict(Counter(r["event_particle_by_final_boundary_gap"] for r in walls))}
            # Outermost body's radial field contributions at t=0, contrasted by result.
            ix=np.asarray([key_to_index[r["key"]] for r in rows],int)
            ar=samples["sample_radial_acceleration_components_and_total"][ix,0]
            vr=[]; outer=[]
            for r in rows:
                state=np.asarray(r["initial_state"])
                radii=np.asarray(r["initial_radius_R"])
                i=int(np.argmax(radii)); j=6*i
                rad=radii[i]
                vr.append((state[j]*state[j+3]+state[j+1]*state[j+4])/rad if rad>1e-12 else np.nan)
                outer.append(i)
            vr=np.asarray(vr); outer=np.asarray(outer)
            for iswall in (False,True):
                m=wall==iswall
                aa=np.asarray([ar[z,outer[z]] for z in range(len(rows))])[m]
                vv=vr[m]
                init_summary["outer_particle_initial_radial_mean_"+("wall" if iswall else "CR")]= {
                    "n":int(m.sum()),"radial_velocity":float(np.nanmean(vv)),
                    "accel_components_RF1_RF2_DC_Coulomb_drag_r_ddot":np.nanmean(aa,axis=0).tolist() if len(aa) else []}
            # Work through contact/end of the 10-cycle window, summed over both bodies.
            work=np.asarray([r["integrated_work_to_early_end_by_particle_RF1_RF2_DC_Coulomb_drag"] for r in rows],float)
            init_summary["mean_total_work_to_early_end_wall_RF1_RF2_DC_Coulomb_drag"] = (
                np.nansum(work[wall],axis=1).mean(axis=0).tolist() if wall.any() else [])
            for b in range(len(RADIAL_EDGES)-1):
                m=rbin==b
                init_summary["radial_bins"].append({"lo":float(RADIAL_EDGES[b]),"hi":float(RADIAL_EDGES[b+1]),
                    "n":int(m.sum()),"wall":int(wall[m].sum()),
                    "rate":float(wall[m].mean()) if m.any() else None})
            diagnostics[candidate][group]=init_summary
    for filename,rows,fields in (
        ("initial_radius_wall_rates.csv",radial_rows,["candidate","group","rmax_lo_R","rmax_hi_R","n","wall","wall_fraction"]),
        ("initial_phase_wall_rates.csv",phase_rows,["candidate","group","phase1_bin","phase2_bin","phase1_center_rad","phase2_center_rad","n","wall","wall_fraction"])):
        with (outdir/filename).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (HERE/"p1_analysis_summary.json").write_text(json.dumps(diagnostics,indent=2)+"\n",encoding="utf-8")
    # Draw 2 x 4 phase-risk panels with sample counts shown per cell.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ri,candidate in enumerate(CANDIDATES):
        for ci,group in enumerate(GROUPS):
            ax=axes[ri,ci]
            selected=[r for r in phase_rows if r["candidate"]==candidate and r["group"]==group]
            nphase=len(PHASE_EDGES)-1
            rate=np.full((nphase,nphase),np.nan); ns=np.zeros((nphase,nphase),int)
            for x in selected:
                rate[x["phase1_bin"],x["phase2_bin"]]=np.nan if x["wall_fraction"] is None else x["wall_fraction"]
                ns[x["phase1_bin"],x["phase2_bin"]]=x["n"]
            im=ax.imshow(rate,origin="lower",vmin=0,vmax=1,cmap="magma",extent=(0,nphase,0,nphase),aspect="auto")
            for a in range(nphase):
                for b in range(nphase):
                    if ns[a,b]: ax.text(b+.5,a+.5,f"{rate[a,b]:.0%}\n{ns[a,b]}",ha="center",va="center",fontsize=6,
                        color="white" if rate[a,b]>.45 else "black")
            ax.set_title(f"{candidate} / {group}")
            ax.set_xlabel("RF phase 2 bin"); ax.set_ylabel("RF phase 1 bin")
            ax.set_xticks(range(nphase));ax.set_yticks(range(nphase))
    fig.colorbar(im,ax=axes.ravel().tolist(),label="wall-contact fraction (all starts in cell)",shrink=.8)
    fig.savefig(outdir/"rf_phase_wall_risk.png",dpi=180)
    plt.close(fig)
    # Radius-conditioned wall rates, separately for each ensemble.
    fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True,sharey=True)
    for ri,candidate in enumerate(CANDIDATES):
        for ci,group in enumerate(GROUPS):
            ax=axes[ri,ci]
            selected=[r for r in radial_rows if r["candidate"]==candidate and r["group"]==group and r["n"]>0]
            x=np.asarray([(r["rmax_lo_R"]+r["rmax_hi_R"])/2 for r in selected])
            y=np.asarray([r["wall_fraction"] for r in selected])
            n=np.asarray([r["n"] for r in selected])
            ax.plot(x,y,"o-",color="#b23a48")
            for xx,yy,nn in zip(x,y,n): ax.annotate(str(nn),(xx,yy),xytext=(0,6),textcoords="offset points",ha="center",fontsize=7)
            ax.set_ylim(0,1);ax.grid(alpha=.25);ax.set_title(f"{candidate} / {group}")
            ax.set_xlabel("initial max(r₁,r₂)/R")
            if ci==0: ax.set_ylabel("wall-contact fraction")
    fig.savefig(outdir/"initial_radius_wall_risk.png",dpi=180)
    plt.close(fig)
    print(json.dumps({"cases":len(cases),"diagnostics":str(HERE/"p1_analysis_summary.json"),
        "maps":["p1_early_diagnostics/rf_phase_wall_risk.png","p1_early_diagnostics/initial_radius_wall_risk.png"]},indent=2))


if __name__=="__main__":main()
