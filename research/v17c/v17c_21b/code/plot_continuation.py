"""Publication-style plots built only from saved V17C.21.2 result tables."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE / "figures"
TABLES = BASE / "tables"
OUT.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)
COLORS = {"CR_PLUS_MINUS": "#147d64", "CR_MINUS_PLUS": "#008da8",
          "CO_PLUS_PLUS": "#d07a00", "CO_MINUS_MINUS": "#b34a4a",
          "BOUND_NONROTATING": "#7867a8", "MIXED_OR_NEAR_ZERO": "#737373"}
PARAMS = ("A", "B", "nu", "beta")

def load_rows(name):
    d=json.loads((BASE/"results"/f"continuation_{name}.json").read_text(encoding="utf-8"))
    rows=[d["baseline"]]+[r for seq in d["directions"].values() for r in seq if r.get("status")=="VERIFIED_ROOT"]
    rows.sort(key=lambda r:r["factor_from_nominal"])
    return d, rows

def export_csv():
    p=TABLES/"m3_1d_continuation.csv"
    fields=["parameter","factor","value","classification","status","r1_R","r2_R","omega1_over_Omega1","omega2_over_Omega1","f1_Hz","f2_Hz","z1_R","z2_R","balance_residual_inf","HB_residual_1","HB_residual_2","wall_time_s"]
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name in PARAMS:
            _,rows=load_rows(name)
            for r in rows:
                root=r["root"]; inn=r.get("inner",[{},{}])
                w.writerow(dict(parameter=name,factor=r.get("factor_from_nominal"),value=r.get("value"),classification=r.get("classification"),status=r.get("status"),r1_R=root["radii_R"][0],r2_R=root["radii_R"][1],omega1_over_Omega1=root["omega_over_Omega1"][0],omega2_over_Omega1=root["omega_over_Omega1"][1],f1_Hz=root["frequency_Hz"][0],f2_Hz=root["frequency_Hz"][1],z1_R=root["mean_z_R"][0],z2_R=root["mean_z_R"][1],balance_residual_inf=r.get("scaled_balance_max_norm"),HB_residual_1=inn[0].get("relative_nonlinear_residual"),HB_residual_2=inn[1].get("relative_nonlinear_residual"),wall_time_s=r.get("wall_time_s")))
    return p

def fig01():
    fig,axs=plt.subplots(2,2,figsize=(9.2,6.2),sharex="col",constrained_layout=True)
    for col,name in enumerate(PARAMS):
        d,rows=load_rows(name); axr=axs[0,col%2]; axw=axs[1,col%2]
        # Two panels per parameter in a 2x2 compact grid is not possible for all four,
        # so each column stores a separate parameter subplot pair in one 4x2 canvas.
    plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(9.2,10.2),constrained_layout=True,sharex="col")
    for i,name in enumerate(PARAMS):
        d,rows=load_rows(name); x=np.array([r["factor_from_nominal"] for r in rows])
        r1=np.array([r["root"]["radii_R"][0] for r in rows]); r2=np.array([r["root"]["radii_R"][1] for r in rows])
        w1=np.array([r["root"]["omega_over_Omega1"][0] for r in rows]); w2=np.array([r["root"]["omega_over_Omega1"][1] for r in rows])
        axs[i,0].plot(x,r1,"o",color="#225ea8",ms=4,label=r"$r_1/R$")
        axs[i,0].plot(x,r2,"s",color="#d95f0e",ms=4,label=r"$r_2/R$")
        axs[i,1].plot(x,w1,"o",color="#225ea8",ms=4,label=r"$\omega_1/\Omega_1$")
        axs[i,1].plot(x,w2,"s",color="#d95f0e",ms=4,label=r"$\omega_2/\Omega_1$")
        axs[i,1].axhline(0,color="black",lw=.7)
        for ax in axs[i]:
            ax.axvline(1,color="#777",lw=.6,ls=":"); ax.grid(alpha=.2); ax.tick_params(labelsize=8)
        axs[i,0].set_ylabel(f"{name} × nominal\n$r_i/R$",fontsize=9)
        axs[i,1].set_ylabel(r"$\omega_i/\Omega_1$",fontsize=9)
        axs[i,0].legend(frameon=False,ncol=2,fontsize=8,loc="best")
        axs[i,1].legend(frameon=False,ncol=2,fontsize=8,loc="best")
    axs[-1,0].set_xlabel("parameter / nominal value"); axs[-1,1].set_xlabel("parameter / nominal value")
    fig.suptitle("V17C.21.2 — continuation of re-solved M3 equilibria\n96×12×12 Fourier grid; points are computed nodes",fontsize=11)
    for ext in ("pdf","svg"): fig.savefig(OUT/f"FIG01_m3_1d_continuation.{ext}")
    fig.savefig(OUT/"FIG01_m3_1d_continuation.png",dpi=600)
    plt.close(fig)

def fig02():
    data=[]
    for name in ("A","nu","beta"):
        d=json.loads((BASE/"results"/f"event_omega1_zero_{name}.json").read_text(encoding="utf-8"))
        for row in d.get("iterations",[]):
            if row.get("status")=="VERIFIED_ROOT":
                data.append((name,row["factor_from_nominal"],row["root"]["omega_over_Omega1"][0]))
    if not data: return
    fig,axs=plt.subplots(1,3,figsize=(9.2,3.1),constrained_layout=True)
    for ax,name in zip(axs,("A","nu","beta")):
        q=[r for r in data if r[0]==name]
        if q: ax.plot([r[1] for r in q],[r[2] for r in q],"o",ms=4,color="#225ea8")
        ax.axhline(0,color="black",lw=.8); ax.grid(alpha=.2); ax.set_title(name); ax.set_xlabel("parameter / nominal")
    axs[0].set_ylabel(r"$\omega_1/\Omega_1$ (M3 root)")
    fig.suptitle("Zero-spin event refinement: verified M3 roots only",fontsize=10)
    for ext in ("pdf","svg"): fig.savefig(OUT/f"FIG02_m3_zero_spin_refinement.{ext}")
    fig.savefig(OUT/"FIG02_m3_zero_spin_refinement.png",dpi=600); plt.close(fig)

def fig03():
    maps=[("A_B","A/A₀","B/B₀"),("A_beta","A/A₀","β/β₀"),("B_nu","B/B₀","ν/ν₀")]
    fig,axs=plt.subplots(1,3,figsize=(9.2,3.25),constrained_layout=True)
    markers={"CR_PLUS_MINUS":"o","CR_MINUS_PLUS":"s","CO_PLUS_PLUS":"^","CO_MINUS_MINUS":"v","BOUND_NONROTATING":"D","MIXED_OR_NEAR_ZERO":"P"}
    for ax,(mid,xlab,ylab) in zip(axs,maps):
        for path in sorted((BASE/"results").glob(f"map_{mid}_y*.json")):
            d=json.loads(path.read_text(encoding="utf-8"));yf=d["y_factor"]
            for row in d.get("points",[]):
                x=row.get("x_factor")
                if x is None: continue
                cls=row.get("classification") if row.get("status")=="VERIFIED_ROOT" else None
                if cls is None:
                    ax.scatter(x,yf,marker="x",s=42,color="#666666",linewidths=1.2,zorder=3)
                else:
                    ax.scatter(x,yf,marker=markers.get(cls,"o"),s=52,color=COLORS.get(cls,"#666666"),edgecolor="black",linewidth=.35,zorder=3)
        ax.set_xlabel(xlab);ax.set_ylabel(ylab);ax.set_title(mid.replace("_","–"),fontsize=10)
        ax.grid(alpha=.2);ax.tick_params(labelsize=8)
    handles=[]
    from matplotlib.lines import Line2D
    for cls,label in (("CR_PLUS_MINUS","CR (+−)"),("CO_MINUS_MINUS","CO (−−)")):
        handles.append(Line2D([0],[0],marker=markers[cls],color="none",markerfacecolor=COLORS[cls],markeredgecolor="black",label=label,markersize=7))
    handles.append(Line2D([0],[0],marker="x",color="#666",label="unresolved",linestyle="none",markersize=7))
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,-.08),fontsize=8)
    fig.suptitle("Sparse M3 branch map: local root class at computed nodes; no interpolation",fontsize=10)
    for ext in ("pdf","svg"):fig.savefig(OUT/f"FIG03_sparse_m3_maps.{ext}",bbox_inches="tight")
    fig.savefig(OUT/"FIG03_sparse_m3_maps.png",dpi=600,bbox_inches="tight");plt.close(fig)

def fig04():
    p=BASE/"results"/"full_rf_validation.json"
    if not p.exists(): return
    d=json.loads(p.read_text(encoding="utf-8")); by={r["label"]+"_"+str(r["numerics"]["steps_per_RF1_period"]):r for r in d["rows"]}
    labels=[("nominal RK4-64","M3_CR_nominal_64"),("nominal RK4-96","M3_CR_nominal_96"),("A/A₀=0.95 RK4-64","M3_CR_A095_long_64")]
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.7),constrained_layout=True)
    colors=("#225ea8","#d95f0e")
    for particle in (0,1):
        for idx,(pretty,key) in enumerate(labels):
            row=by.get(key)
            if not row: continue
            root=row["m3_root"]
            x_m3=root["radii_R"][particle] if idx==2 else (0.5550374649008335 if particle==0 else 0.7513382762325129)
            x_rf=row["mean_radius_R"][particle]
            y_m3=root["frequency_Hz"][particle]
            y_rf=row["frequency_Hz_from_turns"][particle]
            xpos=idx+(particle-.5)*.16
            axs[0].plot([xpos-.035,xpos+.035],[x_m3,x_rf],color=colors[particle],lw=1.4)
            axs[0].scatter([xpos-.035],[x_m3],marker="x",s=35,color=colors[particle],zorder=3)
            axs[0].scatter([xpos+.035],[x_rf],marker="o",s=30,color=colors[particle],zorder=3)
            axs[1].plot([xpos-.035,xpos+.035],[y_m3,y_rf],color=colors[particle],lw=1.4)
            axs[1].scatter([xpos-.035],[y_m3],marker="x",s=35,color=colors[particle],zorder=3)
            axs[1].scatter([xpos+.035],[y_rf],marker="o",s=30,color=colors[particle],zorder=3)
    for ax in axs:
        ax.set_xticks(range(3),[x[0] for x in labels],rotation=12,ha="right");ax.grid(axis="y",alpha=.2);ax.tick_params(labelsize=8)
    axs[0].set_ylabel(r"mean radius $r_i/R$");axs[1].set_ylabel("mean orbital frequency [Hz]")
    axs[0].set_title("M3 (×) vs full RF RK4 (○): radii")
    axs[1].set_title("M3 (×) vs full RF RK4 (○): frequencies")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0],[0],color=colors[0],marker="o",label="particle 1"),Line2D([0],[0],color=colors[1],marker="o",label="particle 2")], ["particle 1","particle 2"],loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,-.12),fontsize=8)
    fig.suptitle("Independent full RF trajectory comparison; point starts at M3 circular means",fontsize=10)
    for ext in ("pdf","svg"):fig.savefig(OUT/f"FIG04_m3_fullrf_comparison.{ext}",bbox_inches="tight")
    fig.savefig(OUT/"FIG04_m3_fullrf_comparison.png",dpi=600,bbox_inches="tight");plt.close(fig)

def fig05():
    p=BASE/"results"/"full_rf_validation.json"
    if not p.exists(): return
    d=json.loads(p.read_text(encoding="utf-8")); wanted=["M3_CR_nominal_rk464_15000p","M3_CO_A_low_rk464_15000p"]
    fig=plt.figure(figsize=(8.5,4.1),constrained_layout=True);ax=fig.add_subplot(111,projection="3d")
    colors=("#147d64","#b34a4a")
    for label,color in zip(wanted,colors):
        path=BASE/"results"/(label+".npz")
        if not path.exists():continue
        with np.load(path) as f:states=f["states"]
        for i,(j,name,ls) in enumerate(((0,"particle 1","-"),(6,"particle 2","--"))):
            ax.plot(states[:,j],states[:,j+1],states[:,j+2],color=color,ls=ls,lw=1.05,label=f"{label.split('_rk4')[0]} — {name}")
    ax.set_xlabel("x/R");ax.set_ylabel("y/R");ax.set_zlabel("z/R");ax.view_init(elev=25,azim=36);ax.legend(frameon=False,fontsize=7,loc="upper left")
    ax.set_title("Sampled full RF trajectories: CR and CO predictions",fontsize=10)
    for ext in ("pdf","svg"):fig.savefig(OUT/f"FIG05_fullrf_trajectories.{ext}",bbox_inches="tight")
    fig.savefig(OUT/"FIG05_fullrf_trajectories.png",dpi=600,bbox_inches="tight");plt.close(fig)

def fig06():
    p=BASE/"results"/"selected_m3_stability.json"
    if not p.exists():return
    d=json.loads(p.read_text(encoding="utf-8"))
    fig,ax=plt.subplots(figsize=(6.4,4.2),constrained_layout=True)
    palette={"A":"#225ea8","B":"#d95f0e","nu":"#147d64","beta":"#8c6bb1","nominal":"#333333"}
    for row in d.get("selected_points",[]):
        slow=row.get("reduced_slowflow")
        if not slow:continue
        vals=np.asarray(slow["eigenvalues_real_imag"],dtype=float)
        label=f'{row["parameter_label"]} × {row["factor_from_nominal"]:.3g}'
        ax.scatter(vals[:,0],vals[:,1],s=25,alpha=.75,color=palette.get(row["parameter_label"],"#777777"),
                   label=label,edgecolors="none")
    ax.axvline(0,color="black",lw=.8);ax.axhline(0,color="#888888",lw=.5)
    ax.grid(alpha=.2);ax.set_xlabel(r"$\operatorname{Re}\lambda$ [$\tau^{-1}$]")
    ax.set_ylabel(r"$\operatorname{Im}\lambda$ [$\tau^{-1}$]")
    ax.set_title("Selected 10D quasi-steady M3 slow-flow spectra\nnot full-RF Lyapunov/Floquet exponents",fontsize=10)
    handles,labels=ax.get_legend_handles_labels()
    # Keep the legend useful without listing repeated points for every eigenvalue.
    unique={label:handle for handle,label in zip(handles,labels)}
    ax.legend(unique.values(),unique.keys(),frameon=False,fontsize=7,ncol=2,loc="best")
    for ext in ("pdf","svg"):fig.savefig(OUT/f"FIG06_m3_slowflow_spectra.{ext}",bbox_inches="tight")
    fig.savefig(OUT/"FIG06_m3_slowflow_spectra.png",dpi=600,bbox_inches="tight");plt.close(fig)

def fig07():
    p=BASE/"results"/"full_rf_validation.json"
    if not p.exists():return
    d=json.loads(p.read_text(encoding="utf-8"))
    chosen=[r for r in d["rows"] if r["label"] in ("M3_A_sign_before","M3_A_sign_after","M3_CO_A_low","M3_CR_A095_long")]
    if not chosen:return
    fig,ax=plt.subplots(figsize=(8.1,4.0),constrained_layout=True)
    for idx,row in enumerate(chosen):
        m3=row["m3_root"]["frequency_Hz"][0];rf=row["frequency_Hz_from_turns"][0]
        ax.plot([idx-.08,idx+.08],[m3,rf],color="#777777",lw=1)
        ax.scatter(idx-.08,m3,marker="x",s=42,color="#225ea8",zorder=3)
        cls=row["classification"]
        color={"CR_PLUS_MINUS":"#147d64","CO_MINUS_MINUS":"#b34a4a","UNRESOLVED":"#777777"}.get(cls,"#777777")
        ax.scatter(idx+.08,rf,marker="o",s=34,color=color,zorder=3)
    ax.axhline(0,color="black",lw=.8);ax.grid(axis="y",alpha=.2)
    class_label={"CR_PLUS_MINUS":"CR (+−)","CO_MINUS_MINUS":"CO (−−)","UNRESOLVED":"unresolved"}
    ax.set_xticks(range(len(chosen)),[r["label"].replace("M3_","")+"\n"+class_label.get(r["classification"],r["classification"]) for r in chosen],fontsize=7)
    ax.set_ylabel(r"particle 1 mean frequency [Hz]")
    ax.set_title("Predicted M3 frequency and independent full-RF measurement\nnear the A-controlled zero-spin event",fontsize=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker="x",linestyle="none",color="#225ea8",label="M3 root"),
                       Line2D([],[],marker="o",linestyle="none",color="#147d64",label="RF: CR"),
                       Line2D([],[],marker="o",linestyle="none",color="#b34a4a",label="RF: CO"),
                       Line2D([],[],marker="o",linestyle="none",color="#777777",label="RF: unresolved")],
              frameon=False,ncol=4,fontsize=7,loc="best")
    for ext in ("pdf","svg"):fig.savefig(OUT/f"FIG07_fullrf_boundary_checks.{ext}",bbox_inches="tight")
    fig.savefig(OUT/"FIG07_fullrf_boundary_checks.png",dpi=600,bbox_inches="tight");plt.close(fig)

def fig08():
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.4),constrained_layout=True)
    for i,name in enumerate(PARAMS):
        d,rows=load_rows(name)
        x=np.array([r["factor_from_nominal"] for r in rows])
        f=np.array([r["root"]["frequency_Hz"] for r in rows])
        orbit=np.max(np.abs(f),axis=1)/50.0
        axs[0].plot(x,orbit,"o",ms=3.5,label=name)
        inner=d["baseline"].get("inner",[{},{}])
        if i==0:
            for r in rows:
                inn=r.get("inner",[{},{}])
                for j,color in ((0,"#225ea8"),(1,"#d95f0e")):
                    v=inn[j].get("rms_micromotion",np.nan)/max(inn[j].get("mean_speed",np.nan),1e-30)
                    axs[1].scatter(r["factor_from_nominal"],v,s=16,color=color,alpha=.8)
    axs[0].set_yscale("log");axs[0].set_xlabel("parameter / nominal");axs[0].set_ylabel(r"$\max_i|f_i|/f_1$")
    axs[0].set_title("Orbital-to-RF frequency ratio");axs[0].legend(frameon=False,ncol=2,fontsize=7)
    axs[1].set_xlabel("A / A₀");axs[1].set_ylabel("RMS micromotion / mean speed")
    axs[1].set_title("Nominal A continuation: micro/slow velocity scale")
    for ax in axs:ax.grid(alpha=.2);ax.tick_params(labelsize=8)
    fig.suptitle("Measured scale-separation proxies; combination resonances need separate checks",fontsize=10)
    for ext in ("pdf","svg"):fig.savefig(OUT/f"FIG08_adiabatic_scale_proxies.{ext}",bbox_inches="tight")
    fig.savefig(OUT/"FIG08_adiabatic_scale_proxies.png",dpi=600,bbox_inches="tight");plt.close(fig)

if __name__=="__main__":
    export_csv(); fig01(); fig02(); fig03(); fig04(); fig05(); fig06(); fig07(); fig08()
