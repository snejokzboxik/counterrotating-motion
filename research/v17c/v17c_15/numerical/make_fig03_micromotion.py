"""FIG03: RF-resolved micromotion and averaged torque budget for Local15 S01."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"results/local15_exact_rf/micromotion"
SUMMARY=json.loads((DATA/"summary.json").read_text(encoding="utf-8"))
COLORS={"species1":"#126782","species2":"#b74f22"}
LABELS={"species1":"species 1","species2":"species 2"}
period=2*np.pi
plot_lo,plot_hi=400,420
stride=1
series=[]
arrays={}
for name in ("species1","species2"):
    d=np.load(DATA/f"{name}_S01_1000periods.npz")
    nperiod=(d["tau"]-d["tau"][0])/period
    mask=(nperiod>=plot_lo)&(nperiod<=plot_hi)
    idx=np.flatnonzero(mask)[::stride]
    arrays[name]=(d,nperiod,idx)
    for j in idx:
        series.append({"species":name,"primary_period":float(nperiod[j]),
          "rho":float(d["rho"][j]),"instantaneous_omega":float(d["omega"][j]),
          "torque_mode1":float(d["torque_mode1"][j]),
          "torque_mode2":float(d["torque_mode2"][j]),
          "drag_torque":float(d["drag_torque"][j])})
with (ROOT/"results/figure_data/fig03_micromotion_series.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(series[0]));w.writeheader();w.writerows(series)
torques=[]
for name in ("species1","species2"):
    for key in ("mean_rf_torque_mode1","mean_rf_torque_mode2","mean_drag_torque"):
        torques.append({"species":name,"component":key,
            "mean_torque":SUMMARY["species"][name][key]})
with (ROOT/"results/figure_data/fig03_torque_means.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(torques[0]));w.writeheader();w.writerows(torques)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":9,
 "axes.titlesize":9,"mathtext.fontset":"dejavusans","pdf.fonttype":42,
 "ps.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,
 "axes.spines.right":False})
fig,axs=plt.subplots(3,1,figsize=(7.0,6.2),constrained_layout=True,
    gridspec_kw={"height_ratios":[1,1.15,1.05]})
ax=axs[0]
for name,(d,nperiod,idx) in arrays.items():
    ax.plot(nperiod[idx],d["rho"][idx],lw=.65,color=COLORS[name],label=LABELS[name])
ax.set_ylabel("$r/R$")
ax.set_xlim(plot_lo,plot_hi)
ax.set_title("(a) RF-resolved radius, 20 periods (blue: species 1; orange: species 2)",loc="left")
ax.grid(True,color="#d8dde3",lw=.5)

ax=axs[1]
for name,(d,nperiod,idx) in arrays.items():
    ax.plot(nperiod[idx],1e3*d["omega"][idx],lw=.55,color=COLORS[name],
        label=f"{LABELS[name]} (mean {1e3*SUMMARY['species'][name]['omega_from_unwrapped_angle']:+.3f})")
ax.axhline(0,color="#555555",lw=.6)
ax.set_ylabel("$10^3\\,\\omega_{\\rm inst}/\\Omega_1$")
ax.set_xlim(plot_lo,plot_hi)
ax.set_title("(b) Instantaneous rate (blue/orange: species 1/2); mean uses full 1000 periods",loc="left")
ax.grid(True,color="#d8dde3",lw=.5)

ax=axs[2]
components=("mean_rf_torque_mode1","mean_rf_torque_mode2","mean_drag_torque")
comp_labels=("RF mode 1","RF mode 2","drag")
x=np.arange(3); width=.34
for pos,name in enumerate(("species1","species2")):
    vals=[1e3*SUMMARY["species"][name][k] for k in components]
    ax.bar(x+(pos-.5)*width,vals,width,label=LABELS[name],color=COLORS[name],
        edgecolor="white",linewidth=.4)
ax.axhline(0,color="#555555",lw=.65)
ax.set_xticks(x,comp_labels)
ax.set_ylabel("$10^3\\,\\langle T_z\\rangle$")
ax.set_title("(c) Mean torques over 1000 RF periods (blue/orange: species 1/2)",loc="left")
ax.grid(True,axis="y",color="#d8dde3",lw=.5)
fig.suptitle("Local15 S01: exact ideal two-mode RF micromotion and angular-momentum transfer",fontsize=10)
for ext,dpi in (("pdf",None),("svg",None),("png",600)):
    kw={} if dpi is None else {"dpi":dpi}
    fig.savefig(ROOT/f"figures/FIG03_LOCAL15_MICROMOTION_TORQUE.{ext}",bbox_inches="tight",**kw)
plt.close(fig)
print("FIG03 saved; torque means are time averages, not ensemble estimates")
