"""FIG01: exact RF late-orbit panel against leading-averaged roots."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/"results/local15_exact_rf/rk4_96_15000.json").read_text(encoding="utf-8"))
audit=json.loads((ROOT/"results/local15_exact_rf/independent_audit.json").read_text(encoding="utf-8"))
moving=json.loads((ROOT/"results/moving_orbit_correction_local15.json").read_text(encoding="utf-8"))
rows=[]
for r in d["results"]:
  name=r["species"]
  for x in r["windows"]:
    if x["horizon_periods"]!=15000: continue
    if x["escape_period"] is not None and x["escape_period"]<=15000: continue
    rows.append({"species":name,"start_id":x["start_id"],"rho_mean":x["rho_mean"],
      "rho_variation_sd":x["rho_std"],"omega_mean":x["omega_mean"],
      "leading_rho":audit["leading_roots"][name]["rho"],
      "leading_omega":audit["leading_roots"][name]["omega"],
      "kinematic_rho":moving["species"][name]["moving_kinematic_only"]["rho"],
      "kinematic_omega":moving["species"][name]["moving_kinematic_only"]["omega"],
      "full_local_rho":moving["species"][name]["moving_orbit_corrected"]["rho"],
      "full_local_omega":moving["species"][name]["moving_orbit_corrected"]["omega"],
      "window_periods":x["window_periods"],"horizon_periods":15000})
with (ROOT/"results/figure_data/fig01_exact_vs_leading.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":9,
 "axes.titlesize":9,"mathtext.fontset":"dejavusans","pdf.fonttype":42,
 "ps.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,
 "axes.spines.right":False})
fig,axs=plt.subplots(1,2,figsize=(7.0,3.0),constrained_layout=True)
colors={"species1":"#126782","species2":"#b74f22"}
labels={"species1":"species 1","species2":"species 2"}
for ax,name in zip(axs,("species1","species2")):
  subset=[x for x in rows if x["species"]==name]
  x=np.array([z["rho_mean"] for z in subset]); y=1e3*np.array([z["omega_mean"] for z in subset])
  xe=np.array([z["rho_variation_sd"] for z in subset])
  ax.hlines(y,x-xe,x+xe,color=colors[name],lw=.65,alpha=.65)
  ax.scatter(x,y,s=19,marker="o",color=colors[name],edgecolor="white",linewidth=.35,zorder=3)
  ax.scatter([subset[0]["leading_rho"]],[1e3*subset[0]["leading_omega"]],
      marker="*",s=95,color="#202020",edgecolor="white",linewidth=.4,zorder=5)
  ax.scatter([subset[0]["kinematic_rho"]],[1e3*subset[0]["kinematic_omega"]],
      marker="s",s=38,facecolor="white",edgecolor="#009E73",linewidth=1.0,zorder=6)
  ax.scatter([subset[0]["full_local_rho"]],[1e3*subset[0]["full_local_omega"]],
      marker="D",s=31,facecolor="#CC79A7",edgecolor="white",linewidth=.4,zorder=6)
  ax.axhline(0,color="#777777",lw=.65)
  ax.set_xlabel("$\\langle r\\rangle/R$")
  ax.set_ylabel("$10^3\\,\\bar\\omega/\\Omega_1$")
  ax.set_title(f"{labels[name]}: 22 retained, 2 escaped",loc="left")
  ax.grid(True,color="#d8dde3",lw=.5)
  if name=="species1":
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker="o",linestyle="None",color=colors[name],label="exact RF starts"),
      Line2D([],[],marker="*",linestyle="None",color="#202020",markersize=9,label="leading average"),
      Line2D([],[],marker="s",linestyle="None",markerfacecolor="white",markeredgecolor="#009E73",label="moving, kinematic"),
      Line2D([],[],marker="D",linestyle="None",color="#CC79A7",label="full local response")]
    ax.legend(handles=handles,loc="lower right",fontsize=6.3,frameon=True,
              framealpha=.94,edgecolor="#bbbbbb")
fig.suptitle("Local15: exact RF panel and reduced predictions; RK4/96, 15 000 periods",fontsize=9.5)
for ext,dpi in (("pdf",None),("svg",None),("png",600)):
  kw={} if dpi is None else {"dpi":dpi}
  fig.savefig(ROOT/f"figures/FIG01_LOCAL15_EXACT_VS_AVERAGED.{ext}",bbox_inches="tight",**kw)
plt.close(fig)
print("FIG01 saved; radial whiskers are variation, not confidence intervals")
