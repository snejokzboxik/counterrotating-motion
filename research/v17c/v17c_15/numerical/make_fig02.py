"""FIG02: discrete horizon convergence of exact orbit metrics vs leading roots."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/"results/local15_exact_rf/independent_audit.json").read_text(encoding="utf-8"))
rows=[]
for h,group in a["horizons"].items():
  for name,stat in group.items():
    rows.append({"horizon_periods":int(h),"species":name,
      "retained":stat["retained"],"rho_error_percent":stat["relative_rho_error_percent"],
      "omega_magnitude_error_percent":stat["relative_abs_omega_error_percent"]})
with (ROOT/"results/figure_data/fig02_horizon_convergence.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":9,
 "axes.titlesize":9,"mathtext.fontset":"dejavusans","pdf.fonttype":42,
 "ps.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,
 "axes.spines.right":False})
fig,axs=plt.subplots(1,2,figsize=(7.0,3.0),constrained_layout=True)
styles={"species1":("#126782","o"),"species2":("#b74f22","s")}
for ax,key,title in ((axs[0],"rho_error_percent","(a) radius error"),
                     (axs[1],"omega_magnitude_error_percent","(b) |frequency| error")):
  ax.axhline(0,color="#555555",lw=.7)
  for name,(color,marker) in styles.items():
    r=sorted([z for z in rows if z["species"]==name],key=lambda z:z["horizon_periods"])
    ax.scatter([z["horizon_periods"] for z in r],[z[key] for z in r],
      s=34,marker=marker,color=color,edgecolor="white",linewidth=.45,zorder=3)
  ax.set_xticks([5000,10000,15000],labels=["5 000","10 000","15 000"])
  ax.set_xlabel("integration horizon (primary RF periods)")
  ax.set_ylabel("relative difference from leading root (%)")
  ax.set_title(title,loc="left")
  ax.grid(True,color="#d8dde3",lw=.5)
fig.suptitle("Local15 horizon check; last 500-period window at each horizon",fontsize=10)
for ext,dpi in (("pdf",None),("svg",None),("png",600)):
  kw={} if dpi is None else {"dpi":dpi}
  fig.savefig(ROOT/f"figures/FIG02_LOCAL15_HORIZON_CONVERGENCE.{ext}",bbox_inches="tight",**kw)
plt.close(fig)
print("FIG02 saved; discrete horizons are not joined by interpolated curves")
