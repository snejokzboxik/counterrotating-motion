"""Publication-style FIG07: fixed-radius analytical Coulomb ring scaling."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
src=ROOT/"results/collective_ring_checks.json"
data=json.loads(src.read_text(encoding="utf-8"))["rows"]
figdir=ROOT/"figures"; datadir=ROOT/"results/figure_data"
figdir.mkdir(parents=True,exist_ok=True); datadir.mkdir(parents=True,exist_ok=True)
fields=list(data[0])
with (datadir/"fig07_collective_ring_scaling.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
N=np.array([d["N"] for d in data])
E=np.array([d["intraring_energy_formula"] for d in data])
Fr=np.array([d["radial_force_per_particle_formula"] for d in data])
lam=np.array([d["vertical_laplacian_lambda_max"] for d in data])
bc=np.array([d["beta2_critical_for_quadrupole_Kz_2beta"] for d in data])
plt.rcParams.update({
 "font.family":"DejaVu Sans","font.size":9,"axes.labelsize":9,
 "axes.titlesize":10,"legend.fontsize":8,"xtick.labelsize":8,
 "ytick.labelsize":8,"mathtext.fontset":"dejavusans",
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
 "axes.spines.top":False,"axes.spines.right":False,
})
fig,axs=plt.subplots(2,2,figsize=(7.1,5.4),constrained_layout=True)
plots=[
 (axs[0,0],E,"$U_{\\rm self}/(\\kappa Q^2/R)$","(a) Coulomb energy"),
 (axs[0,1],Fr,"$F_{r,\\rm self}/(\\kappa Q^2/R^2)$","(b) Outward force per particle"),
 (axs[1,0],lam,"$\\ell_{\\max}R^3$","(c) Axial Coulomb Laplacian"),
 (axs[1,1],bc,"$\\beta_{2,\\rm crit}$","(d) Quadrupole buckling threshold"),
]
for ax,y,label,title in plots:
    ax.scatter(N,y,s=26,color="#174a7e",edgecolor="white",linewidth=0.45,zorder=3)
    ax.set_xlabel("$N$")
    ax.set_ylabel(label)
    ax.set_title(title,loc="left",fontweight="normal")
    ax.grid(True,which="major",color="#d8dde3",linewidth=0.6,alpha=0.8)
    ax.set_xticks(N)
    ax.tick_params(axis="x",rotation=35)
fig.suptitle("Regular Coulomb ring at fixed $R=Q=\\mu=\\kappa=1$",fontsize=11)
for ext,dpi in (("pdf",None),("svg",None),("png",600)):
    kwargs={} if dpi is None else {"dpi":dpi}
    fig.savefig(figdir/f"FIG07_GENERAL_N_RING_SCALING.{ext}",bbox_inches="tight",**kwargs)
plt.close(fig)
print("saved FIG07 PDF/SVG/PNG600 and CSV; discrete exact values shown as markers only")
