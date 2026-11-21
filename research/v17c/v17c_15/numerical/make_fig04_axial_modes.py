"""FIG04: discrete axial Fourier spectrum and exact buckling thresholds."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"numerical"))
from collective_rings import vertical_laplacian_spectrum_formula

rows=[]
for n in range(2,33):
    eig=vertical_laplacian_spectrum_formula(n,1.0)
    for m in range(n//2+1):
        rows.append({"N":n,"mode_m":m,"mode_fraction_m_over_N":m/n,
          "ell_m_R3":float(eig[m]),"beta2_critical_mode":float(eig[m]/2),
          "critical_charge_parameter_kappa_Q_over_R3":1.0})
with (ROOT/"results/figure_data/fig04_axial_mode_thresholds.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":9,
 "axes.titlesize":9,"mathtext.fontset":"dejavusans","pdf.fonttype":42,
 "ps.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,
 "axes.spines.right":False})
fig,axs=plt.subplots(1,3,figsize=(9.0,3.2),constrained_layout=True,
  gridspec_kw={"width_ratios":[1.5,1.0,1.1]})
positive=[r for r in rows if r["mode_m"]>0]
sc=axs[0].scatter([r["N"] for r in positive],
  [r["mode_fraction_m_over_N"] for r in positive],
  c=[r["beta2_critical_mode"] for r in positive],
  s=23,cmap="viridis",norm=LogNorm(),edgecolor="none")
axs[0].scatter([r["N"] for r in rows if r["mode_m"]==0],
  [0 for r in rows if r["mode_m"]==0],marker="x",s=18,color="#777777")
axs[0].set_xlabel("$N$")
axs[0].set_ylabel("$m/N$")
axs[0].set_title("(a) Exact modal buckling threshold",loc="left")
axs[0].set_xticks([2,8,14,20,26,32])
axs[0].grid(True,color="#d8dde3",lw=.45)
cb=fig.colorbar(sc,ax=axs[0],pad=.02)
cb.set_label("$\\beta_{2,\\rm crit}^{(m)}$ (log scale)")

ns=np.arange(2,33)
mstar=np.array([int(np.argmax(vertical_laplacian_spectrum_formula(int(n),1.0))) for n in ns])
axs[1].scatter(ns,mstar/ns,s=24,color="#174a7e",edgecolor="white",linewidth=.35)
axs[1].axhline(.5,color="#777777",lw=.7,ls="--")
axs[1].set_ylim(.25,.55)
axs[1].set_xlabel("$N$")
axs[1].set_ylabel("$m_{\\max}/N$")
axs[1].set_title("(b) First unstable mode",loc="left")
axs[1].set_xticks([2,8,14,20,26,32])
axs[1].grid(True,color="#d8dde3",lw=.45)

crit=np.array([vertical_laplacian_spectrum_formula(int(n),1.0).max()/2 for n in ns])
axs[2].scatter(ns,crit,s=24,color="#b74f22",edgecolor="white",linewidth=.35,label="exact integer-$N$ sum")
ng=np.linspace(8,32,100)
asym=7*1.202056903159594/(32*np.pi**3)*ng**3
axs[2].plot(ng,asym,color="#222222",lw=1,ls="--",label="$7\\zeta(3)N^3/(32\\pi^3)$")
axs[2].set_yscale("log")
axs[2].set_xticks([2,10,20,32],labels=["2","10","20","32"])
axs[2].set_xlabel("$N$")
axs[2].set_ylabel("$\\beta_{2,\\rm crit}^{\\max}$")
axs[2].set_title("(c) Maximum threshold and asymptote",loc="left")
axs[2].legend(frameon=False,fontsize=6.2,loc="upper left")
axs[2].grid(True,which="major",color="#d8dde3",lw=.45)
fig.suptitle("Planar equal-charge Coulomb ring: $R=Q=\\mu=\\kappa=1$, $K_z=2\\beta_2$",fontsize=10)
for ext,dpi in (("pdf",None),("svg",None),("png",600)):
    kw={} if dpi is None else {"dpi":dpi}
    fig.savefig(ROOT/f"figures/FIG04_AXIAL_MODE_THRESHOLDS.{ext}",bbox_inches="tight",**kw)
plt.close(fig)
print("FIG04 saved; points are exact discrete modes, dashed curve is derived large-N asymptote")
