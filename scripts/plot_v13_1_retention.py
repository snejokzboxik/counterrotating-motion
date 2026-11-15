"""Publication-style paired retention curves for the V13.1 ideal pilot."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1"
rows=list(csv.DictReader((BASE/"ideal_pilot/retention_summary.csv").open(encoding="utf-8",newline="")))
eta_color={-.1:"#225ea8",-.05:"#41b6c4",.05:"#e34a33",.1:"#b30000"}
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,constrained_layout=True)
for ri,phase in enumerate((0.,90.)):
 for ci,horizon in enumerate((100,300)):
  ax=axs[ri,ci]
  for model,eta,color,style,label in [
      ("Q",0.,"#202020","-","Q fixed"),
      ("Qopt",0.,"#777777","--","Q coarse voltage selected"),
      *[(m,e,eta_color[e],"-" if m=="SO" else "--",f"{m} η={e:+.2f}") for e in (-.1,-.05,.05,.1) for m in ("SO","RO")]]:
   z=sorted([r for r in rows if float(r["phase_deg"])==phase and int(float(r["horizon_rf"]))==horizon and r["model"]==model and float(r["eta"])==eta],key=lambda r:float(r["mu"]))
   x=np.array([float(r["mu"]) for r in z]);y=np.array([float(r["retention"]) for r in z])
   lo=np.array([float(r["retention_wilson95_low"]) for r in z]);hi=np.array([float(r["retention_wilson95_high"]) for r in z])
   ax.errorbar(x,y,yerr=np.vstack((y-lo,hi-y)),color=color,linestyle=style,marker="o",ms=3.5,lw=1.25,capsize=1.5,label=label)
  ax.set_title(f"RF start phase {phase:.0f}°, horizon {horizon} periods")
  ax.set_xlim(.84,1.16);ax.set_ylim(-.04,1.04);ax.grid(alpha=.24)
  if ri==1:ax.set_xlabel(r"Mass ratio $\mu=(M/Q)/(M_0/Q_0)$")
  if ci==0:ax.set_ylabel("2D finite-time retention fraction")
handles,labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="outside lower center",ncol=5,frameon=False)
fig.suptitle("V13.1 ideal leading-harmonic pilot · 32 shared Sobol starts · Wilson 95% intervals")
for ext in ("pdf","svg","png"):
 fig.savefig(BASE/f"ideal_mass_retention.{ext}",dpi=220 if ext=="png" else None,bbox_inches="tight")
