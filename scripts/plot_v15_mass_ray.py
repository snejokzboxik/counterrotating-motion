"""Publication plots for the computed V15 A9 mass-ray grid (no interpolated samples)."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v15/mass_ray_a9"
OUT=BASE/"figures";OUT.mkdir(parents=True,exist_ok=True)
COLORS={"Q":"#333333","SO_equal_harmonic":"#2a6fbb","SO_equal_rms":"#75aadb","RO_PLUS":"#8d2d8d"}
LABELS={"Q":"Q","SO_equal_harmonic":"Q+SO, равная гармоника",
        "SO_equal_rms":"Q+SO, равная RF-интенсивность",
        "RO_PLUS":"Q+RO± (полное удержание совпадает по зеркалу)"}
rows=list(csv.DictReader((BASE/"retention_summary.csv").open(newline="",encoding="utf8")))
fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
for ax,h in zip(axes,(100,300)):
 for cfg,c in COLORS.items():
  rr=sorted((r for r in rows if r["metric"]=="P_total" and r["config"]==cfg and int(r["horizon_rf"])==h),key=lambda x:float(x["mu"]))
  if not rr:continue
  x=[float(r["mu"]) for r in rr];y=[100*float(r["mean"]) for r in rr]
  lo=[100*(float(r["mean"])-float(r["ci95_low"])) for r in rr];hi=[100*(float(r["ci95_high"])-float(r["mean"])) for r in rr]
  ax.errorbar(x,y,yerr=[lo,hi],marker="o",ms=4,capsize=2,color=c,label=LABELS[cfg])
 ax.set_title(f"N_RF={h}");ax.set_xlabel("Относительная масса μ");ax.grid(alpha=.25)
axes[0].set_ylabel("Конечновременное поперечное удержание (%)")
axes[1].legend(fontsize=8);fig.suptitle("A9 mass ray: фиксированные напряжения, физическая энергия стартов")
fig.tight_layout();fig.savefig(OUT/"v15_a9_mass_retention.png",dpi=220);fig.savefig(OUT/"v15_a9_mass_retention.svg");plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
for ax,h in zip(axes,(100,300)):
 for cfg,c in (("RO_PLUS","#cf4c4c"),("RO_MINUS","#8d2d8d")):
  rr=sorted((r for r in rows if r["metric"]=="Delta_chi" and r["config"]==cfg and int(r["horizon_rf"])==h),key=lambda x:float(x["mu"]))
  if not rr:continue
  x=[float(r["mu"]) for r in rr];y=[100*float(r["mean"]) for r in rr]
  lo=[100*(float(r["mean"])-float(r["ci95_low"])) for r in rr];hi=[100*(float(r["ci95_high"])-float(r["mean"])) for r in rr]
  label="Q+RO+" if cfg=="RO_PLUS" else "Q+RO−"
  ax.errorbar(x,y,yerr=[lo,hi],marker="o",ms=4,capsize=2,color=c,label=label)
 ax.axhline(0,color="black",lw=.8);ax.set_title(f"N_RF={h}");ax.set_xlabel("Относительная масса μ");ax.grid(alpha=.25)
axes[0].set_ylabel("P+ − P− (процентные пункты)");axes[1].legend()
fig.suptitle("Хиральный контраст вдоль массового луча (точки расчёта и QMC-интервалы)")
fig.tight_layout();fig.savefig(OUT/"v15_a9_mass_chirality.png",dpi=220);fig.savefig(OUT/"v15_a9_mass_chirality.svg");plt.close(fig)

p=list(csv.DictReader((BASE/"paired_retention_differences.csv").open(newline="",encoding="utf8")))
fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
for ax,h in zip(axes,(100,300)):
 for contrast,c in (("RO_PLUS-Q","#333333"),("RO_PLUS-SO_equal_harmonic","#2a6fbb"),("RO_PLUS-SO_equal_rms","#75aadb")):
  rr=sorted((r for r in p if r["contrast"]==contrast and int(r["horizon_rf"])==h),key=lambda x:float(x["mu"]))
  if not rr:continue
  x=[float(r["mu"]) for r in rr];y=[float(r["mean_difference_pp"]) for r in rr]
  lo=[y[i]-float(r["ci95_low_pp"]) for i,r in enumerate(rr)];hi=[float(r["ci95_high_pp"])-y[i] for i,r in enumerate(rr)]
  ax.errorbar(x,y,yerr=[lo,hi],marker="o",ms=4,capsize=2,color=c,label=contrast)
 ax.axhline(0,color="black",lw=.8);ax.set_title(f"N_RF={h}");ax.set_xlabel("Относительная масса μ");ax.grid(alpha=.25)
axes[0].set_ylabel("Парное P_RO+ − P_control (п.п.)");axes[1].legend(fontsize=8)
fig.suptitle("Парные контрасты удержания на общих физических стартах")
fig.tight_layout();fig.savefig(OUT/"v15_a9_mass_paired_differences.png",dpi=220);fig.savefig(OUT/"v15_a9_mass_paired_differences.svg");plt.close(fig)
