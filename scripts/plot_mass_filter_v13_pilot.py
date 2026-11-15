"""Plot paired 2D finite-time retention pilot; not a transmission curve."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/pilot_ideal_interior_h20/summary.csv"
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13"
rows = list(csv.DictReader(DATA.open(encoding="utf-8", newline="")))
models = ["Q", "Q+SO", "Q+RO"]
colors = {"Q": "#222222", "Q+SO": "#2878B5", "Q+RO": "#D34E24"}
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True, constrained_layout=True)
for ax, eta in zip(axes, (-0.2, 0.2)):
    for model in models:
        selected = [r for r in rows if float(r["eta"]) == eta and r["model"] == model]
        x = np.array([float(r["mass_ratio"]) for r in selected])
        n = np.array([int(r["n"]) for r in selected])
        k = np.array([int(r["retained"]) for r in selected])
        p = k / n
        # Wilson 95% intervals, displayed as asymmetric binomial error bars.
        z = 1.96
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        ax.errorbar(x, p, yerr=np.vstack((p-np.maximum(0, center-half), np.minimum(1, center+half)-p)),
                    marker="o", ms=4.5, lw=1.5, capsize=2, color=colors[model], label=model)
    ax.set_title(rf"$\eta={eta:+.1f}$")
    ax.set_xlabel(r"Mass ratio $\mu=(M/Q)/(M/Q)_0$")
    ax.set_xlim(0.86, 1.14); ax.set_ylim(-0.04, 1.04)
    ax.grid(True, alpha=.25)
axes[0].set_ylabel("Finite-time retention fraction (20 RF periods)")
axes[1].legend(frameon=False, loc="upper left")
fig.suptitle("Ideal 2D first-exit pilot (16 shared Sobol initial states; Wilson 95% intervals)")
fig.savefig(OUT / "mass_selectivity_pilot.pdf")
fig.savefig(OUT / "mass_selectivity_pilot.svg")
fig.savefig(OUT / "mass_selectivity_pilot.png", dpi=220)
