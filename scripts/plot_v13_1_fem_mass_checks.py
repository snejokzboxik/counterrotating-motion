"""Plot the selected full-P4 FEM mass checks without interpolating uncomputed masses."""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1"
RUNS = [BASE / "fem_h50_p4_mu100_105", BASE / "fem_h50_p4_mu110"]
rows = []
for run in RUNS:
    rows.extend(csv.DictReader((run / "retention_summary.csv").open(encoding="utf-8", newline="")))

models = [("Q", 0.0, "#222222", "o", "Q fixed"),
          ("Qopt", 0.0, "#777777", "s", "Q coarse Floquet candidate"),
          ("SO", -0.05, "#2c7fb8", "o", "SO η=−0.05"),
          ("RO", -0.05, "#7fcdbb", "s", "RO η=−0.05"),
          ("SO", 0.05, "#d95f0e", "o", "SO η=+0.05"),
          ("RO", 0.05, "#fdbb84", "s", "RO η=+0.05")]
phases = sorted({float(r["phase_deg"]) for r in rows})
horizons = (100, 300)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True, constrained_layout=True)
for ri, phase in enumerate(phases):
    for ci, horizon in enumerate(horizons):
        ax = axes[ri, ci]
        for model, eta, color, marker, label in models:
            group = [r for r in rows if float(r["phase_deg"]) == phase and int(r["horizon_rf"]) == horizon
                     and r["model"] == model and float(r["eta"]) == eta]
            group.sort(key=lambda r: float(r["mu"]))
            x = np.array([float(r["mu"]) for r in group])
            y = np.array([float(r["retention"]) for r in group])
            low = np.array([float(r["wilson95_low"]) for r in group])
            high = np.array([float(r["wilson95_high"]) for r in group])
            ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), color=color, marker=marker,
                        linestyle="none", ms=5, capsize=2, label=label)
        ax.set_title(f"Injection phase {phase:g}°, {horizon} RF periods")
        ax.set_xlim(0.98, 1.12); ax.set_ylim(-0.04, 1.04); ax.grid(alpha=0.25)
        if ri == 1: ax.set_xlabel("Mass ratio μ")
        if ci == 0: ax.set_ylabel("2D first-exit retention fraction")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
fig.suptitle("V13.1 selected full h50 P4 FEM checks · points shown only at simulated masses")
for ext in ("pdf", "svg", "png"):
    fig.savefig(BASE / f"fem_selected_mass_checks.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
