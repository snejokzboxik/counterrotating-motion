"""FIG09: finite-time conditional mechanical Lyapunov spectrum."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/"results/nonautonomous_stability/local15_s01_conditional_qr_15000.json"
data = json.loads(source.read_text(encoding="utf-8"))
rows = []
for species, item in data["species"].items():
    drag = float(item["lambda"])
    for horizon, exponents in item["windows_lyapunov_exponents_per_tau"].items():
        for mode, value in enumerate(exponents, start=1):
            rows.append({
                "species": species,
                "horizon_primary_rf_periods": int(horizon),
                "conditional_mechanical_mode_descending": mode,
                "lambda_per_tau": float(value),
                "lambda_over_drag": float(value)/drag,
                "initial_start": "S01",
                "steps_per_period": data["steps_per_period"],
                "panel_sha256": data["panel_sha256"],
            })
csv_path = ROOT/"results/figure_data/fig09_conditional_qr.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 9,
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
colors = {"species1": "#126782", "species2": "#b74f22"}
markers = {"species1": "o", "species2": "s"}
fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.4), sharex=True,
                        constrained_layout=True)
for mode, ax in enumerate(axs.flat, start=1):
    for species in ("species1", "species2"):
        subset = [r for r in rows if r["species"] == species
                  and r["conditional_mechanical_mode_descending"] == mode]
        subset.sort(key=lambda r: r["horizon_primary_rf_periods"])
        ax.scatter([r["horizon_primary_rf_periods"]/1000 for r in subset],
                   [r["lambda_over_drag"] for r in subset],
                   marker=markers[species], s=32,
                   facecolor=colors[species] if species == "species1" else "none",
                   edgecolor=colors[species], linewidth=1.0, zorder=3,
                   label="species 1" if species == "species1" else "species 2")
    ax.axhline(0, color="#555555", lw=0.7, ls="--")
    ax.set_title(rf"conditional mode $\Lambda_{mode}$", loc="left")
    ax.grid(True, color="#d8dde3", lw=0.45)
    if mode in (3, 4):
        ax.set_xlabel(r"horizon ($10^3$ primary RF periods)")
    if mode in (1, 3):
        ax.set_ylabel(r"$\Lambda_j/\lambda_i$")
    ax.set_xlim(4.0, 16.0)
    ax.set_xticks([5, 10, 15], labels=["5", "10", "15"])
axs[0, 0].set_ylim(-0.00003, 0.00031)
axs[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
fig.suptitle("Local15 exact RF: conditional mechanical QR spectrum on S01", fontsize=10)
for ext, dpi in (("pdf", None), ("svg", None), ("png", 600)):
    kwargs = {} if dpi is None else {"dpi": dpi}
    fig.savefig(ROOT/f"figures/FIG09_NONAUTONOMOUS_QR.{ext}",
                bbox_inches="tight", **kwargs)
plt.close(fig)
print("FIG09 saved: 24 horizon/mode rows, no interpolation between horizons")
