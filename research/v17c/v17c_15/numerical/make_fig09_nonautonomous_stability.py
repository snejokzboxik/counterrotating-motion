"""FIG09: panel spread and selected long-horizon conditional QR spectra."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
stability = ROOT / "results/nonautonomous_stability"
panel = json.loads((stability / "local15_panel_conditional_qr_15000.json")
                   .read_text(encoding="utf-8"))
extended = json.loads((stability / "local15_selected_qr_50000.json")
                      .read_text(encoding="utf-8"))
species_order = ("species1", "species2")
panel_horizons = (5000, 10000, 15000)
long_horizons = (5000, 10000, 15000, 30000, 50000)
mode_colors = ("#126782", "#8b6f9c", "#d28b22", "#b74f22")
species_colors = {"species1": "#126782", "species2": "#b74f22"}
category_colors = {
    "S01": "#202020", "min at 15k": "#126782",
    "upper median at 15k": "#57924c", "max at 15k": "#cf741f",
}
species_labels = {"species1": "species 1", "species2": "species 2"}


def diagnostic(record: dict, horizon: int) -> dict:
    return next(row for row in record["periodic_diagnostics"]
                if int(row["period"]) == horizon)


rows = []
for species in species_order:
    for start_id, record in sorted(panel["species"][species].items()):
        drag = float(record["lambda"])
        for horizon in panel_horizons:
            item = diagnostic(record, horizon)
            for mode, value in enumerate(item["conditional_lyapunov_per_tau"], 1):
                rows.append({
                    "cohort": "retained_panel_22",
                    "species": species,
                    "start_id": start_id,
                    "horizon_primary_rf_periods": horizon,
                    "mode_descending": mode,
                    "lambda_per_tau": float(value),
                    "lambda_over_drag": float(value) / drag,
                    "steps_per_period": panel["steps_per_period"],
                    "panel_sha256": panel["panel_sha256"],
                })
    for start_id, record in sorted(extended["species"][species].items()):
        drag = float(record["lambda"])
        for horizon in (30000, 50000):
            item = diagnostic(record, horizon)
            for mode, value in enumerate(item["conditional_lyapunov_per_tau"], 1):
                rows.append({
                    "cohort": "selected_4_extended",
                    "species": species,
                    "start_id": start_id,
                    "horizon_primary_rf_periods": horizon,
                    "mode_descending": mode,
                    "lambda_per_tau": float(value),
                    "lambda_over_drag": float(value) / drag,
                    "steps_per_period": extended["steps_per_period"],
                    "panel_sha256": extended["panel_sha256"],
                })

csv_path = ROOT / "results/figure_data/fig09_conditional_qr.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

selection_category: dict[str, dict[str, str]] = {}
for species, starts in panel["species"].items():
    ranked = sorted((diagnostic(record, 15000)["conditional_lyapunov_per_tau"][0], start_id)
                    for start_id, record in starts.items())
    chosen = {
        "S01": "S01",
        ranked[0][1]: "min at 15k",
        ranked[len(ranked)//2][1]: "upper median at 15k",
        ranked[-1][1]: "max at 15k",
    }
    selection_category[species] = chosen

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.labelsize": 7.6, "axes.titlesize": 8.3,
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, axs = plt.subplots(2, 3, figsize=(10.0, 5.3), constrained_layout=True)

for row_index, species in enumerate(species_order):
    records = panel["species"][species]
    drag = float(next(iter(records.values()))["lambda"])
    ax_spectrum, ax_spread, ax_long = axs[row_index]

    # Four mode medians and min-to-max panel ranges, at measured horizons only.
    for mode in range(4):
        xs, medians, lower, upper = [], [], [], []
        offset = (mode - 1.5) * 0.065
        for horizon in panel_horizons:
            values = np.array([
                diagnostic(records[start_id], horizon)["conditional_lyapunov_per_tau"][mode]
                for start_id in sorted(records)
            ]) / drag
            median = float(np.median(values))
            xs.append(horizon / 1000 + offset)
            medians.append(median)
            lower.append(median - float(values.min()))
            upper.append(float(values.max()) - median)
        ax_spectrum.errorbar(
            xs, medians, yerr=np.array([lower, upper]), fmt="o", ms=2.8,
            color=mode_colors[mode], ecolor=mode_colors[mode], elinewidth=0.7,
            capsize=1.5,
            label=rf"$\Lambda_{mode+1}$" if row_index == 0 else None,
            zorder=3)
    ax_spectrum.axhline(0, color="#555555", lw=0.6, ls="--")
    ax_spectrum.set_title(f"{species_labels[species]}: four modes, 22 starts", loc="left")
    ax_spectrum.set_ylabel(r"median $\Lambda_j/\lambda_i$; bars: panel min–max")
    ax_spectrum.set_xlabel(r"horizon ($10^3$ RF periods)")
    ax_spectrum.set_xlim(4.35, 15.65)
    ax_spectrum.set_xticks([5, 10, 15], labels=["5", "10", "15"])
    ax_spectrum.set_ylim(-1.08, 0.11)
    ax_spectrum.grid(True, color="#d8dde3", lw=0.4)

    # All 22 largest exponents are shown separately; whiskers are deterministic ranges.
    offsets = np.linspace(-0.16, 0.16, len(records))
    for horizon in panel_horizons:
        values = np.array([
            diagnostic(records[start_id], horizon)["conditional_lyapunov_per_tau"][0]
            for start_id in sorted(records)
        ]) / drag
        x0 = horizon / 1000
        ax_spread.scatter(
            x0 + offsets, values, s=9,
            facecolor=species_colors[species] if species == "species1" else "none",
            edgecolor=species_colors[species], linewidth=0.45, alpha=0.72, zorder=2)
        median = float(np.median(values))
        ax_spread.errorbar(
            [x0], [median],
            yerr=[[median - float(values.min())], [float(values.max()) - median]],
            fmt="D", ms=3.5, color="#202020", ecolor="#202020",
            elinewidth=0.8, capsize=1.7, zorder=4)
    ax_spread.axhline(0, color="#555555", lw=0.6, ls="--")
    ax_spread.set_title(f"{species_labels[species]}: leading mode, panel spread", loc="left")
    ax_spread.set_ylabel(r"$\Lambda_1/\lambda_i$; 22 fixed starts")
    ax_spread.set_xlabel(r"horizon ($10^3$ RF periods)")
    ax_spread.set_xlim(4.35, 15.65)
    ax_spread.set_xticks([5, 10, 15], labels=["5", "10", "15"])
    ax_spread.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax_spread.grid(True, color="#d8dde3", lw=0.4)

    # Four selected starts continue to 50k; only discrete values are drawn.
    ext_records = extended["species"][species]
    category_ids = selection_category[species]
    category_order = ("S01", "min at 15k", "upper median at 15k", "max at 15k")
    for category_index, category in enumerate(category_order):
        start_id = next(sid for sid, label in category_ids.items() if label == category)
        x_offset = (category_index - 1.5) * 0.055
        combined = records[start_id]
        for horizon in long_horizons:
            if horizon <= 15000:
                item = diagnostic(combined, horizon)
            else:
                item = diagnostic(ext_records[start_id], horizon)
            value = float(item["conditional_lyapunov_per_tau"][0]) / drag
            ax_long.scatter(
                horizon / 1000 + x_offset, value, s=18,
                marker=("o", "s", "^", "D")[category_index],
                color=category_colors[category], edgecolor="white", linewidth=0.35,
                label=category if row_index == 0 else None, zorder=3)
    ax_long.axhline(0, color="#555555", lw=0.6, ls="--")
    ax_long.set_title(f"{species_labels[species]}: selected starts to 50k", loc="left")
    ax_long.set_ylabel(r"$\Lambda_1/\lambda_i$; no interpolation")
    ax_long.set_xlabel(r"horizon ($10^3$ RF periods)")
    ax_long.set_xlim(4.3, 51.5)
    ax_long.set_xticks([5, 10, 15, 30, 50], labels=["5", "10", "15", "30", "50"])
    ax_long.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax_long.grid(True, color="#d8dde3", lw=0.4)

axs[0, 0].legend(frameon=False, fontsize=6.8, ncol=4, loc="lower left")
axs[0, 2].legend(frameon=False, fontsize=6.6, ncol=2, loc="upper right")
fig.suptitle(
    "Local15 exact RF: conditional mechanical QR spectra across starts and horizons",
    fontsize=9.2)
for extension, dpi in (("pdf", None), ("svg", None), ("png", 600)):
    kwargs = {} if dpi is None else {"dpi": dpi}
    fig.savefig(ROOT / f"figures/FIG09_NONAUTONOMOUS_QR.{extension}",
                bbox_inches="tight", **kwargs)
plt.close(fig)
print(f"FIG09 saved: {len(rows)} mode/start/horizon records; discrete nodes only")
