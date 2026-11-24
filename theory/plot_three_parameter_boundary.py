"""Plot the analytical M0 band and sparse archived M3 samples."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = json.loads((ROOT / "data/three_parameter_criterion_validation.json").read_text())
    nom = result["nominal"]
    nus = [1.5, nom["nu"], 5.0]
    alphas = np.linspace(0.05, 0.98, 400)
    out = ROOT / "data/analytic_phase_boundary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nu_slice", "alpha2", "Pi_lower", "Pi_upper", "M0_CR_exists"])
        for nu in nus:
            for alpha2 in alphas:
                low = max(1.0, alpha2 * (1 + nu))
                high = 1 + nu
                w.writerow([f"{nu:.12g}", f"{alpha2:.12g}", f"{low:.12g}",
                            f"{high:.12g}", str(low < high).lower()])

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                         "axes.titlesize": 11, "legend.fontsize": 8,
                         "figure.dpi": 120, "savefig.dpi": 600,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True,
                             constrained_layout=True)
    m3_rows = result["sparse_M3_continuation_comparison"]["rows"]
    colors = {"CR_PLUS_MINUS": "#167d4a", "CR_MINUS_PLUS": "#2574a8",
              "CO_PLUS_PLUS": "#c98412", "CO_MINUS_MINUS": "#b73b37"}
    labels = {"CR_PLUS_MINUS": "M3: CR (+−)", "CR_MINUS_PLUS": "M3: CR (−+)",
              "CO_PLUS_PLUS": "M3: CO (++)", "CO_MINUS_MINUS": "M3: CO (−−)"}
    for ax, nu in zip(axes, nus):
        low = np.maximum(1, alphas * (1 + nu))
        high = np.full_like(alphas, 1 + nu)
        valid = low < high
        ax.fill_between(alphas, low, high, where=valid, color="#dcefe5",
                        alpha=1, label="M0 CR (+−) band")
        ax.plot(alphas, low, color="#28664a", lw=1.2)
        ax.plot(alphas, high, color="#28664a", lw=1.2)
        ax.set_xlim(0.05, 0.98)
        ax.set_ylim(0.8, 6.5)
        ax.set_xlabel(r"$\alpha_2=(q_2/m_2)/(q_1/m_1)$")
        ax.set_title(rf"$\nu={nu:.3g}$")
        ax.grid(color="#d8d8d8", linewidth=0.45, alpha=0.75)
    axes[0].set_ylabel(r"$\Pi=\beta/(4A^2)$")
    central = axes[1]
    seen = set()
    for row in m3_rows:
        if abs(row["nu"] - nom["nu"]) > 1e-9:
            continue
        status = row["M3_classification"]
        c = colors.get(status, "#6a6a6a")
        label = labels.get(status, status)
        central.scatter(nom["alpha2"], row["Pi"], s=23, marker="o",
                        facecolors=c if status != "UNRESOLVED" else "none",
                        edgecolors=c, linewidths=0.8,
                        label=label if label not in seen else None, zorder=3)
        seen.add(label)
    central.scatter(nom["alpha2"], nom["Pi"], marker="*", s=100,
                    c="#151515", edgecolors="white", linewidths=0.5,
                    label="nominal refine12", zorder=5)
    handles, labels_ = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Leading-order counterrotation band and archived M3 continuation", y=1.04)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(ROOT / f"figures/FIG03_three_parameter_M0_region.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
