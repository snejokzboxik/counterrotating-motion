"""FIG10: discrete error map for local moving-orbit averaging corrections."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/figure_data/moving_orbit_error_map_local15.csv"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.3, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "figure.facecolor": "white",
})
COLORS = {"species1": "#0072B2", "species2": "#D55E00"}
SPECIES_LABELS = {"species1": "Species 1", "species2": "Species 2"}
NU_MARKERS = {2.1: "o", 2.175: "s", 2.25: "^", 2.325: "D", 2.4: "P"}
NU_LABELS = {2.1: "2.100", 2.175: "2.175", 2.25: "2.250",
             2.325: "2.325", 2.4: "2.400"}
MODEL_LABELS = {
    "leading": "Leading averaged root",
    "kinematic_only": "Moving orbit, kinematic terms",
    "full_local_response": "Moving orbit + DC Jacobian",
}


def main() -> None:
    with DATA.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    models = ("leading", "kinematic_only", "full_local_response")
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 7.25), sharex=True)

    for row_index, model in enumerate(models):
        for col_index, ax in enumerate(axes[row_index]):
            for species in ("species1", "species2"):
                selected = [r for r in rows if r["model"] == model
                            and r["species"] == species]
                for row in selected:
                    nu = float(row["nu"])
                    x = float(row["epsilon_mu_max"])
                    if col_index == 0:
                        y = float(row["rho_error_percent_prediction_over_exact_mean"])
                    else:
                        y = 1e4 * float(
                            row["omega_error_absolute_prediction_minus_exact_mean"])
                    ax.scatter(x, y, s=31, marker=NU_MARKERS[nu],
                               facecolor=COLORS[species], edgecolor="white",
                               linewidth=0.5, zorder=3)
            ax.axhline(0, color="#555555", linewidth=0.7,
                       linestyle=(0, (3, 2)), zorder=1)
            ax.grid(color="#dddddd", linewidth=0.45, zorder=0)
            if row_index == 0:
                ax.set_title("Radius error" if col_index == 0
                             else "Rotation-frequency error")
            if col_index == 0:
                ax.set_ylabel(r"$100(\rho_{pred}/\bar\rho_{RF}-1)$ (%)")
                ax.text(0.025, 0.95, MODEL_LABELS[model], transform=ax.transAxes,
                        ha="left", va="top", fontsize=7.3,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                              "pad": 1.5})
            else:
                ax.set_ylabel(r"$10^4(\omega_{pred}-\bar\omega_{RF})$")
            if row_index == len(models) - 1:
                ax.set_xlabel(r"Derived applicability measure $\epsilon_\mu^{max}$")
            ax.set_xlim(0.098, 0.188)

    species_handles = [plt.Line2D([], [], marker="o", linestyle="none",
                                  markerfacecolor=COLORS[s], markeredgecolor="white",
                                  markersize=6, label=SPECIES_LABELS[s])
                       for s in ("species1", "species2")]
    nu_handles = [plt.Line2D([], [], marker=NU_MARKERS[nu], linestyle="none",
                             color="#333333", markersize=5, label=rf"$\nu={NU_LABELS[nu]}$")
                  for nu in NU_MARKERS]
    fig.legend(handles=species_handles + nu_handles, loc="upper center", ncol=7,
               frameon=False, bbox_to_anchor=(0.5, 1.005), handletextpad=0.35,
               columnspacing=0.85)
    fig.text(0.5, 0.012,
             "Discrete archived nodes only; no connecting curves. Exact comparator: "
             "3,000-period fixed panel (24 starts/species, 20–22 retained); "
             "panel spread is not a confidence interval.",
             ha="center", va="bottom", fontsize=7.2)
    fig.tight_layout(rect=(0.02, 0.055, 1, 0.96), h_pad=1.4, w_pad=1.2)

    stem = FIGS / "FIG10_MOVING_ORBIT_ERROR_MAP"
    for ext in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        if ext == "svg":
            svg = path.read_text(encoding="utf-8")
            path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                            encoding="utf-8", newline="\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
