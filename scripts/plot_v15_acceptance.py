"""Plot finite-domain 4D acceptance estimates from the V15 summary table."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v15"
OUT = BASE / "figures"
COLORS = {"Q": "#333333", "SO_equal_harmonic": "#2a6fbb",
          "SO_equal_rms": "#75aadb", "RO_PLUS": "#cf4c4c", "RO_MINUS": "#8d2d8d"}
LABELS = {"Q": "Q", "SO_equal_harmonic": "Q+SO, равная гармоника",
          "SO_equal_rms": "Q+SO, равная RF-интенсивность",
          "RO_PLUS": "Q+RO±, зеркально-эквивалентное полное удержание",
          "RO_MINUS": "Q+RO−"}


def load():
    with (BASE / "finite_domain_acceptance.csv").open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    mass = [r for r in rows if r["source"] == "ideal_mass_ray"]
    if mass:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, horizon in zip(axes, (100, 300)):
            for model, color in COLORS.items():
                if model == "RO_MINUS":
                    continue  # exact mirror pairing makes total RO+ and RO- curves coincide
                rr = sorted((r for r in mass if r["model"] == model
                             and int(r["horizon_rf"]) == horizon), key=lambda r: float(r["mu"]))
                if not rr:
                    continue
                x = [float(r["mu"]) for r in rr]
                y = [float(r["acceptance4_norm"]) for r in rr]
                lo = [y0-float(r["acceptance4_norm_ci95_low"]) for y0, r in zip(y, rr)]
                hi = [float(r["acceptance4_norm_ci95_high"])-y0 for y0, r in zip(y, rr)]
                ax.errorbar(x, y, yerr=[lo, hi], color=color, marker="o", ms=4,
                            capsize=2, label=LABELS[model])
            ax.set_title(f"{horizon} RF-периодов")
            ax.set_xlabel("Относительная масса μ")
            ax.grid(alpha=.25)
        axes[0].set_ylabel(r"Нормированный конечный 4D фазовый объём $a_{4,D}$")
        axes[1].legend(fontsize=8)
        fig.suptitle("A9: конечнодоменный аксептанс вдоль массового луча, идеальные гармоники")
        fig.tight_layout()
        fig.savefig(OUT / "v15_a9_acceptance_mass_ray.png", dpi=220)
        fig.savefig(OUT / "v15_a9_acceptance_mass_ray.svg")
        plt.close(fig)

    fem = [r for r in rows if r["source"] == "fem_confirmation_a9_refined"]
    if fem:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, mesh in zip(axes, (50, 35)):
            for model, color in COLORS.items():
                rr = sorted((r for r in fem if int(r["mesh_um"]) == mesh
                             and r["model"] == model), key=lambda r: int(r["horizon_rf"]))
                if not rr:
                    continue
                x = [int(r["horizon_rf"]) for r in rr]
                y = [float(r["acceptance4_norm"]) for r in rr]
                lo = [y0-float(r["acceptance4_norm_ci95_low"]) for y0, r in zip(y, rr)]
                hi = [float(r["acceptance4_norm_ci95_high"])-y0 for y0, r in zip(y, rr)]
                ax.errorbar(x, y, yerr=[lo, hi], color=color, marker="o", ms=4,
                            capsize=2, label=LABELS[model])
            ax.set_title(f"Полное P4 FEM, h{mesh}")
            ax.set_xlabel("Горизонт, RF-периоды")
            ax.set_xticks((100, 300))
            ax.grid(alpha=.25)
        axes[0].set_ylabel(r"Нормированный конечный 4D фазовый объём $a_{4,D}$")
        axes[1].legend(fontsize=8)
        fig.suptitle("A9: конечнодоменный 4D аксептанс по общему физическому ансамблю")
        fig.tight_layout()
        fig.savefig(OUT / "v15_a9_acceptance_fem.png", dpi=220)
        fig.savefig(OUT / "v15_a9_acceptance_fem.svg")
        plt.close(fig)

    fem_mu = [r for r in rows if r["source"] == "fem_mass_mu1p2_confirmation_fixed"]
    if fem_mu:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, mesh in zip(axes, (50, 35)):
            for model, color in COLORS.items():
                rr = sorted((r for r in fem_mu if int(r["mesh_um"]) == mesh
                             and r["model"] == model), key=lambda r: int(r["horizon_rf"]))
                if not rr:
                    continue
                x = [int(r["horizon_rf"]) for r in rr]
                y = [float(r["acceptance4_norm"]) for r in rr]
                lo = [y0-float(r["acceptance4_norm_ci95_low"]) for y0, r in zip(y, rr)]
                hi = [float(r["acceptance4_norm_ci95_high"])-y0 for y0, r in zip(y, rr)]
                ax.errorbar(x, y, yerr=[lo, hi], color=color, marker="o", ms=4,
                            capsize=2, label=LABELS[model])
            ax.set_title(f"Полное P4 FEM, h{mesh}, μ=1.2")
            ax.set_xlabel("Горизонт, RF-периоды")
            ax.set_xticks((100, 300))
            ax.grid(alpha=.25)
        axes[0].set_ylabel(r"Нормированный конечный 4D фазовый объём $a_{4,D}$")
        axes[1].legend(fontsize=8)
        fig.suptitle("μ=1.2: конечнодоменный 4D аксептанс по общему физическому ансамблю")
        fig.tight_layout()
        fig.savefig(OUT / "v15_a9_acceptance_fem_mu1p2.png", dpi=220)
        fig.savefig(OUT / "v15_a9_acceptance_fem_mu1p2.svg")
        plt.close(fig)

    fem_peak = [r for r in rows if r["source"] == "fem_equal_peak_a9"]
    if fem_peak:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, mesh in zip(axes, (50, 35)):
            for model, color in COLORS.items():
                rr = sorted((r for r in fem_peak if int(r["mesh_um"]) == mesh
                             and r["model"] == model), key=lambda r: int(r["horizon_rf"]))
                if not rr:
                    continue
                x = [int(r["horizon_rf"]) for r in rr]
                y = [float(r["acceptance4_norm"]) for r in rr]
                lo = [y0-float(r["acceptance4_norm_ci95_low"]) for y0, r in zip(y, rr)]
                hi = [float(r["acceptance4_norm_ci95_high"])-y0 for y0, r in zip(y, rr)]
                ax.errorbar(x, y, yerr=[lo, hi], color=color, marker="o", ms=4,
                            capsize=2, label=LABELS[model])
            ax.set_title(f"Полное P4 FEM, h{mesh}")
            ax.set_xlabel("Горизонт, RF-периоды")
            ax.set_xticks((100, 300))
            ax.grid(alpha=.25)
        axes[0].set_ylabel(r"Нормированный конечный 4D фазовый объём $a_{4,D}$")
        axes[1].legend(fontsize=8)
        fig.suptitle("A9 при почти предельном пике электродов 250 V")
        fig.tight_layout()
        fig.savefig(OUT / "v15_equal_peak_acceptance_fem.png", dpi=220)
        fig.savefig(OUT / "v15_equal_peak_acceptance_fem.svg")
        plt.close(fig)

    fem_mass_points = [r for r in rows if r["source"] in
                       ("fem_confirmation_a9_refined", "fem_mass_mu1p2_confirmation_fixed")]
    if fem_mass_points:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True, sharey=True)
        for row, mesh in enumerate((50, 35)):
            for col, horizon in enumerate((100, 300)):
                ax = axes[row, col]
                for model, color in COLORS.items():
                    rr = sorted((r for r in fem_mass_points if int(r["mesh_um"]) == mesh
                                 and int(r["horizon_rf"]) == horizon and r["model"] == model),
                                key=lambda r: float(r["mu"]))
                    if not rr:
                        continue
                    x = [float(r["mu"]) for r in rr]
                    y = [float(r["acceptance4_norm"]) for r in rr]
                    lo = [y0-float(r["acceptance4_norm_ci95_low"]) for y0, r in zip(y, rr)]
                    hi = [float(r["acceptance4_norm_ci95_high"])-y0 for y0, r in zip(y, rr)]
                    ax.errorbar(x, y, yerr=[lo, hi], color=color, marker="o", ms=5,
                                capsize=3, linestyle="none", label=LABELS[model])
                ax.set_title(f"h{mesh}, {horizon} RF")
                ax.grid(alpha=.25)
                ax.set_xlabel("Относительная масса μ")
        axes[0, 0].set_ylabel(r"Нормированный 4D фазовый объём $a_{4,D}$")
        axes[1, 0].set_ylabel(r"Нормированный 4D фазовый объём $a_{4,D}$")
        axes[0, 1].legend(fontsize=8)
        fig.suptitle("P4 FEM: рассчитанные массовые точки μ=1.0 и 1.2 (без интерполяции)")
        fig.tight_layout()
        fig.savefig(OUT / "v15_fem_acceptance_mass_points.png", dpi=220)
        fig.savefig(OUT / "v15_fem_acceptance_mass_points.svg")
        plt.close(fig)


if __name__ == "__main__":
    main()
