"""Plot matched ideal/h50/h35 4D retention and RO-minus-Q differences."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"


def rows(directory: str, name: str) -> list[dict[str, str]]:
    with (BASE / directory / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", default="acceptance_4d_fem_comparison_N100")
    args = parser.parse_args()
    if Path(args.output_directory).name != args.output_directory:
        parser.error("output directory must be a direct child of the V13.3 atlas")
    out = BASE / args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    runs = [
        ("Ideal", "acceptance_4d_core_N100"),
        ("h50 P4", "acceptance_4d_fem_h50_selected_N100"),
        ("h35 P4", "acceptance_4d_fem_h35_selected_N100"),
    ]
    point_ids = ("a9_b2_e0", "ro_gain_negative")
    pcolors = {"Q": "#355c9a", "RO": "#bd4b37"}
    markers = {"Q": "o", "RO": "s"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True, constrained_layout=True)
    for ax, point in zip(axes, point_ids):
        for model, offset in (("Q", -.08), ("RO", .08)):
            xs, means, lows, highs = [], [], [], []
            for i, (_, directory) in enumerate(runs):
                row = next(r for r in rows(directory, "acceptance_summary_N100.csv")
                           if r["point_id"] == point and r["model"] == model and float(r["mu"]) == 1.)
                mean = float(row["phase_average_P_lower_mean"])
                low = float(row["phase_average_P_lower_t95_low"])
                high = float(row["phase_average_P_lower_t95_high"])
                xs.append(i + offset); means.append(mean)
                lows.append(max(0., mean - low)); highs.append(max(0., high - mean))
            ax.errorbar(xs, means, yerr=[lows, highs], marker=markers[model], color=pcolors[model],
                        lw=0, linestyle="none", capsize=3, label=model if point == point_ids[0] else None)
        ax.set_title(point.replace("_", " "))
        ax.set_xticks(range(3), [name for name, _ in runs])
        ax.grid(axis="y", alpha=.25)
        ax.set_ylim(-.01, .32)
    axes[0].set_ylabel(r"$P_{ret}$ over the physical 4D launch domain")
    axes[0].legend(frameon=False)
    fig.suptitle(r"Matched starts, $\mu=1$, $N_{RF}=100$, four injection phases")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fem_4d_retention_field_comparison.{ext}", dpi=220 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for j, point in enumerate(point_ids):
        xs, values, lows, highs = [], [], [], []
        for i, (_, directory) in enumerate(runs):
            row = next(r for r in rows(directory, "paired_acceptance_summary_N100.csv")
                       if r["point_id"] == point and float(r["mu"]) == 1.)
            mean = float(row["paired_delta_P_lower_mean"])
            low = float(row["paired_delta_P_lower_t95_low"])
            high = float(row["paired_delta_P_lower_t95_high"])
            xs.append(i + (j - .5) * .12); values.append(mean)
            lows.append(max(0., mean - low)); highs.append(max(0., high - mean))
        ax.errorbar(xs, values, yerr=[lows, highs], marker="o", capsize=3, lw=0, linestyle="none",
                    label=point.replace("_", " "))
    ax.axhline(0., color="0.25", lw=.8)
    ax.set_xticks(range(3), [name for name, _ in runs])
    ax.set_ylabel(r"paired $P_{ret}(Q+RO)-P_{ret}(Q)$")
    ax.set_title(r"Matched RO−Q effect; $mu=1$, $N_{RF}=100$")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fem_4d_ro_minus_q_field_comparison.{ext}", dpi=220 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
