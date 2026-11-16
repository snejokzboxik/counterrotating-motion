"""Plot replicate-paired A4 retention for the V13.3 delta=0/pi/2 pilot."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    with (root / "relative_phase_replicate_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["model"] == "RO"]
    if not rows:
        raise SystemExit("No paired RO replicate rows")
    rows.sort(key=lambda row: int(row["replica"]))
    d0 = np.array([float(row["P_delta0_lower"]) for row in rows])
    d1 = np.array([float(row["P_comparison_lower"]) for row in rows])
    diff = d1 - d0
    half = student_t.ppf(.975, len(diff) - 1) * np.std(diff, ddof=1) / np.sqrt(len(diff))
    center = np.mean(diff)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].plot(x, d0, "o-", lw=1.4, label=r"$\delta=0$")
    axes[0].plot(x, d1, "s-", lw=1.4, label=r"$\delta=\pi/2$")
    axes[0].set(xlabel="Independent Sobol scramble", ylabel=r"$P_{ret}$", ylim=(-.02, .22),
                title=r"Paired retention, $\eta=-0.05$, $\mu=1$")
    axes[0].set_xticks(x, [str(i + 1) for i in x])
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="0.35", lw=.8)
    axes[1].errorbar([0], [center], yerr=[[half], [half]], fmt="o", capsize=4, color="#8f2d2d",
                     label="Mean ± replicate-level 95% t interval")
    axes[1].scatter(np.full(len(diff), .10), diff, marker="x", color="black", zorder=3,
                    label="Four paired scrambles")
    axes[1].set(xlim=(-.35, .5), ylim=(-.19, .03), ylabel=r"$P_{ret}(\pi/2)-P_{ret}(0)$",
                title="Paired phase effect")
    axes[1].set_xticks([])
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Finite-time 4D-domain retention; four sampled injection phases averaged", fontsize=10)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(root / f"relative_phase_delta0_vs_pi2.{ext}", dpi=220 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
