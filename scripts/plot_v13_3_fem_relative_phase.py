"""Plot h50-P4 paired finite-time 4D retention for delta=0 and pi/2."""
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
        rows = list(csv.DictReader(stream))
    points = sorted({row["point_id"] for row in rows if row["model"] == "RO"})
    if not points:
        raise SystemExit("No paired RO phase summaries found")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for point in points:
        selected = sorted((r for r in rows if r["point_id"] == point and r["model"] == "RO"),
                          key=lambda row: int(row["replica"]))
        if len(selected) != 4:
            raise ValueError(f"expected four independent Sobol scrambles for {point}, got {len(selected)}")
        d0 = np.asarray([float(r["P_delta0_lower"]) for r in selected])
        d1 = np.asarray([float(r["P_comparison_lower"]) for r in selected])
        diff = np.asarray([float(r["paired_delta_lower"]) for r in selected])
        half0 = student_t.ppf(.975, 3) * np.std(d0, ddof=1) / 2
        half1 = student_t.ppf(.975, 3) * np.std(d1, ddof=1) / 2
        halfd = student_t.ppf(.975, 3) * np.std(diff, ddof=1) / 2

        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
        x = np.arange(4)
        axes[0].plot(x, d0, "o-", lw=1.3, label=r"$\delta=0$")
        axes[0].plot(x, d1, "s-", lw=1.3, label=r"$\delta=\pi/2$")
        axes[0].axhline(np.mean(d0), color="#1f77b4", lw=.8, alpha=.45)
        axes[0].axhline(np.mean(d1), color="#ff7f0e", lw=.8, alpha=.45)
        axes[0].set(xlabel="Independent Sobol scramble", ylabel=r"$P_{ret}$", ylim=(-.015, .18),
                    title="Four-phase mean by scramble")
        axes[0].set_xticks(x, [str(i + 1) for i in x])
        axes[0].legend(frameon=False)
        axes[1].axhline(0, color="0.35", lw=.8)
        axes[1].errorbar([0], [np.mean(diff)], yerr=[[halfd], [halfd]], fmt="o", capsize=4,
                         color="#8f2d2d", label="Mean ± replicate-level 95% t interval")
        axes[1].scatter(np.full(4, .10), diff, marker="x", color="black", zorder=3,
                        label="Four paired scrambles")
        axes[1].set(xlim=(-.35, .5), ylim=(-.19, .04),
                    ylabel=r"$P_{ret}(\pi/2)-P_{ret}(0)$", title="Paired phase effect")
        axes[1].set_xticks([])
        axes[1].legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle(f"h50 P4 finite-time 4D retention, {point}, μ=1, N=100", fontsize=10)
        stem = root / f"relative_phase_h50_{point}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=220 if ext == "png" else None,
                        bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
