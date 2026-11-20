"""Plot structure formation through the 120k-period extensions."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "p1_formation_extensions"
TASKS = [
    ("ordered_003", "Ordered control"),
    ("random_azimuth_010", "Random azimuth"),
    ("random_position_rest_004", "Random positions, rest"),
    ("random_position_rest_015", "Random positions, rest; radial exchange"),
    ("random_position_moving_006", "Random positions and velocities"),
]
COLORS = ("#1565c0", "#c62828")


def main() -> None:
    fig, axes = plt.subplots(len(TASKS), 2, figsize=(15.5, 16.5), sharex=True)
    for row, (task, label) in enumerate(TASKS):
        path = OUT / task / "full_120k_trajectory.npz"
        with np.load(path) as data:
            logs = data["trajectory"]
            skip = int(data["post_load_skip"])
        post = logs[skip:]
        cycles = np.arange(len(post), dtype=np.float64) / 1000.0
        theta = post[:, :, 7]
        radius = np.hypot(post[:, :, 1], post[:, :, 2])
        for g, sl in enumerate((slice(0, 3), slice(3, 6))):
            axes[row, 0].plot(cycles, radius[:, sl].mean(axis=1), lw=0.8, color=COLORS[g], label=f"Ring {g+1}" if row == 0 else None)
            psi = np.abs(np.mean(np.exp(3j * theta[:, sl]), axis=1))
            axes[row, 1].plot(cycles, psi, lw=0.8, color=COLORS[g], label=f"Ring {g+1}" if row == 0 else None)
        for ax in axes[row]:
            ax.axvline(15.0, color="0.35", lw=0.9, ls="--")
            ax.grid(alpha=0.22)
        axes[row, 0].set_ylabel(f"{label}\nmean r/R")
        axes[row, 1].set_ylabel("Ψ₃")
        axes[row, 1].set_ylim(0.55, 1.01)
    axes[0, 0].legend(loc="lower right", frameon=False, ncol=2)
    axes[0, 1].legend(loc="lower right", frameon=False, ncol=2)
    axes[-1, 0].set_xlabel("RF periods after loading / 1000")
    axes[-1, 1].set_xlabel("RF periods after loading / 1000")
    fig.suptitle("N=6 local15: radial ordering and triangular order during long continuation\nDashed line: original 15,000-period classification horizon", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "P1_FORMATION_EXTENSIONS.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
