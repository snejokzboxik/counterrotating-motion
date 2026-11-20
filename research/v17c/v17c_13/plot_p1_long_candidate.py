"""Plot stroboscopic trajectories and structure diagnostics for one long #505 run."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "p1_long_candidates" / "P2_505_angle0_phase0_long_trajectory.npz"


def main() -> None:
    with np.load(DATA) as archive:
        logs = archive["trajectory"]
        groups = archive["groups"]
        skip = int(archive["post_load_skip"])
    cycles = logs[:, 0, 0] / (2.0 * np.pi)
    stride = 5
    views = logs[::stride]
    colors = ["#1769aa", "#4285c5", "#79a7d8", "#b3261e", "#dc554a", "#ed8c7f"]
    labels = ["inner 1", "inner 2", "inner 3", "outer 1", "outer 2", "outer 3"]

    fig = plt.figure(figsize=(15, 4.2), constrained_layout=True)
    ax3 = fig.add_subplot(1, 4, 1, projection="3d")
    axxy = fig.add_subplot(1, 4, 2)
    axxz = fig.add_subplot(1, 4, 3)
    axyz = fig.add_subplot(1, 4, 4)
    for i in range(6):
        x, y, z = views[:, i, 1], views[:, i, 2], views[:, i, 3]
        ax3.plot(x, y, z, lw=0.65, color=colors[i], label=labels[i])
        axxy.plot(x, y, lw=0.65, color=colors[i])
        axxz.plot(x, z, lw=0.65, color=colors[i])
        axyz.plot(y, z, lw=0.65, color=colors[i])
    ax3.set(title="3D (stroboscopic)", xlabel="x/R", ylabel="y/R", zlabel="z/R")
    ax3.view_init(elev=28, azim=35)
    ax3.legend(fontsize=7, ncol=2, loc="upper left")
    for ax, title, xlabel, ylabel in ((axxy, "XY", "x/R", "y/R"),
                                      (axxz, "XZ", "x/R", "z/R"),
                                      (axyz, "YZ", "y/R", "z/R")):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("V17C.13 — local15, #505, six free particles; stroboscopic every 5 RF periods")
    fig.savefig(HERE / "P1_LONG_TRAJECTORY_VIEWS.png", dpi=180)
    plt.close(fig)

    post = logs[skip:]
    tau = post[:, 0, 0]
    tcycles = (tau - tau[0]) / (2.0 * np.pi)
    theta = post[:, :, 7]
    radius = np.hypot(post[:, :, 1], post[:, :, 2])
    z = post[:, :, 3]
    psi = [np.abs(np.mean(np.exp(3j * theta[:, g * 3:(g + 1) * 3]), axis=1)) for g in (0, 1)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    for i in range(6):
        ax.plot(tcycles, (theta[:, i] - theta[0, i]) / (2.0 * np.pi), lw=0.8, color=colors[i], label=labels[i])
    ax.set(title="Accumulated turns after loading", xlabel="primary RF periods", ylabel="turns")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for i in range(6):
        ax.plot(tcycles, radius[:, i], lw=0.7, color=colors[i], label=labels[i])
    ax.set(title="Orbit radius", xlabel="primary RF periods", ylabel="r/R", ylim=(0.45, 0.85))
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(tcycles, psi[0], color="#1769aa", lw=0.8, label="inner Ψ₃")
    ax.plot(tcycles, psi[1], color="#b3261e", lw=0.8, label="outer Ψ₃")
    ax.set(title="Triangular order parameter", xlabel="primary RF periods", ylabel="Ψ₃", ylim=(0.8, 1.005))
    ax.grid(alpha=0.25); ax.legend()

    ax = axes[1, 1]
    for g, base, color, label in ((0, 0, "#1769aa", "inner"), (1, 3, "#b3261e", "outer")):
        rel = theta[:, base:base + 3] - theta[:, base:base + 1]
        for j in (1, 2):
            ax.plot(tcycles, np.rad2deg(rel[:, j] - rel[0, j]), lw=0.8,
                    color=color, alpha=0.65 + 0.15 * j, label=f"{label} relative phase {j}")
    ax.set(title="Unwrapped relative phase", xlabel="primary RF periods", ylabel="phase change (deg)")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("V17C.13 — ordered #505 candidate, 15,000 post-load periods")
    fig.savefig(HERE / "P1_LONG_STRUCTURE_DIAGNOSTICS.png", dpi=180)
    plt.close(fig)
    print(HERE / "P1_LONG_TRAJECTORY_VIEWS.png")
    print(HERE / "P1_LONG_STRUCTURE_DIAGNOSTICS.png")


if __name__ == "__main__":
    main()
