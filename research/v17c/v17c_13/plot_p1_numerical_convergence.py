"""Plot full-horizon RK4 convergence and the late DOP853 cross-check."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    d = json.loads((HERE / "p1_numerical_validation_audit.json").read_text())
    rk = d["rk4_full_horizon"]
    dop = d["independent_dop853_late_segment"]
    steps = np.asarray(rk["resolutions_steps_per_period"], dtype=int)
    freqs = np.asarray([rk["mean_frequency_Hz_by_group"][str(n)] for n in steps])
    err = np.asarray([rk["relative_energy_residual_by_resolution"][str(n)] for n in steps])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    ax = axes[0]
    delta = freqs - freqs[-1, :]
    ax.plot(steps, delta[:, 0], "o-", lw=1.8, label="Тройка 1 (+)")
    ax.plot(steps, delta[:, 1], "s-", lw=1.8, label="Тройка 2 (−)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(steps, labels=[str(x) for x in steps])
    ax.set_xlabel("Шагов на RF-период")
    ax.set_ylabel("Смещение средней частоты от RK4/256, Hz")
    ax.set_title("Сходимость частоты на 15 000 периодах")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.loglog(steps, err, "o-", color="#7b4ab5", lw=1.8)
    for x, y in zip(steps, err):
        ax.annotate(f"{y:.1e}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xticks(steps, labels=[str(x) for x in steps])
    ax.set_xlabel("Шагов на RF-период")
    ax.set_ylabel("Максимальный остаток / RF-работа")
    ax.set_title("Сходимость энергетического баланса")
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    rkf = np.asarray(dop["rk4_late_mean_frequency_Hz_by_group"])
    dpf = np.asarray(dop["dop853_late_mean_frequency_Hz_by_group"])
    diff_mhz = (dpf - rkf) * 1000.0
    colors = ["#c05a30" if v >= 0 else "#36769c" for v in diff_mhz]
    bars = ax.bar([0, 1], diff_mhz, color=colors, width=0.58)
    ax.axhline(0, color="black", lw=0.8)
    for bar, value in zip(bars, diff_mhz):
        ax.annotate(f"{value:+.3f} mHz", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 6 if value >= 0 else -14), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks([0, 1], ["Тройка 1 (+)", "Тройка 2 (−)"])
    ax.set_ylabel("DOP853 − RK4/96, mHz")
    ax.set_title("Частотное расхождение на 3 000 периодах")
    ax.grid(alpha=0.25)

    fig.suptitle("V17C.13 — численная проверка 3+3 режима local15", fontsize=13)
    fig.savefig(HERE / "P1_NUMERICAL_CONVERGENCE.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
