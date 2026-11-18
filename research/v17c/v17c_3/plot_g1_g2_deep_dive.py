"""Publication figures for the G1/G2 FEM-frequency and pair screens."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rf_dc_core import RESULTS

ROOT = RESULTS / "g1_g2_joint"


def main() -> None:
    freqs = json.loads((ROOT / "G1_G2_DOP853_translation_frequency_validation.json").read_text(encoding="utf-8"))
    strict = json.loads((ROOT / "strict_classification" / "strict_reclassification_summary.json").read_text(encoding="utf-8"))
    labels, errors = [], []
    for geometry in ("G1", "G2"):
        for idx, mode in enumerate(freqs[geometry]["pseudo_to_DOP853_matches"]):
            labels.append(f"{geometry}, поперечная {idx+1}")
            errors.append(100*mode["relative_error"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = ["#4776a8", "#de8f05", "#4776a8", "#de8f05"]
    x = np.arange(len(labels))
    ax0.bar(x, errors, color=colors)
    ax0.axhline(0, color="black", linewidth=.8)
    ax0.set_xticks(x, labels, rotation=18, ha="right")
    ax0.set_ylabel("(ω Floquet − ω псевдо) / ω псевдо, %")
    ax0.set_title("Поперечные трансляционные частоты")
    ax0.grid(axis="y", alpha=.25)
    for xi, val in zip(x, errors):
        ax0.annotate(f"{val:+.3f}%", (xi, val), xytext=(0, 4 if val >= 0 else -13),
                     textcoords="offset points", ha="center", fontsize=8)

    max_turns = []
    pair_labels = []
    broad = strict["broad_screen"]["per_geometry"]
    targeted = strict["barrier_targeted"]["per_geometry"]
    for dataset, mapping in (("Широкий поиск", broad), ("Барьерный поиск", targeted)):
        for geometry in ("G1", "G2"):
            pair_labels.append(f"{dataset}\n{geometry}")
            max_turns.append(mapping[geometry]["max_minimum_total_turns"])
    xx = np.arange(len(max_turns))
    bars = ax1.bar(xx, max_turns, color=["#4776a8", "#8fb2d3", "#de8f05", "#f2c06b"])
    ax1.axhline(10, color="#b22222", linestyle="--", linewidth=1.4, label="критерий: 10 оборотов каждой")
    ax1.set_xticks(xx, pair_labels)
    ax1.set_ylim(0, 10.8)
    ax1.set_ylabel("Максимум min(|обороты₁|, |обороты₂|)")
    ax1.set_title("Лучший парный запуск против порога")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=.25)
    for rect, val in zip(bars, max_turns):
        ax1.annotate(f"{val:.3f}", (rect.get_x()+rect.get_width()/2, rect.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    fig.suptitle("G1/G2: независимая FEM-частотная проверка и трёхмерные парные сканы")
    fig.savefig(ROOT / "G1_G2_deep_dive_summary.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
