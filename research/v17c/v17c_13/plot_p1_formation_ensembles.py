"""Plot retention, strict 3+3 success, and outcome composition by N=6 ensemble."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "p1_formation_ensembles"
LABELS = {
    "ordered": "Упорядоченные",
    "perturbed": "Возмущённые",
    "random_azimuth": "Случ.\nазимуты",
    "random_position_rest": "Случ.\nпокой",
    "random_position_moving": "Случ.\nдвижение",
}
OUTCOMES = [
    ("COUNTERROTATING_CRYSTALS_3_PLUS_3", "Кристаллы 3+3", "#238b57"),
    ("COUNTERROTATING_UNORDERED", "Встречное, без порядка", "#6c9f76"),
    ("PHASE_SLIP", "Фазовый slip", "#e0a22e"),
    ("RING_EXCHANGE", "Обмен радиусов", "#a888c7"),
    ("AXIS_AMBIGUOUS", "Ось неоднозначна", "#777777"),
    ("WALL_CONTACT", "Контакт со стенкой", "#c84c4c"),
    ("PARTICLE_CONTACT", "Контакт частиц", "#8f2d2d"),
    ("UNRESOLVED", "Неопределённый", "#b6b6b6"),
]


def main() -> None:
    audit = json.loads((DATA / "formation_ensemble_audit.json").read_text(encoding="utf-8"))
    names = list(audit["ensembles"])
    x = np.arange(len(names))
    n = np.asarray([audit["ensembles"][k]["n_total"] for k in names], dtype=float)
    retain = np.asarray([audit["ensembles"][k]["retention_probability_all_starts"] for k in names])
    crystal = np.asarray([audit["ensembles"][k]["P_crystal_all_starts"] for k in names])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.95, 1.45]})

    ax = axes[0]
    width = 0.34
    ax.bar(x - width/2, retain, width, color="#4c78a8", label="Полное удержание")
    ax.bar(x + width/2, crystal, width, color="#238b57", label="Строгий успех 3+3")
    for ix, v in enumerate(retain):
        count = int(round(v*n[ix]))
        ax.text(ix - width/2, max(0.04, v/2), str(count), ha="center", va="center",
                fontsize=8, color="white")
    for ix, v in enumerate(crystal):
        count = int(round(v*n[ix]))
        ax.text(ix + width/2, max(0.04, v/2), str(count), ha="center", va="center",
                fontsize=8, color="white")
    ax.set_xticks(x, [LABELS[k] for k in names])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Доля всех стартов")
    ax.set_title("Удержание и строгие успехи (N=16)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0))

    ax = axes[1]
    bottom = np.zeros(len(names), dtype=float)
    for key, label, color in OUTCOMES:
        values = np.asarray([audit["ensembles"][k]["outcome_counts"].get(key, 0) for k in names], dtype=float)
        if np.all(values == 0):
            continue
        ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.7)
        for ix, v in enumerate(values):
            if v >= 2:
                ax.text(ix, bottom[ix] + v/2, str(int(v)), ha="center", va="center",
                        fontsize=9, color="white" if color not in ("#e0a22e", "#b6b6b6") else "#222222")
        bottom += values
    ax.set_xticks(x, [LABELS[k] for k in names])
    ax.set_ylim(0, 17)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_ylabel("Число исходов (N=16 в каждом ансамбле)")
    ax.set_title("Все исходы; ранние потери включены")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("V17C.13 — самоформирование двух встречновращающихся кристаллов", fontsize=13)
    fig.savefig(HERE / "P1_FORMATION_ENSEMBLES.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
