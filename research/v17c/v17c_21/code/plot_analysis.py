"""Publication-style comparison figure from committed V17C.21 result JSON."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
OUT = BASE / "figures"


def main():
    pred = json.loads((RESULTS / "analytical_predictions.json").read_text(encoding="utf-8"))
    torque = json.loads((RESULTS / "reference_torque_audit.json").read_text(encoding="utf-8"))
    names = ["M1", "M2", "M3"]
    keys = [
        "M1_leading_modes_linear_Stokes",
        "M2_full_MFS_linear_Stokes",
        "M3_full_MFS_SN_torus_Galerkin",
    ]
    colors = ["#4C78A8", "#F2A541", "#278C73"]
    obs = [r"$r_1$", r"$r_2$", r"$f_1$", r"$f_2$"]
    errors = np.array([
        [*pred["models"][k]["radius_relative_error_percent"],
         *pred["models"][k]["frequency_relative_error_percent"]]
        for k in keys
    ])

    ref_window = torque["windows"][-1]["mean_torques"]
    ref_p1 = ref_window["particle1"]
    ref_p2 = ref_window["particle2_own_mass_units"]
    m3 = pred["models"][keys[-1]]["inner_harmonic_balance"]["particles"]
    torque_labels = ["1 RF", "1 drag", "2 RF", "2 drag"]
    scale = 1.0e4
    ref_values = np.array([ref_p1["RF"], ref_p1["drag"], ref_p2["RF"], ref_p2["drag"]]) * scale
    m3_values = np.array([
        m3[0]["mean_total_RF_torque"], m3[0]["mean_total_drag_torque"],
        m3[1]["mean_total_RF_torque"], m3[1]["mean_total_drag_torque"],
    ]) * scale

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    x = np.arange(len(obs))
    width = 0.24
    for i, (name, color) in enumerate(zip(names, colors)):
        ax0.bar(x + (i - 1) * width, errors[i], width=width, label=name,
                color=color, edgecolor="white", linewidth=0.5)
    ax0.axhline(0.0, color="#333333", linewidth=0.8)
    ax0.axhline(5.0, color="#999999", linewidth=0.7, linestyle="--")
    ax0.axhline(-5.0, color="#999999", linewidth=0.7, linestyle="--")
    ax0.set_ylim(-5.5, 7.2)
    ax0.set_xticks(x, obs)
    ax0.set_ylabel("Относительная ошибка относительно V20B, %")
    ax0.set_title("(a) Ошибка средних величин")
    ax0.legend(frameon=False, ncol=3, loc="upper left")
    ax0.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax0.set_axisbelow(True)

    x2 = np.arange(len(torque_labels))
    width2 = 0.36
    ax1.bar(x2 - width2 / 2, ref_values, width=width2, label="V20B RK4, 220–240k",
            color="#777777", edgecolor="white", linewidth=0.5)
    ax1.bar(x2 + width2 / 2, m3_values, width=width2, label="M3 phase-torus",
            color="#278C73", edgecolor="white", linewidth=0.5)
    ax1.axhline(0.0, color="#333333", linewidth=0.8)
    ax1.set_xticks(x2, torque_labels)
    ax1.set_ylabel(r"Средний момент, $10^{-4}$ безразмерных единиц")
    ax1.set_title("(b) Баланс RF и сопротивления")
    ax1.set_ylim(-4.8, 7.2)
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax1.set_axisbelow(True)
    fig.suptitle("V17C.21 — Проверка усреднённой теории на refine12_sobol17", fontsize=12)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    fig.text(0.54, 0.025, "Средний Coulomb-момент V20B: −0.00105 и +0.000115 "
             "в единицах вертикальной шкалы; M3 фазовое среднее = 0.",
             ha="left", va="bottom", fontsize=8, color="#444444")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        out = OUT / f"FIG01_model_accuracy_torque.{ext}"
        fig.savefig(out,
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
        if ext == "svg":
            svg = out.read_text(encoding="utf-8")
            out.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                           encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
