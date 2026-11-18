"""Recreate compact figures from the saved V17C.4 checkpoints (no integration)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pseudopotential():
    data = read_json(RES / "pseudopotential" / "ADCFEB_full_spectrum_pseudopotential.json")
    rows = data["radial_metrics"]
    r = np.array([x["r_over_R"] for x in rows])
    chi = np.array([x["chirality"] for x in rows])
    barrier = np.array([x["angular_barrier_per_alpha2"] for x in rows])
    corr = np.array([x["angular_corrugation"] for x in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.axhspan(0, 0.44959, color="#dcefe3", alpha=.65)
    ax.axhspan(0.44959, 0.88453, color="#f8e4e1", alpha=.55)
    ax.axhline(0, color="0.25", lw=.9)
    ax.plot(r, chi, "o-", color="#2b6cb0", label="усреднённая хиральность")
    ax.set(xlabel="$r/R$", ylabel="Нормированное RF воздействие", xlim=(.18, .68))
    ax2 = ax.twinx()
    ax2.plot(r, barrier, "s--", color="#b7791f", label="барьер / $\\alpha^2$")
    ax2.plot(r, corr, "^:", color="#805ad5", label="corrugation")
    ax2.set_ylabel("Угловой барьер / неоднородность")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [x.get_label() for x in lines], loc="upper left", frameon=True)
    ax.set_title("ADCFEB: знак среднего момента и рост углового барьера")
    fig.savefig(FIG / "pseudopotential_profile.png", dpi=180)
    plt.close(fig)


def coulomb_continuation():
    data = read_json(RES / "coulomb_continuation" / "ADCFEB_000228" / "summary.json")
    rows = data["rows"]
    chi = np.array([float(x["chi"]) for x in rows])
    # Place the uncoupled control at the origin; symlog preserves it visibly.
    order = np.argsort(chi)
    chi = chi[order]
    rows = [rows[i] for i in order]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.4, 7), sharex=True, constrained_layout=True)
    for k, label, color in [(1, "частица 1", "#2b6cb0"), (2, "частица 2", "#c05621")]:
        a.plot(chi, [x[f"turns{k}_after_relax"] for x in rows], "o-", ms=4, label=label, color=color)
        b.plot(chi, [x[f"r{k}_mean"] for x in rows], "o-", ms=4, label=label, color=color)
    original = float(next(x["original_chi"] for x in rows if float(x["original_chi"]) > 0))
    for ax in (a, b):
        ax.axvline(original, color="0.25", ls="--", lw=1, label="исходное $\\chi$" if ax is a else None)
        ax.set_xscale("symlog", linthresh=1e-5)
        ax.grid(True, alpha=.25)
        ax.legend()
    a.axhline(0, color="0.25", lw=.8)
    a.set_ylabel("Обороты после релаксации")
    b.set_ylabel("Средний радиус $r/R$")
    b.set_xlabel("Интенсивность кулоновской связи $\\chi$")
    fig.suptitle("Продолжение Coulomb-связи на одинаковом начальном состоянии")
    fig.savefig(FIG / "coulomb_continuation.png", dpi=180)
    plt.close(fig)


def matched_phase_trajectories():
    src = RES / "phase_controls" / "ADCFEB_AFCBED_matched_000228"
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.3), constrained_layout=True)
    cases = [("ADCFEB_chi_0.npz", "ADCFEB, $\\chi=0$"),
             ("ADCFEB_chi_0.000433921.npz", "ADCFEB, исходное $\\chi$")]
    for col, (file, title) in enumerate(cases):
        z = np.load(src / file)
        t = z["t"]
        state = z["state"].reshape(-1, 2, 13)
        tau_periods = t / (2 * np.pi)
        samples_per_rf = max(1, int(round((len(t) - 1) / tau_periods[-1])))
        phase_locked = np.arange(0, len(t), samples_per_rf, dtype=int)
        for j, color in enumerate(("#2b6cb0", "#c05621")):
            axs[0, col].plot(state[:, j, 0], state[:, j, 1], lw=.7, color=color, label=f"частица {j+1}")
            locked_theta = np.unwrap(np.arctan2(state[phase_locked, j, 1], state[phase_locked, j, 0])) / (2 * np.pi)
            locked_periods = tau_periods[phase_locked]
            idx = locked_periods >= 239.4780149
            axs[1, col].plot(locked_periods[idx] - locked_periods[idx][0], locked_theta[idx] - locked_theta[idx][0], lw=1, color=color, label=f"частица {j+1}")
        axs[0, col].set(title=title, xlabel="$x/R$", ylabel="$y/R$", aspect="equal")
        axs[0, col].add_patch(plt.Circle((0, 0), 1, fill=False, color="0.45", ls="--", lw=.8))
        axs[1, col].set(xlabel="Периоды RF после 5 времён затухания", ylabel="Развёрнутые обороты")
        for ax in (axs[0, col], axs[1, col]):
            ax.grid(True, alpha=.22)
            ax.legend()
    fig.suptitle("Сохранённые полные 3D-парные контроли: XY и накопленные углы")
    fig.savefig(FIG / "matched_pair_trajectories.png", dpi=180)
    plt.close(fig)


def campaign_counts():
    names = [
        "ADCFEB_sobol_800p_n512_s20261118",
        "ADCFEB_focused_1000p_n384_s20261119",
        "ADCFFEB_coulomb_focused_2400p_n192_s20261120",
        "ADCFEB_barrier_challenge_2000p_n128_s20261121",
    ]
    labels = ["широкий\nSobol", "фокус\nрадиус/qm", "сильнее\nКулон", "барьерный\nконтроль"]
    counts = {k: [] for k in ("Стартовые условия", "В апертуре", "Противоположные знаки", "Устойчивая встречная пара")}
    for name in names:
        j = read_json(RES / "full_dynamics" / name / "summary.json")
        with (RES / "full_dynamics" / name / "all_runs.csv").open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        counts["Стартовые условия"].append(len(rows))
        counts["В апертуре"].append(sum(x["escaped"].lower() not in ("true", "1") for x in rows))
        counts["Противоположные знаки"].append(int(j.get("opposite_after_relax", 0)))
        counts["Устойчивая встречная пара"].append(int(j.get("stable_counter_late", 0)))
    fig, ax = plt.subplots(figsize=(8.4, 4.7), constrained_layout=True)
    x = np.arange(len(labels)); width = .19
    palette = ["#718096", "#2b6cb0", "#dd6b20", "#2f855a"]
    for i, (key, vals) in enumerate(counts.items()):
        ax.bar(x + (i - 1.5) * width, vals, width, label=key, color=palette[i])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Число случаев")
    ax.set_title("Исходы четырёх полных RF–DC кампаний")
    ax.grid(axis="y", alpha=.22)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(FIG / "dynamics_campaign_counts.png", dpi=180)
    plt.close(fig)


def hydrodynamic_models():
    path = HERE / "validation_hydrodynamics" / "results" / "matched_hydrodynamics.csv"
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    labels = ["Стокс", "Озен\nself", "Озен\npair", "RPY\npair"]
    col_to_label = {"stokes": labels[0], "oseen_self": labels[1], "oseen_pair": labels[2], "rpy_pair": labels[3]}
    rows.sort(key=lambda x: labels.index(col_to_label[x["model"]]))
    x = np.arange(len(rows)); width = .34
    turns1 = np.array([float(r["turns1_after_relax"]) for r in rows])
    turns2 = np.array([float(r["turns2_after_relax"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    ax.bar(x-width/2, turns1, width, color="#2b6cb0", label="частица 1")
    ax.bar(x+width/2, turns2, width, color="#c05621", label="частица 2")
    ax.axhline(0, color="0.25", lw=.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Обороты после релаксации за ≈799 RF-периодов")
    ax.set_ylim(-.01, .01)
    ax.set_title("Matched-IC гидродинамика: угловое движение практически затухло")
    ax.grid(axis="y", alpha=.22)
    ax.legend()
    fig.savefig(FIG / "hydrodynamic_matched_models.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    pseudopotential()
    coulomb_continuation()
    matched_phase_trajectories()
    campaign_counts()
    hydrodynamic_models()
    print(f"Saved figures to {FIG}")
