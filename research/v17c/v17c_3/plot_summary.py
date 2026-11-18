"""Create reproducible summary plots from the saved V17C.3 CSV outputs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def rows(name):
    with (RES / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scatter(ax, data, x, y, color, title, xlabel, ylabel, cmap="coolwarm"):
    xs = np.array([float(d[x]) for d in data])
    ys = np.array([float(d[y]) for d in data])
    cs = np.array([float(d[color]) for d in data])
    escaped = np.array([d.get("escaped", "False").lower() == "true" for d in data])
    p = ax.scatter(xs[~escaped], ys[~escaped], c=cs[~escaped], cmap=cmap,
                   s=55, edgecolors="black", linewidths=.35, marker="o")
    if escaped.any():
        ax.scatter(xs[escaped], ys[escaped], c=cs[escaped], cmap=cmap,
                   s=65, linewidths=1.1, marker="x")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.25)
    return p


fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
cases = [
    ("G0_alpha_beta_map.csv", "alpha", "beta", "mean_omega",
     "G0: разреженный скан α–β", "α", "средняя ω"),
    ("G0_qm_vdc_map.csv", "q_over_m_scale", "VDC_scale", "mean_omega",
     "G0: связанный скан q/m–VDC", "масштаб q/m", "масштаб VDC"),
    ("G0_vdc_friction_map.csv", "VDC_scale", "lambda", "mean_omega",
     "G0: скан VDC–трение", "масштаб VDC", "λ"),
]
for ax, (file, x, y, color, title, xlabel, ylabel) in zip(axes.flat, cases):
    p = scatter(ax, rows(file), x, y, color, title, xlabel, ylabel)
    fig.colorbar(p, ax=ax, label="средняя ω")

single = rows("single_particle_rf_dc_scan.csv")
colors = {"G0_h50": "tab:blue", "G1": "tab:orange", "G2": "tab:green",
          "G3": "tab:red", "G5": "tab:purple"}
ax = axes.flat[3]
for geom, color in colors.items():
    selected = [d for d in single if d["geometry"] == geom]
    if not selected:
        continue
    for spin in ("rest", "predicted", "opposite"):
        subset = [d for d in selected if d["initial_spin_kind"] == spin]
        if subset:
            live = [d for d in subset if d["escaped"].lower() != "true"]
            lost = [d for d in subset if d["escaped"].lower() == "true"]
            if live:
                ax.scatter([float(d["radius_mean"]) for d in live],
                           [float(d["mean_omega"]) for d in live],
                           color=color, marker={"rest": "o", "predicted": "^", "opposite": "v"}[spin],
                           s=42, label=f"{geom}, {spin}", alpha=.85)
            if lost:
                ax.scatter([float(d["radius_mean"]) for d in lost],
                           [float(d["mean_omega"]) for d in lost],
                           color=color, marker="x", linewidths=1.2, s=55,
                           label=f"{geom}, выход")
ax.axhline(0, color="black", linewidth=.8)
ax.set_title("Одиночные старты: радиус и средняя ω")
ax.set_xlabel("средний радиус / R")
ax.set_ylabel("средняя ω")
ax.grid(alpha=.25)
ax.legend(fontsize=7, ncol=2)
fig.suptitle("V17C.3 — контрольные RF–DC сканы (крестик: выход)", fontsize=14)
fig.savefig(RES / "rf_dc_control_maps.png", dpi=180)
plt.close(fig)

# Plot the sign of the averaged RF torque from the full harmonic spectra.
torque = rows("radial_torque_dc_maps.csv")
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for geom in ("G0_h50", "G1", "G2", "G3", "G5"):
    subset = [d for d in torque if d["geometry"] == geom]
    if subset:
        ax.plot([float(d["r_over_R"]) for d in subset],
                [float(d["omega_avg_estimate"]) for d in subset],
                label=geom, linewidth=1.6)
ax.axhline(0, color="black", linewidth=.8)
ax.axvline(1, color="black", linewidth=.8, linestyle="--", label="апертура")
ax.set_xlim(0, 1.01)
ax.set_xlabel("r / R")
ax.set_ylabel("усреднённая оценка угловой скорости")
ax.set_title("Знак RF-момента по полному доступному спектру")
ax.grid(alpha=.25)
ax.legend()
fig.savefig(RES / "averaged_torque_sign_roots.png", dpi=180)
plt.close(fig)

# Full 3D path and the three requested orthogonal projections for the
# longest RF-supported orbit.  One point per RF period is saved in the NPZ.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

traj = np.load(RES / "G0_h50_positive_50000RF.npz")
state = traj["state"]
stride = 5
x, y, z = (state[::stride, j] for j in range(3))
t = traj["t"][::stride] / (2 * np.pi)
fig = plt.figure(figsize=(12, 9), constrained_layout=True)
ax3d = fig.add_subplot(2, 2, 1, projection="3d")
segments = len(x)
sc = ax3d.scatter(x, y, z, c=t, s=1.0, cmap="viridis")
ax3d.set_xlabel("x / R"); ax3d.set_ylabel("y / R"); ax3d.set_zlabel("z / R")
ax3d.set_title("3D, цвет — время")
for ax, a, b, xl, yl, title in (
    (fig.add_subplot(2, 2, 2), x, y, "x / R", "y / R", "XY"),
    (fig.add_subplot(2, 2, 3), x, z, "x / R", "z / R", "XZ"),
    (fig.add_subplot(2, 2, 4), y, z, "y / R", "z / R", "YZ"),
):
    ax.scatter(a, b, c=t, s=1.0, cmap="viridis", rasterized=True)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title); ax.grid(alpha=.25)
fig.colorbar(sc, ax=fig.axes, label="RF-периоды", shrink=.75)
fig.suptitle("G0 h50: подтверждённая положительная одиночная орбита")
fig.savefig(RES / "G0_positive_trajectory_projections.png", dpi=180)
plt.close(fig)

# Initial-condition Coulomb coupling scan: all shown points escaped before
# the ten-turn criterion; this plot is diagnostic, not an orbit map.
pair = [d for d in rows("targeted_G1_pair_scan.csv")
        if d["initial_set"] == "V17C3_canonical_mu2p7" and d["spin_case"] == "opposite"]
pair.sort(key=lambda d: float(d["chi"]))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for ax, key, label in ((axes[0], "duration_RF_periods", "до выхода, RF-периодов"),
                       (axes[1], "min_distance", "минимальное расстояние / R")):
    ax.plot([float(d["chi"]) for d in pair], [float(d[key]) for d in pair],
            "o-", color="tab:blue")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_xlabel("χ, масштаб Кулона")
    ax.set_ylabel(label)
    ax.grid(alpha=.25)
axes[0].set_title("Короткие парные старты")
axes[1].set_title("Кулоновский контроль")
fig.suptitle("G1, старт .40R/.75R, μ₂/μ₁=2.7, противоположный спин")
fig.savefig(RES / "G1_pair_coulomb_scan_diagnostic.png", dpi=180)
plt.close(fig)
