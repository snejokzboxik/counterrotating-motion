"""Publication-style plots from stored, converged M3 diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140, "savefig.dpi": 600})

d = json.loads((ROOT / "data/m3_nominal_torque_decomposition.json").read_text())
fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
colors = ["#2166ac", "#b2182b"]
for p, color in zip(d["particles"], colors):
    ax.bar(np.arange(8) + (0.18 if p["index"] == 1 else -0.18),
           [x["torque"] for x in p["m3_modes"]], width=.34,
           color=color, label=f"particle {p['index']} (r/R={p['r_over_R']:.3f})")
ax.axhline(0, color="black", lw=.7)
ax.set_xticks(np.arange(8), [str(x["k"]) for x in d["particles"][0]["m3_modes"]])
ax.set_xlabel(r"azimuthal MFS index $k$")
ax.set_ylabel(r"mean RF torque $M/(m_iR^2\Omega_1^2)$")
ax.legend(frameon=False, ncols=2)
fig.savefig(FIG / "FIG01_m3_harmonic_torques.pdf")
fig.savefig(FIG / "FIG01_m3_harmonic_torques.svg")
fig.savefig(FIG / "FIG01_m3_harmonic_torques.png", dpi=600)
plt.close(fig)

v = json.loads((ROOT / "data/m3_fixed_radius_torque_vs_omega.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True, constrained_layout=True)
for ax, particle, color in zip(axes, (1, 2), colors):
    rows = sorted((x for x in v["rows"] if x["particle"] == particle),
                  key=lambda x: x["omega_over_Omega1"])
    omega = np.asarray([x["omega_over_Omega1"] for x in rows])
    rf = np.asarray([x["RF_torque"] for x in rows])
    drag = np.asarray([x["drag_torque"] for x in rows])
    net = np.asarray([x["net_torque_without_ring_averaged_Coulomb"] for x in rows])
    ax.plot(omega, rf, "o", ms=4, color=color, label="RF")
    ax.plot(omega, drag, "s", ms=3.5, color="#4d9221", label="drag")
    ax.plot(omega, net, "D", ms=3.2, color="#762a83", label="RF + drag")
    ax.axhline(0, color="black", lw=.65)
    ax.axvline(rows[0]["omega_over_Omega1"], color="0.45", lw=.8, ls=":")
    ax.set_title(f"particle {particle}, fixed r/R={rows[0]['r_over_R_fixed']:.3f}")
    ax.set_xlabel(r"slow angular rate $\omega/\Omega_1$")
axes[0].set_ylabel(r"mean torque $M/(m_iR^2\Omega_1^2)$")
axes[1].legend(frameon=False)
fig.savefig(FIG / "FIG02_m3_fixed_radius_torque_vs_omega.pdf")
fig.savefig(FIG / "FIG02_m3_fixed_radius_torque_vs_omega.svg")
fig.savefig(FIG / "FIG02_m3_fixed_radius_torque_vs_omega.png", dpi=600)
plt.close(fig)

