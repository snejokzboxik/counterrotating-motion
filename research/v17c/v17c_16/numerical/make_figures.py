"""Build compact, publication-style V17C.16 figures from saved results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from mfs_breakdown import gradient_matrix
from mfs_two_particle_dynamics import MFSRFField
from two_particle_oseen import local15_trial_model


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COLORS = ("#0072B2", "#D55E00")


def load_summary(tag: str) -> dict:
    return json.loads((ROOT/"results"/f"trajectory_validation_{tag}"/"summary.json")
                      .read_text(encoding="utf-8"))


def save_all(fig, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        output = FIG/f"{stem}.{ext}"
        fig.savefig(output, dpi=600 if ext == "png" else None)
        if ext == "svg":
            # Matplotlib inserts trailing spaces at SVG line wraps in path
            # data; XML whitespace is insignificant here and those spaces
            # otherwise overwhelm whitespace audits of the research branch.
            lines = output.read_text(encoding="utf-8").splitlines()
            output.write_text("\n".join(line.rstrip() for line in lines)+"\n",
                              encoding="utf-8", newline="\n")
    plt.close(fig)


def figure_01_mfs_dynamics() -> dict:
    folder = ROOT/"results"/"trajectory_validation_mfs1000_s16"
    with np.load(folder/"oseen_coulomb_nominal_start01.npz") as data:
        tau, state = data["tau"], data["state"]
    periods = tau/(2*np.pi)
    radius = np.linalg.norm(state[:, :, :2], axis=2)
    theta = np.unwrap(np.arctan2(state[:, :, 1], state[:, :, 0]), axis=0)
    theta_relative = theta-theta[0:1, :]
    z = state[:, :, 2]
    summary = json.loads((folder/"oseen_coulomb_nominal_start01.json")
                         .read_text(encoding="utf-8"))
    mean_r = summary["mean_radius_dimensionless_last_80pct"]
    mean_w = summary["mean_angular_velocity_dimensionless_last_80pct"]
    mean_z = summary["mean_axial_position_over_R_last_80pct"]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True,
                             constrained_layout=True)
    for i in range(2):
        axes[0].plot(periods, radius[:, i], color=COLORS[i], lw=0.48,
                     alpha=0.78, rasterized=True, label=f"particle {i+1}")
        axes[0].axhline(mean_r[i], color=COLORS[i], ls="--", lw=0.9)
        axes[1].plot(periods, theta_relative[:, i], color=COLORS[i], lw=0.65,
                     alpha=0.8, rasterized=True)
        start = int(0.2*len(periods))
        fit = theta_relative[start, i] + mean_w[i]*(tau[start:]-tau[start])
        axes[1].plot(periods[start:], fit, color=COLORS[i], ls="--", lw=1.0)
        axes[2].plot(periods, z[:, i], color=COLORS[i], lw=0.55,
                     alpha=0.8, rasterized=True)
        axes[2].axhline(mean_z[i], color=COLORS[i], ls="--", lw=0.9)
    for ax, ylabel in zip(axes, (r"$r_i/R$", r"$\theta_i-\theta_i(0)$ (rad)",
                                 r"$z_i/R$")):
        ax.axvline(200.0, color="0.4", ls=":", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=2, loc="upper right", frameon=False)
    axes[0].set_title("Two-particle trajectory in full 2D MFS RF field; 1000 primary periods")
    axes[-1].set_xlabel(r"time / primary RF period, $t/T_1$")
    axes[-1].set_xlim(0, periods[-1])
    save_all(fig, "fig01_mfs_trajectory")
    return {
        "question": "Does the finite-horizon MFS trajectory remain in opposite rotation and what are its radial/axial excursions?",
        "conditions": summary,
        "display": "Full sampled trajectory; dashed horizontal lines are radius/axial means, dashed sloped lines are last-80% angle-fit guides; vertical dotted line marks averaging-window start.",
    }


def figure_02_model_comparison() -> dict:
    analytic = json.loads((ROOT/"results"/"analytic_screen.json").read_text(encoding="utf-8"))
    ideal_1000 = load_summary("1000p_s16")
    mfs_1000 = load_summary("mfs1000_s16")
    ideal_by_case = {r["case"]: r for r in ideal_1000["results"]}
    mfs_by_case = {r["case"]: r for r in mfs_1000["results"]}
    models = ["Stokes avg.", "Oseen avg.", "unsteady Stokes\ncontrol",
              "exact Stokes\nideal", "exact Oseen\nideal", "exact Oseen\nMFS"]
    particle_data = []
    for i in range(2):
        a = analytic[f"particle_{i+1}"]
        particle_data.append({
            "rho": [a["stokes"]["rho"], a["oseen"]["rho"],
                    a["unsteady_stokes_control"]["rho"],
                    ideal_by_case["stokes_coulomb_nominal_control"]["mean_radius_dimensionless_last_80pct"][i],
                    ideal_by_case["oseen_coulomb_nominal_start01"]["mean_radius_dimensionless_last_80pct"][i],
                    mfs_by_case["oseen_coulomb_nominal_start01"]["mean_radius_dimensionless_last_80pct"][i]],
            "omega": [a["stokes"]["omega"], a["oseen"]["omega"],
                      a["unsteady_stokes_control"]["omega"],
                      ideal_by_case["stokes_coulomb_nominal_control"]["mean_angular_velocity_dimensionless_last_80pct"][i],
                      ideal_by_case["oseen_coulomb_nominal_start01"]["mean_angular_velocity_dimensionless_last_80pct"][i],
                      mfs_by_case["oseen_coulomb_nominal_start01"]["mean_angular_velocity_dimensionless_last_80pct"][i]],
            "radius_std_exact_ideal": ideal_by_case["oseen_coulomb_nominal_start01"]["radius_std_dimensionless_last_80pct"][i],
            "radius_std_exact_mfs": mfs_by_case["oseen_coulomb_nominal_start01"]["radius_std_dimensionless_last_80pct"][i],
        })

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), constrained_layout=True)
    x = np.arange(len(models))
    for i in range(2):
        rho = particle_data[i]["rho"]
        omega = particle_data[i]["omega"]
        axes[0, i].scatter(x, rho, color=COLORS[i], s=28, zorder=3)
        axes[0, i].errorbar(4, rho[4], yerr=particle_data[i]["radius_std_exact_ideal"],
                            fmt="none", ecolor=COLORS[i], capsize=2, lw=0.9)
        axes[0, i].errorbar(5, rho[5], yerr=particle_data[i]["radius_std_exact_mfs"],
                            fmt="none", ecolor=COLORS[i], capsize=2, lw=0.9)
        axes[0, i].set_title(f"particle {i+1}: mean radius")
        axes[0, i].set_ylabel(r"$\langle r\rangle/R$")
        axes[0, i].set_xticks(x, models, rotation=32, ha="right")
        axes[0, i].grid(axis="y", alpha=0.2)
        axes[1, i].scatter(x, omega, color=COLORS[i], s=28, zorder=3)
        axes[1, i].axhline(0.0, color="0.25", lw=0.7)
        axes[1, i].set_title(f"particle {i+1}: signed mean angular velocity")
        axes[1, i].set_ylabel(r"$\langle\omega\rangle/\Omega_1$")
        axes[1, i].set_xticks(x, models, rotation=32, ha="right")
        axes[1, i].grid(axis="y", alpha=0.2)
    fig.suptitle("Averaged branches and exact trajectories (unsteady control is not an exact ODE reference)")
    save_all(fig, "fig02_averaging_vs_trajectories")
    return {
        "question": "How do Stokes, Oseen and linear unsteady-Stokes averaged predictions compare with exact Stokes/Oseen trajectories?",
        "categorical_models": models,
        "particle_values": particle_data,
        "fixed_parameters": "35/45 um PS in air; 47.5/147.811 Hz; frozen Local15 A,B,beta; Coulomb=1 for coupled exact cases; ideal vs full 2D MFS RF field.",
        "initial_conditions": "The exact ODE runs use the nominal analytic Oseen/Stokes branch radii and velocities, gravity-set z offsets, angles (0, pi/2), RF phases (0,0).",
        "horizon_and_classifier": "1000 primary RF periods; mean over last 80%; DOP853; opposite signs classify counterrotation; uncertainty bars show radial standard deviation, not confidence intervals.",
    }


def figure_03_mfs_field_and_surface() -> dict:
    model = local15_trial_model(coulomb_scale=1.0, gravity=False)
    field = MFSRFField(model, order_per_rod=24)
    lim = 1.18
    grid = np.linspace(-lim, lim, 105)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    centers_r = 1.0/0.95
    rod_r = 0.05/0.95
    shield_r = 1.10/0.95
    keep = np.linalg.norm(points, axis=1) < shield_r
    for k in range(24):
        c = centers_r*np.array([np.cos(2*np.pi*k/24), np.sin(2*np.pi*k/24)])
        keep &= np.linalg.norm(points-c, axis=1) > rod_r*1.03
    values = np.full(len(points), np.nan)
    for lo in range(0, len(points), 256):
        ids = np.flatnonzero(keep[lo:lo+256])+lo
        if len(ids):
            target = field.r_over_D*(points[ids, 0]+1j*points[ids, 1])
            gx, gy = gradient_matrix(target, field.sources)
            p1, p2 = field.U1*field.r_over_D/field.vscale, \
                     field.U2*field.r_over_D/field.vscale
            g1x, g1y = gx@field.strengths[:, 0], gy@field.strengths[:, 0]
            g2x, g2y = gx@field.strengths[:, 1], gy@field.strengths[:, 1]
            ex = np.real(p1*g1x+p2*g2x)
            ey = np.real(p1*g1y+p2*g2y)
            values[ids] = np.hypot(ex, ey)
    zz = values.reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    ax = axes[0]
    levels = np.geomspace(np.nanpercentile(zz, 5), np.nanpercentile(zz, 99.7), 22)
    image = ax.contourf(xx, yy, zz, levels=levels, norm=matplotlib.colors.LogNorm(),
                        cmap="viridis", extend="both")
    for k in range(24):
        c = centers_r*np.array([np.cos(2*np.pi*k/24), np.sin(2*np.pi*k/24)])
        ax.add_patch(Circle(c, rod_r, fc="white", ec="black", lw=0.45, zorder=4))
    ax.add_patch(Circle((0, 0), shield_r, fill=False, ec="black", lw=0.8))
    for i, rho in enumerate((0.549365, 0.765686)):
        ax.add_patch(Circle((0, 0), rho, fill=False, ec=COLORS[i], ls="--", lw=1.0))
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(r"$x/R$")
    ax.set_ylabel(r"$y/R$")
    ax.set_title(r"(a) $|\nabla\phi_{\rm RF}|$ at RF phases $(0,0)$")
    cb = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.02)
    cb.set_label(r"$|\nabla\phi_{\rm RF}|$ (dimensionless)")

    audit = json.loads((ROOT/"results"/"mfs_surface_audit.json").read_text(encoding="utf-8"))
    orders = [x["p_collocation_per_rod"] for x in audit["MFS_convergence"]]
    field_scale_V_m = field.vscale/field.R
    rod = [x["rf_only_rod_max_surface_field_V_m"]/field_scale_V_m
           for x in audit["MFS_convergence"]]
    shield = [x["rf_only_shield_max_surface_field_V_m"]/field_scale_V_m
              for x in audit["MFS_convergence"]]
    reference_3MV = 3.0e6/field_scale_V_m
    ax = axes[1]
    ax.plot(orders, rod, "o", color=COLORS[0], label="rod surface")
    ax.plot(orders, shield, "s", color=COLORS[1], label="shield surface")
    ax.axhline(reference_3MV, color="0.25", ls=":", lw=0.9,
               label="3 MV/m reference (not a breakdown criterion)")
    ax.set_xlabel("MFS sources/collocation points per rod")
    ax.set_ylabel(r"$q_1 E_{\rm surf}/(m_1R\Omega_1^2)$")
    ax.set_title("(b) independent RF-phase-torus surface maxima")
    ax.set_xticks(orders)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.suptitle("24-electrode infinite-rod MFS field and boundary convergence")
    save_all(fig, "fig03_mfs_field_and_surface")
    return {
        "question": "What transverse RF field does the 24-rod geometry produce at one phase snapshot, and do the MFS surface maxima converge with basis order?",
        "phase_snapshot": [0.0, 0.0],
        "field_map": "Dimensionless RF gradient; electrodes/shield masked, dashed circles show mean radii of the full-MFS exact run.",
        "surface_data": audit["MFS_convergence"],
        "dimensionless_field_scale_V_m": field_scale_V_m,
        "caveat": "The 3 MV/m line is a familiar reference scale only; this figure does not establish corona or breakdown. Surface values use independent phase-torus maxima, while panel (a) is a single phase snapshot.",
    }


def figure_04_polarization_sensitivity() -> dict:
    folders = {
        "point-charge baseline": ROOT/"results"/"trajectory_validation_mfs1000_s16",
        r"induced dipole, $\epsilon_r=2.6$": ROOT/"results"/"trajectory_validation_mfs1000_dep_eps26",
    }
    window_periods = 50.0
    rows = []
    styles = [("point-charge baseline", "o", True),
              (r"induced dipole, $\epsilon_r=2.6$", "s", False)]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True,
                             constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.11,
                        hspace=0.08)
    for label, marker, filled in styles:
        folder = folders[label]
        data = np.load(folder/"oseen_coulomb_nominal_start01.npz")
        tau, state = data["tau"], data["state"]
        periods = tau/(2*np.pi)
        radius = np.linalg.norm(state[:, :, :2], axis=2)
        theta = np.unwrap(np.arctan2(state[:, :, 1], state[:, :, 0]), axis=0)
        omega = np.gradient(theta, tau, axis=0)
        edges = np.arange(0.0, 1000.0+window_periods, window_periods)
        centers = 0.5*(edges[:-1]+edges[1:])
        means_r, means_w, std_w = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (periods >= lo) & (periods < hi)
            means_r.append(np.mean(radius[mask], axis=0))
            means_w.append(np.mean(omega[mask], axis=0))
            std_w.append(np.std(omega[mask], axis=0))
        means_r, means_w, std_w = map(np.asarray, (means_r, means_w, std_w))
        rows.append({"case": label, "window_periods": window_periods,
                     "window_centers_primary_periods": centers.tolist(),
                     "mean_radius_over_R_by_window": means_r.tolist(),
                     "mean_omega_over_Omega1_by_window": means_w.tolist(),
                     "within_window_omega_std_not_statistical_error": std_w.tolist()})
        for i in range(2):
            axes[0].plot(centers, means_w[:, i], linestyle="none", marker=marker,
                         ms=4.0, mfc=COLORS[i] if filled else "none",
                         mec=COLORS[i], alpha=0.86,
                         label=(f"p{i+1} baseline" if filled
                                else rf"p{i+1} + dipole ($\epsilon_r=2.6$)"))
            axes[1].plot(centers, means_r[:, i], linestyle="none", marker=marker,
                         ms=4.0, mfc=COLORS[i] if filled else "none",
                         mec=COLORS[i], alpha=0.86)
    axes[0].axhline(0.0, color="0.2", lw=0.8)
    axes[0].set_ylabel(r"50-period mean $\langle\omega_i\rangle/\Omega_1$")
    axes[1].set_ylabel(r"50-period mean $\langle r_i\rangle/R$")
    axes[1].set_xlabel(r"time / primary RF period, $t/T_1$")
    for ax in axes:
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, fontsize=8,
               loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Sensitivity of MFS trajectories to an illustrative induced-dipole force",
                 y=0.995)
    save_all(fig, "fig04_polarization_sensitivity")
    return {
        "question": "Does a dielectric-sphere induced-dipole force materially change the two-particle counterrotating trajectory?",
        "fixed_parameters": "35/45 um polystyrene pair in air; same charges, RF/DC drive, Coulomb scale=1, MFS order=24, nominal initial state and RF phases in both runs.",
        "comparison": "Point-charge Oseen baseline versus point charge plus instantaneous real-sphere polarizability with illustrative epsilon_r=2.6.",
        "window_and_classifier": "Non-overlapping 50-primary-period windows; markers are window means, not capture statistics. Hollow squares identify induced-dipole sensitivity case.",
        "scope": "A real, frequency-independent epsilon_r is a sensitivity model only; complex dielectric response, surface charging, and mutual dipole effects are not included.",
        "series": rows,
    }


def main() -> None:
    manifest = {
        "figures": [figure_01_mfs_dynamics(), figure_02_model_comparison(),
                    figure_03_mfs_field_and_surface(),
                    figure_04_polarization_sensitivity()],
        "formats": ["PDF", "SVG", "PNG at 600 dpi"],
        "style": "DejaVu Sans, minimal grid, mathematical labels, no decorative effects",
    }
    (FIG/"FIGURE_DATA.json").write_text(json.dumps(manifest, indent=2,
                                                   ensure_ascii=False)+"\n",
                                         encoding="utf-8")
    print(FIG)


if __name__ == "__main__":
    main()
