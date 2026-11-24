"""Reproduce V17C.22 audits and publication-style summaries from frozen V20A/B/V21 data."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
V20A = ROOT / "research/v17c/v17c_20a"
V20B = ROOT / "research/v17c/v17c_20b"
V21A = ROOT / "research/v17c/v17c_21a"
OUT = ROOT / "research/v17c/v17c_22"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_figure(fig, stem: str) -> None:
    for ext in ("pdf", "svg"):
        path = FIGURES / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        if ext == "svg":
            # Matplotlib emits harmless trailing spaces in path data; trim them
            # so generated vector files pass whitespace and patch audits.
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines).rstrip() + "\n", encoding="utf-8")
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def group_perturbations(rows):
    out = {}
    for r in rows:
        q = out.setdefault(r["direction"], {"eps": [], "ratio": [], "max_ratio": [], "rate": [],
                                                "max_abs_dr": [], "max_abs_df": []})
        eps = float(r["epsilon_metric"])
        q["eps"].append(eps)
        q["ratio"].append(float(r["final_distance_metric"]) / float(r["initial_distance_metric"]))
        q["max_ratio"].append(float(r["max_distance_metric"]) / float(r["initial_distance_metric"]))
        q["rate"].append(float(r["finite_time_separation_rate_per_tau_below_0p1"]))
        q["max_abs_dr"].append(max(abs(float(p["mean_radius_R"] - n["mean_radius_R"]))
                                    for p, n in zip(r["perturbed_observables"], r["nominal_late_observables"])))
        q["max_abs_df"].append(max(abs(float(p["frequency_Hz"] - n["frequency_Hz"]))
                                    for p, n in zip(r["perturbed_observables"], r["nominal_late_observables"])))
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    qr50_path = V20A / "results/lyapunov/refine12_sobol17_240000p_seed__100p_transient__50000p_qr__s32_q20/result.json"
    ext_path = V20A / "results/extended_qr/refine12_sobol17_240000p_seed__100p_transient__20000p_extqr__s32_q20/result.json"
    fits_path = V20A / "results/lyapunov_horizon_fits.json"
    perturb_path = V20A / "results/perturbations/refine12_sobol17_240000p_seed_5000p_phase_locked.json"
    short_path = RESULTS / "short_dop853_perturbation_summary.json"
    short_npz_path = RESULTS / "short_dop853_perturbation_trajectories.npz"
    strobe_path = V20A / "results/stroboscopic/refine12_sobol17_240000p_seed_100p_transient_20000p_every_rf1.npz"
    m3_path = V21A / "results/m3_slowflow_stability.json"
    baseline_path = V20B / "results/refine12_240k_corrected_drag_summary.json"
    keys = [qr50_path, ext_path, fits_path, perturb_path, strobe_path, m3_path, baseline_path]
    for p in keys:
        if not p.exists():
            raise FileNotFoundError(p)
    qr50, ext, fits, pert, m3, baseline = [load_json(p) for p in
                                           [qr50_path, ext_path, fits_path, perturb_path, m3_path, baseline_path]]
    candidate_key = next(k for k in fits["fits"] if "refine12" in k)
    fit = fits["fits"][candidate_key]
    variant = m3["variants"][0]
    m3_eig = np.asarray([complex(x[0], x[1]) for x in variant["eigenvalues_real_imag"]])
    cond = np.asarray(qr50["lyapunov_spectrum_per_tau_descending"], float)
    extexp = np.asarray(ext["spectrum_per_tau_descending"], float)
    horizons = np.asarray(fit["horizons_periods"], float)
    top_by_horizon = np.asarray(fit["exponents_at_horizons_per_tau"], float)
    rows = pert["rows"]
    groups = group_perturbations(rows)
    st = np.load(strobe_path)
    pos = np.asarray(st["slow_positions_R"])
    phase2 = np.mod(np.asarray(st["phase2_rad"]), 2 * np.pi)
    relphase = np.mod(np.asarray(st["relative_orbital_phase_rad"]), 2 * np.pi)
    radii = np.asarray(st["radial_R"])
    period_index = np.asarray(st["period_index"])
    # The full 12D vector field divergence is the trace of the analytic state
    # Jacobian; archived QR values were independently integrated along the run.
    volume_error = abs(float(qr50["sum_exponents_per_tau"]) -
                       float(qr50["integrated_divergence_per_tau"]))
    fit_rows = []
    for i in range(top_by_horizon.shape[1]):
        for j, horizon in enumerate(horizons):
            fit_rows.append({"conditional_exponent_rank": i + 1, "horizon_RF1_periods": int(horizon),
                             "lambda_per_tau": float(top_by_horizon[j, i])})
    with (RESULTS / "conditional_qr_horizon_spectrum.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fit_rows[0])); w.writeheader(); w.writerows(fit_rows)
    perturb_rows = []
    for direction, d in groups.items():
        for i, eps in enumerate(d["eps"]):
            perturb_rows.append({"direction": direction, "epsilon": eps, "distance_ratio_final": d["ratio"][i],
                                 "distance_ratio_max": d["max_ratio"][i], "finite_rate_per_tau": d["rate"][i],
                                 "max_abs_mean_radius_difference_R": d["max_abs_dr"][i],
                                 "max_abs_mean_frequency_difference_Hz": d["max_abs_df"][i]})
    with (RESULTS / "phase_locked_perturbation_reanalysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perturb_rows[0])); w.writeheader(); w.writerows(perturb_rows)

    # Figure 01: two weak conditional exponents and the ten strongly contracting
    # mechanical exponents; all points are measured finite-time quantities.
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.1, 1.3]})
    rank = np.arange(1, 13)
    ax[0].scatter(rank, cond, marker="o", s=32, label="Full RF, conditional 12D QR (50k periods)", color="#156082")
    ax[0].scatter(np.arange(1, 11), np.sort(m3_eig.real)[::-1], marker="x", s=42,
                  label="Adiabatic M3 slow-flow eigenvalue real parts", color="#c04b22")
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set(xlabel="Descending rank", ylabel=r"Rate [$\tau^{-1}$]", title="Finite-time spectra")
    ax[0].set_xticks(rank); ax[0].grid(alpha=.2); ax[0].legend(fontsize=7, loc="upper right")
    for idx, color, label in [(0, "#156082", r"$\lambda_1$"), (1, "#6b8e23", r"$\lambda_2$")]:
        ax[1].plot(horizons, top_by_horizon[:, idx], "o-", color=color, label=label)
    ax[1].scatter([50000, 50000], cond[:2], marker="s", color=["#156082", "#6b8e23"], s=28)
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set(xlabel="Accumulation horizon [RF1 periods]", ylabel=r"Conditional $\lambda$ [$\tau^{-1}$]",
                                         title="Near-zero directions")
    ax[1].grid(alpha=.2); ax[1].legend(frameon=False)
    fig.suptitle("V17C.22 — Full RF and M3 stability evidence", fontsize=11)
    fig.tight_layout(); save_figure(fig, "FIG01_lyapunov_and_m3_spectra")

    # Figure 02: end distance measured at fixed carrier phases grows for phase
    # sensitive perturbations even as averaged observables remain nearly equal.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    styles = {"azimuth_position_p1": ("#b7472a", "P1 azimuth"), "azimuth_position_p2": ("#e0a02b", "P2 azimuth"),
              "radial_position_p1": ("#2878a5", "P1 radial position"), "tangential_velocity_p1": ("#548235", "P1 tangential velocity"),
              "random_full_state": ("#73559b", "Random full state")}
    for name, d in groups.items():
        c, label = styles[name]
        ax.loglog(d["eps"], d["ratio"], "o-", color=c, label=label)
    ax.axhline(1.0, color="black", lw=.8, ls="--")
    ax.set(xlabel="Initial perturbation norm", ylabel="Final / initial phase-locked state distance",
           title="5000-period perturbations at fixed external RF phases")
    ax.grid(which="both", alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); save_figure(fig, "FIG02_phase_locked_perturbation_growth")

    # Figure 03: raw-state separation is largely phase-like; late averaged
    # radii/frequencies barely change.  Frequency error is shown absolute.
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for name, d in groups.items():
        c, label = styles[name]
        ax[0].loglog(d["eps"], np.maximum(d["max_abs_dr"], 1e-18), "o-", color=c, label=label)
        ax[1].loglog(d["eps"], np.maximum(d["max_abs_df"], 1e-18), "o-", color=c, label=label)
    ax[0].set(xlabel="Initial perturbation norm", ylabel=r"Max $|\Delta\bar r|$ [R]", title="Late mean radii")
    ax[1].set(xlabel="Initial perturbation norm", ylabel=r"Max $|\Delta f|$ [Hz]", title="Late mean frequencies")
    for a in ax: a.grid(which="both", alpha=.2)
    ax[1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Small changes in averaged observables do not imply phase-space convergence")
    fig.tight_layout(); save_figure(fig, "FIG03_phase_dephasing_vs_observables")

    # Figure 04: scale separation proxies.  The gray region is excluded from
    # the fast inverse by the documented cutoff, not a complete spectrum gap.
    omega_orb = np.asarray([0.0010944455997807332, -0.0009747968388241109])
    slow = float(variant["max_real_eigenvalue"])
    vals = [abs(omega_orb[0]), abs(omega_orb[1]), 1.0, 3.2196773638017477, abs(slow), 0.05]
    labels = [r"$|\omega_1|$", r"$|\omega_2|$", r"$\Omega_1$", r"$\Omega_2$", r"$|\lambda_{M3,max}|$", r"$\kappa_c$ cutoff"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = ["#2878a5", "#2878a5", "#b7472a", "#b7472a", "#548235", "#73559b"]
    ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.set_yscale("log"); ax.set_xticks(np.arange(len(vals)), labels)
    ax.set_ylabel(r"Frequency / decay rate [$\tau^{-1}$]")
    ax.set_title("Equilibrium orbital rates, RF carriers, slow-flow decay and HB cutoff")
    ax.grid(axis="y", which="both", alpha=.2)
    ax.text(0.02, 0.95, "At the fixed orbit: |ω| ≪ ΩRF.\nAfter perturbation: |Re λslow| = 0.106, above κc=0.05.",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout(); save_figure(fig, "FIG04_timescale_and_cutoff_audit")

    # Figure 06: projected RF1-section cloud.  This is an observed trajectory
    # projection, not a fitted invariant torus or a dimension estimate.
    theta1 = np.arctan2(pos[:, 1], pos[:, 0])
    theta2 = np.arctan2(pos[:, 4], pos[:, 3])
    rel = np.mod(theta1 - theta2, 2 * np.pi)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    sc = ax[0].scatter(phase2, rel, c=period_index, s=2, cmap="viridis", rasterized=True)
    ax[0].set(xlabel=r"RF2 phase on RF1 section [rad]", ylabel=r"Projected relative orbital phase [rad]",
              title="One 20,000-period RF1-section record")
    ax[0].set_xlim(0, 2*np.pi); ax[0].set_ylim(0, 2*np.pi)
    ax[1].scatter(radii[:, 0], radii[:, 1], c=phase2, s=2, cmap="twilight", rasterized=True)
    ax[1].set(xlabel=r"$r_1/R$", ylabel=r"$r_2/R$", title="Observed radius projection")
    fig.colorbar(sc, ax=ax[0], label="RF1 period index", fraction=.047)
    fig.suptitle("Observed section geometry — insufficient by itself to identify a torus")
    fig.tight_layout(); save_figure(fig, "FIG06_observed_invariant_set_projection")

    # Figure 05: short independent DOP853 response from two late archived states.
    # Distances are phase-locked full-state distances; the figure intentionally
    # does not label them transverse distances.
    short_summary = load_json(short_path)
    short = np.load(short_npz_path)
    times = np.asarray(short["periods"])
    cubes = np.asarray(short["states"])
    names = [r["case"] for r in short_summary["runs"][0]["rows"]]
    vscale = np.array([0.0038162658249025747] * 3 + [0.004575900271655408] * 3)
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 5.6), sharex=True, sharey=True)
    perturb_order = ["p1_radial_position_1e-4R", "p1_azimuth_1e-4rad",
                     "p1_axial_position_1e-4R", "p1_tangential_velocity_1e-4scale"]
    labels = {"p1_radial_position_1e-4R": "Radial position",
              "p1_azimuth_1e-4rad": "Azimuth",
              "p1_axial_position_1e-4R": "Axial position",
              "p1_tangential_velocity_1e-4scale": "Tangential velocity"}
    for panel, name in zip(axs.ravel(), perturb_order):
        k = names.index(name)
        for seed_idx, seed_label, color, style in [(0, "238000-period seed", "#2878a5", "-"),
                                                    (1, "240000-period seed", "#b7472a", "--")]:
            y = cubes[seed_idx, :, :, k]
            ref = cubes[seed_idx, :, :, 0]
            delta = y - ref
            metric = np.sqrt(np.sum(delta[:, [0, 1, 2, 6, 7, 8]] ** 2, axis=1) +
                             np.sum((delta[:, [3, 4, 5, 9, 10, 11]] / vscale) ** 2, axis=1))
            panel.semilogy(times[seed_idx], np.maximum(metric / metric[0], 1e-9),
                           color=color, ls=style, lw=1.1, label=seed_label)
        panel.axhline(1.0, color="black", lw=.7, ls=":")
        panel.set_title(labels[name], fontsize=9)
        panel.grid(which="both", alpha=.2)
        panel.legend(frameon=False, fontsize=6, loc="best")
    for panel in axs[-1, :]: panel.set_xlabel("Elapsed RF1 periods")
    for panel in axs[:, 0]: panel.set_ylabel("Phase-locked distance / initial")
    fig.suptitle("DOP853 perturbation response; 300 RF1 periods; fixed carrier phases", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96)); save_figure(fig, "FIG05_short_full_rf_perturbation_response")

    # Figure 07: M3 and full-RF conditional spectra have ten strongly
    # contracting rates in the same broad band; two extra weak directions are
    # not represented by the orbit-averaged M3 state.
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.scatter(np.arange(1, 13), cond, s=36, marker="o", color="#156082", label="Full RF conditional 12D QR")
    ax.scatter(np.arange(1, 11), np.sort(m3_eig.real)[::-1], s=48, marker="x", color="#c04b22", label="M3 10D slow-flow Re eigenvalues")
    ax.scatter(np.arange(1, 15), extexp, s=16, marker=".", color="#8c6d31", alpha=.65,
               label="Extended 14D QR (20k; includes 2 exact RF-phase zeros)")
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="Rank within each spectrum (not a mode correspondence)", ylabel=r"Rate [$\tau^{-1}$]",
           title="Spectral comparison; rank matching is not a proof of modal identity")
    ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); save_figure(fig, "FIG07_m3_vs_full_rf_stability_spectra")

    reanalysis = {
        "source_commit": "894c6896ce5927371708037ffa4a4021189d2b54",
        "baseline_parameters": qr50["parameters"],
        "source_file_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in keys},
        "conditional_qr": {
            "horizon_periods": fit["horizons_periods"],
            "first_two_exponents_at_horizons_per_tau": [[float(x[0]), float(x[1])] for x in top_by_horizon],
            "spectrum_50000_periods_per_tau": cond.tolist(),
            "sum_per_tau": float(qr50["sum_exponents_per_tau"]),
            "integrated_trace_per_tau": float(qr50["integrated_divergence_per_tau"]),
            "sum_trace_absolute_difference_per_tau": volume_error,
            "max_step_per_period": qr50["integration"]["steps_per_rf_period"],
            "qr_interval_periods": qr50["integration"]["qr_interval_periods"],
        },
        "extended_qr_20000_periods_per_tau": extexp.tolist(),
        "extended_exact_external_phase_rates": ext["phase_rates"],
        "m3_adiabatic_slow_flow": {
            "root_state": variant["root_state"],
            "eigenvalues_real_imag": variant["eigenvalues_real_imag"],
            "max_real_per_tau": float(variant["max_real_eigenvalue"]),
            "inner_solve_status": variant["observables_inner_solve_status"],
            "inner_residuals": variant["observables_inner_residual"],
            "approximation": variant["approximation"],
        },
        "phase_locked_perturbations_5000_periods": {
            "initial_rf_phase": pert["rf_phase_rad_fixed"],
            "directions": {name: {"final_over_initial_distance": d["ratio"],
                                  "max_over_initial_distance": d["max_ratio"],
                                  "finite_rate_per_tau": d["rate"],
                                  "max_abs_late_mean_radius_difference_R": d["max_abs_dr"],
                                  "max_abs_late_frequency_difference_Hz": d["max_abs_df"]}
                           for name, d in groups.items()},
            "source_period": pert["source_period"],
        },
        "section_record": {
            "samples": int(len(period_index)), "start_rf1_period": float(period_index[0]),
            "end_rf1_period": float(period_index[-1]),
            "sampling_interval_rf1_periods": float(np.median(np.diff(period_index))),
            "phase2_start_end_rad": [float(phase2[0]), float(phase2[-1])],
            "interpretation": "Single projected orbit record; not a reconstruction proof, no invariant dimension assigned.",
        },
        "short_full_rf_perturbation_recheck": {
            "summary_file_sha256": sha256(short_path),
            "trajectory_file_sha256": sha256(short_npz_path),
            "source_periods": [int(x["source_period"]) for x in short_summary["runs"]],
            "horizon_rf1_periods": 300,
            "method": "Independent Python NumPy RHS with SciPy DOP853; 12D state variational Jacobian checked separately against centered differences.",
            "maximum_steps_per_rf1_period": 96,
            "rtol_atol": [1e-9, 1e-11],
            "interpretation": "Short-horizon local response only. Radial/azimuthal phase-locked state distances grow to about 2e-3 to 2.7e-3 from 1e-4 seeds, while their late-window mean radii/frequencies remain close; axial perturbations decay strongly in this sample. The horizon does not resolve long-term orbital frequencies or prove attraction.",
        },
        "numerical_conclusion": {
            "positive_conditional_exponent_seen_in_archives": bool(np.max(cond) > 1e-5),
            "two_weak_conditional_directions_finite_horizon": bool(np.sum(np.abs(cond) < 1e-4) >= 2),
            "m3_has_ten_stable_linear_modes": bool(np.max(m3_eig.real) < 0),
            "attractor_or_torus_proved": False,
            "strongest_status": "LONG_LIVED_COUNTERROTATION; weak phase-like directions remain unresolved between true neutrality and very slow contraction",
        },
    }
    (RESULTS / "stability_evidence_audit.json").write_text(
        json.dumps(reanalysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str((RESULTS / "stability_evidence_audit.json").relative_to(ROOT)),
                      "qr_top2_50k": cond[:2].tolist(), "m3_max_real": float(variant["max_real_eigenvalue"]),
                      "qr_trace_error": volume_error, "n_perturb_runs": len(rows),
                      "section_samples": len(period_index), "figures": 7}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
