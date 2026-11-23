"""Build compact tabular audits and publication figures from saved M3 campaigns."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
FIG = BASE / "figures"
TABLES = BASE / "tables"
FIG.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "lines.linewidth": 1.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 600,
})


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def write_csv(name, rows):
    rows = list(rows)
    with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)


def main():
    full = load("full_grid_campaign.json")
    stages = full["stages"]
    finest = stages[-1]["root"]
    rows = []
    for s in stages:
        root = s["root"]
        rows.append({
            "ntheta": s["grid"][0], "nphase1": s["grid"][1], "nphase2": s["grid"][2],
            "status": s["status"], "outer_nfev": s["outer_nfev"],
            "scaled_balance_max_norm": s["scaled_balance_max_norm"],
            "wall_time_s": s["wall_time_s"],
            "r1_R": root["radii_R"][0], "r2_R": root["radii_R"][1],
            "f1_Hz": root["frequency_Hz"][0], "f2_Hz": root["frequency_Hz"][1],
            "z1_R": root["mean_z_R"][0], "z2_R": root["mean_z_R"][1],
            "dr1_finest_percent": 100*(root["radii_R"][0]/finest["radii_R"][0]-1),
            "dr2_finest_percent": 100*(root["radii_R"][1]/finest["radii_R"][1]-1),
            "df1_finest_percent": 100*(root["frequency_Hz"][0]/finest["frequency_Hz"][0]-1),
            "df2_finest_percent": 100*(root["frequency_Hz"][1]/finest["frequency_Hz"][1]-1),
        })
    write_csv("full_root_grid_convergence.csv", rows)

    # FIG01: Full root re-solved on each grid; relative values versus finest.
    x = np.arange(len(stages))
    labels = ["96×8×8", "96×12×12", "128×12×12", "192×16×16"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for p, marker in enumerate(("o", "s")):
        vals = [r[f"dr{p+1}_finest_percent"] for r in rows]
        axs[0].plot(x, vals, linestyle="none", marker=marker, label=fr"$r_{p+1}/R$")
        valsf = [r[f"df{p+1}_finest_percent"] for r in rows]
        axs[1].plot(x, valsf, linestyle="none", marker=marker, label=fr"$f_{p+1}$")
    for ax in axs:
        ax.axhline(0, color="0.35", lw=.8)
        ax.set_xticks(x, labels)
        ax.grid(True, alpha=.25)
        ax.legend(frameon=False)
    axs[0].set_ylabel("Relative change vs finest (%)")
    axs[0].set_title("Mean radii")
    axs[1].set_title("Rotation frequencies")
    fig.suptitle("M3 full six-balance roots re-solved at each Fourier grid")
    fig.tight_layout()
    savefig(fig, "FIG01_full_root_grid_convergence")

    solver = load("solver_benchmark.json")["rows"]
    write_csv("inner_solver_benchmark.csv", [
        {k: r.get(k) for k in ("particle", "method", "status", "converged", "iterations",
                               "wall_time_s", "relative_nonlinear_residual",
                               "relative_original_fast_equation_residual",
                               "minimum_fast_mode_denominator")}
        for r in solver
    ])
    # FIG02: Residual and time, preserving failed/capped solver statuses.
    methods = ["fixed_point", "adaptive", "anderson", "newton_krylov"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4))
    width = .18
    for p, offset in ((1, -.10), (2, .10)):
        rr = {r["method"]: r for r in solver if r["particle"] == p}
        xx = np.arange(len(methods)) + offset
        residuals = [max(rr[m]["relative_nonlinear_residual"], 1e-16) for m in methods]
        times = [rr[m]["wall_time_s"] for m in methods]
        bars = axs[0].bar(xx, residuals, width=width, label=f"particle {p}")
        axs[1].bar(xx, times, width=width, label=f"particle {p}")
        for bar, m in zip(bars, methods):
            if not rr[m]["converged"]:
                bar.set_hatch("///")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Relative nonlinear-map residual")
    axs[1].set_ylabel("Wall time (s)")
    for ax in axs:
        ax.set_xticks(np.arange(len(methods)), ["fixed point", "adaptive", "Anderson", "JFNK"], rotation=18)
        ax.grid(True, axis="y", alpha=.25)
        ax.legend(frameon=False)
    axs[0].set_title("Inner solve accuracy (hatched = failed)")
    axs[1].set_title("Cost at 192×16×16")
    fig.tight_layout()
    savefig(fig, "FIG02_inner_solver_comparison")

    trace_data = load("solver_residual_trace.json")
    trace_rows = []
    fig, ax = plt.subplots(figsize=(6.7, 4.0))
    for row in trace_data["rows"]:
        vals = [r for r in row["residual_trace"] if "relative_nonlinear_residual" in r]
        it = [r["iteration"] for r in vals]
        rv = [max(r["relative_nonlinear_residual"], 1e-16) for r in vals]
        trace_rows.extend({"method": row["method"], "iteration": r["iteration"],
                           "relative_nonlinear_residual": r["relative_nonlinear_residual"]}
                          for r in vals)
        ax.plot(it, rv, marker="o", ms=2.4, label=row["method"])
    ax.axhline(1e-5, color="0.35", ls="--", lw=.8, label="map tolerance")
    ax.set_yscale("log")
    ax.set_xlabel("Inner iterations (particle 1 benchmark)")
    ax.set_ylabel("Relative nonlinear-map residual")
    ax.set_title("Residual history: fast fixed-point methods can stall or diverge")
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "FIG08_inner_nonlinear_residual_history")
    write_csv("inner_solver_residual_trace.csv", trace_rows)

    # FIG03: Fourier energy by signed RF phase indices; orbital phase averaged.
    with np.load(RES / "xi_seed_192_16.npz") as z:
        coeffs = [z["xi1_hat"], z["xi2_hat"]]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True, sharey=True)
    mode_rows = []
    nphase = coeffs[0].shape[1]
    inds = np.fft.fftfreq(nphase) * nphase
    order = np.fft.fftshift(inds.astype(int))
    for p, (ax, c) in enumerate(zip(axs, coeffs), start=1):
        # Sum over both displacement components and average orbital phase.
        power = np.mean(np.sum(np.abs(c)**2, axis=-1), axis=0)
        power = np.fft.fftshift(power, axes=(0, 1))
        db = 10*np.log10(np.maximum(power / np.max(power), 1e-16))
        im = ax.imshow(db, origin="lower", aspect="equal", cmap="viridis",
                       extent=[order[0]-.5, order[-1]+.5, order[0]-.5, order[-1]+.5],
                       vmin=-100, vmax=0)
        ax.set_title(f"Particle {p}")
        ax.set_xlabel("RF2 Fourier index")
        ax.set_xticks([-8, -4, 0, 4, 7])
        ax.set_yticks([-8, -4, 0, 4, 7])
        for j, k in enumerate(order):
            for l, m in enumerate(order):
                mode_rows.append({"particle": p, "rf1_index": int(k), "rf2_index": int(m),
                                  "relative_power": float(power[j, l]/np.max(power)),
                                  "relative_power_dB": float(db[j, l])})
    axs[0].set_ylabel("RF1 Fourier index")
    cb = fig.colorbar(im, ax=axs, fraction=.035, pad=.04)
    cb.set_label("Relative displacement power (dB; phase-averaged over orbit)")
    fig.suptitle("Converged M3 micromotion spectrum, 192×16×16; retained displacement")
    fig.subplots_adjust(top=.84, wspace=.12, right=.88)
    savefig(fig, "FIG03_micromotion_fourier_spectrum")
    write_csv("micromotion_mode_power.csv", mode_rows)

    cutoff = load("cutoff_sensitivity.json")
    crows = []
    for s in cutoff["stages"]:
        roots = s["root"]
        ir = s["inner"]
        for p in range(2):
            crows.append({
                "cutoff": s["cutoff"], "particle": p+1, "status": s["status"],
                "r_R": roots["radii_R"][p], "frequency_Hz": roots["frequency_Hz"][p],
                "excluded_modes": ir[p]["number_of_excluded_rf_combination_modes"],
                "minimum_fast_denominator": ir[p]["minimum_fast_mode_denominator"],
                "relative_fast_equation_residual": ir[p]["relative_original_fast_equation_residual"],
                "rms_unresolved_slow_force": ir[p]["rms_unresolved_slow_force"],
            })
    write_csv("fast_slow_cutoff.csv", crows)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4))
    cutoffs = sorted(set(r["cutoff"] for r in crows))
    for p, marker in ((1, "o"), (2, "s")):
        rr = [r for r in crows if r["particle"] == p]
        base_r = rr[2]["r_R"]
        base_f = rr[2]["frequency_Hz"]
        axs[0].plot(cutoffs, [100*(r["r_R"]/base_r-1) for r in rr], linestyle="none", marker=marker, label=f"particle {p} radius")
        axs[1].plot(cutoffs, [100*(r["frequency_Hz"]/base_f-1) for r in rr], linestyle="none", marker=marker, label=f"particle {p} frequency")
    for ax in axs:
        ax.axhline(0, color="0.35", lw=.8)
        ax.set_xlabel(r"Fast/slow threshold $|\kappa|$")
        ax.grid(True, alpha=.25)
        ax.legend(frameon=False)
    axs[0].set_ylabel("Change from cutoff 0.05 (%)")
    axs[0].set_title("Re-solved mean radii")
    axs[1].set_title("Re-solved frequencies")
    fig.suptitle("Sensitivity to fast/slow mode partition, full M3 roots")
    fig.tight_layout()
    savefig(fig, "FIG04_cutoff_sensitivity")

    # FIG05: M2/M3 vs the same archived finite-window nominal trajectory.
    import sys
    sys.path.insert(0, str(BASE.parents[0] / "v17c_21" / "code"))
    import analytical_model as model
    d = model.load_inputs()
    nominal = stages[-1]["root"]
    v20b = d["candidate"]["published_long_replay_observables"]
    v21doc = json.loads((Path(model.__file__).resolve().parents[1] / "results" / "sensitivity_and_stability.json").read_text(encoding="utf-8"))
    m2state = v21doc["baseline_state"]
    v21 = {"radii_R": [m2state[0], m2state[2]],
           "frequency_Hz": [m2state[1]*d["f1_Hz"], m2state[3]*d["f1_Hz"]]}
    models = [
        ("V20B finite-window", v20b["late_mean_radius_R"], v20b["late_frequency_Hz"]),
        ("V21 M2", v21["radii_R"], v21["frequency_Hz"]),
        ("V21.1 M3", nominal["radii_R"], nominal["frequency_Hz"]),
    ]
    model_rows=[]
    for name, rad, freq in models:
        for p in range(2):
            model_rows.append({"model": name, "particle": p+1, "radius_R": rad[p], "frequency_Hz": freq[p],
                               "radius_error_percent_vs_V20B": 100*(rad[p]/v20b["late_mean_radius_R"][p]-1),
                               "frequency_error_percent_vs_V20B": 100*(freq[p]/v20b["late_frequency_Hz"][p]-1)})
    write_csv("M2_M3_reference_comparison.csv", model_rows)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for j, (label, key) in enumerate((("Radius", "radius_error_percent_vs_V20B"), ("Frequency", "frequency_error_percent_vs_V20B"))):
        ax=axs[j]
        for p, marker in ((1,"o"),(2,"s")):
            subset=[r for r in model_rows if r["particle"]==p]
            vals=[r[key] for r in subset]
            ax.plot([0,1,2], vals, linestyle="none", marker=marker, label=f"particle {p}")
        ax.axhline(0,color="0.35",lw=.8); ax.grid(True,alpha=.25)
        ax.set_xticks([0,1,2],["V20B ref.","M2","M3"])
        ax.set_ylabel("Error vs V20B finite-window estimate (%)")
        ax.set_title(label)
        ax.legend(frameon=False)
    fig.suptitle("Same nominal refine12_sobol17 configuration; V20B is a finite-window reference")
    fig.tight_layout()
    savefig(fig, "FIG05_M2_M3_reference_errors")

    # FIG06: conditional adiabatic M3 slow-flow eigenvalues.
    stability=load("m3_slowflow_stability.json")
    eig=np.asarray(stability["variants"][0]["eigenvalues_real_imag"],float)
    fig, ax=plt.subplots(figsize=(5.2,3.8))
    ax.scatter(eig[:,0],eig[:,1],s=28,facecolor="#2878b5",edgecolor="black",linewidth=.45)
    ax.axvline(0,color="#b22222",lw=1,ls="--")
    ax.axhline(0,color="0.45",lw=.7)
    ax.set_xlabel(r"Re$(\lambda)$ (per $\tau$)")
    ax.set_ylabel(r"Im$(\lambda)$ (per $\tau$)")
    ax.set_title("Approximate adiabatic M3 slow-flow Jacobian (10 modes)")
    ax.grid(True,alpha=.25)
    fig.text(.5,.01,"Conditional closure: quasi-steady micromotion; phase-averaged ring Coulomb. Not full-system Floquet/Lyapunov exponents.",ha="center",fontsize=7.5)
    fig.tight_layout(rect=[0,.06,1,1])
    savefig(fig, "FIG06_adiabatic_M3_slowflow_spectrum")
    write_csv("M3_slowflow_eigenvalues.csv", [
        {"mode":i+1,"real_per_tau":float(a),"imag_per_tau":float(b)} for i,(a,b) in enumerate(eig)
    ])

    # FIG07: finite-window uncertainty in the archived RK4 frequency estimate.
    win=load("reference_window_audit.json")["windows"]
    write_csv("reference_window_sensitivity.csv", [
        {"window_periods":w["window_periods"],"particle":p+1,
         **w["particles"][p]} for w in win for p in range(2)
    ])
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.3))
    window_x=np.arange(len(win))
    window_labels=[f"{w['window_periods']//1000}k" for w in win]
    for p,marker in ((0,"o"),(1,"s")):
        fs=np.array([w["particles"][p]["frequency_endpoint_Hz"] for w in win])
        axs[0].plot(window_x,fs,linestyle="none",marker=marker,label=f"particle {p+1}")
        rs=np.array([w["particles"][p]["mean_radius_R"] for w in win])
        axs[1].plot(window_x,rs,linestyle="none",marker=marker,label=f"particle {p+1}")
    axs[0].set_ylabel("Endpoint turn-count frequency (Hz)")
    axs[1].set_ylabel("Window mean radius / R")
    for ax in axs:
        ax.set_xticks(window_x,window_labels)
        ax.set_xlabel("Late analysis window (RF1 periods)"); ax.grid(True,alpha=.25); ax.legend(frameon=False)
    axs[0].set_title("Frequency window sensitivity")
    axs[1].set_title("Mean radius and slow radial oscillation")
    fig.suptitle("Archived V20B trajectory; no new integration")
    fig.tight_layout()
    savefig(fig,"FIG07_reference_window_sensitivity")

    torque=load("m3_zero_rf_torque_refine.json")
    torque_rows=[]
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.6))
    for ax,pdata in zip(axs,torque["particles"]):
        samples=sorted(pdata["samples"],key=lambda r:r["r_R"])
        radii=np.array([r["r_R"] for r in samples])
        rf=np.array([r["mean_RF_torque"] for r in samples])
        drag=np.array([r["mean_drag_torque"] for r in samples])
        ax.scatter(radii,rf,s=24,marker="o",label=r"$\langle M_{RF}\rangle$")
        ax.scatter(radii,rf+drag,s=23,marker="s",label=r"$\langle M_{RF}+M_D\rangle$")
        ax.axhline(0,color="0.35",lw=.8)
        ax.axvline(torque["leading_estimate_rcr_over_R"],color="#b22222",ls="--",lw=1,
                   label="leading estimate")
        if pdata.get("zero_torque_bracket_R"):
            lo,hi=pdata["zero_torque_bracket_R"]
            ax.axvspan(lo,hi,color="#e69f00",alpha=.22,label="M3 RF-torque bracket")
        ax.set_title(f"Particle {pdata['particle']}: {pdata['status']}")
        ax.set_xlabel(r"Radius $r/R$ at $\omega=0$")
        ax.grid(True,alpha=.25)
        ax.legend(frameon=False,fontsize=7)
        for r in samples:
            torque_rows.append({"particle":pdata["particle"],**r,
                "mean_total_torque":r["mean_RF_torque"]+r["mean_drag_torque"],
                "status":pdata["status"]})
    axs[0].set_ylabel("Dimensionless averaged torque (code normalization)")
    fig.suptitle(r"M3 zero-RF-torque radius in the isolated $\omega=0$ slice; discrete converged samples")
    fig.tight_layout()
    savefig(fig,"FIG09_M3_zero_RF_torque_slice")
    write_csv("M3_zero_RF_torque_samples.csv",torque_rows)

    held=load("heldout_m3_resolved.json")
    write_csv("holdout_m3_errors.csv", [
        {"parameter":r["parameter"],"factor":r["factor"],"status":r["status"],
         "r1_error_percent":r["radius_error_percent"][0],"r2_error_percent":r["radius_error_percent"][1],
         "f1_error_percent":r["frequency_error_percent"][0],"f2_error_percent":r["frequency_error_percent"][1],
         "reference_horizon_periods":r["RK4_reference"]["horizon_periods"]}
        for r in held["rows"]
    ])

    print(json.dumps({"figures": sorted(p.name for p in FIG.glob("FIG*")),
                      "tables": sorted(p.name for p in TABLES.glob("*.csv"))}, indent=2))


if __name__ == "__main__":
    main()
