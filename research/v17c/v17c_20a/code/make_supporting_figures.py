from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
}
COLORS = {"candidate52": "#0072B2", "local15": "#D55E00",
          "refine12_sobol17": "#009E73", "refine12_B115": "#CC79A7"}
LABELS = {"candidate52": "candidate52", "local15": "Local15",
          "refine12_sobol17": "refine12 baseline", "refine12_B115": "refine12, B×1.15"}


def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def multistart_rows():
    folder = RES / "multistart"
    base = json.loads((folder / "same_phase_multistart_summary.json").read_text(encoding="utf-8"))
    b115 = json.loads((folder / "same_phase_multistart_B115_summary.json").read_text(encoding="utf-8"))
    rows = []
    for record, variant in ((base, None), (b115, "refine12_B115")):
        for r in record["results"]:
            if r["status"] != 0:
                rows.append({"variant": variant or r["candidate"], "candidate": r["candidate"],
                             "start": r["run_id"], "outcome": r["outcome"], "retained": False})
                continue
            rows.append({"variant": variant or r["candidate"], "candidate": r["candidate"],
                         "start": r["run_id"], "outcome": r["outcome"], "retained": True,
                         "frequency_Hz": r["late_frequency_Hz"],
                         "mean_radius_R": r["late_mean_radius_R"],
                         "turns": r["late_turns"],
                         "ensemble": r["ensemble"]})
    return rows


def figure05(rows):
    names = ["candidate52", "local15", "refine12_sobol17", "refine12_B115"]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), constrained_layout=True)
    measures = [("frequency_Hz", 0, "particle 1 mean frequency (Hz)"),
                ("frequency_Hz", 1, "particle 2 mean frequency (Hz)"),
                ("mean_radius_R", None, "late mean radius (R)")]
    rng = np.random.default_rng(11)
    for ax, (key, index, ylabel) in zip(axes, measures):
        for i, name in enumerate(names):
            subset = [r for r in rows if r["variant"] == name and r["retained"]]
            vals = []
            for r in subset:
                if index is None:
                    vals.extend(r[key])
                else:
                    vals.append(r[key][index])
            if vals:
                # Deterministic horizontal jitter only separates discrete starts.
                jitter = rng.uniform(-0.10, 0.10, len(vals))
                ax.scatter(i + jitter, vals, s=22, color=COLORS[name], alpha=.8,
                           edgecolor="white", linewidth=.35, zorder=3)
                ax.plot([i-.14, i+.14], [np.mean(vals)]*2, color="black", lw=1.1)
        ax.axhline(0, color="0.55", lw=.7) if "frequency" in key else None
        ax.set_xticks(x, [LABELS[n] for n in names], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.9", lw=.6)
    axes[0].set_title("Rotation sign and frequency")
    axes[1].set_title("Opposite rotation branch")
    axes[2].set_title("Late radii across starts")
    fig.suptitle("Same RF phases within each candidate; 15,000 RF1 periods, last 10,000 analyzed", fontsize=10)
    save(fig, "FIG05_same_phase_multistart")


def figure06(rows):
    names = ["candidate52", "local15", "refine12_sobol17", "refine12_B115"]
    groups = []
    for name in names:
        subset = [r for r in rows if r["variant"] == name]
        ncr = sum(r["retained"] and r["outcome"] == "CR(+−)" for r in subset)
        # The non-success category is kept explicit; this small panel does not resolve
        # the nature of the failed trajectory beyond the integrator's recorded event.
        failures = len(subset) - ncr
        groups.append((ncr, failures, len(subset)))
    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    x = np.arange(len(names))
    cr = np.array([g[0] for g in groups]); other = np.array([g[1] for g in groups])
    ax.bar(x, cr, color="#009E73", label="CR(+/−) retained")
    ax.bar(x, other, bottom=cr, color="#D9D9D9", edgecolor="0.55", label="escape / other recorded outcome")
    for i, (ncr, nfail, n) in enumerate(groups):
        ax.text(i, n + .15, f"{ncr}/{n} CR", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [LABELS[n] for n in names], rotation=20, ha="right")
    ax.set_ylabel("Number of starts (n=8 each)")
    ax.set_ylim(0, 9.5)
    ax.set_title("Finite same-phase start panel: outcome counts, not basin probabilities")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    ax.grid(axis="y", color="0.9", lw=.6)
    save(fig, "FIG06_same_phase_outcome_counts")
    (RES / "multistart" / "FIG06_data.json").write_text(json.dumps({
        "question": "Do these selected same-phase mechanical starts converge to the same observed outcome class?",
        "horizon_periods": 15000, "measurement_periods": 10000,
        "starts_per_variant": 8, "variants": [
            {"name": n, "counterrotating_retained": int(a), "other_or_escape": int(b), "n": int(c)}
            for n, (a, b, c) in zip(names, groups)],
        "caveat": "Selected V20B frozen-start panel at fixed forcing phases; counts are not universal basin probabilities."
    }, indent=2) + "\n", encoding="utf-8")


def figure08():
    fits = json.loads((RES / "lyapunov_horizon_fits.json").read_text(encoding="utf-8"))["fits"]
    names = ["candidate52", "local15", "refine12_sobol17"]
    full = {}
    for n in names:
        full[n] = fits[n]["extra_50k"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.45]})
    colors = [COLORS[n] for n in names]
    x = np.arange(3)
    for mode, offset, marker, label in ((0, -.12, "o", r"$lambda_1$"), (1, .12, "s", r"$lambda_2$")):
        axes[0].scatter(x + offset, [full[n][mode] for n in names], color=colors,
                        marker=marker, s=40, label=label, zorder=3)
    axes[0].axhline(0, color="0.45", lw=.8)
    axes[0].set_xticks(x, ["candidate52", "Local15", "refine12"], rotation=20, ha="right")
    axes[0].set_ylabel(r"Conditional mechanical exponent ($\tau^{-1}$)")
    axes[0].set_title("Two leading directions at 50k")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="0.9", lw=.6)

    # Show all 12 exponents as discrete markers. Symlog retains sign and resolves
    # the near-zero pair without hiding the strongly contracting directions.
    for i, n in enumerate(names):
        vals = np.asarray(full[n])
        axes[1].scatter(np.arange(1, 13) + (i-1)*.10, vals, s=20,
                        color=COLORS[n], alpha=.82, label=LABELS[n])
    axes[1].axhline(0, color="0.45", lw=.8)
    axes[1].set_yscale("symlog", linthresh=1e-5, linscale=.9)
    axes[1].set_xticks(np.arange(1, 13))
    axes[1].set_xlabel("Ordered mechanical QR mode")
    axes[1].set_ylabel(r"Exponent ($\tau^{-1}$), symlog scale")
    axes[1].set_title("Full 12D conditional spectra")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(axis="y", color="0.9", lw=.6)
    fig.suptitle("All candidates: two near-neutral modes and ten contracting modes at finite horizon", fontsize=10)
    save(fig, "FIG08_candidate_spectrum_comparison")
    (RES / "candidate_spectra_50k.json").write_text(json.dumps({
        "question": "How do the full conditional mechanical QR spectra compare across the three validated candidates?",
        "horizon_periods": 50000, "units": "inverse dimensionless time tau",
        "spectra_per_tau_descending": full,
        "interpretation": "Finite-time spectra. The first two values are near neutral at this horizon; regression and start sensitivity prevent assigning exact asymptotic signs."
    }, indent=2) + "\n", encoding="utf-8")


def main():
    rows = multistart_rows()
    (RES / "multistart" / "FIG05_data.json").write_text(json.dumps({
        "question": "Do independent same-phase starts converge to similar late rotations/radii?",
        "horizon_periods": 15000, "skip_periods": 5000,
        "data": rows,
        "caveat": "Selected starts and finite windows; late-observable clustering is not proof of an attractor."
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figure05(rows)
    figure06(rows)
    figure08()


if __name__ == "__main__":
    main()
