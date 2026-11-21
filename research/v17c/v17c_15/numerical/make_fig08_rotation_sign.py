"""FIG08: discrete analytic-vs-exact RF rotation-sign screen/refinement."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/rotation_sign_scan"
FIGS = ROOT / "figures"
TABLES = ROOT / "results/figure_data"
FIGS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "figure.facecolor": "white",
})
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#777777"


def load_rows():
    docs = [json.loads((DATA / "local15_sign_screen.json").read_text(encoding="utf-8"))]
    docs.extend(json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(DATA.glob("local15_sign_refinement_*.json")))
    records = []
    roots = {}
    for d in docs:
        for res in d["results"]:
            key = (float(res["nu"]), res["species"])
            roots[key] = res["leading_roots"]
            for row in res["rows"]:
                records.append({**row, "periods": d["periods"],
                                "steps_per_primary_period": d["steps_per_primary_period"],
                                "panel_sha256": d["panel_sha256"]})
    long_path = DATA / "local15_sign_transition_50000.json"
    if long_path.exists():
        long_doc = json.loads(long_path.read_text(encoding="utf-8"))
        for ensemble in long_doc["ensembles"]:
            key = (float(ensemble["nu"]), ensemble["species"])
            roots[key] = ensemble["leading_roots"]
            for trajectory in ensemble["trajectory_records"]:
                final = next(row for row in trajectory["horizons"]
                             if row["horizon_periods"] == 50_000)
                records.append({
                    "species": ensemble["species"], "nu": float(ensemble["nu"]),
                    "start_id": trajectory["start_id"],
                    "rho_mean_last_500": final["rho_mean_last_500"],
                    "omega_last_500": final["omega_last_500"],
                    "omega_mean_15k_to_50k": trajectory["omega_mean_15k_to_50k"],
                    "max_rho": trajectory["max_rho"],
                    "escaped": final["escaped_by_horizon"],
                    "escape_period": trajectory["escape_period"],
                    "sign": final["sign"], "periods": 50_000,
                    "steps_per_primary_period": long_doc["steps_per_primary_period"],
                    "panel_sha256": long_doc["panel_sha256"],
                })
    extension_path = DATA / "local15_sign_transition_120000_mixed_node.json"
    if extension_path.exists():
        extension = json.loads(extension_path.read_text(encoding="utf-8"))
        for trajectory in extension["trajectory_records"]:
            final = next(row for row in trajectory["horizons"]
                         if row["horizon_periods"] == 120_000)
            records.append({
                "species": extension["species"], "nu": float(extension["nu"]),
                "start_id": trajectory["start_id"],
                "rho_mean_last_500": final["rho_mean_last_500"],
                "omega_last_500": final["omega_last_500"],
                "omega_mean_15k_to_50k": "",
                "omega_mean_50k_to_120k": trajectory["omega_mean_50k_to_120k"],
                "omega_mean_15k_to_120k": trajectory["omega_mean_15k_to_120k"],
                "max_rho": trajectory["max_rho"],
                "escaped": final["escaped_by_horizon"],
                "escape_period": trajectory["escape_period"],
                "sign": final["sign"], "periods": 120_000,
                "steps_per_primary_period": extension["steps_per_primary_period"],
                "panel_sha256": extension["panel_sha256"],
            })
    # Keep each measured horizon as a discrete comparison; deduplicate only
    # identical start/frequency/horizon records.
    selected = {}
    for row in records:
        k = (row["species"], row["nu"], row["start_id"], row["periods"])
        selected[k] = row
    return list(selected.values()), roots


def main():
    rows, roots = load_rows()
    out_rows = []
    for row in rows:
        root_rows = roots.get((float(row["nu"]), row["species"]), [])
        stable = next((r for r in root_rows if r["stable_reduced"]), None)
        out_rows.append({**row,
            "omega_mean_15k_to_50k": row.get("omega_mean_15k_to_50k", ""),
            "omega_mean_50k_to_120k": row.get("omega_mean_50k_to_120k", ""),
            "omega_mean_15k_to_120k": row.get("omega_mean_15k_to_120k", ""),
            "leading_stable_rho": "" if stable is None else stable["rho"],
            "leading_stable_omega": "" if stable is None else stable["omega"]})
    with (TABLES / "fig08_rotation_sign.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for row in out_rows for k in row))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)

    fig, axs = plt.subplots(1, 2, figsize=(7.3, 3.35), sharey=False)
    for ax, species, color, title in zip(axs, ("species1", "species2"), (BLUE, ORANGE),
                                          ("Species 1", "Species 2")):
        frequencies = sorted({float(r["nu"]) for r in rows if r["species"] == species})
        label_by_horizon = set()
        label_cumulative = label_root = label_120k = label_cumulative_120k = False
        for nu in frequencies:
            group = [r for r in rows if r["species"] == species and float(r["nu"]) == nu]
            for horizon in sorted({int(r["periods"]) for r in group}):
                if horizon == 120_000:
                    continue
                cohort = sorted((r for r in group if int(r["periods"]) == horizon
                                 and not r["escaped"]), key=lambda x: x["start_id"])
                jitter = np.linspace(-0.006, 0.006, max(1, len(cohort)))
                marker, offset = (("^", 0.002) if horizon >= 50_000 else
                                  (("s", 0.0) if horizon >= 15_000 else ("o", -0.002)))
                label = None
                if horizon not in label_by_horizon:
                    label_by_horizon.add(horizon)
                    label = f"Exact RF, {horizon // 1000}k window"
                ax.scatter(nu + jitter + offset,
                           [r["omega_last_500"] for r in cohort],
                           s=14 if horizon >= 15_000 else 10, marker=marker,
                           facecolor=color, edgecolor="white", linewidth=0.35,
                           alpha=0.82, label=label, zorder=3)
                if horizon >= 50_000:
                    if not label_cumulative:
                        cumulative_label, label_cumulative = "Cumulative mean, 15k–50k", True
                    else:
                        cumulative_label = None
                    ax.scatter(nu + jitter + offset,
                               [r["omega_mean_15k_to_50k"] for r in cohort],
                               s=13, marker="x", color="#303030", linewidth=0.75,
                               label=cumulative_label, zorder=3)
            if species == "species1" and nu == 2.23125:
                cohort = sorted((r for r in group if int(r["periods"]) == 120_000
                                 and not r["escaped"]), key=lambda x: x["start_id"])
                jitter = np.linspace(-0.006, 0.006, max(1, len(cohort)))
                ax.scatter(nu + jitter + 0.015,
                           [r["omega_last_500"] for r in cohort],
                           s=19, marker="v", facecolor=color, edgecolor="white",
                           linewidth=0.4, alpha=0.9,
                           label="Exact RF, 120k window" if not label_120k else None,
                           zorder=4)
                label_120k = True
                ax.scatter(nu + jitter + 0.029,
                           [r["omega_mean_50k_to_120k"] for r in cohort],
                           s=18, marker="+", color="#303030", linewidth=0.9,
                           label="Cumulative mean, 50k–120k"
                           if not label_cumulative_120k else None, zorder=4)
                label_cumulative_120k = True
            for root in roots.get((nu, species), []):
                if root["stable_reduced"]:
                    ax.scatter([nu], [root["omega"]], marker="D", s=34,
                               facecolor="white", edgecolor="black", linewidth=0.85,
                               label="Leading stable root" if not label_root else None,
                               zorder=4)
                    label_root = True
        ax.axhline(0, color=GRAY, linewidth=0.7, linestyle=(0, (3, 2)))
        ax.set_title(title)
        ax.set_xlabel(r"Frequency ratio $\nu=\omega_2/\Omega_1$")
        ax.set_xlim(2.08, 2.42)
        ax.grid(axis="y", color="#dddddd", linewidth=0.45)
        if species == "species1":
            window_doc = json.loads((DATA / "local15_sign_window_audit.json").read_text(encoding="utf-8"))
            band_lo, band_hi = window_doc["observed_exact_panel_transition_band"]
            ax.axvspan(band_lo, band_hi, color="#999999", alpha=0.12,
                       label="Panel-dependent sign band", zorder=0)
            A = 0.03716194907442204
            beta = 0.017273303124515367
            lam = 0.20975541409005094
            nu_lead = beta / (4 * A**2 / (1 + lam**2)) - 1
            ax.axvline(nu_lead, color="#333333", linestyle=(0, (1, 2)), linewidth=0.8,
                       label="Leading-model zero")
            ax.text(nu_lead + 0.002, 0.91, r"$\nu_{0,\mathrm{lead}}$",
                    transform=ax.get_xaxis_transform(), fontsize=8, va="top")
            ax.set_ylim(-4.6e-4, 4.7e-4)
        else:
            ax.set_ylim(-2.0e-3, 1.0e-4)
            ax.set_ylabel(r"Late mean rotation $\bar\omega/\Omega_1$")
    axs[0].set_ylabel(r"Late mean rotation $\bar\omega/\Omega_1$")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Exact RF rotation signs vs leading averaged roots", y=1.12, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    stem = FIGS / "FIG08_LOCAL15_ROTATION_SIGN_SCREEN"
    for ext in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
        if ext == "svg":
            # Matplotlib emits trailing spaces in multiline path data; strip
            # them so the vector source passes repository whitespace checks.
            svg = path.read_text(encoding="utf-8")
            path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                            encoding="utf-8", newline="\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
