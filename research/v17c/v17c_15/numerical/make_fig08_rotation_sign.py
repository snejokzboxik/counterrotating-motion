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
    # If a refined 15k result exists at the same nu, retain only that horizon.
    selected = {}
    for row in records:
        k = (row["species"], row["nu"], row["start_id"])
        if k not in selected or row["periods"] > selected[k]["periods"]:
            selected[k] = row
    return list(selected.values()), roots


def main():
    rows, roots = load_rows()
    out_rows = []
    for row in rows:
        root_rows = roots.get((float(row["nu"]), row["species"]), [])
        stable = next((r for r in root_rows if r["stable_reduced"]), None)
        out_rows.append({**row,
            "leading_stable_rho": "" if stable is None else stable["rho"],
            "leading_stable_omega": "" if stable is None else stable["omega"]})
    with (TABLES / "fig08_rotation_sign.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)

    fig, axs = plt.subplots(1, 2, figsize=(7.3, 3.35), sharey=False)
    for ax, species, color, title in zip(axs, ("species1", "species2"), (BLUE, ORANGE),
                                          ("Species 1", "Species 2")):
        frequencies = sorted({float(r["nu"]) for r in rows if r["species"] == species})
        label_3k = label_15k = label_root = False
        for nu in frequencies:
            group = [r for r in rows if r["species"] == species and float(r["nu"]) == nu]
            kept = sorted((r for r in group if not r["escaped"]), key=lambda x: x["start_id"])
            jitter = np.linspace(-0.006, 0.006, max(1, len(kept)))
            refined = bool(kept and max(r["periods"] for r in kept) >= 15000)
            label = None
            if refined and not label_15k:
                label, label_15k = "Exact RF, 15k", True
            elif not refined and not label_3k:
                label, label_3k = "Exact RF, 3k", True
            ax.scatter(nu + jitter, [r["omega_last_500"] for r in kept],
                       s=13 if refined else 10, marker="s" if refined else "o",
                       facecolor=color, edgecolor="white", linewidth=0.35, alpha=0.82,
                       label=label,
                       zorder=3)
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
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
