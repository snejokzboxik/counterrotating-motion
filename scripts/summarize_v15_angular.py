"""Summarize matched V15 ideal-field angular-momentum pilot checkpoints."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v15"
CHECKPOINT = OUT / "ideal_chiral_checkpoint.jsonl"


def ci(values):
    a = np.asarray(values, float)
    m = float(a.mean())
    h = float(student_t.ppf(.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else math.nan
    return m, m-h, m+h


def main():
    records = [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf8").splitlines() if line]
    groups = defaultdict(list)
    for r in records:
        groups[(r["point_id"], r["delta_rad"], r["tau0_rad"], r["config"], r["replica"])].append(r)
    summary = []
    replicate = {}
    for key, rr in groups.items():
        rr.sort(key=lambda x: (x["pair_index"], -x["Lz_sign"]))
        vals = {
            "P_plus": np.mean([x["status"] == "retained" for x in rr if x["Lz_sign"] > 0]),
            "P_minus": np.mean([x["status"] == "retained" for x in rr if x["Lz_sign"] < 0]),
            "Delta_chi": np.mean([x["status"] == "retained" for x in rr if x["Lz_sign"] > 0]) - np.mean([x["status"] == "retained" for x in rr if x["Lz_sign"] < 0]),
            "P_total": np.mean([x["status"] == "retained" for x in rr]),
            "Delta_Lz_all": np.mean([x["Lz_end_dimensionless"] - x["Lz0_dimensionless"] for x in rr]),
            "Delta_Lz_survivors": np.mean([x["Lz_end_dimensionless"] - x["Lz0_dimensionless"] for x in rr if x["status"] == "retained"]) if any(x["status"] == "retained" for x in rr) else math.nan,
            "Delta_Lz_escaped": np.mean([x["Lz_end_dimensionless"] - x["Lz0_dimensionless"] for x in rr if x["status"] == "escaped"]) if any(x["status"] == "escaped" for x in rr) else math.nan,
            "Torque_integral": np.mean([x["torque_integral_dimensionless"] for x in rr]),
            "Torque_residual_abs_max": max(abs(x["torque_identity_residual"]) for x in rr),
            "Median_exit_rf": float(np.median([x["exit_rf_step_estimate"] for x in rr if x["status"] == "escaped"])) if any(x["status"] == "escaped" for x in rr) else math.nan,
        }
        replicate[key] = vals
    base = defaultdict(lambda: defaultdict(list))
    for (point, delta, tau0, config, rep), vals in replicate.items():
        for metric, value in vals.items():
            base[(point, delta, tau0, config)][metric].append((rep, value))
    outrows = []
    for (point, delta, tau0, config), metrics in sorted(base.items()):
        for metric, pairs in metrics.items():
            if metric == "Torque_residual_abs_max":
                mean, lo, hi = max(v for _, v in pairs), None, None
                reps = [v for _, v in pairs]
            else:
                reps = [v for _, v in pairs if np.isfinite(v)]
                mean, lo, hi = ci(reps) if reps else (None, None, None)
            outrows.append({"point_id":point,"delta_rad":delta,"tau0_rad":tau0,"config":config,
                            "metric":metric,"mean":mean,"ci95_low":lo,"ci95_high":hi,
                            "replicate_values":json.dumps(reps,separators=(",",":"))})
    path = OUT / "ideal_angular_momentum_summary.csv"
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(outrows[0])); w.writeheader(); w.writerows(outrows)

    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    # Phase-conditioned chirality: Q and standing-wave controls should be zero.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True, sharey=True)
    points = ("A9", "gain_negative", "gain_positive", "floquet_stable")
    colors = {"RO_PLUS":"#a51c30", "RO_MINUS":"#2455a4", "SO_equal_harmonic":"#3c8d5a"}
    for ax, point in zip(axes.flat, points):
        for config in colors:
            dvals = sorted({d for (p,d,t,c) in base if p == point and c == config})
            means=[]; lows=[]; highs=[]
            for d in dvals:
                # Display tau0=0 slice; other injection phases are present in CSV.
                vals = replicate.get((point,d,0.0,config,0))
                allrep = [replicate[(point,d,0.0,config,r)]["Delta_chi"] for r in range(4)] if all((point,d,0.0,config,r) in replicate for r in range(4)) else []
                if not allrep: continue
                m,lo,hi=ci(allrep); means.append(m); lows.append(m-lo); highs.append(hi-m)
            if means:
                ax.errorbar(np.degrees(dvals[:len(means)]), np.array(means)*100,
                            yerr=np.array([lows,highs])*100, fmt="o", capsize=3,
                            color=colors[config], label=config)
        ax.axhline(0,color="black",lw=.7)
        ax.set_title(point); ax.grid(alpha=.25)
    axes[1,0].set_xlabel("Относительная фаза δ, град")
    axes[1,1].set_xlabel("Относительная фаза δ, град")
    axes[0,0].set_ylabel("Δχ = P₊ − P₋, п.п.")
    axes[1,0].set_ylabel("Δχ = P₊ − P₋, п.п.")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("V15: направленное удержание, идеальное поле, τ₀=0°, N_RF=100")
    fig.tight_layout()
    fig.savefig(figdir/"v15_ideal_chirality_by_delta.png",dpi=220)
    fig.savefig(figdir/"v15_ideal_chirality_by_delta.svg")
    plt.close(fig)

    # Angular momentum balance at A9, δ=τ0=0.
    configs=("Q","SO_equal_harmonic","SO_equal_rms","RO_PLUS","RO_MINUS")
    fig,ax=plt.subplots(figsize=(9,5.2))
    for config in configs:
        arr=[replicate[("A9",0.0,0.0,config,r)]["Delta_Lz_all"] for r in range(4)]
        m,lo,hi=ci(arr)
        ax.errorbar(config,m,yerr=[[m-lo],[hi-m]],fmt="o",capsize=4,label=config)
        ax.text(config,m+(.003 if m>=0 else -.006),f"{m:+.3f}",ha="center",fontsize=8)
    ax.axhline(0,color="black",lw=.7)
    ax.set_ylabel("⟨Lz(100T) − Lz(0)⟩, безразмерно")
    ax.set_title("V15 A9: передача углового момента, физический ансамбль, δ=τ₀=0")
    ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(figdir/"v15_ideal_angular_momentum_a9.png",dpi=220); fig.savefig(figdir/"v15_ideal_angular_momentum_a9.svg"); plt.close(fig)
    print(f"wrote {path}; groups={len(base)}; max torque closure={max(r['mean'] for r in outrows if r['metric']=='Torque_residual_abs_max'):.3e}")


if __name__ == "__main__":
    main()
