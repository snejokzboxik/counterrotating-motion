"""Summarize paired full-FEM V13.1 pilot retention and uncertainties."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def paired_summary(left: dict[int, bool], right: dict[int, bool]) -> dict[str, float | int]:
    ids = sorted(set(left) & set(right))
    d = np.array([int(left[i]) - int(right[i]) for i in ids], dtype=float)
    n = len(d)
    mean = float(d.mean()) if n else math.nan
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    # Paired t interval for the mean of the per-start binary differences.
    # This preserves the common initial-condition pairing.
    half = float(student_t.ppf(0.975, n - 1)) * se if n > 1 else math.nan
    return {
        "n_pairs": n,
        "left_gains": int((d == 1).sum()),
        "right_gains": int((d == -1).sum()),
        "ties": int((d == 0).sum()),
        "difference_left_minus_right": mean,
        "paired_ci95_low": max(-1.0, mean - half) if n > 1 else math.nan,
        "paired_ci95_high": min(1.0, mean + half) if n > 1 else math.nan,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    rows = list(csv.DictReader((args.directory / "trajectory_results.csv").open(encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("trajectory_results.csv is empty")
    mu_values = sorted({float(r["mu"]) for r in rows})
    phases = sorted({float(r["phase_deg"]) for r in rows})
    horizons = (100, 300)
    models = [("Q", 0.0), ("Qopt", 0.0), ("SO", -0.05), ("RO", -0.05), ("SO", 0.05), ("RO", 0.05)]
    step_overrides: dict[tuple[float, float, str, float, int, int], str] = {}
    step_unresolved: set[tuple[float, float, str, float, int, int]] = set()
    sensitivity_file = args.directory.parent / "fem_step_refinement/sensitive_refinement.csv"
    if sensitivity_file.exists():
        for r in csv.DictReader(sensitivity_file.open(encoding="utf-8", newline="")):
            base_key = (float(r["mu"]), float(r["phase_deg"]), r["model"], float(r["eta"]), int(r["initial_index"]))
            for h in horizons:
                s256, s512 = r[f"status{h}_tau256"], r[f"status{h}_tau512"]
                key = (*base_key, h)
                if s256 != s512:
                    step_unresolved.add(key)
                else:
                    step_overrides[key] = s512
    summary: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    lookup: dict[tuple[float, float, int, str, float], dict[int, bool]] = {}
    for mu in mu_values:
        for phase in phases:
            for horizon in horizons:
                field = f"status_{horizon}"
                for model, eta in models:
                    grp = [r for r in rows if float(r["mu"]) == mu and float(r["phase_deg"]) == phase
                           and r["model"] == model and float(r["eta"]) == eta]
                    key_for = lambda r: (float(r["mu"]), float(r["phase_deg"]), r["model"], float(r["eta"]), int(r["index"]), horizon)
                    resolved = [r for r in grp if r[field] in ("retained", "escaped") and key_for(r) not in step_unresolved]
                    ret = sum(step_overrides.get(key_for(r), r[field]) == "retained" for r in resolved)
                    lo, hi = wilson(ret, len(resolved))
                    lookup[(mu, phase, horizon, model, eta)] = {
                        int(r["index"]): step_overrides.get(key_for(r), r[field]) == "retained" for r in resolved
                    }
                    n_unresolved = len(grp) - len(resolved)
                    summary.append({
                        "mu": mu, "phase_deg": phase, "horizon_rf": horizon, "model": model, "eta": eta,
                        "n_total": len(grp), "n_resolved": len(resolved),
                        "n_unresolved": n_unresolved,
                        "n_step_sensitive": sum(key_for(r) in step_unresolved for r in grp), "n_retained": ret,
                        "retention": ret / len(resolved) if resolved else math.nan,
                        "retention_basis": "resolved cases only; unresolved excluded",
                        "retention_bound_low_all_n": ret / len(grp) if grp else math.nan,
                        "retention_bound_high_all_n": (ret + n_unresolved) / len(grp) if grp else math.nan,
                        "wilson95_low": lo, "wilson95_high": hi,
                    })
                q = lookup[(mu, phase, horizon, "Q", 0.0)]
                for model, eta in models[1:]:
                    m = lookup[(mu, phase, horizon, model, eta)]
                    result = paired_summary(m, q)
                    paired.append({"mu": mu, "phase_deg": phase, "horizon_rf": horizon,
                                   "contrast": f"{model}(eta={eta:+.2f}) - Q", **result})
                for eta in (-0.05, 0.05):
                    so = lookup[(mu, phase, horizon, "SO", eta)]
                    ro = lookup[(mu, phase, horizon, "RO", eta)]
                    result = paired_summary(ro, so)
                    paired.append({"mu": mu, "phase_deg": phase, "horizon_rf": horizon,
                                   "contrast": f"RO(eta={eta:+.2f}) - SO(eta={eta:+.2f})", **result})

    def write(name: str, data: list[dict[str, object]]) -> None:
        path = args.directory / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    write("retention_summary.csv", summary)
    write("paired_differences.csv", paired)

    colors = {"Q": "#222222", "Qopt": "#777777", "SO-0.05": "#2c7fb8", "RO-0.05": "#7fcdbb",
              "SO+0.05": "#d95f0e", "RO+0.05": "#fdbb84"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True, constrained_layout=True)
    for ri, phase in enumerate(phases):
        for ci, horizon in enumerate(horizons):
            ax = axes[ri, ci]
            for model, eta in models:
                key = (model if model in ("Q", "Qopt") else f"{model}{eta:+.2f}")
                grp = [r for r in summary if r["phase_deg"] == phase and r["horizon_rf"] == horizon
                       and r["model"] == model and r["eta"] == eta]
                grp.sort(key=lambda r: r["mu"])
                x = np.array([r["mu"] for r in grp], dtype=float)
                y = np.array([r["retention"] for r in grp], dtype=float)
                low = np.array([r["wilson95_low"] for r in grp], dtype=float)
                high = np.array([r["wilson95_high"] for r in grp], dtype=float)
                ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), color=colors[key], marker="o",
                            ms=4, lw=1.3, linestyle="--" if model in ("Qopt", "RO") else "-",
                            capsize=2, label=key)
            ax.set_title(f"Start phase {phase:g}°, {horizon} RF periods")
            ax.set_xlim(min(mu_values) - 0.02, max(mu_values) + 0.02)
            ax.set_ylim(-0.04, 1.04)
            ax.grid(alpha=0.25)
            if ri == 1:
                ax.set_xlabel("Mass ratio μ")
            if ci == 0:
                ax.set_ylabel("2D first-exit retention fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("V13.1 full h50 P4 FEM pilot · 32 paired Sobol starts · Wilson 95% intervals")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(args.directory / f"fem_mass_retention.{ext}", dpi=220 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote summaries for {len(rows)} trajectories; unresolved={sum(r['n_unresolved'] for r in summary)}")


if __name__ == "__main__":
    main()
