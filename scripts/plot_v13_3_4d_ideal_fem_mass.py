"""Plot paired ideal-vs-h50 P4 4D retention at computed mass nodes."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bounds(status: str) -> tuple[int, int]:
    if status == "retained":
        return 1, 1
    if status == "escaped":
        return 0, 0
    if status == "unresolved":
        return 0, 1
    raise ValueError(f"unknown trajectory outcome {status!r}")


def ci(values: list[float]) -> tuple[float, float, float]:
    a = np.asarray(values, float)
    mean = float(np.mean(a))
    half = float(student_t.ppf(.975, len(a) - 1) * np.std(a, ddof=1) / np.sqrt(len(a)))
    return mean, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-directory", required=True)
    args = parser.parse_args()
    if Path(args.comparison_directory).name != args.comparison_directory:
        parser.error("comparison directory must be a direct child of the V13.3 atlas")
    root = BASE / args.comparison_directory
    pairs = read(root / "field_comparison_trajectories.csv")
    if not pairs:
        raise ValueError("no paired field trajectories in comparison table")
    manifests = json.loads((root / "field_comparison_manifest.json").read_text(encoding="utf-8"))
    ref_by_rep = defaultdict(list)
    comparison_by_rep = defaultdict(list)
    diff_by_rep = defaultdict(list)
    unresolved = defaultdict(int)
    for row in pairs:
        k = (row["comparison"], row["point_id"], row["model"], float(row["mu"]), int(row["replica"]))
        ilo, ihi = bounds(row["reference_status"])
        flo, fhi = bounds(row["comparison_status"])
        ref_by_rep[k].append((ilo, ihi))
        comparison_by_rep[k].append((flo, fhi))
        diff_by_rep[k].append((flo - ihi, fhi - ilo))
        unresolved[k] += row["reference_status"] == "unresolved" or row["comparison_status"] == "unresolved"

    grouped = defaultdict(list)
    delta_grouped = defaultdict(list)
    for k in sorted(ref_by_rep):
        comparison, point, model, mu, replica = k
        ref = ref_by_rep[k]
        fem = comparison_by_rep[k]
        delta = diff_by_rep[k]
        n = len(ref)
        row = {"comparison_directory": comparison, "point_id": point, "model": model,
               "mu": mu, "replica": replica, "paired_launches": n,
               "P_ideal_lower": float(np.mean([x[0] for x in ref])),
               "P_ideal_upper": float(np.mean([x[1] for x in ref])),
               "P_h50_lower": float(np.mean([x[0] for x in fem])),
               "P_h50_upper": float(np.mean([x[1] for x in fem])),
               "delta_h50_minus_ideal_lower": float(np.mean([x[0] for x in delta])),
               "delta_h50_minus_ideal_upper": float(np.mean([x[1] for x in delta])),
               "unresolved_pairs": unresolved[k]}
        grouped[(comparison, point, model, mu)].append(row)
        delta_grouped[(comparison, point, model, mu)].append(row)

    summary = []
    for (comparison, point, model, mu), reps in sorted(grouped.items()):
        if len(reps) != 4:
            raise ValueError(f"expected four independent scrambles at {(comparison, point, model, mu)}")
        out = {"comparison_directory": comparison, "point_id": point, "model": model, "mu": mu,
               "horizon_RF": manifests["horizon_RF"], "replicates": 4,
               "paired_launches": sum(r["paired_launches"] for r in reps),
               "unresolved_pairs": sum(r["unresolved_pairs"] for r in reps)}
        for metric in ("P_ideal_lower", "P_h50_lower", "delta_h50_minus_ideal_lower"):
            mean, low, high = ci([r[metric] for r in reps])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_t95_low"] = low
            out[f"{metric}_t95_high"] = high
        summary.append(out)
    write(root / "ideal_h50_mass_replicate_summary.csv",
          [r for group in grouped.values() for r in group])
    write(root / "ideal_h50_mass_summary.csv", summary)

    point_ids = sorted({row["point_id"] for row in summary})
    comparisons = sorted({row["comparison_directory"] for row in summary})
    colors = {"Q": "#355c9a", "RO": "#bd4b37"}
    markers = {"Q": "o", "RO": "s"}
    markers_by_field = {"ideal": "filled", "h50_P4": "open"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(point_ids), figsize=(5.0 * len(point_ids), 3.8),
                             squeeze=False, constrained_layout=True)
    for ax, point in zip(axes[0], point_ids):
        for model in ("Q", "RO"):
            color = colors[model]
            for field, key, label in (("ideal", "P_ideal_lower", "ideal"),
                                      ("h50_P4", "P_h50_lower", "h50 P4")):
                subset = sorted((r for r in summary if r["point_id"] == point and r["model"] == model),
                                key=lambda r: r["mu"])
                # If separate comparison batches include distinct masses, merge by a unique μ.
                dedup = {}
                for row in subset:
                    dedup[row["mu"]] = row
                subset = [dedup[mu] for mu in sorted(dedup)]
                xs, means, lowerr, higherr = [], [], [], []
                for row in subset:
                    xs.append(float(row["mu"]))
                    means.append(float(row[f"{key}_mean"]))
                    lowerr.append(max(0., means[-1] - float(row[f"{key}_t95_low"])))
                    higherr.append(max(0., float(row[f"{key}_t95_high"]) - means[-1]))
                ax.errorbar(xs, means, yerr=[lowerr, higherr], color=color,
                            marker=markers[model], markerfacecolor=color if field == "ideal" else "white",
                            markeredgecolor=color, linestyle="none", capsize=2.5, ms=4.2,
                            label=f"{model}, {label}")
        ax.set_title(point.replace("_", " "))
        ax.set_xlabel(r"Normalized mass $\mu$")
        ax.set_ylim(-.02, 1.02)
        ax.grid(axis="y", alpha=.22)
    axes[0][0].set_ylabel(r"$P_{ret}$ on the shared physical 4D domain")
    axes[0][0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Ideal harmonics vs full h50 P4 FEM; sampled mass nodes only")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(root / f"ideal_vs_h50_mass_retention.{ext}", dpi=220 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"summary_rows": len(summary), "points": point_ids,
                      "comparison_directories": comparisons,
                      "output_directory": str(root)}, indent=2))


if __name__ == "__main__":
    main()
