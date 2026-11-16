"""Replicate-aware summaries for compact-domain V13.3 mass-ray scans."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def interval(values: list[float]) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    mean = float(np.mean(data))
    if len(data) < 2:
        return mean, math.nan, math.nan
    half = float(student_t.ppf(0.975, len(data) - 1) * np.std(data, ddof=1) / np.sqrt(len(data)))
    return mean, mean - half, mean + half


def status_bounds(status: str) -> tuple[int, int]:
    if status == "retained":
        return 1, 1
    if status == "escaped":
        return 0, 0
    if status == "unresolved":
        return 0, 1
    raise ValueError(f"unknown trajectory status {status!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True,
                        help="direct child run directory below the V13.3 atlas")
    args = parser.parse_args()
    if Path(args.directory).name != args.directory:
        parser.error("directory must be a direct child of the V13.3 atlas")
    out = BASE / args.directory
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    trajectories = read_csv(out / "trajectory_results.csv")
    seeds = [int(seed) for seed in manifest["replica_seeds"]]
    nrep = len(seeds)
    nstate = int(manifest["states_per_replica"])
    phases = list(enumerate(float(v) for v in manifest["phases_rad"]))
    mus = sorted(float(v) for v in manifest["mu_grid"])
    expected = len(mus) * len(phases) * nrep * nstate * 2
    if len(trajectories) != expected:
        raise ValueError(f"expected {expected} trajectories, got {len(trajectories)}")

    index: dict[tuple, dict[str, str]] = {}
    for row in trajectories:
        key = (round(float(row["mu"]), 12), int(row["phase_i"]), int(row["replica"]),
               int(row["state_i"]), row["model"])
        if key in index:
            raise ValueError(f"duplicate trajectory key {key}")
        if row["model"] not in ("Q", "RO"):
            raise ValueError(f"unexpected model {row['model']!r}")
        index[key] = row
    if len(index) != expected:
        raise ValueError(f"unique trajectory count mismatch: {len(index)} != {expected}")

    replicate_rows = []
    for mu in mus:
        for model in ("Q", "RO"):
            for phase_i, phase in phases:
                for replica, seed in enumerate(seeds):
                    group = [index[(round(mu, 12), phase_i, replica, state, model)]
                             for state in range(nstate)]
                    counts = defaultdict(int)
                    for row in group:
                        counts[row["status"]] += 1
                    escaped = [float(row["exit_periods"]) for row in group
                               if row["status"] == "escaped" and row["exit_periods"]]
                    replicate_rows.append({
                        "mu": mu, "model": model, "phase_i": phase_i,
                        "tau0_deg": math.degrees(phase), "replica": replica, "seed": seed,
                        "n_total": nstate, "n_retained": counts["retained"],
                        "n_escaped": counts["escaped"], "n_unresolved": counts["unresolved"],
                        "P_ret_lower": counts["retained"] / nstate,
                        "P_ret_upper": (counts["retained"] + counts["unresolved"]) / nstate,
                        "median_exit_RF": float(np.median(escaped)) if escaped else "",
                    })
    write_csv(out / "replicate_retention.csv", replicate_rows)

    phase_average: dict[tuple, list[dict]] = defaultdict(list)
    for row in replicate_rows:
        phase_average[(row["mu"], row["model"], row["replica"])].append(row)
    averaged = []
    for (mu, model, replica), group in sorted(phase_average.items()):
        if len(group) != len(phases):
            raise ValueError(f"missing phase slice for {(mu, model, replica)}")
        averaged.append({
            "mu": mu, "model": model, "replica": replica,
            "seed": seeds[replica],
            "phase_average_P_ret_lower": float(np.mean([float(r["P_ret_lower"]) for r in group])),
            "phase_average_P_ret_upper": float(np.mean([float(r["P_ret_upper"]) for r in group])),
        })
    write_csv(out / "replica_phase_average_retention.csv", averaged)

    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for row in averaged:
        by_group[(row["mu"], row["model"])].append(row)
    summary = []
    for (mu, model), group in sorted(by_group.items()):
        if len(group) != nrep:
            raise ValueError(f"expected {nrep} independent replicas at {(mu, model)}")
        lo, lo1, lo2 = interval([float(r["phase_average_P_ret_lower"]) for r in group])
        hi, hi1, hi2 = interval([float(r["phase_average_P_ret_upper"]) for r in group])
        exits = [float(r["median_exit_RF"]) for r in replicate_rows
                 if float(r["mu"]) == mu and r["model"] == model and r["median_exit_RF"] != ""]
        summary.append({
            "mu": mu, "model": model, "replicas": nrep, "states_per_replica": nstate,
            "phase_quadrature_deg": json.dumps([math.degrees(v) for _, v in phases]),
            "P_ret_mean": lo, "P_ret_t95_low": lo1, "P_ret_t95_high": lo2,
            "P_ret_upper_mean": hi, "P_ret_upper_t95_low": hi1, "P_ret_upper_t95_high": hi2,
            "median_exit_RF_median_of_replicate_phase_medians": float(np.median(exits)) if exits else "",
            "unresolved_total": sum(int(r["n_unresolved"]) for r in replicate_rows
                                     if float(r["mu"]) == mu and r["model"] == model),
        })
    write_csv(out / "mass_retention_summary.csv", summary)

    pair_replica: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for mu in mus:
        for replica, _seed in enumerate(seeds):
            for phase_i, _phase in phases:
                for state in range(nstate):
                    q = index[(round(mu, 12), phase_i, replica, state, "Q")]["status"]
                    ro = index[(round(mu, 12), phase_i, replica, state, "RO")]["status"]
                    qlo, qhi = status_bounds(q)
                    rlo, rhi = status_bounds(ro)
                    pair_replica[(mu, replica)].append((rlo - qhi, rhi - qlo))
    pair_rows = []
    for (mu, replica), values in sorted(pair_replica.items()):
        pair_rows.append({
            "mu": mu, "replica": replica, "seed": seeds[replica],
            "paired_launches": len(values),
            "RO_minus_Q_lower": float(np.mean([v[0] for v in values])),
            "RO_minus_Q_upper": float(np.mean([v[1] for v in values])),
        })
    write_csv(out / "paired_replica_difference.csv", pair_rows)
    paired_summary = []
    for mu in mus:
        group = [row for row in pair_rows if float(row["mu"]) == mu]
        if len(group) != nrep:
            raise ValueError(f"expected {nrep} paired replicate means at μ={mu}")
        lower = interval([float(r["RO_minus_Q_lower"]) for r in group])
        upper = interval([float(r["RO_minus_Q_upper"]) for r in group])
        paired_summary.append({
            "mu": mu, "replicas": nrep,
            "RO_minus_Q_mean": lower[0], "RO_minus_Q_t95_low": lower[1],
            "RO_minus_Q_t95_high": lower[2], "upper_bound_mean": upper[0],
            "upper_bound_t95_low": upper[1], "upper_bound_t95_high": upper[2],
            "unresolved_pairs": sum(int(r["n_unresolved"]) for r in replicate_rows
                                     if float(r["mu"]) == mu),
        })
    write_csv(out / "paired_mass_difference_summary.csv", paired_summary)

    colors = {"Q": "#315a91", "RO": "#bd4b38"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for model, marker in (("Q", "o"), ("RO", "s")):
        rows = sorted((r for r in summary if r["model"] == model), key=lambda r: float(r["mu"]))
        x = np.asarray([float(r["mu"]) for r in rows])
        y = np.asarray([float(r["P_ret_mean"]) for r in rows])
        low = np.maximum(0.0, y - np.asarray([float(r["P_ret_t95_low"]) for r in rows]))
        high = np.maximum(0.0, np.asarray([float(r["P_ret_t95_high"]) for r in rows]) - y)
        axes[0].errorbar(x, y, yerr=np.vstack((low, high)), linestyle="none", marker=marker,
                         color=colors[model], capsize=2, ms=4, label=model)
    axes[0].set(xlabel="Normalized mass μ", ylabel="Phase-averaged compact-domain retention",
                ylim=(-0.03, 1.03))
    axes[0].legend(frameon=False)
    x = np.asarray([float(r["mu"]) for r in paired_summary])
    y = np.asarray([float(r["RO_minus_Q_mean"]) for r in paired_summary])
    low = np.maximum(0.0, y - np.asarray([float(r["RO_minus_Q_t95_low"]) for r in paired_summary]))
    high = np.maximum(0.0, np.asarray([float(r["RO_minus_Q_t95_high"]) for r in paired_summary]) - y)
    axes[1].errorbar(x, y, yerr=np.vstack((low, high)), linestyle="none", marker="D",
                     color="#7a3e9d", capsize=2, ms=4)
    axes[1].axhline(0.0, color="0.35", lw=0.8)
    axes[1].set(xlabel="Normalized mass μ", ylabel="Paired RO−Q retention difference")
    fig.suptitle(f"{manifest['point']['candidate_id']}: compact-domain mass response, N=100")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"mass_retention_curve.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"directory": str(out), "trajectories": len(trajectories),
                      "mass_nodes": len(mus), "replicates": nrep,
                      "unresolved": sum(row["status"] == "unresolved" for row in trajectories)},
                     indent=2))


if __name__ == "__main__":
    main()
