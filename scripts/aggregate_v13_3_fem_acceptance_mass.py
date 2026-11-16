"""Combine independently run h50/h35 V13.3 4D QMC mass nodes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directories", nargs="+", required=True,
                        help="run directories with acceptance_summary_N*.csv")
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    if Path(args.output_directory).name != args.output_directory:
        parser.error("output directory must be a direct child of the V13.3 atlas")
    out = BASE / args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    combined: dict[tuple, dict] = {}
    paired: dict[tuple, dict] = {}
    manifests: dict[str, dict] = {}
    source_hashes: dict[str, str] = {}
    for name in args.directories:
        if Path(name).name != name:
            parser.error("input directories must be direct children of the V13.3 atlas")
        root = BASE / name
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[name] = {key: manifest.get(key) for key in
                           ("field_mode", "mu_values", "horizons", "relative_delta_rad",
                            "states_per_replica", "replica_seeds", "solver")}
        source_hashes[name] = sha256(manifest_path)
        field_mode = str(manifest["field_mode"])
        for summary_path in sorted(root.glob("acceptance_summary_N*.csv")):
            horizon = int(summary_path.stem.split("N")[-1])
            for row in read_csv(summary_path):
                mu = float(row["mu"])
                key = (row["point_id"], row["model"], mu, horizon)
                if key in combined:
                    raise ValueError(f"duplicate mass/mode/horizon result {key}")
                combined[key] = {"source_directory": name, "field_mode": field_mode,
                                 "point_id": row["point_id"], "model": row["model"],
                                 "mu": mu, "horizon_RF": horizon,
                                 "replicates": int(row["replicates"]),
                                 "domain_volume_m4_s2": float(row["domain_volume_m4_s2"]),
                                 "P_ret_mean": float(row["phase_average_P_lower_mean"]),
                                 "P_ret_t95_low": float(row["phase_average_P_lower_t95_low"]),
                                 "P_ret_t95_high": float(row["phase_average_P_lower_t95_high"]),
                                 "A4_m4_s2_mean": float(row["phase_average_A4_lower_m4_s2_mean"]),
                                 "A4_m4_s2_t95_low": float(row["phase_average_A4_lower_m4_s2_t95_low"]),
                                 "A4_m4_s2_t95_high": float(row["phase_average_A4_lower_m4_s2_t95_high"]),
                                 "a4_dimensionless_mean": float(row["phase_average_a4_lower_mean"]),
                                 "unresolved_bounds_differ": not np.isclose(
                                     float(row["phase_average_P_lower_mean"]),
                                     float(row["phase_average_P_upper_mean"]), rtol=0.0, atol=1e-14)}
        for pair_path in sorted(root.glob("paired_acceptance_summary_N*.csv")):
            horizon = int(pair_path.stem.split("N")[-1])
            for row in read_csv(pair_path):
                key = (row["point_id"], float(row["mu"]), horizon)
                if key in paired:
                    raise ValueError(f"duplicate paired mass/horizon result {key}")
                paired[key] = {"RO_minus_Q_mean": float(row["paired_delta_P_lower_mean"]),
                               "RO_minus_Q_t95_low": float(row["paired_delta_P_lower_t95_low"]),
                               "RO_minus_Q_t95_high": float(row["paired_delta_P_lower_t95_high"])}

    for (point, mu, horizon), delta in paired.items():
        q = combined.get((point, "Q", mu, horizon))
        ro = combined.get((point, "RO", mu, horizon))
        if q is None or ro is None:
            raise ValueError(f"paired result has no matching Q/RO rows: {(point, mu, horizon)}")
        domain_volume = q["domain_volume_m4_s2"]
        delta_a4 = delta["RO_minus_Q_mean"] * domain_volume
        for row in (q, ro):
            row.update(delta)
            row["RO_minus_Q_A4_m4_s2_mean"] = delta_a4
            row["RO_minus_Q_A4_m4_s2_t95_low"] = delta["RO_minus_Q_t95_low"] * domain_volume
            row["RO_minus_Q_A4_m4_s2_t95_high"] = delta["RO_minus_Q_t95_high"] * domain_volume
        if q["A4_m4_s2_mean"] > 0:
            ro["A4_RO_over_Q"] = ro["A4_m4_s2_mean"] / q["A4_m4_s2_mean"]
        else:
            ro["A4_RO_over_Q"] = "undefined_Q_A4_zero"

    rows = [combined[key] for key in sorted(combined)]
    if not rows:
        raise ValueError("no acceptance summary rows were found")
    write_csv(out / "fem_acceptance_mass_curve.csv", rows)
    (out / "fem_acceptance_mass_manifest.json").write_text(
        json.dumps({"input_directories": args.directories,
                    "input_manifest_sha256": source_hashes,
                    "protocols": manifests,
                    "metrics": "phase-averaged A4 and retention; 95% t intervals across four independent scrambled Sobol replicate means",
                    "paired_metric": "RO-Q difference on common Sobol launches; never pool Sobol points as independent replicates"},
                   indent=2) + "\n", encoding="utf-8")

    points = sorted({row["point_id"] for row in rows})
    horizons = sorted({row["horizon_RF"] for row in rows})
    colors = {"Q": "#1f77b4", "RO": "#d62728"}
    markersizes = {100: 3.8, 300: 4.6, 1000: 5.2}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for point in points:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
        for horizon in horizons:
            for model in ("Q", "RO"):
                subset = sorted((row for row in rows if row["point_id"] == point and
                                 row["horizon_RF"] == horizon and row["model"] == model),
                                key=lambda row: row["mu"])
                if not subset:
                    continue
                x = np.asarray([row["mu"] for row in subset])
                for axis, mean_key, low_key, high_key, label in (
                    (axes[0], "P_ret_mean", "P_ret_t95_low", "P_ret_t95_high", r"$P_{ret}$"),
                    (axes[1], "A4_m4_s2_mean", "A4_m4_s2_t95_low", "A4_m4_s2_t95_high", r"$A_4$"),
                ):
                    y = np.asarray([row[mean_key] for row in subset])
                    lo = np.maximum(0.0, y - np.asarray([row[low_key] for row in subset]))
                    hi = np.maximum(0.0, np.asarray([row[high_key] for row in subset]) - y)
                    axis.errorbar(x, y, yerr=np.vstack((lo, hi)), color=colors[model],
                                  linestyle="none", marker="o" if model == "Q" else "s",
                                  ms=markersizes.get(horizon, 4.5), capsize=2,
                                  label=f"{model}, N={horizon}")
        for horizon in horizons:
            subset = sorted((row for row in rows if row["point_id"] == point and
                             row["horizon_RF"] == horizon and row["model"] == "RO"),
                            key=lambda row: row["mu"])
            if not subset:
                continue
            x = np.asarray([row["mu"] for row in subset])
            y = np.asarray([row["RO_minus_Q_A4_m4_s2_mean"] for row in subset])
            lo = y - np.asarray([row["RO_minus_Q_A4_m4_s2_t95_low"] for row in subset])
            hi = np.asarray([row["RO_minus_Q_A4_m4_s2_t95_high"] for row in subset]) - y
            axes[2].errorbar(x, y, yerr=np.vstack((lo, hi)), color="#7a3e9d",
                             linestyle="none", marker="D", ms=markersizes.get(horizon, 4.5),
                             capsize=2, lw=1.1, label=f"N={horizon}")
        axes[0].set(xlabel=r"Normalized mass $\mu$", ylabel=r"$P_{ret}$", ylim=(-.02, 1.02))
        axes[1].set(xlabel=r"Normalized mass $\mu$", ylabel=r"$A_4$ (m$^4$ s$^{-2}$)")
        axes[2].axhline(0, color="0.35", lw=.8)
        axes[2].set(xlabel=r"Normalized mass $\mu$", ylabel=r"$\Delta A_4=A_4(RO)-A_4(Q)$ (m$^4$ s$^{-2}$)")
        axes[0].legend(frameon=False, fontsize=7, ncol=2)
        axes[1].legend(frameon=False, fontsize=7, ncol=2)
        axes[2].legend(frameon=False, fontsize=7)
        fig.suptitle(f"Full h50 P4 physical-domain acceptance, {point}")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"fem_A4_vs_mass_{point}.{ext}", dpi=220 if ext == "png" else None,
                        bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({"rows": len(rows), "points": points, "horizons": horizons,
                      "output_directory": str(out)}, indent=2))


if __name__ == "__main__":
    main()
