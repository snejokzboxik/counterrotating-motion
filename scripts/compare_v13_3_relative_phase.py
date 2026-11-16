"""Paired, four-phase QMC comparison of V13.3 relative octupole phases."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def t_interval(values: list[float]) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, math.nan, math.nan
    half = float(student_t.ppf(.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_key(row: dict[str, str]) -> tuple:
    return (row["point_id"], row["model"], round(float(row["mu"]), 12),
            int(row["phase_i"]), int(row["replica"]), int(row["state_i"]), int(row["horizon"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta0-directory", required=True)
    parser.add_argument("--comparison-directory", required=True)
    parser.add_argument("--point-ids", nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    for name in (args.delta0_directory, args.comparison_directory):
        if Path(name).name != name:
            parser.error("input directories must be direct children of the V13.3 atlas")

    d0_dir, d1_dir = BASE / args.delta0_directory, BASE / args.comparison_directory
    m0 = json.loads((d0_dir / "manifest.json").read_text(encoding="utf-8"))
    m1 = json.loads((d1_dir / "manifest.json").read_text(encoding="utf-8"))
    if m0["field_mode"] != m1["field_mode"]:
        raise ValueError("field modes differ; compare matching field models only")
    if m0["phase_rad"] != m1["phase_rad"]:
        raise ValueError("injection phase quadrature differs")
    if m0["replica_seeds"] != m1["replica_seeds"]:
        raise ValueError("independent Sobol replica seeds differ")
    for mass in set(map(float, m0["domain_volumes"])) & set(map(float, m1["domain_volumes"])):
        v0 = m0["domain_volumes"][str(mass)]["D_volume_m4_s2"]
        v1 = m1["domain_volumes"][str(mass)]["D_volume_m4_s2"]
        if not math.isclose(float(v0), float(v1), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"physical phase-space domain differs at mu={mass}")

    rows0 = [r for r in read_csv(d0_dir / "trajectory_results.csv")
             if int(r["horizon"]) == args.horizon and r["model"] in ("Q", "RO")]
    rows1 = [r for r in read_csv(d1_dir / "trajectory_results.csv")
             if int(r["horizon"]) == args.horizon and r["model"] in ("Q", "RO")]
    if args.point_ids:
        rows0 = [r for r in rows0 if r["point_id"] in args.point_ids]
        rows1 = [r for r in rows1 if r["point_id"] in args.point_ids]
    index0 = {trajectory_key(row): row for row in rows0}
    index1 = {trajectory_key(row): row for row in rows1}
    if len(index0) != len(rows0) or len(index1) != len(rows1):
        raise ValueError("duplicate trajectory keys in one relative-phase input")
    common = sorted(set(index0) & set(index1))
    if not common:
        raise ValueError("no paired trajectories share all physical and Sobol keys")

    paired_rows = []
    by_replicate = defaultdict(list)
    phase_p = defaultdict(lambda: defaultdict(list))
    initial_fields = ("x0_rho", "y0_rho", "vx0_rho_tau", "vy0_rho_tau")
    for key in common:
        a, b = index0[key], index1[key]
        for field in initial_fields:
            if not math.isclose(float(a[field]), float(b[field]), rel_tol=0, abs_tol=1e-14):
                raise ValueError(f"initial conditions differ for key={key}, field={field}")
        if a["status"] == "unresolved" or b["status"] == "unresolved":
            # Keep unresolved cases explicit; indicator bounds are computed below.
            pass
        row = {"point_id": key[0], "model": key[1], "mu": key[2], "phase_i": key[3],
               "phase_deg": 90 * key[3], "replica": key[4], "state_i": key[5], "horizon_RF": key[6],
               "x0_rho": a["x0_rho"], "y0_rho": a["y0_rho"],
               "vx0_rho_tau": a["vx0_rho_tau"], "vy0_rho_tau": a["vy0_rho_tau"],
               "delta0_status": a["status"], "comparison_status": b["status"],
               "delta0_exit_RF": a["exit_periods"], "comparison_exit_RF": b["exit_periods"],
               "delta0_rho_max": a["rho_max"], "comparison_rho_max": b["rho_max"],
               "status_changed": a["status"] != b["status"]}
        paired_rows.append(row)
        def bounds(status: str) -> tuple[int, int]:
            return (1, 1) if status == "retained" else (0, 0) if status == "escaped" else (0, 1)
        lo0, hi0 = bounds(a["status"])
        lo1, hi1 = bounds(b["status"])
        by_replicate[(key[0], key[1], key[2], key[4])].append((lo0, hi0, lo1, hi1))
        phase_p[(key[0], key[1], key[2], key[3], key[4])]["n"] .append(1)
        phase_p[(key[0], key[1], key[2], key[3], key[4])]["d0_lo"].append(lo0)
        phase_p[(key[0], key[1], key[2], key[3], key[4])]["d0_hi"].append(hi0)
        phase_p[(key[0], key[1], key[2], key[3], key[4])]["d1_lo"].append(lo1)
        phase_p[(key[0], key[1], key[2], key[3], key[4])]["d1_hi"].append(hi1)

    rep_rows, summary_rows = [], []
    grouped = defaultdict(list)
    for (point, model, mu, rep), vals in sorted(by_replicate.items()):
        n = len(vals)
        d0_lo = sum(v[0] for v in vals) / n
        d0_hi = sum(v[1] for v in vals) / n
        d1_lo = sum(v[2] for v in vals) / n
        d1_hi = sum(v[3] for v in vals) / n
        diff_lo = sum(v[2] - v[1] for v in vals) / n
        diff_hi = sum(v[3] - v[0] for v in vals) / n
        rep_row = {"point_id": point, "model": model, "mu": mu, "replica": rep,
                   "n_paired_phase_states": n, "P_delta0_lower": d0_lo, "P_delta0_upper": d0_hi,
                   "P_comparison_lower": d1_lo, "P_comparison_upper": d1_hi,
                   "paired_delta_lower": diff_lo, "paired_delta_upper": diff_hi,
                   "unresolved_pairs": sum(v[1] != v[0] or v[3] != v[2] for v in vals)}
        rep_rows.append(rep_row)
        grouped[(point, model, mu)].append(rep_row)

    domain = {float(k): float(v["D_volume_m4_s2"]) for k, v in m0["domain_volumes"].items()}
    for (point, model, mu), reps in sorted(grouped.items()):
        if len(reps) != 4:
            raise ValueError(f"expected four independent scrambles for {(point, model, mu)}")
        result = {"point_id": point, "model": model, "mu": mu, "horizon_RF": args.horizon,
                  "delta0_rad": float(m0["relative_delta_rad"]),
                  "comparison_delta_rad": float(m1["relative_delta_rad"]),
                  "replicates": 4, "Vol_D_m4_s2": domain[mu]}
        for metric in ("P_delta0_lower", "P_delta0_upper", "P_comparison_lower", "P_comparison_upper",
                       "paired_delta_lower", "paired_delta_upper"):
            mean, low, high = t_interval([float(r[metric]) for r in reps])
            result[metric + "_mean"] = mean
            result[metric + "_t95_low"] = low
            result[metric + "_t95_high"] = high
        result["A4_delta0_lower_mean_m4_s2"] = result["P_delta0_lower_mean"] * domain[mu]
        result["A4_comparison_lower_mean_m4_s2"] = result["P_comparison_lower_mean"] * domain[mu]
        result["paired_delta_A4_lower_mean_m4_s2"] = result["paired_delta_lower_mean"] * domain[mu]
        result["paired_delta_A4_t95_low_m4_s2"] = result["paired_delta_lower_t95_low"] * domain[mu]
        result["paired_delta_A4_t95_high_m4_s2"] = result["paired_delta_lower_t95_high"] * domain[mu]
        result["note"] = "unresolved outcomes widen lower/upper bounds; phase average over the four sampled injection phases"
        summary_rows.append(result)

    out = d1_dir
    write_csv(out / "relative_phase_paired_trajectories.csv", paired_rows)
    write_csv(out / "relative_phase_replicate_summary.csv", rep_rows)
    write_csv(out / "relative_phase_comparison_summary.csv", summary_rows)
    metadata = {"delta0_directory": args.delta0_directory, "comparison_directory": args.comparison_directory,
                "horizon_RF": args.horizon, "point_ids": sorted({r["point_id"] for r in paired_rows}),
                "matched_keys": len(common), "initial_conditions_match_abs_tol": 1e-14,
                "delta0_manifest_sha256": sha256(d0_dir / "manifest.json"),
                "comparison_manifest_sha256": sha256(d1_dir / "manifest.json"),
                "delta0_trajectory_sha256": sha256(d0_dir / "trajectory_results.csv"),
                "comparison_trajectory_sha256": sha256(d1_dir / "trajectory_results.csv"),
                "summary_uncertainty": "95% Student-t intervals across four independent Sobol-replicate means; paired lower/upper bounds preserve unresolved outcomes",
                "comparison_script_sha256": sha256(Path(__file__))}
    (out / "relative_phase_comparison_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matched_keys": len(common), "summary_rows": len(summary_rows),
                      "changed_statuses": sum(r["status_changed"] for r in paired_rows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
