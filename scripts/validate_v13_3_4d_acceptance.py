"""Integrity checks for V13.3 scrambled-Sobol 4D acceptance runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
R = 0.008
OMEGA = 2 * math.pi * 100_000.0
START_FIELDS = ("x0_rho", "y0_rho", "vx0_rho_tau", "vy0_rho_tau")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    if Path(args.directory).name != args.directory:
        parser.error("directory must be a direct child of the V13.3 atlas")
    out = BASE / args.directory
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in read(out / "trajectory_results.csv")
            if int(row["horizon"]) == args.horizon]
    points = sorted({row["point_id"] for row in rows})
    requested = sorted(manifest.get("point_ids_with_continuation_requests", points))
    if points != requested:
        raise ValueError(f"trajectory point set differs from requested points: {points} != {requested}")
    masses = [float(x) for x in manifest["mu_values"]]
    phases = list(range(len(manifest["phase_rad"])))
    seeds = list(range(len(manifest["replica_seeds"])))
    nstate = int(manifest["states_per_replica"])
    expected = len(points) * len(masses) * 2 * len(phases) * len(seeds) * nstate
    if len(rows) != expected:
        raise ValueError(f"expected {expected} trajectory rows at N={args.horizon}, found {len(rows)}")

    keys = {}
    for row in rows:
        key = (row["point_id"], row["model"], float(row["mu"]), int(row["phase_i"]),
               int(row["replica"]), int(row["state_i"]), int(row["horizon"]))
        if key in keys:
            raise ValueError(f"duplicate trajectory key: {key}")
        if row["status"] not in {"retained", "escaped", "unresolved"}:
            raise ValueError(f"unknown outcome status {row['status']!r}")
        keys[key] = row

    paired = 0
    unresolved = 0
    for key, row in keys.items():
        if row["model"] != "RO":
            continue
        qkey = (key[0], "Q", *key[2:])
        if qkey not in keys:
            raise ValueError(f"missing paired Q trajectory: {qkey}")
        qrow = keys[qkey]
        for field in START_FIELDS:
            if not math.isclose(float(row[field]), float(qrow[field]), rel_tol=0, abs_tol=1e-14):
                raise ValueError(f"Q/RO launch mismatch at {key}, field={field}")
        paired += 1
        unresolved += row["status"] == "unresolved" or qrow["status"] == "unresolved"
    if paired * 2 != len(rows):
        raise ValueError(f"Q/RO pair coverage is incomplete: {paired} pairs for {len(rows)} rows")

    volume = {float(k): float(v["D_volume_m4_s2"]) for k, v in manifest["domain_volumes"].items()}
    summaries = read(out / f"acceptance_summary_N{args.horizon}.csv")
    if len(summaries) != len(points) * len(masses) * 2:
        raise ValueError("acceptance summary row count does not match point/model/mass product")
    for row in summaries:
        mu = float(row["mu"])
        expected_a4 = float(row["phase_average_P_lower_mean"]) * volume[mu]
        observed_a4 = float(row["phase_average_A4_lower_m4_s2_mean"])
        if not math.isclose(expected_a4, observed_a4, rel_tol=2e-12, abs_tol=1e-12):
            raise ValueError(f"A4 != P_ret*Vol(D) for {row['point_id']} {row['model']} mu={mu}")
        expected_a4_hi = float(row["phase_average_P_upper_mean"]) * volume[mu]
        observed_a4_hi = float(row["phase_average_A4_upper_m4_s2_mean"])
        if not math.isclose(expected_a4_hi, observed_a4_hi, rel_tol=2e-12, abs_tol=1e-12):
            raise ValueError(f"A4 upper bound != P_ret upper*Vol(D) for {row['point_id']} {row['model']} mu={mu}")
        expected_a4_dimensionless = observed_a4 / (R**4 * OMEGA**2)
        if not math.isclose(expected_a4_dimensionless, float(row["phase_average_a4_lower_mean"]),
                            rel_tol=2e-12, abs_tol=1e-14):
            raise ValueError(f"dimensionless a4 scaling mismatch for {row['point_id']} {row['model']} mu={mu}")

    pair_summaries = read(out / f"paired_acceptance_summary_N{args.horizon}.csv")
    if len(pair_summaries) != len(points) * len(masses):
        raise ValueError("paired summary row count does not match point/mass product")
    for row in pair_summaries:
        mu = float(row["mu"])
        for bound in ("lower", "upper"):
            dp = float(row[f"paired_delta_P_{bound}_mean"])
            da4 = float(row[f"paired_delta_A4_{bound}_m4_s2_mean"])
            if not math.isclose(da4, dp * volume[mu], rel_tol=2e-12, abs_tol=1e-12):
                raise ValueError(f"paired ΔA4 scaling mismatch for {row['point_id']} mu={mu} bound={bound}")

    print(json.dumps({"directory": args.directory, "horizon_RF": args.horizon,
                      "trajectory_rows": len(rows), "unique_keys": len(keys),
                      "paired_launches": paired, "unresolved_launch_pairs": int(unresolved),
                      "acceptance_summary_rows": len(summaries),
                      "paired_summary_rows": len(pair_summaries),
                      "A4_measure_and_pair_checks": "passed"}, indent=2))


if __name__ == "__main__":
    main()
