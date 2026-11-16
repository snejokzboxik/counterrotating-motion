"""Paired ideal/FEM/mesh comparison for V13.3 4D acceptance ensembles."""
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
START_FIELDS = ("x0_rho", "y0_rho", "vx0_rho_tau", "vy0_rho_tau")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def interval(values: list[float]) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    mean = float(np.mean(v))
    if len(v) < 2:
        return mean, math.nan, math.nan
    half = float(student_t.ppf(.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v)))
    return mean, mean - half, mean + half


def status_bounds(status: str) -> tuple[int, int]:
    if status == "retained":
        return 1, 1
    if status == "escaped":
        return 0, 0
    if status == "unresolved":
        return 0, 1
    raise ValueError(f"unknown status: {status}")


def key(row: dict[str, str]) -> tuple:
    return (row["point_id"], row["model"], float(row["mu"]), int(row["phase_i"]),
            int(row["replica"]), int(row["state_i"]), int(row["horizon"]))


def load_run(name: str, horizon: int) -> tuple[dict, dict[tuple, dict[str, str]]]:
    if Path(name).name != name:
        raise ValueError("run directory names must be direct children of the atlas")
    root = BASE / name
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [r for r in read_csv(root / "trajectory_results.csv") if int(r["horizon"]) == horizon]
    index = {key(r): r for r in rows}
    if len(index) != len(rows):
        raise ValueError(f"duplicate trajectory keys in {name}")
    return manifest, index


def compare(ref_name: str, ref: dict, other_name: str, other: dict, horizon: int) -> tuple[list[dict], list[dict]]:
    common = sorted(set(ref) & set(other))
    if not common:
        raise ValueError(f"no paired launches between {ref_name} and {other_name}")
    pair_rows = []
    reps = defaultdict(list)
    transitions = defaultdict(lambda: defaultdict(int))
    for k in common:
        a, b = ref[k], other[k]
        for field in START_FIELDS:
            if not math.isclose(float(a[field]), float(b[field]), rel_tol=0, abs_tol=1e-14):
                raise ValueError(f"initial-state mismatch at {k}, field={field}")
        alo, ahi = status_bounds(a["status"])
        blo, bhi = status_bounds(b["status"])
        diff_lo, diff_hi = blo - ahi, bhi - alo
        pair_rows.append({"reference": ref_name, "comparison": other_name,
                          "point_id": k[0], "model": k[1], "mu": k[2], "phase_i": k[3],
                          "phase_deg": 90 * k[3], "replica": k[4], "state_i": k[5],
                          "horizon_RF": horizon, "reference_status": a["status"],
                          "comparison_status": b["status"], "status_changed": a["status"] != b["status"],
                          "reference_exit_RF": a["exit_periods"], "comparison_exit_RF": b["exit_periods"],
                          "reference_rho_max": a["rho_max"], "comparison_rho_max": b["rho_max"],
                          "delta_P_lower_indicator": diff_lo, "delta_P_upper_indicator": diff_hi})
        reps[(k[0], k[1], k[2], k[4])].append((diff_lo, diff_hi))
        transitions[(k[0], k[1])][f"{a['status']}→{b['status']}"] += 1

    rep_rows = []
    grouped = defaultdict(list)
    for (point, model, mu, replica), vals in sorted(reps.items()):
        n = len(vals)
        row = {"reference": ref_name, "comparison": other_name, "point_id": point, "model": model,
               "mu": mu, "replica": replica, "paired_launches": n,
               "delta_P_lower": float(np.mean([x[0] for x in vals])),
               "delta_P_upper": float(np.mean([x[1] for x in vals])),
               "unresolved_pairs": sum(x[0] != x[1] for x in vals)}
        rep_rows.append(row)
        grouped[(point, model, mu)].append(row)

    summary = []
    for (point, model, mu), rows in sorted(grouped.items()):
        if len(rows) != 4:
            raise ValueError(f"expected 4 independent Sobol replicas for {(point, model, mu)}")
        out = {"reference": ref_name, "comparison": other_name, "point_id": point,
               "model": model, "mu": mu, "horizon_RF": horizon, "replicates": 4,
               "paired_launches": sum(int(r["paired_launches"]) for r in rows)}
        for metric in ("delta_P_lower", "delta_P_upper"):
            mean, low, high = interval([float(r[metric]) for r in rows])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_t95_low"] = low
            out[f"{metric}_t95_high"] = high
        out["unresolved_pairs"] = sum(int(r["unresolved_pairs"]) for r in rows)
        summary.append(out)

    return pair_rows, summary


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-directory", required=True)
    parser.add_argument("--comparison-directories", nargs="+", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    if Path(args.output_directory).name != args.output_directory:
        parser.error("output directory must be a direct child of the atlas")
    out = BASE / args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    ref_manifest, ref = load_run(args.reference_directory, args.horizon)
    reference_delta = float(ref_manifest.get("relative_delta_rad", 0.0))
    manifests = {args.reference_directory: ref_manifest}
    all_pairs, all_summaries = [], []
    runs = [(args.reference_directory, ref)]
    for name in args.comparison_directories:
        manifest, index = load_run(name, args.horizon)
        comparison_delta = float(manifest.get("relative_delta_rad", 0.0))
        if not math.isclose(reference_delta, comparison_delta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"relative octupole phase mismatch: {args.reference_directory} has "
                f"delta={reference_delta}, {name} has delta={comparison_delta}"
            )
        manifests[name] = manifest
        pairs, summaries = compare(args.reference_directory, ref, name, index, args.horizon)
        all_pairs.extend(pairs)
        all_summaries.extend(summaries)
        runs.append((name, index))
    # Explicit second-grid agreement is measured on common h50/h35 launch keys.
    mesh_rows, mesh_summary = [], []
    if len(runs) >= 3:
        a_name, a = runs[1]
        b_name, b = runs[2]
        pairs, summaries = compare(a_name, a, b_name, b, args.horizon)
        mesh_rows.extend(pairs)
        mesh_summary.extend(summaries)
    write_csv(out / "field_comparison_trajectories.csv", all_pairs)
    write_csv(out / "field_comparison_summary.csv", all_summaries)
    if mesh_rows:
        write_csv(out / "mesh_pair_trajectories.csv", mesh_rows)
        write_csv(out / "mesh_pair_summary.csv", mesh_summary)
    metadata = {"reference_directory": args.reference_directory,
                "comparison_directories": args.comparison_directories,
                "horizon_RF": args.horizon,
                "relative_delta_rad": reference_delta,
                "common_initial_state_abs_tolerance": 1e-14,
                "uncertainty": "paired differences are averaged within each of four independent Sobol scrambles; Student-t interval across four replicate means; unresolved statuses widen bounds",
                "source_manifest_sha256": {name: sha256(BASE / name / "manifest.json") for name in manifests},
                "field_modes": {name: manifests[name]["field_mode"] for name in manifests},
                "comparison_script_sha256": sha256(Path(__file__))}
    (out / "field_comparison_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reference": args.reference_directory,
                      "comparisons": args.comparison_directories,
                      "comparison_trajectory_rows": len(all_pairs),
                      "summary_rows": len(all_summaries),
                      "mesh_pair_summary_rows": len(mesh_summary), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
