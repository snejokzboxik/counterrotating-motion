"""Run checkpointed fixed-voltage mass rays for atlas-selected V13.3 points."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
sys.path.insert(0, str(ROOT / "scripts"))
import run_v13_3_fine_mass_scan as runner  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_manifest_peak(path: Path, point: dict, peak_v: float, entry_hash: str) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entry_script"] = "scripts/run_v13_3_adaptive_mass_rays.py"
    manifest["entry_script_sha256"] = entry_hash
    manifest["reason"] = point.get("reason", "atlas-selected fixed-voltage mass ray")
    manifest["physical_constants"]["max_electrode_peak_abs_V"] = peak_v
    manifest["physical_constants"]["electrode_peak_method"] = (
        "max over 12 electrodes of |VDC Q2| + hypot(U2 Q2 + U4 C4, U4 S4)"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _init_worker(point: dict) -> None:
    import run_v13_3_fine_mass_scan as worker_runner
    worker_runner.POINT = point


def electrode_peak(point: dict[str, str]) -> float:
    e = 1.602176634e-19
    amu = 1.66053906660e-27
    m0 = 3200.04652261404 * amu
    omega = 2 * math.pi * 100_000.0
    radius = 0.008
    c2 = d2 = 0.7718255558274212
    c4 = 0.5413395699439091
    scale = m0 * omega**2 * radius**2 / e
    alpha, beta, kappa = (float(point[k]) for k in ("alpha", "beta", "kappa"))
    u2 = beta * scale / c2
    vdc = alpha * scale / d2
    u4 = kappa * scale / c4
    electrode_file = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1/electrode_voltage_peaks.csv"
    rows = [r for r in read_csv(electrode_file) if r["scheme"] == "Q"]
    values = [
        abs(vdc * float(r["Q2"]))
        + math.hypot(u2 * float(r["Q2"]) + u4 * float(r["C4"]), u4 * float(r["S4"]))
        for r in rows
    ]
    return max(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", default=str(BASE / "fem_nonlinear_gain_points.csv"))
    parser.add_argument("--output-prefix", default="adaptive_mass_ray")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mu-min", type=float, default=0.80)
    parser.add_argument("--mu-max", type=float, default=1.20)
    parser.add_argument("--mu-step", type=float, default=0.025)
    parser.add_argument("--states-per-replica", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in 1..8")
    if Path(args.output_prefix).name != args.output_prefix:
        parser.error("output-prefix must be one directory name")

    points = read_csv(Path(args.points))
    if not points:
        parser.error("points file is empty")
    entry_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for row in points:
        point = {"candidate_id": row["candidate_id"], **{
            key: float(row[key]) for key in ("alpha", "beta", "kappa")
        }}
        peak_v = electrode_peak(row)
        if peak_v > 250.0:
            raise ValueError(f"{point['candidate_id']} exceeds 250 V: {peak_v:.6f}")
        runner.POINT = point
        runner.ProcessPoolExecutor = lambda max_workers, p=point: ProcessPoolExecutor(
            max_workers=max_workers, initializer=_init_worker, initargs=(p,)
        )
        runner.OUT = BASE / f"{args.output_prefix}_{point['candidate_id']}"
        sys.argv = [
            sys.argv[0], "--workers", str(args.workers),
            "--mu-min", str(args.mu_min), "--mu-max", str(args.mu_max),
            "--mu-step", str(args.mu_step),
            "--states-per-replica", str(args.states_per_replica),
        ]
        runner.main()
        write_manifest_peak(runner.OUT, row, peak_v, entry_hash)
        print(json.dumps({"candidate_id": point["candidate_id"], "max_peak_V": peak_v,
                          "directory": str(runner.OUT)}, indent=2))


if __name__ == "__main__":
    main()
