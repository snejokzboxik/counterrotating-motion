"""Independently audit the archived 50k sign-transition continuation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "results/rotation_sign_scan"
INPUT = SCAN / "local15_sign_transition_50000.json"
OUTPUT = SCAN / "local15_sign_transition_50000_audit.json"
HORIZONS = (5_000, 10_000, 15_000, 20_000, 30_000, 40_000, 50_000)
STEPS_PER_PERIOD = 96
WINDOW = 500


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=2e-13, atol=2e-15))


def main() -> None:
    report = json.loads(INPUT.read_text(encoding="utf-8"))
    panel = ROOT / "results/frozen_local15_single_particle_panel.csv"
    panel_hash = sha256(panel)
    assert report["status"] == "completed"
    assert report["panel_sha256"] == panel_hash
    assert report["steps_per_primary_period"] == STEPS_PER_PERIOD
    assert tuple(report["horizons"]) == HORIZONS
    assert len(report["ensembles"]) == 5

    checked = []
    total_starts = total_window_rows = 0
    for ensemble in report["ensembles"]:
        checkpoint = ROOT / ensemble["checkpoint"]
        assert checkpoint.is_file()
        with np.load(checkpoint, allow_pickle=False) as data:
            step = int(data["step"])
            nu = float(data["nu"])
            species = str(data["species"].item())
            radii = data["rho_samples"]
            angles = data["angle_samples"]
            escape_periods = data["escape_period"].copy()
            max_rho = data["maxrho"].copy()

        n = len(ensemble["trajectory_records"])
        assert n == 24
        assert nu == float(ensemble["nu"])
        assert species == ensemble["species"]
        assert step == 50_000 * STEPS_PER_PERIOD
        assert radii.shape == angles.shape == (50_001, n)
        assert np.isfinite(radii).all() and np.isfinite(angles).all()
        recomputed_counts = {}

        for j, record in enumerate(ensemble["trajectory_records"]):
            escape = None if not np.isfinite(escape_periods[j]) else float(escape_periods[j])
            assert record["start_id"]
            assert record["escape_period"] == escape
            assert close(float(record["max_rho"]), float(max_rho[j]))
            cumulative = float((angles[50_000, j] - angles[15_000, j]) /
                               (2 * np.pi * 35_000))
            assert close(float(record["omega_mean_15k_to_50k"]), cumulative)

            for horizon in HORIZONS:
                item = record["horizons"][HORIZONS.index(horizon)]
                mean_rho = float(np.mean(radii[horizon-WINDOW:horizon+1, j]))
                omega = float((angles[horizon, j] - angles[horizon-WINDOW, j]) /
                              (2 * np.pi * WINDOW))
                escaped = escape is not None and escape <= horizon
                sign = 0 if escaped else int(np.sign(omega))
                assert item["horizon_periods"] == horizon
                assert close(float(item["rho_mean_last_500"]), mean_rho)
                assert close(float(item["omega_last_500"]), omega)
                assert item["escaped_by_horizon"] is escaped
                assert item["sign"] == sign
                counts = recomputed_counts.setdefault(str(horizon),
                    {"positive": 0, "negative": 0, "escaped": 0, "n_starts": n})
                counts["positive"] += sign > 0
                counts["negative"] += sign < 0
                counts["escaped"] += escaped
                total_window_rows += 1

        assert ensemble["sign_counts"] == recomputed_counts
        retained = [r["omega_mean_15k_to_50k"] for r in ensemble["trajectory_records"]
                    if r["escape_period"] is None or r["escape_period"] > 50_000]
        expected_cumulative = {
            "positive": sum(x > 0 for x in retained),
            "negative": sum(x < 0 for x in retained),
            "escaped": n - len(retained),
            "n_starts": n,
            "retained_min_median_max": ([float(min(retained)), float(np.median(retained)),
                                         float(max(retained))] if retained else []),
        }
        actual_cumulative = ensemble["cumulative_15k_to_50k"]
        assert {k: actual_cumulative[k] for k in ("positive", "negative", "escaped", "n_starts")} == {
            k: expected_cumulative[k] for k in ("positive", "negative", "escaped", "n_starts")}
        assert all(close(a, b) for a, b in zip(actual_cumulative["retained_min_median_max"],
                                               expected_cumulative["retained_min_median_max"]))
        total_starts += n
        checked.append({"species": species, "nu": nu, "starts": n,
                        "step": step, "checkpoint_sha256": sha256(checkpoint),
                        "recomputed_sign_counts_match": True,
                        "recomputed_cumulative_summary_match": True})

    with (SCAN / "local15_sign_transition_50000.csv").open(
            newline="", encoding="utf-8") as stream:
        csv_rows = sum(1 for _ in csv.DictReader(stream))
    assert csv_rows == total_window_rows == 840
    assert report["continued_trajectory_segments"] == total_starts == 120
    assert report["additional_particle_periods"] == 4_488_000

    result = {
        "status": "passed",
        "input": INPUT.relative_to(ROOT).as_posix(),
        "panel_sha256": panel_hash,
        "ensembles_checked": len(checked),
        "trajectory_records_checked": total_starts,
        "trajectory_window_rows_recomputed": total_window_rows,
        "csv_rows_checked": csv_rows,
        "nominal_continuation_particle_periods": report["additional_particle_periods"],
        "checks": [
            "panel SHA256 matches the archived fixed-start panel",
            "all checkpoints reach the exact target step and have finite histories",
            "JSON window means, rotation signs, escape counts, and cumulative means recompute from NPZ arrays",
            "CSV row count and nominal trajectory/particle-period inventory match the raw ensembles",
        ],
        "ensembles": checked,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "status", "ensembles_checked", "trajectory_records_checked",
        "trajectory_window_rows_recomputed", "csv_rows_checked",
        "nominal_continuation_particle_periods")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
