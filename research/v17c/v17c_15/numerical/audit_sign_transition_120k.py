"""Independent integrity and statistic audit of the 120k continuation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "results/rotation_sign_scan"
REPORT = SCAN / "local15_sign_transition_120000_mixed_node.json"
CSV = SCAN / "local15_sign_transition_120000_mixed_node.csv"
PANEL = ROOT / "results/frozen_local15_single_particle_panel.csv"
SOURCE = SCAN / "extended_sign_transition_50000_checkpoints/nu_2.231_species1.npz"
COMBINED = SCAN / "extended_sign_transition_120000_checkpoints/nu_2.231_species1.npz"
TARGET_DIR = SCAN / "extended_sign_transition_120000_checkpoints"
STEPS = 96
HORIZONS = (50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 110_000, 120_000)
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
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    panel_hash = sha256(PANEL)
    assert report["status"] == "completed"
    assert report["panel_sha256"] == panel_hash
    assert report["source_checkpoint_sha256"] == sha256(SOURCE)
    assert report["target_checkpoint_sha256"] == sha256(COMBINED)
    assert report["steps_per_primary_period"] == STEPS
    assert tuple(report["window_horizons"]) == HORIZONS
    assert report["continued_trajectory_segments"] == 24
    assert report["additional_particle_periods"] == 24 * 70_000

    with np.load(SOURCE, allow_pickle=False) as src, np.load(COMBINED, allow_pickle=False) as dst:
        src_rho, src_angle = src["rho_samples"], src["angle_samples"]
        rho, angle = dst["rho_samples"], dst["angle_samples"]
        escape_periods = dst["escape_period"].copy()
        maxrho = dst["maxrho"].copy()
        active = dst["active"].copy()
        escaped = dst["escaped"].copy()
        assert int(src["step"]) == 50_000 * STEPS
        assert int(dst["step"]) == 120_000 * STEPS
        assert float(dst["nu"]) == 2.23125
        assert str(dst["species"].item()) == "species1"
        assert rho.shape == angle.shape == (120_001, 24)
        assert src_rho.shape == src_angle.shape == (50_001, 24)
        assert np.isfinite(rho).all() and np.isfinite(angle).all()
        assert np.array_equal(rho[:50_001], src_rho)
        assert np.array_equal(angle[:50_001], src_angle)

    with PANEL.open(newline="", encoding="utf-8") as stream:
        panel_rows = list(csv.DictReader(stream))
    assert len(panel_rows) == 24
    records = report["trajectory_records"]
    assert len(records) == len(panel_rows)
    counts: dict[str, dict[str, int]] = {
        str(h): {"negative": 0, "positive": 0, "escaped": 0, "n_starts": 24}
        for h in HORIZONS
    }
    by_id = {r["start_id"]: r for r in records}
    assert set(by_id) == {r["start_id"] for r in panel_rows}
    recomputed_cum50: list[float] = []
    recomputed_cum15: list[float] = []
    csv_expected = {}
    per_start_hashes = []
    for i, row in enumerate(panel_rows):
        record = by_id[row["start_id"]]
        one = TARGET_DIR / row["start_id"] / "nu_2.231_species1.npz"
        assert one.is_file()
        with np.load(one, allow_pickle=False) as individual:
            assert int(individual["step"]) == 120_000 * STEPS
            assert individual["rho_samples"].shape == (120_001, 1)
            assert np.array_equal(individual["rho_samples"][:, 0], rho[:, i])
            assert np.array_equal(individual["angle_samples"][:, 0], angle[:, i])
            assert np.array_equal(individual["rho_samples"][:50_001, 0], src_rho[:, i])
            assert np.array_equal(individual["angle_samples"][:50_001, 0], src_angle[:, i])
        per_start_hashes.append(sha256(one))

        ep = None if not np.isfinite(escape_periods[i]) else float(escape_periods[i])
        assert record["escape_period"] == ep
        assert close(record["max_rho"], float(maxrho[i]))
        cum50 = float((angle[120_000, i] - angle[50_000, i]) /
                      (2 * np.pi * 70_000))
        cum15 = float((angle[120_000, i] - angle[15_000, i]) /
                      (2 * np.pi * 105_000))
        assert close(record["omega_mean_50k_to_120k"], cum50)
        assert close(record["omega_mean_15k_to_120k"], cum15)
        retained = ep is None or ep > 120_000
        if retained:
            recomputed_cum50.append(cum50)
            recomputed_cum15.append(cum15)
        for j, horizon in enumerate(HORIZONS):
            item = record["horizons"][j]
            omega = float((angle[horizon, i] - angle[horizon-WINDOW, i]) /
                          (2 * np.pi * WINDOW))
            mean_rho = float(np.mean(rho[horizon-WINDOW:horizon+1, i]))
            escaped_by_h = ep is not None and ep <= horizon
            sign = 0 if escaped_by_h else int(np.sign(omega))
            assert item["horizon_periods"] == horizon
            assert close(item["omega_last_500"], omega)
            assert close(item["rho_mean_last_500"], mean_rho)
            assert item["escaped_by_horizon"] is escaped_by_h
            assert item["sign"] == sign
            counts[str(horizon)]["negative"] += sign < 0
            counts[str(horizon)]["positive"] += sign > 0
            counts[str(horizon)]["escaped"] += escaped_by_h
            csv_expected[(row["start_id"], horizon)] = {
                "omega_last_500": omega, "rho_mean_last_500": mean_rho,
                "sign": sign, "escaped_by_horizon": escaped_by_h,
            }
    assert np.array_equal(escaped, ~active)
    assert report["window_sign_counts"] == counts

    for key, values in (("cumulative_50k_to_120k", recomputed_cum50),
                        ("cumulative_15k_to_120k", recomputed_cum15)):
        actual = report[key]
        expected = {"positive": sum(v > 0 for v in values),
                    "negative": sum(v < 0 for v in values),
                    "escaped": 24 - len(values), "n_starts": 24,
                    "retained_min_median_max": [float(min(values)),
                        float(np.median(values)), float(max(values))] if values else []}
        assert {k: actual[k] for k in expected if k != "retained_min_median_max"} == {
            k: expected[k] for k in expected if k != "retained_min_median_max"}
        assert all(close(a, b) for a, b in zip(
            actual["retained_min_median_max"], expected["retained_min_median_max"]))

    with CSV.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 24 * len(HORIZONS)
    for row in csv_rows:
        key = (row["start_id"], int(row["horizon_periods"]))
        exp = csv_expected[key]
        assert close(float(row["omega_last_500"]), exp["omega_last_500"])
        assert close(float(row["rho_mean_last_500"]), exp["rho_mean_last_500"])
        assert int(row["sign"]) == exp["sign"]
        assert (row["escaped_by_horizon"].lower() == "true") is exp["escaped_by_horizon"]

    result = {
        "status": "passed",
        "input_report_sha256": sha256(REPORT),
        "panel_sha256": panel_hash,
        "source_50k_checkpoint_sha256": sha256(SOURCE),
        "combined_120k_checkpoint_sha256": sha256(COMBINED),
        "trajectory_segments_checked": 24,
        "per_start_checkpoints_checked": len(per_start_hashes),
        "sample_rows_checked": 120_001 * 24,
        "trajectory_window_rows_recomputed": len(csv_rows),
        "additional_particle_periods_recomputed": 24 * 70_000,
        "sign_counts_120k": counts["120000"],
        "cumulative_50k_to_120k": report["cumulative_50k_to_120k"],
        "cumulative_15k_to_120k": report["cumulative_15k_to_120k"],
        "checks": [
            "all 24 start histories retain an exact bitwise match to the archived 50k prefix",
            "combined checkpoint is exactly reproducible by its 24 per-start checkpoints",
            "step, metadata, finite arrays, escape flags, and panel provenance are valid",
            "all local-window means, signs, escape counts, cumulative means, and CSV values recompute",
        ],
    }
    output = SCAN / "local15_sign_transition_120000_mixed_node_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in (
        "status", "trajectory_segments_checked", "sample_rows_checked",
        "trajectory_window_rows_recomputed", "sign_counts_120k",
        "cumulative_50k_to_120k", "cumulative_15k_to_120k")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
