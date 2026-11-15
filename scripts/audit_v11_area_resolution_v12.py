"""Reconcile all V11 XY-map area estimates against their saved row data.

This is a source-only audit. It does not integrate trajectories or write into
the V11 output tree. It checks all equal-voltage 21x21, 41x41, and 61x61 maps
for phases 0, 45, and 90 degrees against the published area_resolution.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V11 = ROOT / "outputs/octupole_comparison_home/confinement_maps_v11"
OUT = ROOT / "outputs/octupole_comparison_home/dimensionless_acceptance_v12/validation"
AREA = V11 / "v11_completion/postprocessing/area_resolution.csv"
OLD_21 = {
    0: V11 / "xy/4ceec3a67c8fdfcd",
    45: V11 / "xy/7c854b0d1f8f059f",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_dir(phase: int, grid: int) -> Path:
    if grid == 21 and phase in OLD_21:
        return OLD_21[phase]
    return V11 / f"v11_completion/xy/phase{phase:02d}_{grid}"


def close(a: float, b: float, atol: float = 1e-10) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=atol)


def main() -> None:
    area_rows = [row for row in read_csv(AREA)
                 if row["protocol"] == "equal" and int(row["grid"]) in (21, 41, 61)]
    out_rows: list[dict] = []
    errors: list[str] = []

    for area in area_rows:
        phase = int(round(float(area["phase_deg"])))
        grid = int(area["grid"])
        model = area["model"]
        directory = source_dir(phase, grid)
        config_path = directory / "config.json"
        initial_path = directory / "initial_states.csv"
        results_path = directory / "results.csv"
        if not all(path.is_file() for path in (config_path, initial_path, results_path)):
            errors.append(f"missing source file for phase={phase}, grid={grid}: {directory}")
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        initial = read_csv(initial_path)
        results = read_csv(results_path)
        subset_initial = [row for row in initial if row["model"] == model]
        subset_results = [row for row in results if row["model"] == model]
        statuses = {name: sum(row["status"] == name for row in subset_results)
                    for name in ("retained", "escaped", "numerical_error")}
        coordinates = {(float(row["x0_m"]), float(row["y0_m"])) for row in subset_initial}
        result_coordinates = {(float(row["x0_m"]), float(row["y0_m"])) for row in subset_results}
        xs = sorted({x for x, _ in coordinates})
        ys = sorted({y for _, y in coordinates})
        dx_mm = (xs[1] - xs[0]) * 1e3 if len(xs) > 1 else math.nan
        dy_mm = (ys[1] - ys[0]) * 1e3 if len(ys) > 1 else math.nan
        step_mm = float(area["step_mm"])
        sampled = len(subset_results)
        retained_area = statuses["retained"] * step_mm**2
        upper_area = (statuses["retained"] + statuses["numerical_error"]) * step_mm**2
        source_area = float(area["retained_cell_area_mm2"])
        config_ok = (
            int(config.get("grid", -1)) == grid
            and int(round(float(config.get("phase_deg", -999)))) == phase
            and config.get("protocol") == "equal"
            and config.get("horizon_RF") == 1000
            and config.get("mesh_um") == 50
            and all(close(float(config.get("voltage_V", {}).get(name, math.nan)), 250.0)
                    for name in ("A", "B", "C"))
        )
        coordinate_match = coordinates == result_coordinates
        pass_row = (
            config_ok and coordinate_match and close(dx_mm, step_mm)
            and close(dy_mm, step_mm) and sampled == int(area["sampled"])
            and statuses["retained"] == int(area["retained"])
            and statuses["escaped"] == int(area["escaped"])
            and statuses["numerical_error"] == int(area["numerical_error"])
            and close(retained_area, source_area)
            and close(upper_area, float(area["upper_cell_area_mm2"]))
        )
        if not pass_row:
            errors.append(f"source reconciliation failed for phase={phase}, grid={grid}, model={model}")
        out_rows.append({
            "phase_deg": phase, "grid": grid, "model": model,
            "source_dir": directory.relative_to(ROOT).as_posix(),
            "initial_states_csv_sha256": sha256(initial_path),
            "results_csv_sha256": sha256(results_path),
            "config_json_sha256": sha256(config_path),
            "sampled": sampled,
            "retained": statuses["retained"], "escaped": statuses["escaped"],
            "numerical_error": statuses["numerical_error"],
            "step_mm_source": step_mm, "step_mm_recomputed_x": dx_mm,
            "step_mm_recomputed_y": dy_mm,
            "retained_cell_area_mm2_recomputed": retained_area,
            "upper_cell_area_mm2_recomputed": upper_area,
            "retained_cell_area_mm2_source": source_area,
            "config_match": config_ok,
            "coordinate_sets_match": coordinate_match,
            "pass": pass_row,
        })

    expected_rows = 27
    if len(area_rows) != expected_rows:
        errors.append(f"expected {expected_rows} equal-area rows, found {len(area_rows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "v11_area_source_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out_rows[0]) if out_rows else [])
        writer.writeheader()
        writer.writerows(out_rows)
    manifest = {
        "study": "V12 source-only reconciliation of V11 XY-map area estimates",
        "status": "passed" if not errors and len(out_rows) == expected_rows else "failed",
        "area_resolution_source": AREA.relative_to(ROOT).as_posix(),
        "equal_protocol_rows_checked": len(out_rows),
        "expected_rows": expected_rows,
        "grid_sizes": [21, 41, 61],
        "phases_deg": [0, 45, 90],
        "checks": [
            "V11 physical configuration and voltage protocol",
            "initial/result coordinate set equality per model",
            "saved XY lattice spacing",
            "retained/escaped/numerical-error row counts",
            "retained and upper node-cell area calculations",
            "source-file SHA-256 provenance",
        ],
        "errors": errors,
        "results_csv": csv_path.name,
    }
    (OUT / "v11_area_source_audit.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
