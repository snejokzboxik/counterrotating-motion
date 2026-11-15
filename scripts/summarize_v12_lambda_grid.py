"""Export the configured physical Lambda and leading-mode kappa grid for V12.

This is metadata postprocessing only. It reads the frozen broad-run config and
FEM harmonic coefficients; it does not integrate trajectories or infer outcomes.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/octupole_comparison_home/dimensionless_acceptance_v12/runs/e6bf3a33795e4a5354a8_equal_broad_lambda_h50_4x256"
OUT = ROOT / "outputs/octupole_comparison_home/dimensionless_acceptance_v12/validation"
EXPECTED_KINDS = ("A", "B", "C", "idealA", "idealB", "idealC")
ROTATING = {"A", "idealA"}


def close(a: float, b: float, rel: float = 2e-13) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-15)


def main() -> None:
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    specs = json.loads((RUN / "model_coefficients.json").read_text(encoding="utf-8"))
    mass = float(config["physical"]["mass_kg"])
    charge = float(config["physical"]["charge_C"])
    frequency = float(config["physical"]["frequency_Hz"])
    omega = 2.0 * math.pi * frequency
    length = float(config["physical"]["length_m"])
    scales = [float(value) for value in config["scales"]]
    voltage_base = 250.0

    rows: list[dict] = []
    for spec in specs:
        kind = spec["kind"]
        if kind not in EXPECTED_KINDS:
            continue
        order = int(spec["leading_order"])
        cosine = complex(*spec["cosine"][order])
        sine = complex(*spec["sine"][order])
        quadrature_norm = math.sqrt(abs(cosine) ** 2 + abs(sine) ** 2)
        if kind in ROTATING:
            if not close(abs(cosine), abs(sine)):
                raise ValueError(f"Expected equal leading quadratures for rotating {kind}")
            c_m = quadrature_norm / math.sqrt(2.0)
            convention = "equal cosine/sine quadratures: c_m=norm/sqrt(2)"
        else:
            if not (close(abs(cosine), 0.0) or close(abs(sine), 0.0)):
                raise ValueError(f"Expected a single leading quadrature for standing {kind}")
            c_m = quadrature_norm
            convention = "single standing quadrature: c_m=quadrature norm"

        voltage = float(spec["voltage_V"])
        lam = float(spec["coupling"])
        expected_lam = charge * voltage / (mass * omega**2 * length**2)
        if not close(lam, expected_lam):
            raise ValueError(f"Lambda mismatch for {spec['name']}: {lam} vs {expected_lam}")
        if not any(close(float(spec["scale_factor"]), scale) for scale in scales):
            raise ValueError(f"Unexpected scale in {spec['name']}")
        rows.append({
            "model": spec["name"], "kind": kind, "mesh_um": int(spec["mesh_um"]),
            "scale_factor": float(spec["scale_factor"]), "voltage_V": voltage,
            "Lambda": lam, "leading_order_m": order,
            "leading_quadrature_norm": quadrature_norm,
            "leading_c_m_convention": c_m,
            "kappa_m_leading_Lambda_c_m": lam * c_m,
            "leading_rms_field_per_volt_V_m": spec["leading_rms_field_per_volt_V_m"],
            "coefficient_convention": convention,
            "interpretation": "leading-mode control parameter only; full FEM model retains higher harmonics",
        })

    if len(rows) != len(EXPECTED_KINDS) * len(scales):
        raise ValueError(f"Expected {len(EXPECTED_KINDS)*len(scales)} rows, found {len(rows)}")
    rows.sort(key=lambda row: (EXPECTED_KINDS.index(row["kind"]), row["scale_factor"]))
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "lambda_scan_parameters_v12.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    base = [row for row in rows if close(row["scale_factor"], 1.0)]
    manifest = {
        "study": "V12 broad equal-peak scan parameter provenance",
        "status": "configured_only_no_trajectory_results",
        "source_run": RUN.relative_to(ROOT).as_posix(),
        "source_config_hash": config["config_hash"],
        "source_protocol": "equal 250 V electrode peaks at scale 1, scaled together for Lambda scan",
        "scales": scales,
        "Lambda_by_scale": {
            str(scale): next(row["Lambda"] for row in rows
                             if row["kind"] == "A" and close(row["scale_factor"], scale))
            for scale in scales
        },
        "base_scale_leading_parameters": [
            {key: row[key] for key in (
                "kind", "voltage_V", "Lambda", "leading_order_m",
                "leading_quadrature_norm", "leading_c_m_convention", "kappa_m_leading_Lambda_c_m"
            )}
            for row in base
        ],
        "rows": len(rows),
        "csv": csv_path.name,
        "note": "Kappa values describe only the leading ideal-multipole component extracted from each FEM field; they are not total-field equivalences.",
    }
    json_path = OUT / "lambda_scan_parameters_v12.json"
    json_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
