#!/usr/bin/env python3
"""Compare native P4 controls with the exact V12 baseline Sobol state.

This is postprocessing only. It integrates no trajectories and solves no FEM
problems. The P4 results are produced by scripts/run_v11_fem_checks.py; the
N48 h50 values are read from the already completed acceptance checkpoints.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V12 = ROOT / "outputs/octupole_comparison_home/dimensionless_acceptance_v12"
RUN = V12 / "runs/e6bf3a33795e4a5354a8_equal_broad_lambda_h50_4x256"
CHECK = V12 / "validation/native_p4_baseline_r00_000000"
MODELS = ("A", "B", "C")
STATE = ("x0_m", "y0_m", "vx0_m_s", "vy0_m_s")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(a: float, b: float, atol: float = 1e-14) -> bool:
    return abs(a - b) <= atol


def main() -> None:
    config = read_json(RUN / "config.json")
    if config["config_hash"] != "e6bf3a33795e4a5354a8":
        raise ValueError("Unexpected source acceptance configuration")
    if config["horizon_RF"] != 1000 or config["phase_deg"] != 0.0:
        raise ValueError("Native-P4 comparison requires the phase-zero 1000-RF baseline")

    sample_rows = read_csv(RUN / "initial_states_r00.csv")
    sample = next(row for row in sample_rows if row["sample_index"] == "0")
    candidates = read_csv(CHECK / "candidates.csv")
    if {row["model"] for row in candidates} != set(MODELS) or len(candidates) != 3:
        raise ValueError("Candidate file must contain exactly one matched A/B/C state")
    for candidate in candidates:
        for name in STATE:
            if not close(float(candidate[name]), float(sample[name])):
                raise ValueError(f"Candidate does not equal Sobol r00_000000 for {name}")
        if not close(float(candidate["phase_rad"]), float(sample["phase_rad"])):
            raise ValueError("Candidate RF phase differs from the Sobol sample")
        if not close(float(candidate["voltage_V"]), 250.0, 1e-12):
            raise ValueError("Native-P4 comparison voltage must be 250 V")

    p4_config = read_json(CHECK / "config.json")
    if p4_config["horizon_RF"] != 1000 or p4_config["mesh_um"] != [35, 50]:
        raise ValueError("Unexpected native-P4 control horizon or meshes")
    p4_rows = read_csv(CHECK / "results.csv")
    if len(p4_rows) != 6 or {(row["model"], int(row["mesh_um"])) for row in p4_rows} != {
        (model, mesh) for model in MODELS for mesh in (35, 50)
    }:
        raise ValueError("Expected exactly six native-P4 A/B/C × h35/h50 results")

    results = []
    for p4 in sorted(p4_rows, key=lambda row: (row["model"], int(row["mesh_um"]))):
        model, mesh = p4["model"], int(p4["mesh_um"])
        cp = read_json(RUN / f"checkpoints/{model}_h50_scale1/r00_000000.json")
        if cp.get("status") not in {"retained", "escaped"} or not cp.get("complete"):
            raise ValueError(f"N48 h50 baseline state is unresolved for {model}")
        if cp.get("config_hash") != config["config_hash"] or cp.get("sample_id") != "r00_000000":
            raise ValueError(f"N48 checkpoint provenance mismatch for {model}")
        if p4["status"] not in {"retained", "escaped"}:
            raise ValueError(f"Native P4 outcome is unresolved for {model} h{mesh}")

        p4_exit = None if not p4["escape_RF"] else float(p4["escape_RF"])
        n48_exit = None if cp.get("escape_RF") is None else float(cp["escape_RF"])
        p4_status = p4["status"]
        n48_status = cp["status"]
        field_relative = (
            f"outputs/abc_validation/direct_fem_validation/v2/h{mesh}_P4/sector_solution.npz"
            if model in "AB" else
            f"outputs/octupole_comparison_home/confinement_maps_v11/fields/h{mesh}_P4_C/sector_solution.npz"
        )
        row = {
            "model": model,
            "mesh_um": mesh,
            "sample_id": "r00_000000",
            "x0_m": float(sample["x0_m"]),
            "y0_m": float(sample["y0_m"]),
            "vx0_m_s": float(sample["vx0_m_s"]),
            "vy0_m_s": float(sample["vy0_m_s"]),
            "phase_rad": float(sample["phase_rad"]),
            "voltage_V": 250.0,
            "horizon_RF": 1000,
            "native_P4_status": p4_status,
            "native_P4_exit_RF": p4_exit,
            "N48_h50_status": n48_status,
            "N48_h50_exit_RF": n48_exit,
            "status_match": p4_status == n48_status,
            "native_P4_minus_N48_h50_exit_RF": (
                None if p4_exit is None or n48_exit is None else p4_exit - n48_exit
            ),
            "native_P4_nfev": int(p4["nfev"]),
            "native_P4_field_sha256": p4_config["input_sha256"][field_relative],
        }
        results.append(row)

    grouped = {model: [r for r in results if r["model"] == model] for model in MODELS}
    for model, rows in grouped.items():
        if len(rows) != 2 or rows[0]["status_match"] is not True or rows[1]["status_match"] is not True:
            raise ValueError(f"Native P4/N48 classification mismatch for {model}")
        h35 = next(r for r in rows if r["mesh_um"] == 35)
        h50 = next(r for r in rows if r["mesh_um"] == 50)
        if h35["native_P4_status"] != h50["native_P4_status"]:
            raise ValueError(f"Native P4 h35/h50 status differs for {model}")
        h50["native_P4_h35_minus_h50_exit_RF"] = (
            None if h35["native_P4_exit_RF"] is None or h50["native_P4_exit_RF"] is None
            else h35["native_P4_exit_RF"] - h50["native_P4_exit_RF"]
        )

    output_csv = CHECK / "native_p4_baseline_comparison.csv"
    fields = list(results[0]) + ["native_P4_h35_minus_h50_exit_RF"]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    h50_diffs = [abs(r["native_P4_minus_N48_h50_exit_RF"]) for r in results
                 if r["mesh_um"] == 50 and r["native_P4_minus_N48_h50_exit_RF"] is not None]
    summary = {
        "study": "V12 exact baseline Sobol point: independent native P4 versus N48",
        "scope": "One predeclared 4D validation point; excluded from acceptance integral as an extra sample",
        "sample_id": "r00_000000",
        "initial_state": {name: float(sample[name]) for name in (*STATE, "phase_rad")},
        "energy_eV": float(sample["E0_eV"]),
        "mass_kg": config["physical"]["mass_kg"],
        "charge_C": config["physical"]["charge_C"],
        "frequency_Hz": config["physical"]["frequency_Hz"],
        "voltage_V": 250.0,
        "horizon_RF": 1000,
        "native_P4_method": p4_config["solver"],
        "native_P4_cases": len(results),
        "all_native_P4_h35_h50_statuses_agree": True,
        "all_native_P4_statuses_match_N48_h50": True,
        "maximum_absolute_h50_native_P4_minus_N48_exit_RF": max(h50_diffs, default=None),
        "models": {
            model: {
                "native_P4_h35_status": next(r["native_P4_status"] for r in rows if r["mesh_um"] == 35),
                "native_P4_h50_status": next(r["native_P4_status"] for r in rows if r["mesh_um"] == 50),
                "native_P4_h35_exit_RF": next(r["native_P4_exit_RF"] for r in rows if r["mesh_um"] == 35),
                "native_P4_h50_exit_RF": next(r["native_P4_exit_RF"] for r in rows if r["mesh_um"] == 50),
                "N48_h50_status": next(r["N48_h50_status"] for r in rows if r["mesh_um"] == 50),
                "N48_h50_exit_RF": next(r["N48_h50_exit_RF"] for r in rows if r["mesh_um"] == 50),
            }
            for model, rows in grouped.items()
        },
        "input_config_sha256": p4_config["input_sha256"],
        "comparison_csv": output_csv.name,
        "interpretation_limit": "A single start validates this trajectory and the tested FEM discretizations only; it does not bound the full 4D acceptance error.",
    }
    (CHECK / "native_p4_baseline_comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
