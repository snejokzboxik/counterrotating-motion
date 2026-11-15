"""Replay the saved V11 21x21 XY maps with the V12 dimensionless RF model.

Running this script without ``--run`` performs a source-only audit: it checks
the saved V11 coordinates, physical configuration, status rows, and cell-area
calculations, then writes a manifest under V12/validation. No trajectories
are integrated unless ``--run`` is supplied.

For a later pilot, for example::

    python scripts/replay_v11_maps_v12.py --run --phases 0 --limit 4 --workers 4

``--limit`` counts XY coordinates per phase; all A/B/C rows for each selected
coordinate are kept together. Completed per-trajectory records and 100-period
checkpoints are written only below V12/validation, never into V11.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from three_phase_trap.dimensionless_dynamics_v12 import (  # noqa: E402
    DimensionlessRFModel,
    integrate_to_horizon,
)
from three_phase_trap.rf_reconstruction import (  # noqa: E402
    HarmonicQuadratures,
    RFReconstruction,
)


V11_ROOT = ROOT / "outputs/octupole_comparison_home/confinement_maps_v11"
V12_ROOT = ROOT / "outputs/octupole_comparison_home/dimensionless_acceptance_v12"
OUT = V12_ROOT / "validation"
AREA_CSV = V11_ROOT / "v11_completion/postprocessing/area_resolution.csv"
MAP_DIRS = {
    0: V11_ROOT / "xy/4ceec3a67c8fdfcd",
    45: V11_ROOT / "xy/7c854b0d1f8f059f",
    90: V11_ROOT / "v11_completion/xy/phase90_21",
}
FIELD_PATHS = {
    "A": ROOT / "outputs/abc_validation/direct_fem_validation/v2/h50_P4/diagnostic_N48.json",
    "B": ROOT / "outputs/abc_validation/direct_fem_validation/v2/h50_P4/diagnostic_N48.json",
    "C": V11_ROOT / "fields/h50_P4_C/reconstruction_N48_C.json",
}

E_CHARGE = 1.602176634e-19
MASS_KG = 100_000.0 * 1.66053906660e-27
FREQUENCY_HZ = 100_000.0
OMEGA = 2.0 * math.pi * FREQUENCY_HZ
APERTURE_M = 8.0e-3
INJECTION_RADIUS_M = 7.5e-3
LENGTH_M = APERTURE_M
MODELS = "ABC"
HORIZON_RF = 1000
SEGMENT_RF = 100
RTOL = 2.0e-9
MAX_STEP_RF = 1.0 / 50.0
SAMPLES_PER_RF = 4
ATOL = np.array([
    2.5e-11,
    2.5e-11,
    2.0e-11 / (LENGTH_M * OMEGA),
    2.0e-11 / (LENGTH_M * OMEGA),
])
REPLAY_CSV = OUT / "dimensionless_replay_of_v11_xy.csv"
MANIFEST_PATH = OUT / "v11_map_replay_source_audit.json"

_WORKER_Q: dict[str, HarmonicQuadratures] = {}
_WORKER_FIELD_SHA: dict[str, str] = {}


@lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def near(a: float, b: float, atol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=atol)


def state_key(row: dict[str, str]) -> tuple[str, float, float, float, float, float]:
    return (
        row["model"],
        float(row["x0_m"]),
        float(row["y0_m"]),
        float(row["vx0_m_s"]),
        float(row["vy0_m_s"]),
        float(row["phase_rad"]),
    )


def source_row_key(row: dict[str, str]) -> tuple[str, int, int, int]:
    # XY grid values are exact millimetre fractions represented in SI. Round
    # only to micrometre precision for a stable identifier; retain full values
    # in the emitted result row.
    return (
        row["model"],
        round(float(row["x0_m"]) * 1e6),
        round(float(row["y0_m"]) * 1e6),
        round(float(row["phase_rad"]) * 1e9),
    )


def _phase_area_rows() -> dict[tuple[int, str], dict[str, str]]:
    rows = read_csv(AREA_CSV)
    return {
        (int(round(float(row["phase_deg"]))), row["model"]): row
        for row in rows
        if row["protocol"] == "equal" and int(row["grid"]) == 21
    }


def audit_phase(phase: int, area_rows: dict[tuple[int, str], dict[str, str]]) -> dict[str, Any]:
    directory = MAP_DIRS[phase]
    config_path = directory / "config.json"
    initial_path = directory / "initial_states.csv"
    results_path = directory / "results.csv"
    for path in (config_path, initial_path, results_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    initial = read_csv(initial_path)
    results = read_csv(results_path)
    errors: list[str] = []

    expected_config = {
        "grid": 21,
        "phase_deg": float(phase),
        "protocol": "equal",
        "mesh_um": 50,
        "horizon_RF": HORIZON_RF,
        "segment_RF": SEGMENT_RF,
        "frequency_Hz": FREQUENCY_HZ,
        "mass_kg": MASS_KG,
        "charge_C": E_CHARGE,
        "aperture_m": APERTURE_M,
    }
    for key, expected in expected_config.items():
        actual = config.get(key)
        if isinstance(expected, float):
            ok = isinstance(actual, (int, float)) and near(float(actual), expected, 1e-14)
        else:
            ok = actual == expected
        if not ok:
            errors.append(f"config {key}: expected {expected!r}, got {actual!r}")
    voltage = config.get("voltage_V", {})
    for model_name in MODELS:
        if not near(float(voltage.get(model_name, math.nan)), 250.0, 1e-12):
            errors.append(f"config voltage for {model_name} is not 250 V")
    solver = config.get("solver", {})
    for key, expected in {
        "method": "DOP853",
        "rtol": RTOL,
        "max_step_RF": MAX_STEP_RF,
    }.items():
        actual = solver.get(key)
        if isinstance(expected, float):
            ok = isinstance(actual, (int, float)) and near(float(actual), expected, 1e-14)
        else:
            ok = actual == expected
        if not ok:
            errors.append(f"solver {key}: expected {expected!r}, got {actual!r}")
    if solver.get("atol_SI") != [2e-13, 2e-13, 2e-11, 2e-11]:
        errors.append(f"solver atol_SI mismatch: got {solver.get('atol_SI')!r}")

    expected_phase_rad = math.radians(phase)
    initial_by_key: dict[tuple[str, float, float, float, float, float], dict[str, str]] = {}
    result_by_key: dict[tuple[str, float, float, float, float, float], dict[str, str]] = {}
    for row in initial:
        key = state_key(row)
        if key in initial_by_key:
            errors.append(f"duplicate initial row {key}")
        initial_by_key[key] = row
        if row["model"] not in MODELS:
            errors.append(f"unexpected model {row['model']}")
        if not near(float(row["vx0_m_s"]), 0.0, 1e-15) or not near(float(row["vy0_m_s"]), 0.0, 1e-15):
            errors.append(f"nonzero initial velocity at {key}")
        if not near(float(row["phase_rad"]), expected_phase_rad, 1e-12):
            errors.append(f"initial phase mismatch at {key}")
        if math.hypot(float(row["x0_m"]), float(row["y0_m"])) > INJECTION_RADIUS_M + 1e-12:
            errors.append(f"start outside 7.5 mm map domain at {key}")
    for row in results:
        key = state_key(row)
        if key in result_by_key:
            errors.append(f"duplicate result row {key}")
        result_by_key[key] = row
        if row["status"] not in {"retained", "escaped", "numerical_error"}:
            errors.append(f"unknown V11 status {row['status']!r}")
        if not near(float(row["voltage_V"]), 250.0, 1e-12):
            errors.append(f"non-250 V result at {key}")
        if not near(float(row["phase_rad"]), expected_phase_rad, 1e-12):
            errors.append(f"result phase mismatch at {key}")

    if set(initial_by_key) != set(result_by_key):
        errors.append("initial-state keys and result keys do not match exactly")

    coordinates_by_model = {
        name: {
            (float(row["x0_m"]), float(row["y0_m"]))
            for row in initial
            if row["model"] == name
        }
        for name in MODELS
    }
    coordinate_sets_match = len({frozenset(v) for v in coordinates_by_model.values()}) == 1
    if not coordinate_sets_match:
        errors.append("A/B/C do not use identical XY coordinates")
    coordinates = coordinates_by_model["A"]

    unique_x = sorted({x for x, _ in coordinates})
    unique_y = sorted({y for _, y in coordinates})
    grid_size = int(config.get("grid", 0))
    if len(unique_x) < 2 or len(unique_y) < 2:
        errors.append(f"expected multiple coordinates along both axes, got {len(unique_x)} x {len(unique_y)}")
    dx_values = [b - a for a, b in zip(unique_x, unique_x[1:])]
    dy_values = [b - a for a, b in zip(unique_y, unique_y[1:])]
    dx_m = float(np.median(dx_values)) if dx_values else math.nan
    dy_m = float(np.median(dy_values)) if dy_values else math.nan
    if any(not near(step, dx_m, 1e-13) for step in dx_values) or any(
        not near(step, dy_m, 1e-13) for step in dy_values
    ) or not near(dx_m, dy_m, 1e-13):
        errors.append("XY coordinates are not on a uniform square lattice")
    if grid_size < 3 or not near(dx_m * (grid_size - 1), 2.0 * INJECTION_RADIUS_M, 1e-12):
        errors.append("inferred square lattice span is inconsistent with grid size and 7.5 mm map radius")
    full_axis = np.linspace(-INJECTION_RADIUS_M, INJECTION_RADIUS_M, grid_size) if grid_size >= 3 else []
    in_disk = {
        (float(x), float(y))
        for x in full_axis
        for y in full_axis
        if math.hypot(float(x), float(y)) < INJECTION_RADIUS_M
    }
    if coordinates != in_disk:
        errors.append("saved coordinates do not equal the strict r<7.5 mm subset of the configured Cartesian lattice")

    model_summaries: dict[str, Any] = {}
    cell_area_mm2 = (dx_m * 1e3) ** 2
    for model_name in MODELS:
        model_results = [row for row in results if row["model"] == model_name]
        statuses = {status: sum(row["status"] == status for row in model_results)
                    for status in ("retained", "escaped", "numerical_error")}
        retained_area = statuses["retained"] * cell_area_mm2
        area_row = area_rows.get((phase, model_name))
        if area_row is None:
            errors.append(f"area_resolution.csv has no equal/21 row for phase={phase}, model={model_name}")
            source_area = None
            source_sampled = None
            source_retained = None
            source_step = None
            source_escaped = None
            area_matches = False
        else:
            source_area = float(area_row["retained_cell_area_mm2"])
            source_sampled = int(area_row["sampled"])
            source_retained = int(area_row["retained"])
            source_escaped = int(area_row["escaped"])
            source_step = float(area_row["step_mm"])
            area_matches = (
                source_sampled == len(coordinates)
                and source_retained == statuses["retained"]
                and source_escaped == statuses["escaped"]
                and near(source_step, dx_m * 1e3, 1e-12)
                and near(source_area, retained_area, 1e-10)
            )
            if not area_matches:
                errors.append(f"area_resolution mismatch for phase={phase}, model={model_name}")
        model_summaries[model_name] = {
            "result_rows": len(model_results),
            "statuses": statuses,
            "retained_node_cell_area_mm2_recomputed": retained_area,
            "area_resolution_csv": {
                "sampled": source_sampled,
                "retained": source_retained,
                "escaped": source_escaped,
                "step_mm": source_step,
                "retained_cell_area_mm2": source_area,
            },
            "area_matches": area_matches,
        }

    field_hashes = config.get("field_sha256", {})
    for model_name in MODELS:
        expected_path = FIELD_PATHS[model_name]
        actual_hash = sha256(expected_path)
        if field_hashes.get(model_name) != actual_hash:
            errors.append(f"V11 {model_name} field hash does not match its saved h50 source file")

    return {
        "phase_deg": phase,
        "source_directory": directory.relative_to(ROOT).as_posix(),
        "source_files_sha256": {
            "config.json": sha256(config_path),
            "initial_states.csv": sha256(initial_path),
            "results.csv": sha256(results_path),
        },
        "source_grid": config.get("grid"),
        "coordinate_count": len(coordinates),
        "trajectory_count": len(results),
        "per_model_rows": {name: sum(row["model"] == name for row in results) for name in MODELS},
        "zero_velocity": all(
            near(float(row["vx0_m_s"]), 0.0, 1e-15) and near(float(row["vy0_m_s"]), 0.0, 1e-15)
            for row in initial
        ),
        "injection_radius_mm": INJECTION_RADIUS_M * 1e3,
        "grid_step_mm_recomputed": dx_m * 1e3,
        "models": model_summaries,
        "field_sources": {
            name: {
                "path": FIELD_PATHS[name].relative_to(ROOT).as_posix(),
                "sha256": sha256(FIELD_PATHS[name]),
            }
            for name in MODELS
        },
        "checks_passed": not errors,
        "errors": errors,
    }


def audit_sources(phases: list[int]) -> dict[str, Any]:
    area_rows = _phase_area_rows()
    phase_summaries = [audit_phase(phase, area_rows) for phase in phases]
    coordinate_sets = []
    for phase in phases:
        rows = read_csv(MAP_DIRS[phase] / "initial_states.csv")
        coordinate_sets.append({(float(row["x0_m"]), float(row["y0_m"])) for row in rows})
    common_across_phases = not coordinate_sets or all(
        coords == coordinate_sets[0] for coords in coordinate_sets[1:]
    )
    errors = [error for summary in phase_summaries for error in summary["errors"]]
    if not common_across_phases:
        errors.append("selected phases do not share exactly the same XY coordinate set")
    for name, path in FIELD_PATHS.items():
        if not path.is_file():
            errors.append(f"missing h50 N48 field source for {name}: {path}")

    return {
        "study": "V12 source reconciliation for V11 equal-voltage 21x21 XY maps",
        "schema_version": 1,
        "audit_only": True,
        "trajectory_integrations_performed": 0,
        "fem_solves_performed": 0,
        "selected_phases_deg": phases,
        "expected_physical_parameters": {
            "mass_Th": 100_000,
            "mass_kg": MASS_KG,
            "charge_C": E_CHARGE,
            "frequency_Hz": FREQUENCY_HZ,
            "aperture_mm": APERTURE_M * 1e3,
            "equal_peak_voltage_V": 250.0,
            "initial_velocity_m_s": [0.0, 0.0],
            "horizon_RF": HORIZON_RF,
            "mesh_um": 50,
            "harmonic_reconstruction": "N48",
            "injection_domain": "uniform Cartesian 21x21 bounding lattice nodes with r<7.5 mm (305 retained XY coordinates)",
        },
        "area_definition": (
            "retained node count times square grid spacing squared; this is the saved V11 cell-count estimate, "
            "not a continuous boundary-area estimate"
        ),
        "same_xy_coordinate_set_across_selected_phases": common_across_phases,
        "phase_summaries": phase_summaries,
        "status": "source_reconciled" if not errors else "source_audit_failed",
        "errors": errors,
    }


def parse_phases(value: str) -> list[int]:
    try:
        phases = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("phases must be comma-separated values from 0,45,90") from exc
    if not phases or len(set(phases)) != len(phases) or any(p not in MAP_DIRS for p in phases):
        raise argparse.ArgumentTypeError("phases must be unique values from 0,45,90")
    return phases


def coordinate_selection(directory: Path, limit: int | None) -> list[tuple[float, float]]:
    rows = read_csv(directory / "initial_states.csv")
    coords = sorted({(float(row["x0_m"]), float(row["y0_m"])) for row in rows}, key=lambda xy: (xy[1], xy[0]))
    return coords if limit is None else coords[:limit]


def field_for_worker(_: dict[str, str]) -> None:
    global _WORKER_Q, _WORKER_FIELD_SHA
    _WORKER_Q = {}
    _WORKER_FIELD_SHA = {}
    for name in MODELS:
        path = FIELD_PATHS[name]
        q = RFReconstruction.load(path).unit_quadratures
        if name == "B":
            q = HarmonicQuadratures(np.zeros_like(q.cosine), q.sine.copy(), q.reference_radius)
        _WORKER_Q[name] = q
        _WORKER_FIELD_SHA[name] = sha256(path)


def trajectory_fingerprint(phase: int, model_name: str, source: dict[str, str]) -> str:
    inputs = {
        "schema_version": 1,
        "phase_deg": phase,
        "model": model_name,
        "initial_state_si": [
            float(source["x0_m"]), float(source["y0_m"]),
            float(source["vx0_m_s"]), float(source["vy0_m_s"]),
        ],
        "phase_rad": float(source["phase_rad"]),
        "voltage_V": float(source["voltage_V"]),
        "reference_status_V11": source["status"],
        "reference_exit_RF": float(source["escape_RF"]) if source.get("escape_RF") else None,
        "source_config_sha256": sha256(MAP_DIRS[phase] / "config.json"),
        "field_sha256": sha256(FIELD_PATHS[model_name]),
        "mass_kg": MASS_KG,
        "charge_C": E_CHARGE,
        "omega_rad_s": OMEGA,
        "length_m": LENGTH_M,
        "aperture_m": APERTURE_M,
        "horizon_RF": HORIZON_RF,
        "segment_RF": SEGMENT_RF,
        "rtol": RTOL,
        "atol": ATOL.tolist(),
        "max_step_rf": MAX_STEP_RF,
        "samples_per_rf": SAMPLES_PER_RF,
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_previous_replay() -> dict[tuple[int, str, int, int], dict[str, str]]:
    if not REPLAY_CSV.is_file():
        return {}
    previous = read_csv(REPLAY_CSV)
    reusable: dict[tuple[int, str, int, int], dict[str, str]] = {}
    for row in previous:
        if int(row.get("mesh_um", "0")) != 50 or row.get("model") not in MODELS:
            continue
        if row.get("dimensionless_status") not in {"retained", "escaped"}:
            continue
        try:
            phase = int(round(math.degrees(float(row["phase_rad"]))))
            if phase not in MAP_DIRS or row.get("field_sha256") != sha256(FIELD_PATHS[row["model"]]):
                continue
            key = (phase, row["model"], round(float(row["x0_m"]) * 1e6), round(float(row["y0_m"]) * 1e6))
            reusable[key] = row
        except (KeyError, TypeError, ValueError):
            continue
    return reusable


def make_job(phase: int, model_name: str, xy: tuple[float, float], source: dict[str, str], checkpoint: Path) -> dict[str, Any]:
    return {
        "phase_deg": phase,
        "model": model_name,
        "source": source,
        "x0_m": xy[0],
        "y0_m": xy[1],
        "checkpoint": str(checkpoint),
        "fingerprint": trajectory_fingerprint(phase, model_name, source),
    }


def find_source_rows(phase: int) -> dict[tuple[str, int, int], dict[str, str]]:
    rows = read_csv(MAP_DIRS[phase] / "results.csv")
    result = {}
    for row in rows:
        key = (row["model"], round(float(row["x0_m"]) * 1e6), round(float(row["y0_m"]) * 1e6))
        result[key] = row
    return result


def reuse_row(job: dict[str, Any], previous: dict[str, str]) -> dict[str, Any]:
    source = job["source"]
    status = previous["dimensionless_status"]
    source_exit = float(source["escape_RF"]) if source.get("escape_RF") else None
    dim_exit = float(previous["dimensionless_exit_RF"]) if previous.get("dimensionless_exit_RF") else None
    return {
        "run_status": "complete",
        "result_source": "reused_existing_v12_validation",
        "fingerprint": job["fingerprint"],
        "phase_deg": job["phase_deg"],
        "model": job["model"],
        "x0_m": float(source["x0_m"]),
        "y0_m": float(source["y0_m"]),
        "vx0_m_s": float(source["vx0_m_s"]),
        "vy0_m_s": float(source["vy0_m_s"]),
        "phase_rad": float(source["phase_rad"]),
        "voltage_V": float(source["voltage_V"]),
        "field_sha256": sha256(FIELD_PATHS[job["model"]]),
        "reference_status_V11": source["status"],
        "dimensionless_status": status,
        "status_match": source["status"] == status,
        "reference_exit_RF": source_exit,
        "dimensionless_exit_RF": dim_exit,
        "exit_time_difference_RF": None if source_exit is None or dim_exit is None else dim_exit - source_exit,
        "nfev": int(float(previous.get("nfev", 0))),
        "final_state_dimensionless": None,
        "checkpoint_period_RF": HORIZON_RF if status == "retained" else dim_exit,
        "message": "Reused an exactly matching completed h50 row in dimensionless_replay_of_v11_xy.csv.",
    }


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = job["fingerprint"]
    prior: dict[str, Any] | None = None
    if checkpoint.is_file():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Checkpoint fingerprint mismatch: {checkpoint}")
        if prior.get("run_status") == "complete":
            return prior

    source = job["source"]
    model_name = job["model"]
    q = _WORKER_Q[model_name]
    model = DimensionlessRFModel.from_physical(
        q.cosine,
        q.sine,
        charge_c=E_CHARGE,
        mass_kg=MASS_KG,
        voltage_v=float(source["voltage_V"]),
        omega_rad_s=OMEGA,
        length_m=LENGTH_M,
        reference_radius_m=q.reference_radius,
        aperture_m=APERTURE_M,
        phase_rad=float(source["phase_rad"]),
        name=f"{model_name}_h50_v11_phase{job['phase_deg']}",
    )
    state_si = np.array([
        float(source["x0_m"]), float(source["y0_m"]),
        float(source["vx0_m_s"]), float(source["vy0_m_s"]),
    ])
    initial_d = np.r_[state_si[:2] / LENGTH_M, state_si[2:] / (LENGTH_M * OMEGA)]
    resume_period = float(prior.get("checkpoint_period_RF", 0.0)) if prior else 0.0
    resume_state = np.asarray(prior["state_dimensionless"], dtype=float) if prior and prior.get("state_dimensionless") else None
    previous_max = float(prior.get("max_radius_sampled_rho", 0.0)) if prior else 0.0
    previous_nfev = int(prior.get("nfev", 0)) if prior else 0
    latest_checkpoint: dict[str, Any] = {
        "period": resume_period,
        "state": initial_d.copy() if resume_state is None else resume_state.copy(),
        "max_radius": max(previous_max, float(np.linalg.norm(initial_d[:2]))),
        "nfev": previous_nfev,
    }

    def save_segment(period: float, state: np.ndarray, max_radius: float, nfev: int) -> None:
        latest_checkpoint["period"] = period
        latest_checkpoint["state"] = state.copy()
        latest_checkpoint["max_radius"] = max(float(latest_checkpoint["max_radius"]), max_radius)
        latest_checkpoint["nfev"] = previous_nfev + nfev
        atomic_json(checkpoint, {
            "run_status": "partial",
            "fingerprint": fingerprint,
            "phase_deg": job["phase_deg"],
            "model": model_name,
            "x0_m": float(source["x0_m"]),
            "y0_m": float(source["y0_m"]),
            "vx0_m_s": float(source["vx0_m_s"]),
            "vy0_m_s": float(source["vy0_m_s"]),
            "phase_rad": float(source["phase_rad"]),
            "voltage_V": float(source["voltage_V"]),
            "field_sha256": _WORKER_FIELD_SHA[model_name],
            "checkpoint_period_RF": latest_checkpoint["period"],
            "state_dimensionless": state.tolist(),
            "max_radius_sampled_rho": latest_checkpoint["max_radius"],
            "nfev": latest_checkpoint["nfev"],
        })

    result = integrate_to_horizon(
        model,
        initial_d,
        HORIZON_RF,
        segment_periods=SEGMENT_RF,
        rtol=RTOL,
        atol=ATOL,
        max_step_rf=MAX_STEP_RF,
        samples_per_rf=SAMPLES_PER_RF,
        checkpoint_callback=save_segment,
        resume_state=resume_state,
        resume_period=resume_period,
    )
    v12_status = result.status
    source_status = source["status"]
    total_nfev = previous_nfev + int(result.nfev)
    max_radius = max(previous_max, float(result.max_radius_sampled_rho))
    complete = v12_status in {"retained", "escaped"}
    record = {
        "run_status": "complete" if complete else "numerical_error",
        "result_source": "new_v12_dimensionless_integration",
        "fingerprint": fingerprint,
        "phase_deg": job["phase_deg"],
        "model": model_name,
        "x0_m": float(source["x0_m"]),
        "y0_m": float(source["y0_m"]),
        "vx0_m_s": float(source["vx0_m_s"]),
        "vy0_m_s": float(source["vy0_m_s"]),
        "phase_rad": float(source["phase_rad"]),
        "voltage_V": float(source["voltage_V"]),
        "field_sha256": _WORKER_FIELD_SHA[model_name],
        "reference_status_V11": source_status,
        "dimensionless_status": v12_status,
        "status_match": source_status == v12_status,
        "reference_exit_RF": float(source["escape_RF"]) if source.get("escape_RF") else None,
        "dimensionless_exit_RF": result.escape_period,
        "exit_time_difference_RF": (
            None if not source.get("escape_RF") or result.escape_period is None
            else float(result.escape_period) - float(source["escape_RF"])
        ),
        # If an integration fails mid-segment, retain the last successful
        # segment checkpoint. A later invocation can retry that segment
        # without trusting an incomplete solver state.
        "checkpoint_period_RF": result.reached_period if complete else latest_checkpoint["period"],
        "state_dimensionless": result.final_state.tolist() if complete else latest_checkpoint["state"].tolist(),
        "final_state_dimensionless": result.final_state.tolist() if complete else None,
        "max_radius_sampled_rho": max_radius if complete else latest_checkpoint["max_radius"],
        "nfev": total_nfev if complete else latest_checkpoint["nfev"],
        "message": result.message,
    }
    atomic_json(checkpoint, record)
    return record


def load_checkpoint_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if path.is_file():
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return rows


def write_replay_aggregates(records: list[dict[str, Any]], phases: list[int], out_dir: Path) -> None:
    csv_path = out_dir / "v11_map_replay_v12_results.csv"
    json_path = out_dir / "v11_map_replay_v12_results.json"
    columns = [
        "phase_deg", "model", "x0_m", "y0_m", "vx0_m_s", "vy0_m_s", "phase_rad", "voltage_V",
        "reference_status_V11", "dimensionless_status", "status_match", "reference_exit_RF",
        "dimensionless_exit_RF", "exit_time_difference_RF", "run_status", "result_source",
        "checkpoint_period_RF", "max_radius_sampled_rho", "nfev", "field_sha256", "message",
    ]
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(records, key=lambda r: (r.get("phase_deg", -1), r.get("model", ""), r.get("y0_m", 0), r.get("x0_m", 0))):
            writer.writerow(row)
    os.replace(tmp, csv_path)
    summary = {
        "study": "V12 dimensionless replay of V11 21x21 equal-voltage XY maps",
        "selected_phases_deg": phases,
        "rows": len(records),
        "complete": sum(row.get("run_status") == "complete" for row in records),
        "partial_or_error": sum(row.get("run_status") != "complete" for row in records),
        "status_matches": sum(row.get("run_status") == "complete" and row.get("status_match") is True for row in records),
        "status_mismatches": sum(row.get("run_status") == "complete" and row.get("status_match") is False for row in records),
        "escaped_exit_comparisons": sum(row.get("reference_exit_RF") is not None and row.get("dimensionless_exit_RF") is not None for row in records),
        "max_abs_exit_time_difference_RF": max(
            (abs(float(row["exit_time_difference_RF"])) for row in records if row.get("exit_time_difference_RF") is not None),
            default=None,
        ),
        "csv": csv_path.name,
        "results": records,
    }
    atomic_json(json_path, summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", type=parse_phases, default=parse_phases("0,45,90"),
                        help="comma-separated V11 phases; default: 0,45,90")
    parser.add_argument("--limit", type=int, default=None,
                        help="maximum XY coordinates per phase, preserving A/B/C rows together")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel worker processes when --run is supplied")
    parser.add_argument("--run", action="store_true",
                        help="integrate missing V12 trajectories; without this flag only audit V11 source data")
    parser.add_argument("--output-root", type=Path, default=OUT,
                        help="V12 output root; default is dimensionless_acceptance_v12/validation")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    audit = audit_sources(args.phases)
    audit["requested_coordinate_limit_per_phase"] = args.limit
    audit["run_requested"] = bool(args.run)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / MANIFEST_PATH.name, audit)
    if audit["status"] != "source_reconciled":
        print(json.dumps({"status": audit["status"], "errors": audit["errors"]}, indent=2))
        return 2

    print(json.dumps({
        "status": audit["status"],
        "phases": [
            {
                "phase_deg": row["phase_deg"],
                "coordinates": row["coordinate_count"],
                "trajectories": row["trajectory_count"],
                "areas_mm2": {name: row["models"][name]["retained_node_cell_area_mm2_recomputed"] for name in MODELS},
            }
            for row in audit["phase_summaries"]
        ],
        "integrations_run": 0,
        "manifest": str(output_root / MANIFEST_PATH.name),
    }, indent=2))
    if not args.run:
        return 0

    # Hold an exclusive run lock so two invocations cannot write the same
    # trajectory checkpoints concurrently. A terminated process leaves the
    # lock as a clear signal to inspect before resuming.
    run_id = hashlib.sha256(b"v11-map-replay-v12-h50-n48-1000rf-v1").hexdigest()[:16]
    run_dir = output_root / "v11_map_replay_v12" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f"run lock already exists at {lock_path}; check whether another replay is active")
    os.write(lock_fd, f"pid={os.getpid()}\n".encode())
    os.close(lock_fd)

    try:
        existing = load_previous_replay()
        jobs: list[dict[str, Any]] = []
        expected_paths: list[Path] = []
        for phase in args.phases:
            source_rows = find_source_rows(phase)
            coords = coordinate_selection(MAP_DIRS[phase], args.limit)
            for x, y in coords:
                for model_name in MODELS:
                    source = source_rows[(model_name, round(x * 1e6), round(y * 1e6))]
                    checkpoint = run_dir / f"phase{phase:02d}" / model_name / f"xy_{round(y*1e6):+06d}_{round(x*1e6):+06d}.json"
                    job = make_job(phase, model_name, (x, y), source, checkpoint)
                    expected_paths.append(checkpoint)
                    if checkpoint.is_file():
                        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                        if saved.get("fingerprint") != job["fingerprint"]:
                            raise RuntimeError(f"Saved checkpoint does not match current source: {checkpoint}")
                        if saved.get("run_status") == "complete":
                            continue
                    previous_key = (phase, model_name, round(x * 1e6), round(y * 1e6))
                    old = existing.get(previous_key)
                    if old is not None:
                        old_exit = float(old["reference_exit_RF"]) if old.get("reference_exit_RF") else None
                        source_exit = float(source["escape_RF"]) if source.get("escape_RF") else None
                        old_matches = (
                            near(float(old["x0_m"]), float(source["x0_m"]), 1e-15)
                            and near(float(old["y0_m"]), float(source["y0_m"]), 1e-15)
                            and near(float(old["vx0_m_s"]), float(source["vx0_m_s"]), 1e-15)
                            and near(float(old["vy0_m_s"]), float(source["vy0_m_s"]), 1e-15)
                            and near(float(old["phase_rad"]), float(source["phase_rad"]), 1e-14)
                            and near(float(old["voltage_V"]), float(source["voltage_V"]), 1e-12)
                            and old.get("reference_status_V11") == source["status"]
                            and ((old_exit is None and source_exit is None)
                                 or (old_exit is not None and source_exit is not None and near(old_exit, source_exit, 1e-12)))
                        )
                        if not old_matches:
                            old = None
                    if old is not None and not checkpoint.exists():
                        atomic_json(checkpoint, reuse_row(job, old))
                        continue
                    jobs.append(job)

        if jobs:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                     initializer=field_for_worker, initargs=({},)) as pool:
                futures = {pool.submit(run_one, job): job for job in jobs}
                for completed, future in enumerate(as_completed(futures), 1):
                    job = futures[future]
                    try:
                        result = future.result()
                    except BaseException as exc:
                        print(f"FAILED {job['phase_deg']} {job['model']} ({job['x0_m']},{job['y0_m']}): {exc}", flush=True)
                        continue
                    if completed % 25 == 0 or completed == len(futures):
                        print(f"replay {completed}/{len(futures)}; last={result['phase_deg']} {result['model']} {result['dimensionless_status']}", flush=True)

        all_checkpoint_paths = sorted(run_dir.glob("phase*/[ABC]/xy_*.json"))
        all_records = load_checkpoint_rows(all_checkpoint_paths)
        completed_phases = sorted({int(row["phase_deg"]) for row in all_records if "phase_deg" in row})
        write_replay_aggregates(all_records, completed_phases or args.phases, output_root)
        print(json.dumps({"records": len(all_records), "checkpoints": str(run_dir),
                          "results": str(output_root / "v11_map_replay_v12_results.json")}, indent=2))
    finally:
        lock_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
