"""Postprocess conditional retention in the V12 baseline trajectory table.

This script does not integrate trajectories.  It consumes a completed
``trajectory_results.csv`` and, when needed, joins launch coordinates and
velocities from the adjacent ``initial_states_r*.csv`` files.  Conditional
fractions are stratified by independent Sobol replicate; they are not
contributions to the four-dimensional acceptance A4.

Example (run directory or direct CSV path)::

    python scripts/analyze_v12_baseline_directions.py --run path/to/baseline_run

Without ``--run`` the script only displays help and does not access study data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
from scipy.stats import t as student_t


DEFAULT_ENERGY_EDGES_EV = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
ELEMENTARY_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_KG = 1.66053906660e-27
BASELINE_MASS_KG = 100_000 * ATOMIC_MASS_KG
BASE_MODELS = ("A", "B", "C")
SIGN_CATEGORIES = ("negative", "zero", "positive")
VELOCITY_ZERO_TOLERANCE_M_S = 1e-12


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_float(row: dict[str, str], names: Iterable[str]) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(result):
                return result
    return None


def _kind_scale_mesh(row: dict[str, str]) -> tuple[str | None, float | None, int | None]:
    model = str(row.get("model", "")).strip()
    kind = str(row.get("kind", "")).strip()
    if kind not in BASE_MODELS:
        match = re.match(r"^(A|B|C)(?:$|[_-])", model)
        kind = match.group(1) if match else (model if model in BASE_MODELS else "")
    scale = _first_float(row, ("scale_factor", "scale"))
    if scale is None:
        match = re.search(r"(?:^|[_-])scale([0-9.eE+-]+)(?:$|[_-])?", model)
        if match:
            try:
                scale = float(match.group(1))
            except ValueError:
                scale = None
    mesh = _first_float(row, ("mesh_um", "mesh_h_um"))
    if mesh is None:
        match = re.search(r"(?:^|[_-])h(\d+)(?:$|[_-])", model)
        if match:
            mesh = float(match.group(1))
    return (kind if kind in BASE_MODELS else None), scale, (None if mesh is None else int(mesh))


def _replicate(row: dict[str, str]) -> int | None:
    raw = row.get("replicate")
    if raw not in (None, ""):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            pass
    sample_id = str(row.get("sample_id", ""))
    match = re.match(r"^r(\d+)_", sample_id)
    return int(match.group(1)) if match else None


def _load_config(run_dir: Path) -> dict:
    for path in (run_dir / "config.json", run_dir.parent / "config.json"):
        if path.is_file():
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
    return {}


def _config_path(run_dir: Path) -> Path | None:
    for path in (run_dir / "config.json", run_dir.parent / "config.json"):
        if path.is_file():
            return path
    return None


def _mass_from_config(config: dict) -> float:
    try:
        value = float(config["physical"]["mass_kg"])
        if value > 0 and math.isfinite(value):
            return value
    except (KeyError, TypeError, ValueError):
        pass
    return BASELINE_MASS_KG


def _initial_state_lookup(run_dir: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    candidates = set(run_dir.glob("initial_states_r*.csv"))
    candidates.update(run_dir.parent.glob("initial_states_r*.csv"))
    for path in sorted(candidates):
        for row in _read_csv(path):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                replicate = _replicate(row)
                sample_index = _first_float(row, ("sample_index", "index"))
                if replicate is not None and sample_index is not None:
                    sample_id = f"r{replicate:02d}_{int(sample_index):06d}"
            if sample_id:
                if sample_id in lookup:
                    raise ValueError(f"duplicate sample_id in launch tables: {sample_id}")
                lookup[sample_id] = row
    return lookup


def _get_initial_value(result: dict[str, str], launch: dict[str, str], names: Iterable[str]) -> float | None:
    value = _first_float(result, names)
    return value if value is not None else _first_float(launch, names)


def _sign(value: float, tolerance: float = 0.0) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def _energy_category(energy_ev: float, edges: tuple[float, ...]) -> str:
    if energy_ev < edges[0] or energy_ev > edges[-1]:
        return "outside_bins"
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if lo <= energy_ev < hi or (index == len(edges) - 2 and energy_ev == hi):
            right_bracket = "]" if index == len(edges) - 2 else ")"
            return f"[{lo:g},{hi:g}{right_bracket} eV"
    return "outside_bins"


def _parse_energy_edges(value: str) -> tuple[float, ...]:
    try:
        edges = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("energy edges must be comma-separated numbers") from exc
    if len(edges) < 2 or any(not math.isfinite(edge) for edge in edges) or any(
        right <= left for left, right in zip(edges[:-1], edges[1:])
    ):
        raise argparse.ArgumentTypeError("energy edges must be finite and strictly increasing")
    return edges


def _resolve_input(run_arg: Path) -> tuple[Path, Path]:
    supplied = run_arg.expanduser().resolve()
    if supplied.is_dir():
        run_dir = supplied
        results_path = run_dir / "trajectory_results.csv"
    else:
        results_path = supplied
        run_dir = supplied.parent
    if not results_path.is_file():
        raise FileNotFoundError(
            f"trajectory_results.csv not found: {results_path}. Pass its path or the containing run directory."
        )
    return results_path, run_dir


def _prepare_rows(results_path: Path, run_dir: Path, mass_kg: float,
                  energy_edges: tuple[float, ...], mesh_um: int) -> tuple[list[dict], dict]:
    raw_rows = _read_csv(results_path)
    if not raw_rows:
        raise ValueError(f"trajectory table is empty: {results_path}")
    launch_lookup = _initial_state_lookup(run_dir)
    config = _load_config(run_dir)

    if config.get("protocol") not in (None, "equal"):
        raise ValueError("This postprocessor is for the equal-peak baseline; config protocol is not 'equal'.")
    if config.get("phase_mode") not in (None, "fixed"):
        raise ValueError("This postprocessor expects the fixed-phase baseline, not a phase-averaged run.")
    phase_deg = config.get("phase_deg")
    if phase_deg is not None and not math.isclose(float(phase_deg), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("This postprocessor expects the baseline RF phase 0 degrees.")

    config_scales = config.get("scales")
    prepared: list[dict] = []
    excluded = {"non_ABC_model": 0, "nonbaseline_scale": 0, "nonbaseline_mesh": 0, "missing_launch_state": 0,
                "invalid_replicate": 0, "invalid_status": 0}
    for result in raw_rows:
        kind, scale, row_mesh = _kind_scale_mesh(result)
        if kind not in BASE_MODELS:
            excluded["non_ABC_model"] += 1
            continue
        if scale is None and isinstance(config_scales, list) and len(config_scales) > 1:
            raise ValueError(
                "Run config has multiple scales but the trajectory table has no scale field or suffixed model name; "
                "cannot isolate the scale=1 equal-peak baseline safely."
            )
        if scale is not None and not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
            excluded["nonbaseline_scale"] += 1
            continue
        configured_meshes = config.get("meshes_um")
        if row_mesh is None and isinstance(configured_meshes, list) and len(configured_meshes) > 1:
            raise ValueError(
                "Run config has multiple meshes but the trajectory table has no mesh field or h35/h50 model suffix; "
                "cannot isolate the requested FEM mesh safely."
            )
        if row_mesh is not None and row_mesh != mesh_um:
            excluded["nonbaseline_mesh"] += 1
            continue

        sample_id = str(result.get("sample_id", ""))
        launch = launch_lookup.get(sample_id, {})
        replicate = _replicate(result)
        if replicate is None:
            replicate = _replicate(launch)
        if replicate is None:
            excluded["invalid_replicate"] += 1
            continue

        x = _get_initial_value(result, launch, ("x0_m", "x_m", "x0"))
        y = _get_initial_value(result, launch, ("y0_m", "y_m", "y0"))
        vx = _get_initial_value(result, launch, ("vx0_m_s", "vx_m_s", "vx0"))
        vy = _get_initial_value(result, launch, ("vy0_m_s", "vy_m_s", "vy0"))
        if None in (x, y, vx, vy):
            excluded["missing_launch_state"] += 1
            continue
        radius = math.hypot(x, y)
        lz = mass_kg * (x * vy - y * vx)
        if radius > 0.0:
            radial_velocity = (x * vx + y * vy) / radius
            tangential_velocity = (x * vy - y * vx) / radius
        else:
            radial_velocity = None
            tangential_velocity = None
        energy_ev = _get_initial_value(result, launch, ("E0_eV", "energy0_eV", "initial_energy_eV"))
        if energy_ev is None:
            energy_ev = 0.5 * mass_kg * (vx * vx + vy * vy) / ELEMENTARY_CHARGE_C
        status = str(result.get("status", "")).strip().lower()
        if status not in ("retained", "escaped"):
            # Numerical errors, incomplete records, and unknown labels remain unresolved.
            status = "unresolved"
            excluded["invalid_status"] += 1

        prepared.append({
            "sample_id": sample_id,
            "replicate": replicate,
            "model": kind,
            "scale_factor": 1.0 if scale is None else scale,
            "mesh_um": mesh_um if row_mesh is None else row_mesh,
            "status": status,
            "x0_m": x, "y0_m": y, "vx0_m_s": vx, "vy0_m_s": vy,
            "radius0_m": radius,
            "Lz0_kg_m2_s": lz,
            "v_radial0_m_s": radial_velocity,
            "v_tangential0_m_s": tangential_velocity,
            "E0_eV": energy_ev,
            # Since mass and r are positive, classify the sign through the
            # algebraically equivalent v_theta to use one zero tolerance.
            "Lz0_sign": "zero" if tangential_velocity is None else
                _sign(tangential_velocity, VELOCITY_ZERO_TOLERANCE_M_S),
            "radial_velocity_sign": "undefined_at_center" if radial_velocity is None else
                _sign(radial_velocity, VELOCITY_ZERO_TOLERANCE_M_S),
            "tangential_velocity_sign": "undefined_at_center" if tangential_velocity is None else
                _sign(tangential_velocity, VELOCITY_ZERO_TOLERANCE_M_S),
            "initial_energy_bin": _energy_category(energy_ev, energy_edges),
        })

    if not prepared:
        raise ValueError("No A/B/C scale=1 records with complete initial states were found.")
    keys = [(row["model"], row["sample_id"]) for row in prepared]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate model/sample_id records remain after scale=1 selection.")
    info = {
        "input_rows": len(raw_rows),
        "analyzed_rows": len(prepared),
        "excluded_counts": excluded,
        "replicates": sorted({row["replicate"] for row in prepared}),
        "models": sorted({row["model"] for row in prepared}),
        "mass_kg": mass_kg,
        "config_path": str(_config_path(run_dir)) if _config_path(run_dir) else None,
        "phase_mode": config.get("phase_mode"),
        "phase_deg": config.get("phase_deg"),
        "horizon_RF": config.get("horizon_RF"),
        "protocol": config.get("protocol", "equal (baseline assumption; no config.json)"),
        "voltage_V": config.get("voltage_V"),
        "meshes_um": config.get("meshes_um"),
        "selected_mesh_um": mesh_um,
        "field_sha256": config.get("field_sha256"),
        "energy_edges_eV": list(energy_edges),
    }
    return prepared, info


def _condition_specs(edges: tuple[float, ...]) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []
    for condition, categories in (
        ("Lz0_sign", SIGN_CATEGORIES),
        ("radial_velocity_sign", (*SIGN_CATEGORIES, "undefined_at_center")),
        ("tangential_velocity_sign", (*SIGN_CATEGORIES, "undefined_at_center")),
    ):
        specs.extend((condition, category, "") for category in categories)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        right = "]" if index == len(edges) - 2 else ")"
        specs.append(("initial_energy_bin", f"[{lo:g},{hi:g}{right} eV", ""))
    specs.append(("initial_energy_bin", "outside_bins", ""))
    return specs


def _summarize(prepared: list[dict], replicates: list[int], energy_edges: tuple[float, ...]
               ) -> tuple[list[dict], list[dict]]:
    replicate_rows: list[dict] = []
    summary_rows: list[dict] = []
    models = sorted({row["model"] for row in prepared})
    specs = _condition_specs(energy_edges)

    for model in models:
        model_rows = [row for row in prepared if row["model"] == model]
        for condition, category, _ in specs:
            fractions_by_rep: list[float | None] = []
            total_by_rep: list[int] = []
            retained_by_rep: list[int] = []
            escaped_by_rep: list[int] = []
            unresolved_by_rep: list[int] = []
            for replicate in replicates:
                selected = [row for row in model_rows
                            if row["replicate"] == replicate and row[condition] == category]
                n_total = len(selected)
                n_retained = sum(row["status"] == "retained" for row in selected)
                n_escaped = sum(row["status"] == "escaped" for row in selected)
                n_unresolved = sum(row["status"] == "unresolved" for row in selected)
                complete_fraction = (n_retained / n_total) if n_total > 0 and n_unresolved == 0 else None
                lower = n_retained / n_total if n_total > 0 else None
                upper = (n_retained + n_unresolved) / n_total if n_total > 0 else None
                total_by_rep.append(n_total)
                retained_by_rep.append(n_retained)
                escaped_by_rep.append(n_escaped)
                unresolved_by_rep.append(n_unresolved)
                fractions_by_rep.append(complete_fraction)
                replicate_rows.append({
                    "model": model, "replicate": replicate,
                    "condition": condition, "category": category,
                    "n_initial": n_total, "n_retained": n_retained,
                    "n_escaped": n_escaped, "n_unresolved": n_unresolved,
                    "conditional_retention_fraction": complete_fraction,
                    "unresolved_lower_bound": lower,
                    "unresolved_upper_bound": upper,
                    "interpretation": "conditional fraction within this subgroup; not a contribution to A4",
                })

            fully_observed = all(value is not None for value in fractions_by_rep)
            valid_fractions = [float(value) for value in fractions_by_rep if value is not None]
            if fully_observed and len(valid_fractions) == len(replicates):
                mean = float(np.mean(valid_fractions))
                se = float(np.std(valid_fractions, ddof=1) / math.sqrt(len(valid_fractions))) if len(valid_fractions) > 1 else None
                if se is None:
                    ci_low = ci_high = None
                else:
                    half_width = float(student_t.ppf(0.975, df=len(valid_fractions)-1) * se)
                    ci_low = max(0.0, mean - half_width)
                    ci_high = min(1.0, mean + half_width)
            else:
                mean = se = ci_low = ci_high = None

            summary_rows.append({
                "model": model, "condition": condition, "category": category,
                "conditional_retention_fraction_mean": mean,
                "replica_SE": se, "student_t_95pct_CI_low": ci_low,
                "student_t_95pct_CI_high": ci_high,
                "replicates_expected": len(replicates),
                "replicates_fully_observed": sum(value is not None for value in fractions_by_rep),
                "n_initial_by_replicate_json": json.dumps(dict(zip(replicates, total_by_rep)), sort_keys=True),
                "n_retained_by_replicate_json": json.dumps(dict(zip(replicates, retained_by_rep)), sort_keys=True),
                "n_escaped_by_replicate_json": json.dumps(dict(zip(replicates, escaped_by_rep)), sort_keys=True),
                "n_unresolved_by_replicate_json": json.dumps(dict(zip(replicates, unresolved_by_rep)), sort_keys=True),
                "replicate_fractions_json": json.dumps(dict(zip(replicates, fractions_by_rep)), sort_keys=True),
                "interpretation": "conditional fractions; not a contribution to full four-dimensional A4",
            })
    return replicate_rows, summary_rows


def _contrast_subgroups(replicate_rows: list[dict], replicates: list[int]) -> list[dict]:
    """Paired scramble-level contrasts between predeclared directional/energy strata."""
    pairs = (
        ("Lz0_sign", "negative", "positive", "positive_minus_negative"),
        ("radial_velocity_sign", "negative", "positive", "outward_minus_inward"),
        ("initial_energy_bin", "[0,0.5) eV", "[3,5] eV", "high_minus_low_energy"),
    )
    models = sorted({row["model"] for row in replicate_rows})
    by_key = {(row["model"], row["replicate"], row["condition"], row["category"]): row
              for row in replicate_rows}
    output = []
    for model in models:
        for condition, lower_category, upper_category, contrast_name in pairs:
            differences = []
            n_lower, n_upper = [], []
            for replicate in replicates:
                lower = by_key[(model, replicate, condition, lower_category)]
                upper = by_key[(model, replicate, condition, upper_category)]
                if (lower["n_initial"] < 1 or upper["n_initial"] < 1
                        or lower["n_unresolved"] or upper["n_unresolved"]):
                    differences = []
                    break
                lower_fraction = lower["n_retained"] / lower["n_initial"]
                upper_fraction = upper["n_retained"] / upper["n_initial"]
                differences.append(upper_fraction - lower_fraction)
                n_lower.append(lower["n_initial"])
                n_upper.append(upper["n_initial"])
            if len(differences) == len(replicates) and len(differences) > 1:
                mean = float(np.mean(differences))
                se = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
                half_width = float(student_t.ppf(0.975, df=len(differences)-1) * se)
                ci_low, ci_high = max(-1.0, mean-half_width), min(1.0, mean+half_width)
            else:
                mean = se = ci_low = ci_high = None
            output.append({
                "model": model,
                "contrast": contrast_name,
                "condition": condition,
                "lower_category": lower_category,
                "upper_category": upper_category,
                "upper_minus_lower_retention_fraction": mean,
                "paired_replica_SE": se,
                "paired_student_t_95pct_CI_low": ci_low,
                "paired_student_t_95pct_CI_high": ci_high,
                "replicates_expected": len(replicates),
                "replicates_fully_observed": len(differences),
                "lower_n_by_replicate_json": json.dumps(dict(zip(replicates, n_lower)), sort_keys=True),
                "upper_n_by_replicate_json": json.dumps(dict(zip(replicates, n_upper)), sort_keys=True),
                "replicate_differences_json": json.dumps(differences),
                "interpretation": "paired contrast of conditional subgroup fractions across independent Sobol scrambles; not an A4 contribution",
            })
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary_rows: list[dict], output_dir: Path, metadata: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    panels = [
        ("Lz0_sign", "Начальный угловой момент"),
        ("radial_velocity_sign", "Радиальная скорость"),
        ("tangential_velocity_sign", "Тангенциальная скорость"),
        ("initial_energy_bin", "Начальная кинетическая энергия"),
    ]
    models = [model for model in BASE_MODELS if any(row["model"] == model for row in summary_rows)]
    colors = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=False)
    axes_flat = axes.ravel()
    for axis, (condition, title) in zip(axes_flat, panels):
        rows = [row for row in summary_rows if row["condition"] == condition]
        category_order = []
        if condition.endswith("sign"):
            category_order = ["negative", "zero", "positive"]
            if condition != "Lz0_sign":
                category_order.append("undefined_at_center")
        else:
            category_order = list(dict.fromkeys(row["category"] for row in rows))
        category_order = [category for category in category_order if any(
            candidate["category"] == category and any(
                int(count) > 0 for count in json.loads(candidate["n_initial_by_replicate_json"]).values()
            ) for candidate in rows
        )]
        if not category_order:
            axis.text(0.5, 0.5, "Нет точек в этих срезах", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(title)
            axis.set_ylim(-0.03, 1.03)
            axis.set_ylabel("Условная доля удержания до N RF")
            axis.grid(axis="y", alpha=0.25)
            continue
        x = np.arange(len(category_order), dtype=float)
        width = 0.22 if len(models) == 3 else 0.28
        for model_index, model in enumerate(models):
            offset = (model_index - (len(models)-1)/2) * width
            means, lower, upper = [], [], []
            for category in category_order:
                row = next((candidate for candidate in rows
                            if candidate["model"] == model and candidate["category"] == category), None)
                value = None if row is None else row["conditional_retention_fraction_mean"]
                lo = None if row is None else row["student_t_95pct_CI_low"]
                hi = None if row is None else row["student_t_95pct_CI_high"]
                means.append(np.nan if value is None else float(value))
                lower.append(0.0 if value is None or lo is None else float(value)-float(lo))
                upper.append(0.0 if value is None or hi is None else float(hi)-float(value))
            mean_values = np.asarray(means, dtype=float)
            err = np.vstack((np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)))
            axis.errorbar(x + offset, mean_values, yerr=err, marker="o", markersize=5,
                          linewidth=1.2, capsize=3, color=colors[model], label=model)
        labels = [
            {"negative": "−", "zero": "0", "positive": "+", "undefined_at_center": "центр"}.get(value, value)
            for value in category_order
        ]
        axis.set_xticks(x, labels)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("Условная доля удержания до N RF")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if condition == "initial_energy_bin":
            axis.tick_params(axis="x", labelrotation=25)
    title = "Условное удержание в срезах начального фазового пространства"
    subtitle_bits = []
    if metadata.get("horizon_RF") is not None:
        subtitle_bits.append(f"N={metadata['horizon_RF']} RF")
    if metadata.get("phase_deg") is not None:
        subtitle_bits.append(f"φ₀={metadata['phase_deg']}°")
    if subtitle_bits:
        fig.suptitle(title + " — " + ", ".join(subtitle_bits), y=0.99, fontsize=13)
    else:
        fig.suptitle(title, y=0.99, fontsize=13)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(models), frameon=False)
    fig.text(0.5, 0.015,
             "Среднее по полностью наблюдённым Sobol-репликам; интервалы — 95% t-интервалы между репликами. "
             "Условные доли не являются вкладом в полный A₄.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.91))
    output_names = []
    for extension in ("pdf", "svg", "png"):
        path = output_dir / f"conditional_retention_directions_v12.{extension}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        output_names.append(path.name)
    plt.close(fig)
    return output_names


def analyze(run_arg: Path, output_arg: Path | None,
            energy_edges: tuple[float, ...], mesh_um: int) -> dict:
    results_path, run_dir = _resolve_input(run_arg)
    config = _load_config(run_dir)
    mass = _mass_from_config(config)
    prepared, metadata = _prepare_rows(results_path, run_dir, mass, energy_edges, mesh_um)
    replicates = metadata["replicates"]
    replicate_rows, summary_rows = _summarize(prepared, replicates, energy_edges)
    contrast_rows = _contrast_subgroups(replicate_rows, replicates)

    # For any noncentral state, Lz=M*r*v_tan, so their sign strata must match.
    sign_mismatches = sum(
        row["radius0_m"] > 0.0 and row["Lz0_sign"] != row["tangential_velocity_sign"]
        for row in prepared
    )
    if sign_mismatches:
        raise ArithmeticError("Lz0 and tangential-velocity sign strata disagree for noncentral launches")

    output_dir = output_arg.expanduser().resolve() if output_arg else run_dir / "direction_analysis_v12"
    output_dir.mkdir(parents=True, exist_ok=True)
    by_rep_path = output_dir / "conditional_retention_by_replicate_v12.csv"
    summary_path = output_dir / "conditional_retention_summary_v12.csv"
    contrast_path = output_dir / "conditional_retention_contrasts_v12.csv"
    _write_csv(by_rep_path, replicate_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(contrast_path, contrast_rows)
    figure_names = _plot(summary_rows, output_dir, metadata)

    run_info = {
        "study": "V12 conditional retention by initial direction and energy",
        "interpretation": "Every reported fraction is conditional within a subgroup; none is an A4 contribution.",
        "input_trajectory_results": str(results_path),
        "input_sha256": _sha256(results_path),
        "output_directory": str(output_dir),
        "mass_source": "adjacent config.json physical.mass_kg" if metadata.get("config_path") else "V12 baseline default: 100000 Th",
        "status_policy": "Only retained and escaped are resolved. Numerical errors, incomplete, and unknown statuses stay unresolved and are never labeled retained/escaped.",
        "uncertainty_method": "Mean and Student-t 95% interval across independent replica-level conditional fractions; a summary interval is withheld unless every replicate has a nonempty fully resolved subgroup.",
        "sign_zero_tolerance": {
            "radial_and_tangential_velocity_m_s": VELOCITY_ZERO_TOLERANCE_M_S,
            "Lz_kg_m2_s": "mass * radius * velocity_tolerance, ensuring sign(Lz)=sign(v_tangential) away from the center",
        },
        "energy_bins_eV": list(energy_edges),
        "energy_bin_convention": "left-closed/right-open, except the final bin includes its upper edge",
        "direction_definitions": {
            "Lz0": "M*(x0*vy0-y0*vx0) in kg m^2/s",
            "radial_velocity": "(x0*vx0+y0*vy0)/r0; undefined at exactly r0=0",
            "tangential_velocity": "(x0*vy0-y0*vx0)/r0; undefined at exactly r0=0",
            "identity": "For r0>0 and M>0, sign(Lz0)=sign(v_tangential); these two requested views are algebraically redundant.",
        },
        "counts": metadata,
        "n_Lz_vs_tangential_sign_mismatches_noncentral": sign_mismatches,
        "files": {
            "replicate_csv": by_rep_path.name,
            "summary_csv": summary_path.name,
            "contrast_csv": contrast_path.name,
            "figures": figure_names,
        },
        "figures": "One figure saved as PDF, SVG, and 600 dpi PNG.",
    }
    json_path = output_dir / "conditional_retention_directions_v12.json"
    json_path.write_text(json.dumps(run_info, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"json": str(json_path), "replicate_csv": str(by_rep_path),
            "summary_csv": str(summary_path), "contrast_csv": str(contrast_path),
            "figures": [str(output_dir / name) for name in figure_names],
            "analyzed_rows": metadata["analyzed_rows"], "models": metadata["models"],
            "replicates": metadata["replicates"], "output_directory": str(output_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        help="run directory containing trajectory_results.csv, or the CSV file itself; presence enables data access")
    parser.add_argument("--output", type=Path,
                        help="optional output directory (default: <run>/direction_analysis_v12)")
    parser.add_argument("--energy-edges-ev", type=_parse_energy_edges,
                        default=DEFAULT_ENERGY_EDGES_EV,
                        help="fixed initial-energy edges in eV; default: 0,0.5,1,2,3,5")
    parser.add_argument("--mesh-um", type=int, default=50,
                        help="FEM mesh resolution to include when the table contains multiple meshes (default: 50)")
    args = parser.parse_args(argv)
    if args.run is None:
        parser.print_help()
        return 0
    try:
        result = analyze(args.run, args.output, args.energy_edges_ev, args.mesh_um)
    except (OSError, ValueError, json.JSONDecodeError, ArithmeticError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
