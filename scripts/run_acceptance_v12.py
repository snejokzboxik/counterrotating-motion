"""Checkpointed, paired Sobol estimate of finite-time transverse acceptance.

Examples
--------
Quick performance pilot (not an acceptance result)::

    python scripts/run_acceptance_v12.py --sample-power 5 --replicates 1 \
        --horizon 5 --models A,B,C --meshes 50 --scales 1 --max-jobs 96 --tag pilot

Initial production stage (phase 0, equal peak, 4 independent scrambles)::

    python scripts/run_acceptance_v12.py --sample-power 8 --replicates 4 \
        --horizon 1000 --segment 250 --models A,B,C,idealA,idealB,idealC \
        --meshes 50 --scales 0.25,0.5,0.75,1,1.5,2,4 --workers 8 \
        --tag equal_broad_lambda_h50

Each launch point is shared across all models/meshes/scales. Per-trajectory
atomic JSON checkpoints make interrupted runs resumable without repeating
completed trajectories. Each worker handles a distinct checkpoint path.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import numpy as np
import scipy
from scipy.stats import qmc, t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap.dimensionless_dynamics_v12 import DimensionlessRFModel, integrate_to_horizon
from three_phase_trap.rf_reconstruction import HarmonicQuadratures, RFReconstruction


E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
MASS = 100_000.0 * AMU
FREQUENCY = 100_000.0
OMEGA = 2.0 * math.pi * FREQUENCY
LENGTH = 8.0e-3
INJECTION_RADIUS_M = 7.5e-3
MAX_ENERGY_EV = 5.0
_WORKER_MODELS = {}
_WORKER_META = {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_q(mesh: int, kind: str) -> tuple[HarmonicQuadratures, Path]:
    if kind in ("A", "B", "idealA", "idealB"):
        path = ROOT / f"outputs/abc_validation/direct_fem_validation/v2/h{mesh}_P4/diagnostic_N48.json"
        q = RFReconstruction.load(path).unit_quadratures
        if kind in ("B", "idealB"):
            q = HarmonicQuadratures(np.zeros_like(q.cosine), q.sine.copy(), q.reference_radius)
        return q, path
    if kind in ("C", "idealC"):
        path = ROOT / f"outputs/octupole_comparison_home/confinement_maps_v11/fields/h{mesh}_P4_C/reconstruction_N48_C.json"
        return RFReconstruction.load(path).unit_quadratures, path
    raise ValueError(f"unknown field kind {kind}")


def _angular_rf_strength(q: HarmonicQuadratures, radius_m: float, points: int = 2048) -> float:
    angle = (np.arange(points) + 0.371) * (2 * np.pi / points)
    xy = radius_m * np.column_stack((np.cos(angle), np.sin(angle)))
    ec, es = q.field_quadratures(xy)
    return float(np.mean(np.sum(ec * ec + es * es, axis=1)))


def _voltage_table(mesh: int, protocol: str, reference_mm: float) -> dict[str, float]:
    qa, _ = _load_q(mesh, "A")
    qc, _ = _load_q(mesh, "C")
    if protocol == "equal":
        return {name: 250.0 for name in ("A", "B", "C", "idealA", "idealB", "idealC")}
    result = {name: 250.0 for name in ("A", "B", "C", "idealA", "idealB", "idealC")}
    result["B"] = result["idealB"] = 250.0 * math.sqrt(2.0)
    radius = reference_mm * 1e-3
    result["C"] = result["idealC"] = 250.0 * math.sqrt(
        _angular_rf_strength(qa, radius) / _angular_rf_strength(qc, radius))
    return result


def _active_coefficients(q: HarmonicQuadratures, kind: str) -> tuple[np.ndarray, np.ndarray]:
    c, s = q.cosine.copy(), q.sine.copy()
    if kind.startswith("idealA") or kind == "idealA":
        keep = 4
    elif kind.startswith("idealB") or kind == "idealB":
        keep = 4
        c[:] = 0
    elif kind.startswith("idealC") or kind == "idealC":
        keep = 6
    else:
        return c, s
    mask = np.ones(len(c), dtype=bool)
    mask[keep] = False
    c[mask] = 0
    s[mask] = 0
    return c, s


def _complex_pairs(array: np.ndarray) -> list[list[float]]:
    return [[float(z.real), float(z.imag)] for z in np.asarray(array, dtype=complex)]


def _make_model_specs(kinds: list[str], meshes: list[int], scales: list[float], protocol: str,
                      reference_mm: float, phase_deg: float) -> tuple[list[dict], dict]:
    specs, hashes, voltages = [], {}, {}
    for mesh in meshes:
        volts = _voltage_table(mesh, protocol, reference_mm)
        for kind in kinds:
            if kind.startswith("ideal") and mesh != min(meshes):
                # Ideal controls use the finest FEM coefficient fit only.
                continue
            q, source = _load_q(mesh, kind)
            hashes[f"{kind}_h{mesh}"] = _sha(source)
            c, s = _active_coefficients(q, kind)
            leading_order = 6 if kind.endswith("C") else 4
            leading_c = np.zeros_like(c)
            leading_s = np.zeros_like(s)
            leading_c[leading_order] = c[leading_order]
            leading_s[leading_order] = s[leading_order]
            leading_q = HarmonicQuadratures(leading_c, leading_s, q.reference_radius)
            model_q = HarmonicQuadratures(c, s, q.reference_radius)
            comparison_radius_m = reference_mm * 1e-3
            leading_rms2 = _angular_rf_strength(leading_q, comparison_radius_m)
            total_rms2 = _angular_rf_strength(model_q, comparison_radius_m)
            leading_coefficient_norm = float(math.sqrt(
                abs(leading_c[leading_order])**2 + abs(leading_s[leading_order])**2))
            v0 = volts[kind]
            for scale in scales:
                name = f"{kind}_h{mesh}_scale{scale:g}"
                spec = {
                    "name": name, "kind": kind, "mesh_um": mesh,
                    "voltage_V": v0 * scale, "scale_factor": scale,
                    "coupling": E_CHARGE * v0 * scale / (MASS * OMEGA**2 * LENGTH**2),
                    "leading_order": leading_order,
                    "leading_quadrature_coefficient_norm": leading_coefficient_norm,
                    "comparison_radius_m": comparison_radius_m,
                    "leading_rms_field_per_volt_V_m": math.sqrt(leading_rms2),
                    "total_rms_field_per_volt_V_m": math.sqrt(total_rms2),
                    "cosine": _complex_pairs(c), "sine": _complex_pairs(s),
                    "reference_radius_m": q.reference_radius, "length_m": LENGTH,
                    "aperture_rho": 1.0, "phase_deg": phase_deg,
                    "source_sha256": _sha(source),
                }
                specs.append(spec)
                voltages[name] = v0 * scale
    return specs, {"field_sha256": hashes, "voltage_V": voltages}


def _decode_complex(pairs):
    return np.asarray([complex(float(a), float(b)) for a, b in pairs], dtype=complex)


def _init_worker(specs: list[dict], meta: dict):
    global _WORKER_MODELS, _WORKER_META
    _WORKER_MODELS = {}
    _WORKER_META = meta
    for spec in specs:
        model = DimensionlessRFModel.from_physical(
            _decode_complex(spec["cosine"]), _decode_complex(spec["sine"]),
            charge_c=E_CHARGE, mass_kg=MASS, voltage_v=spec["voltage_V"],
            omega_rad_s=OMEGA, length_m=spec["length_m"],
            reference_radius_m=spec["reference_radius_m"],
            aperture_m=spec["aperture_rho"] * spec["length_m"],
            phase_rad=math.radians(spec["phase_deg"]), name=spec["name"],
        )
        # Guard against accidental inconsistency between recorded and applied Lambda.
        if not np.isclose(model.coupling, spec["coupling"], rtol=2e-15, atol=0.0):
            raise ValueError(f"coupling mismatch in {spec['name']}")
        _WORKER_MODELS[spec["name"]] = (model, spec)


def _job_worker(job: tuple) -> dict:
    (job_id, sample_id, model_name, state, start_period, resume_state,
     checkpoint_text, horizon, segment, rtol, atol, max_step_rf, samples_per_rf,
     config_hash, phase_rad) = job
    checkpoint = Path(checkpoint_text)
    existing = None
    if checkpoint.exists():
        try:
            existing = json.loads(checkpoint.read_text(encoding="utf-8"))
        except Exception:
            existing = None
    if existing and existing.get("config_hash") == config_hash:
        if existing.get("complete"):
            return existing
        if existing.get("status") == "retained" and existing.get("reached_RF", 0) < horizon:
            start_period = float(existing["reached_RF"])
            resume_state = np.asarray(existing["state_dimensionless"], dtype=float)
    model, spec = _WORKER_MODELS[model_name]
    model = replace(model, phase_rad=float(phase_rad))
    y0 = np.asarray(state, dtype=float)
    continued = y0 if resume_state is None else np.asarray(resume_state, dtype=float)

    def checkpoint_segment(period, current, max_radius, nfev):
        _atomic_json(checkpoint, {
            "config_hash": config_hash, "job_id": job_id, "sample_id": sample_id,
            "model": model_name, "status": "retained", "complete": False,
            "reached_RF": period, "state_dimensionless": np.asarray(current).tolist(),
            "escape_RF": None, "max_radius_sampled_rho": max_radius,
            "nfev": nfev,
        })

    began = time.perf_counter()
    result = integrate_to_horizon(
        model, y0, horizon, segment_periods=segment, rtol=rtol, atol=atol,
        max_step_rf=max_step_rf, samples_per_rf=samples_per_rf,
        checkpoint_callback=checkpoint_segment, resume_state=continued,
        resume_period=start_period,
    )
    elapsed = time.perf_counter() - began
    spec_row = {k: spec[k] for k in ("kind", "mesh_um", "voltage_V", "scale_factor", "coupling",
                                      "leading_order", "leading_quadrature_coefficient_norm",
                                      "comparison_radius_m", "leading_rms_field_per_volt_V_m",
                                      "total_rms_field_per_volt_V_m", "source_sha256")}
    row = {
        "config_hash": config_hash, "job_id": job_id, "sample_id": sample_id,
        "model": model_name, **spec_row,
        "horizon_RF": horizon, "phase_rad": phase_rad,
        "status": result.status, "complete": result.status != "numerical_error",
        "reached_RF": result.reached_period, "escape_RF": result.escape_period,
        "state_dimensionless": result.final_state.tolist(),
        "max_radius_sampled_rho": result.max_radius_sampled_rho,
        "nfev": result.nfev, "wall_s": elapsed, "message": result.message,
    }
    _atomic_json(checkpoint, row)
    return row


def _generate_samples(rep: int, power: int, seed: int, phase_deg: float | None,
                      injection_radius_m: float, max_energy_ev: float):
    dimensions = 5 if phase_deg is None else 4
    engine = qmc.Sobol(d=dimensions, scramble=True, seed=seed + 104729 * rep)
    points = engine.random_base2(power)
    rho_max = injection_radius_m / LENGTH
    u_max = math.sqrt(2 * max_energy_ev * E_CHARGE / (MASS * (LENGTH * OMEGA)**2))
    rho = rho_max * np.sqrt(points[:, 0])
    theta = 2 * np.pi * points[:, 1]
    speed = u_max * np.sqrt(points[:, 2])
    beta = 2 * np.pi * points[:, 3]
    phase = np.full(len(points), math.radians(phase_deg) if phase_deg is not None else 0.0)
    if phase_deg is None:
        phase = 2 * np.pi * points[:, 4]
    states = np.column_stack((rho*np.cos(theta), rho*np.sin(theta),
                              speed*np.cos(beta), speed*np.sin(beta)))
    table = {
        "replicate": np.full(len(points), rep, dtype=int),
        "sample_index": np.arange(len(points), dtype=int),
        "rho0": rho, "theta0_rad": theta,
        "x0_m": LENGTH * states[:, 0], "y0_m": LENGTH * states[:, 1],
        "u0": speed, "beta_rad": beta,
        "vx0_m_s": LENGTH * OMEGA * states[:, 2],
        "vy0_m_s": LENGTH * OMEGA * states[:, 3],
        "E0_eV": 0.5 * MASS * (LENGTH * OMEGA * speed)**2 / E_CHARGE,
        "phase_rad": phase,
    }
    return states, table


def _write_samples(path: Path, table: dict[str, np.ndarray]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table))
        writer.writeheader()
        for i in range(len(next(iter(table.values())))):
            writer.writerow({key: float(value[i]) if np.issubdtype(np.asarray(value).dtype, np.floating)
                             else int(value[i]) for key, value in table.items()})


def _summarize(root: Path, specs: list[dict], replicates: int, sample_power: int,
               domain_dimless: float, physical_factor: float, full_run: bool,
               records: list[dict] | None = None):
    if records is None:
        records = []
        for path in (root / "checkpoints").rglob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if row.get("status") in ("retained", "escaped", "numerical_error"):
                records.append(row)
    # Keep the latest record for each task. A numerical-error attempt can be
    # retried on a later invocation and must not coexist with its successful
    # replacement in the acceptance denominator.
    latest = {r["job_id"]: r for r in records if r.get("job_id")}
    records = list(latest.values())
    valid_statuses = {"retained", "escaped"}
    valid_records = [r for r in records if r.get("complete") and r.get("status") in valid_statuses]
    numerical_errors = sum(r.get("status") == "numerical_error" for r in records)
    n_per_rep = 2**sample_power
    models = sorted({s["name"] for s in specs})
    estimates = []
    for model in models:
        fractions, n_ok = [], []
        subset = [r for r in valid_records if r["model"] == model]
        for rep in range(replicates):
            group = [r for r in subset if r["sample_id"].startswith(f"r{rep:02d}_")]
            sample_ids = {r["sample_id"] for r in group}
            n = len(sample_ids)
            if n == n_per_rep and len(group) == n_per_rep:
                fractions.append(sum(r["status"] == "retained" for r in group) / n)
            n_ok.append(n)
        # Do not publish a model estimate until every planned point in every
        # independent scramble has a valid binary outcome.
        if len(fractions) == replicates:
            mean = float(np.mean(fractions))
            se = float(np.std(fractions, ddof=1) / math.sqrt(len(fractions))) if len(fractions) > 1 else None
            if se is None:
                retention_ci = None
            else:
                half_width = float(student_t.ppf(0.975, df=replicates-1) * se)
                retention_ci = [max(0.0, mean-half_width), min(1.0, mean+half_width)]
            spec = next(s for s in specs if s["name"] == model)
            estimates.append({
                "model": model, "kind": spec["kind"], "mesh_um": spec["mesh_um"],
                "scale_factor": spec["scale_factor"], "voltage_V": spec["voltage_V"],
                "coupling_Lambda": spec["coupling"], "horizon_RF": subset[0]["horizon_RF"],
                "leading_order": spec["leading_order"],
                "leading_quadrature_coefficient_norm": spec["leading_quadrature_coefficient_norm"],
                "comparison_radius_m": spec["comparison_radius_m"],
                "leading_rms_field_at_comparison_radius_V_m": spec["voltage_V"] * spec["leading_rms_field_per_volt_V_m"],
                "total_rms_field_at_comparison_radius_V_m": spec["voltage_V"] * spec["total_rms_field_per_volt_V_m"],
                "samples_per_replicate": n_per_rep, "complete_replicates": len(fractions),
                "retention_fraction_mean": mean, "retention_fraction_replica_SE": se,
                "retention_fraction_student_t_95pct_ci": json.dumps(retention_ci),
                "a4_dimensionless_mean": mean * domain_dimless,
                "a4_dimensionless_replica_SE": None if se is None else se * domain_dimless,
                "a4_dimensionless_student_t_95pct_ci": json.dumps(None if retention_ci is None else
                                                                    [retention_ci[0]*domain_dimless,
                                                                     retention_ci[1]*domain_dimless]),
                "A4_m4_per_s2_mean": mean * domain_dimless * physical_factor,
                "A4_m4_per_s2_replica_SE": None if se is None else se * domain_dimless * physical_factor,
                "A4_m4_per_s2_student_t_95pct_ci": json.dumps(None if retention_ci is None else
                                                               [retention_ci[0]*domain_dimless*physical_factor,
                                                                retention_ci[1]*domain_dimless*physical_factor]),
                "replicate_fractions": json.dumps(fractions),
                "complete_counts_by_replicate": json.dumps(n_ok),
            })
    if estimates and full_run:
        with (root / "acceptance_estimates.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(estimates[0]))
            writer.writeheader(); writer.writerows(estimates)
    # Paired differences use the identical sample_id for each model pair.
    pair_rows = []
    by_model_sample = {(r["model"], r["sample_id"]): r for r in valid_records}
    factor_groups = sorted({(s["mesh_um"], s["scale_factor"]) for s in specs})
    for mesh, scale in factor_groups:
        candidates = [s for s in specs if s["mesh_um"] == mesh and s["scale_factor"] == scale]
        kinds_present = {s["kind"] for s in candidates}
        pairs = [("A", "B"), ("A", "C"), ("B", "C"),
                 ("idealA", "idealB"), ("idealA", "idealC"), ("idealB", "idealC"),
                 # Same-order FEM versus one-harmonic ablation isolates the
                 # net effect of the higher spatial harmonics on these starts.
                 ("A", "idealA"), ("B", "idealB"), ("C", "idealC")]
        for left, right in pairs:
            if left not in kinds_present or right not in kinds_present:
                continue
            left_id = f"{left}_h{mesh}_scale{scale:g}"
            right_id = f"{right}_h{mesh}_scale{scale:g}"
            delta_reps, left_reps, right_reps, discordant = [], [], [], 0
            total_pairs = 0
            for rep in range(replicates):
                sample_ids = [f"r{rep:02d}_{i:06d}" for i in range(n_per_rep)]
                rows_left = [by_model_sample.get((left_id, sid)) for sid in sample_ids]
                rows_right = [by_model_sample.get((right_id, sid)) for sid in sample_ids]
                if not all(rows_left) or not all(rows_right):
                    continue
                ind_left = np.array([r["status"] == "retained" for r in rows_left], dtype=float)
                ind_right = np.array([r["status"] == "retained" for r in rows_right], dtype=float)
                delta = ind_left - ind_right
                delta_reps.append(float(np.mean(delta)))
                left_reps.append(float(ind_left.mean()))
                right_reps.append(float(ind_right.mean()))
                total_pairs += len(delta)
                discordant += int(np.count_nonzero(delta))
            # Paired comparisons also require a complete valid ensemble for
            # every scramble; errors are unresolved states, never escapes.
            if len(delta_reps) != replicates:
                continue
            delta_mean = float(np.mean(delta_reps))
            delta_se = float(np.std(delta_reps, ddof=1)/math.sqrt(len(delta_reps))) if len(delta_reps) > 1 else None
            if delta_se is None:
                delta_ci = None
            else:
                half_width = float(student_t.ppf(0.975, df=replicates-1) * delta_se)
                delta_ci = [max(-1.0, delta_mean-half_width), min(1.0, delta_mean+half_width)]
            p_left_mean = float(np.mean(left_reps))
            p_right_mean = float(np.mean(right_reps))
            ratio_of_means = None if p_right_mean == 0 else p_left_mean / p_right_mean
            ratio_ci = None
            if replicates > 1 and p_right_mean > 0:
                seed_material = f"{mesh}|{scale:.12g}|{left}|{right}|{sample_power}|{replicates}"
                bootstrap_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(bootstrap_seed)
                indices = rng.integers(0, replicates, size=(5000, replicates))
                b_left = np.mean(np.asarray(left_reps)[indices], axis=1)
                b_right = np.mean(np.asarray(right_reps)[indices], axis=1)
                ratios = b_left[b_right > 0] / b_right[b_right > 0]
                if len(ratios):
                    ratio_ci = [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]
            pair_rows.append({
                "mesh_um": mesh, "scale_factor": scale, "model_left": left_id,
                "model_right": right_id, "left_minus_right_retention_fraction": delta_mean,
                "paired_replica_SE": delta_se,
                "left_minus_right_retention_student_t_95pct_ci": json.dumps(delta_ci),
                "delta_a4_dimensionless": delta_mean*domain_dimless,
                "delta_a4_dimensionless_student_t_95pct_ci": json.dumps(None if delta_ci is None else
                                                                          [delta_ci[0]*domain_dimless,
                                                                           delta_ci[1]*domain_dimless]),
                "delta_A4_m4_per_s2": delta_mean*domain_dimless*physical_factor,
                "delta_A4_m4_per_s2_student_t_95pct_ci": json.dumps(None if delta_ci is None else
                                                                      [delta_ci[0]*domain_dimless*physical_factor,
                                                                       delta_ci[1]*domain_dimless*physical_factor]),
                "ratio_of_mean_retention_fractions": ratio_of_means,
                "ratio_of_means_bootstrap_95pct": json.dumps(ratio_ci),
                "discordant_pairs": discordant, "total_pairs": total_pairs,
                "discordant_pair_fraction": discordant/total_pairs if total_pairs else None,
                "complete_replicates": len(delta_reps),
            })
    if pair_rows and full_run:
        with (root / "paired_acceptance_differences.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
            writer.writeheader(); writer.writerows(pair_rows)
    summary = {
        "complete_trajectory_records": len(valid_records), "planned_trajectories": replicates*n_per_rep*len(specs),
        "numerical_error_records": numerical_errors,
        "full_run": full_run, "complete_estimates": len(estimates) if full_run else 0,
        "estimates_csv": "acceptance_estimates.csv" if estimates and full_run else None,
        "paired_comparisons": len(pair_rows) if full_run else 0,
        "paired_differences_csv": "paired_acceptance_differences.csv" if pair_rows and full_run else None,
        "models_without_complete_estimate": sorted(set(models) - {r["model"] for r in estimates}),
        "uncertainty_method": "replica-level standard error and 95% Student-t interval across independent scrambled Sobol replicates; paired ratio interval by resampling replicates",
        "note": "Only complete retained/escaped outcomes enter estimates. Numerical errors remain unresolved and prevent an estimate for any incomplete model or paired comparison.",
    }
    _atomic_json(root / "progress_summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-power", type=int, default=10)
    p.add_argument("--replicates", type=int, default=4)
    p.add_argument("--seed", type=int, default=120112)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--segment", type=int, default=100)
    p.add_argument("--models", default="A,B,C")
    p.add_argument("--meshes", default="35,50")
    p.add_argument("--scales", default="0.5,1,2")
    p.add_argument("--protocol", choices=("equal", "matched"), default="equal")
    p.add_argument("--reference-mm", type=float, default=4.0)
    p.add_argument("--phase-deg", type=float, default=0.0,
                   help="fixed initial RF phase; use --phase-average to sample it uniformly")
    p.add_argument("--phase-average", action="store_true")
    p.add_argument("--injection-radius-mm", type=float, default=INJECTION_RADIUS_M * 1e3,
                   help="uniform coordinate-disk radius inside the 8 mm aperture")
    p.add_argument("--max-energy-ev", type=float, default=MAX_ENERGY_EV,
                   help="uniform upper limit of the velocity-disk kinetic-energy measure")
    p.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2)))
    p.add_argument("--rtol", type=float, default=2e-10)
    p.add_argument("--atol", type=float, default=2e-12)
    p.add_argument("--max-step-rf", type=float, default=1/64)
    p.add_argument("--samples-per-rf", type=int, default=4)
    p.add_argument("--max-jobs", type=int, default=0, help="test-only task cap; capped runs are not reported as estimates")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if not (0 <= args.sample_power <= 24) or args.replicates < 1 or args.horizon < 1 or args.segment < 1:
        p.error("invalid sample, replicate or horizon arguments")
    if args.horizon % args.segment:
        p.error("horizon must be divisible by checkpoint segment")
    if args.workers < 1 or args.workers > (os.cpu_count() or args.workers):
        p.error("workers exceeds available logical CPU count")
    kinds = [s.strip() for s in args.models.split(",") if s.strip()]
    valid = {"A", "B", "C", "idealA", "idealB", "idealC"}
    if set(kinds) - valid:
        p.error(f"unknown models: {set(kinds)-valid}")
    meshes = [int(s) for s in args.meshes.split(",") if s.strip()]
    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    if any(m not in (35, 50) for m in meshes) or any(s <= 0 for s in scales):
        p.error("meshes must be 35/50 and all field scales positive")
    injection_radius_m = args.injection_radius_mm * 1e-3
    if not (0 < injection_radius_m < LENGTH) or args.max_energy_ev <= 0:
        p.error("injection radius must be inside the 8 mm aperture and maximum energy must be positive")
    if not (0 < args.reference_mm < LENGTH * 1e3):
        p.error("reference radius must lie strictly inside the 8 mm aperture")
    if args.phase_average and args.phase_deg != 0.0:
        p.error("--phase-average cannot be combined with nonzero --phase-deg")

    fixed_phase = None if args.phase_average else args.phase_deg
    specs, field_meta = _make_model_specs(kinds, meshes, scales, args.protocol,
                                          args.reference_mm, args.phase_deg if fixed_phase is not None else 0.0)
    cfg = {
        "study": "V12 finite-time transverse 4D acceptance, paired randomized Sobol",
        "sample_power": args.sample_power, "replicates": args.replicates,
        "seed": args.seed, "horizon_RF": args.horizon, "segment_RF": args.segment,
        "models": kinds, "meshes_um": meshes, "scales": scales,
        "protocol": args.protocol, "matched_reference_radius_mm": args.reference_mm if args.protocol == "matched" else None,
        "field_diagnostic_radius_mm": args.reference_mm,
        "phase_mode": "uniform-average" if fixed_phase is None else "fixed",
        "phase_deg": fixed_phase, "domain": {"injection_radius_m": injection_radius_m,
                                                  "injection_radius_mm": args.injection_radius_mm,
                                                  "max_energy_eV": args.max_energy_ev,
                                                  "coordinate_measure": "uniform disk area",
                                                  "velocity_measure": "uniform disk area in vx,vy"},
        "physical": {"mass_kg": MASS, "charge_C": E_CHARGE, "frequency_Hz": FREQUENCY,
                     "length_m": LENGTH, "aperture_m": LENGTH},
        "solver": {"method": "DOP853", "rtol": args.rtol, "atol_dimensionless": args.atol,
                   "max_step_rf_fraction": args.max_step_rf, "samples_per_rf_for_max_radius": args.samples_per_rf},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "compute": {"workers": args.workers, "logical_cpus": os.cpu_count(),
                    "max_jobs_test_cap": args.max_jobs or None,
                    "mode": "per-trajectory atomic checkpoint; one parent journal writer"},
        **field_meta,
    }
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode()).hexdigest()[:20]
    cfg["config_hash"] = config_hash
    suffix = f"_{args.tag}" if args.tag else ""
    out_root = ROOT / f"outputs/octupole_comparison_home/dimensionless_acceptance_v12/runs/{config_hash}{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)
    config_path = out_root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != cfg:
        raise ValueError("run folder contains a different config; refusing to overwrite")
    _atomic_json(config_path, cfg)
    spec_path = out_root / "model_coefficients.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != specs:
        raise ValueError("model coefficients differ from saved run; refusing to overwrite")
    _atomic_json(spec_path, specs)

    rho_max = injection_radius_m / LENGTH
    u_max = math.sqrt(2 * args.max_energy_ev * E_CHARGE / (MASS * (LENGTH * OMEGA)**2))
    volume_dimless = math.pi**2 * rho_max**2 * u_max**2
    physical_factor = LENGTH**4 * OMEGA**2
    meta = {"horizon_RF": args.horizon, "config_hash": config_hash}

    journal = out_root / "completed_jobs.jsonl"
    known = set()
    completed_rows = []
    if journal.exists():
        latest_rows = {}
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("status") in ("retained", "escaped", "numerical_error"):
                    latest_rows[row["job_id"]] = row
                    if row.get("complete") and row.get("status") in ("retained", "escaped"):
                        known.add(row["job_id"])
            except Exception: pass
        completed_rows = list(latest_rows.values())
    jobs = []
    sample_hashes = {}
    for rep in range(args.replicates):
        states, table = _generate_samples(rep, args.sample_power, args.seed, fixed_phase,
                                          injection_radius_m, args.max_energy_ev)
        sample_path = out_root / f"initial_states_r{rep:02d}.csv"
        _write_samples(sample_path, table)
        sample_hashes[sample_path.name] = _sha(sample_path)
        for i, state in enumerate(states):
            sample_id = f"r{rep:02d}_{i:06d}"
            phase_rad = float(table["phase_rad"][i])
            for spec in specs:
                model_name = spec["name"]
                job_id = hashlib.sha256(f"{config_hash}|{sample_id}|{model_name}".encode()).hexdigest()[:24]
                if job_id in known:
                    continue
                # RF phase is sampled in the initial-state table for a phase-average run.
                worker_spec_name = model_name
                ckpt = out_root / "checkpoints" / model_name / f"{sample_id}.json"
                existing = None
                if ckpt.exists():
                    try: existing = json.loads(ckpt.read_text(encoding="utf-8"))
                    except Exception: pass
                resume_state = None
                start_period = 0.0
                if existing and existing.get("config_hash") == config_hash and not existing.get("complete"):
                    resume_state = existing.get("state_dimensionless")
                    start_period = float(existing.get("reached_RF", 0.0))
                jobs.append((job_id, sample_id, worker_spec_name, state.tolist(), start_period,
                             resume_state, str(ckpt), args.horizon, args.segment, args.rtol,
                             args.atol, args.max_step_rf, args.samples_per_rf, config_hash, phase_rad))
    _atomic_json(out_root / "initial_state_manifest.json", {
        "sampler": "scipy.stats.qmc.Sobol(scramble=True)",
        "software": cfg["software"], "seed": args.seed,
        "sample_power_per_replicate": args.sample_power, "replicates": args.replicates,
        "phase_mode": cfg["phase_mode"], "sha256_by_file": sample_hashes,
    })
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    _atomic_json(out_root / "run_state.json", {"status": "running", "planned_jobs": args.replicates*2**args.sample_power*len(specs),
                                               "queued_jobs_this_invocation": len(jobs), "config_hash": config_hash,
                                               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({"out": str(out_root), "config_hash": config_hash,
                      "queued_jobs": len(jobs), "specs": len(specs), "workers": args.workers,
                      "sample_points": args.replicates*2**args.sample_power,
                      "Dprime_volume": volume_dimless, "u_max": u_max,
                      "phase_mode": "uniform" if args.phase_average else f"fixed {args.phase_deg} deg"}, indent=2), flush=True)

    started = time.perf_counter()
    done = 0
    progress_every = 500
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(specs, meta)) as pool, journal.open("a", encoding="utf-8") as stream:
        futures = [pool.submit(_job_worker, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            if row["job_id"] not in known:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                done += 1
                completed_rows = [r for r in completed_rows if r.get("job_id") != row["job_id"]]
                completed_rows.append(row)
                if done % progress_every == 0:
                    stream.flush(); os.fsync(stream.fileno())
                    summary = _summarize(out_root, specs, args.replicates, args.sample_power,
                                         volume_dimless, physical_factor, not args.max_jobs,
                                         records=completed_rows)
                    print(f"{done}/{len(jobs)} new tasks, wall={time.perf_counter()-started:.1f}s, "
                          f"complete={summary['complete_trajectory_records']}/{summary['planned_trajectories']}", flush=True)
                if row.get("complete") and row.get("status") in ("retained", "escaped"):
                    known.add(row["job_id"])
        stream.flush(); os.fsync(stream.fileno())
    summary = _summarize(out_root, specs, args.replicates, args.sample_power,
                         volume_dimless, physical_factor, not args.max_jobs,
                         records=completed_rows)
    run_state = {"status": "complete" if not args.max_jobs and summary["complete_trajectory_records"] == summary["planned_trajectories"] else "incomplete",
                 "planned_trajectories": summary["planned_trajectories"], "complete_trajectories": summary["complete_trajectory_records"],
                 "completed_estimates": summary["complete_estimates"], "wall_seconds_this_invocation": time.perf_counter()-started,
                 "config_hash": config_hash,
                 "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _atomic_json(out_root / "run_state.json", run_state)
    print(json.dumps(run_state, indent=2), flush=True)


if __name__ == "__main__":
    main()
