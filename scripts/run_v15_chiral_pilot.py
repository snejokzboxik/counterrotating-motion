"""Checkpointed paired V15 ideal-field chirality pilot.

This is a search-stage calculation. RK4 exit times are step estimates and are
refined with DOP853 before any quantitative headline is reported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v15"
sys.path.insert(0, str(ROOT / "src"))
MODEL_PATH = ROOT / "src/three_phase_trap/v15_dynamics.py"
MASS_SELECTIVITY = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas/MASS_SELECTIVITY_RU.md"
V14_REPORT = ROOT / "outputs/octupole_comparison_home/mass_filter_v14/REPORT_RU.md"

E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
OMEGA = 2.0 * math.pi * 100_000.0
R_M = 8.0e-3
M0_AMU = 3200.04652261404
M0_KG = M0_AMU * AMU
C2 = D2 = 0.7718255558274212
C4 = 0.5413395699439091
SEEDS = (151515, 260705, 20261117, 314159)
PHASES = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
DELTAS = (0.0, math.pi / 4, math.pi / 2)
N_PAIRS_PER_REPLICA = 16

POINTS = {
    "A9": (0.00625, 0.075, -0.015),
    "gain_negative": (0.0125, 0.1125, -0.005625),
    "gain_positive": (-0.0125, 0.1125, 0.005625),
    "floquet_stable": (0.005, 0.15, 0.015),
}
CONFIGS = (
    ("Q", "Q", 0.0),
    ("SO_equal_harmonic", "SO", 1.0),
    ("SO_equal_rms", "SO", math.sqrt(2.0)),
    ("RO_PLUS", "RO_PLUS", 1.0),
    ("RO_MINUS", "RO_MINUS", 1.0),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_pairs(seed: int, mu: float, n: int = N_PAIRS_PER_REPLICA) -> np.ndarray:
    """Return `[pair, sign, x,y,x',y']` states in the physical V14 domain.

    Plus states have positive angular momentum; minus states are exact y-mirror
    images, so r, E_perp, radial velocity, |v_theta|, and tau0 are matched.
    The radial velocity stratum is balanced among inward, zero, and outward.
    """
    power = int(round(math.log2(n)))
    if 2**power != n:
        raise ValueError("n must be a power of two for a scrambled Sobol net")
    u = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)
    radius = 0.0075 * np.sqrt(u[:, 0]) / R_M
    theta = 2.0 * math.pi * u[:, 1]
    energy_ev = 5.0 * u[:, 2]
    speed = np.sqrt(2.0 * energy_ev * E_CHARGE / (M0_KG * mu)) / (R_M * OMEGA)
    radial_fraction = np.resize(np.array([-0.6, 0.0, 0.6, 0.0]), n)
    vr = speed * radial_fraction
    vt = speed * np.sqrt(1.0 - radial_fraction**2)
    plus = np.column_stack((radius * np.cos(theta), radius * np.sin(theta),
                            vr * np.cos(theta) - vt * np.sin(theta),
                            vr * np.sin(theta) + vt * np.cos(theta)))
    minus = plus.copy()
    minus[:, 1] *= -1.0
    minus[:, 3] *= -1.0
    return np.stack((plus, minus), axis=1)


def field_voltage_peaks(alpha: float, beta: float, kappa: float, model: str,
                        delta: float) -> np.ndarray:
    """Peak |V_j| for ideal electrode harmonics at all 12 actual sectors."""
    voltage_scale = M0_KG * OMEGA**2 * R_M**2 / E_CHARGE
    u2 = beta * voltage_scale / C2
    vdc = alpha * voltage_scale / D2
    u4 = kappa * voltage_scale / C4
    model_upper = model.upper()
    if model_upper.startswith("SO_"):
        model = "SO"
    if model == "Q":
        u4 = 0.0
    peaks = []
    for j in range(12):
        theta = math.pi * j / 6.0
        q2 = math.cos(2.0 * theta)
        c4 = math.cos(4.0 * theta)
        s4 = math.sin(4.0 * theta)
        phase_cos, phase_sin = math.cos(delta), math.sin(delta)
        if model in ("RO_PLUS", "RO_MINUS"):
            sigma = 1.0 if model == "RO_PLUS" else -1.0
            a = u2 * q2 + u4 * (c4 * phase_cos - sigma * s4 * phase_sin)
            b = u4 * (c4 * phase_sin + sigma * s4 * phase_cos)
        elif model == "SO":
            a = u2 * q2 + u4 * c4 * phase_cos
            b = u4 * c4 * phase_sin
        else:
            a, b = u2 * q2, 0.0
        peaks.append(abs(vdc * q2) + math.hypot(a, b))
    return np.asarray(peaks)


def run_job(task):
    point_id, alpha, beta, kappa, delta, tau0, config_id, model, multiplier, horizon, steps, mu = task
    from three_phase_trap.v15_dynamics import integrate_rk4_batch, dimensionless_angular_momentum

    seeds = []
    labels = []
    starts = []
    for replica, seed in enumerate(SEEDS):
        pairs = initial_pairs(seed, mu)
        for pair_index in range(N_PAIRS_PER_REPLICA):
            for sign_index in (0, 1):
                starts.append(pairs[pair_index, sign_index])
                seeds.append((replica, seed, pair_index, 1 if sign_index == 0 else -1))
    starts = np.asarray(starts)
    kappa_applied = kappa * multiplier
    result = integrate_rk4_batch(starts, alpha / mu, beta / mu, kappa_applied / mu,
                                 model, delta=delta, tau0=tau0, horizon_rf=horizon,
                                 steps_per_rf=steps, aperture=1.0)
    peaks = field_voltage_peaks(alpha, beta, kappa * multiplier, config_id, delta)
    max_voltage = float(np.max(peaks))
    rows = []
    for index, (replica, seed, pair_index, sign) in enumerate(seeds):
        initial = starts[index]
        final = result["final_state"][index]
        rows.append({
            "job_key": "|".join((point_id, f"mu{mu:.6f}", f"d{delta:.9f}", f"t{tau0:.9f}", config_id)),
            "point_id": point_id, "replica": replica, "seed": seed,
            "pair_index": pair_index, "Lz_sign": sign,
            "config": config_id, "model": model,
            "alpha_base": alpha, "beta_base": beta, "kappa_base": kappa,
            "alpha": alpha / mu, "beta": beta / mu, "kappa_applied": kappa_applied / mu,
            "mu": mu, "delta_rad": delta, "tau0_rad": tau0,
            "horizon_rf": horizon, "steps_per_rf": steps,
            "peak_electrode_V": max_voltage, "within_250V": max_voltage <= 250.0,
            "x0": initial[0], "y0": initial[1], "vx0": initial[2], "vy0": initial[3],
            "Eperp0_eV": 0.5 * M0_KG * mu * (R_M * OMEGA)**2 * (initial[2]**2 + initial[3]**2) / E_CHARGE,
            "rho0": float(np.linalg.norm(initial[:2])),
            "Lz0_dimensionless": float(dimensionless_angular_momentum(initial)),
            "status": str(result["status"][index]),
            "exit_rf_step_estimate": None if not np.isfinite(result["exit_rf_step_estimate"][index]) else float(result["exit_rf_step_estimate"][index]),
            "rho_max_stage_sampled": float(result["rho_max_stage_sampled"][index]),
            "Lz_end_dimensionless": float(result["ell_end"][index]),
            "torque_integral_dimensionless": float(result["torque_integral"][index]),
            "torque_identity_residual": float(result["torque_identity_residual"][index]),
            "xend": float(final[0]), "yend": float(final[1]),
            "vxend": float(final[2]), "vyend": float(final[3]),
        })
    return rows


def protocol_for(args):
    point_ids = ("A9",) if args.stage == "a9" else tuple(POINTS)
    deltas = DELTAS if args.stage == "a9" else DELTAS
    configs = list(CONFIGS)
    tasks = []
    for point_id in point_ids:
        alpha, beta, kappa = POINTS[point_id]
        for delta in deltas:
            for tau0 in PHASES:
                for config_id, model, multiplier in configs:
                    if config_id == "Q" and delta != 0.0:
                        continue  # delta has no meaning for Q; reuse its one field result.
                    peaks = field_voltage_peaks(alpha, beta, kappa * multiplier, config_id, delta)
                    if float(peaks.max()) > 250.0:
                        continue
                    tasks.append((point_id, alpha, beta, kappa, delta, tau0,
                                  config_id, model, multiplier, args.horizon, args.steps_per_rf, args.mu))
    return point_ids, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a9", "all"), default="a9")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--steps-per-rf", type=int, default=128)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    if args.horizon <= 0 or args.steps_per_rf <= 0 or args.mu <= 0:
        raise ValueError("horizon, steps_per_rf, and mu must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    point_ids, tasks = protocol_for(args)
    protocol = {
        "study": "V15 paired chirality search, ideal leading harmonics",
        "point_ids": tuple(POINTS), "points": POINTS,
        "mass_ratio_mu": args.mu, "physical_reference": {"M0_Th": M0_AMU, "Q_C": E_CHARGE,
                              "R_m": R_M, "frequency_hz": OMEGA/(2*math.pi)},
        "domain": "r0 uniform in area to 7.5 mm; Eperp uniform 0..5 eV; radial fraction balanced [-0.6,0,+0.6,0]; exact mirror pair",
        "sobol_seeds": SEEDS, "pairs_per_replica": N_PAIRS_PER_REPLICA,
        "injection_phases_rad": PHASES, "relative_octupole_phases_rad": DELTAS,
        "models": CONFIGS, "horizon_rf": args.horizon, "integrator": "vectorized fixed-step RK4",
        "steps_per_rf": args.steps_per_rf, "aperture_rho": 1.0,
        "hardware_limit_V": 250.0, "task_count": len(tasks),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                          (Path(__file__), MODEL_PATH, MASS_SELECTIVITY, V14_REPORT)},
    }
    core_protocol = {k: v for k, v in protocol.items() if k != "task_count"}
    core_hash = hashlib.sha256(json.dumps(core_protocol, sort_keys=True).encode()).hexdigest()
    manifest_path = OUT / "ideal_chiral_manifest.json"
    run_hash = core_hash
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf8"))
        old_protocol = {k: v for k, v in old.items()
                        if k not in ("run_hash", "core_protocol_hash", "requested_stage", "requested_point_ids")}
        old_core = {k: v for k, v in old_protocol.items() if k != "task_count"}
        current_core = dict(core_protocol)
        # JSON round-trip normalizes tuples in the live protocol to arrays as
        # stored in the manifest.
        old_core = json.loads(json.dumps(old_core, sort_keys=True))
        current_core = json.loads(json.dumps(current_core, sort_keys=True))
        # Permit a compatible checkpoint-reader/orchestration edit while still
        # requiring the field model and all scientific input documents to match.
        old_source = dict(old_core.get("source_sha256", {}))
        current_source = dict(current_core.get("source_sha256", {}))
        old_source.pop(str(Path(__file__).relative_to(ROOT)), None)
        current_source.pop(str(Path(__file__).relative_to(ROOT)), None)
        old_core["source_sha256"] = old_source
        current_core["source_sha256"] = current_source
        if old_core != current_core:
            raise RuntimeError(f"existing manifest scientific protocol mismatch at {manifest_path}")
        # V15 A9 pilot predates separating task scope from physical protocol.
        # Preserve its row hash when extending the same checkpoint to all points.
        run_hash = old["run_hash"]
    manifest_path.write_text(json.dumps({**protocol, "requested_stage": args.stage,
                                         "requested_point_ids": point_ids,
                                         "core_protocol_hash": core_hash,
                                         "run_hash": run_hash}, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
    checkpoint = OUT / "ideal_chiral_checkpoint.jsonl"
    done_rows = {}
    if checkpoint.exists():
        with checkpoint.open(encoding="utf8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["run_hash"] != run_hash:
                    raise RuntimeError("checkpoint hash differs from current V15 protocol")
                if row["key"] in done_rows:
                    raise RuntimeError(f"duplicate checkpoint key {row['key']}")
                done_rows[row["key"]] = row
    expected = 2 * N_PAIRS_PER_REPLICA * len(SEEDS)
    job_counts = {}
    for row in done_rows.values():
        job_counts[row["job_key"]] = job_counts.get(row["job_key"], 0) + 1
    complete_jobs = {job for job, nrows in job_counts.items() if nrows == expected}
    pending = [t for t in tasks if "|".join((t[0], f"mu{t[-1]:.6f}", f"d{t[4]:.9f}", f"t{t[5]:.9f}", t[6])) not in complete_jobs]
    print(f"V15 {args.stage}: tasks={len(tasks)}, complete={len(complete_jobs)}, pending={len(pending)}, workers={args.workers}, RAM/worker is small for ideal-field batches", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool, checkpoint.open("a", encoding="utf8") as stream:
            futures = {pool.submit(run_job, task): task for task in pending}
            for index, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                rows = future.result()
                for row in rows:
                    row["run_hash"] = run_hash
                    key = "|".join((row["job_key"], str(row["replica"]), str(row["pair_index"]), str(row["Lz_sign"])))
                    row["key"] = key
                    if key not in done_rows:
                        stream.write(json.dumps(row, separators=(",", ":"))+"\n")
                        done_rows[key] = row
                stream.flush()
                os.fsync(stream.fileno())
                if index % 8 == 0 or index == len(pending):
                    print(f"completed {index}/{len(pending)} jobs; rows={len(done_rows)}", flush=True)
    print("rows", len(done_rows), "unresolved", sum(r["status"] == "unresolved" for r in done_rows.values()),
          "max torque identity residual", max((abs(float(r["torque_identity_residual"])) for r in done_rows.values()), default=0.0))


if __name__ == "__main__":
    main()
