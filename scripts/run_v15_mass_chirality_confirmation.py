"""Independent Sobol confirmation of the ideal A9 mu=1.2 chirality candidate."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import qmc, sem, t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v15/mass_chirality_mu1p2_confirmation"
ALPHA, BETA, KAPPA = 0.00625, 0.075, -0.015
MU = 1.2
M0_TH = 3200.04652261404
R = 0.008
F = 100_000.0
OMEGA = 2 * math.pi * F
AMU = 1.66053906660e-27
E_CHARGE = 1.602176634e-19
M0_KG = M0_TH * AMU
C2 = 0.7718255558274212
C4 = 0.5413395699439091
SEEDS = (927401, 927403, 927407, 927409, 927421, 927431, 927433, 927437)
N_PAIRS = 128
STEPS = 256
MODELS = (("Q", "Q", 0.0), ("SO_equal_harmonic", "SO", 1.0),
          ("SO_equal_rms", "SO", math.sqrt(2)), ("RO_PLUS", "RO_PLUS", 1.0),
          ("RO_MINUS", "RO_MINUS", 1.0))


def starts(seed: int) -> np.ndarray:
    u = qmc.Sobol(4, scramble=True, seed=seed).random_base2(7)
    radius = 0.0075 * np.sqrt(u[:, 0]) / R
    theta = 2 * np.pi * u[:, 1]
    energy_ev = 5 * u[:, 2]
    speed = np.sqrt(2 * energy_ev * E_CHARGE / M0_KG) / (R * OMEGA)
    phi = np.pi * u[:, 3]
    vr, vt = speed * np.cos(phi), speed * np.sin(phi)
    plus = np.column_stack((radius * np.cos(theta), radius * np.sin(theta),
                            vr * np.cos(theta) - vt * np.sin(theta),
                            vr * np.sin(theta) + vt * np.cos(theta)))
    minus = plus.copy()
    minus[:, 1] *= -1
    minus[:, 3] *= -1
    return np.stack((plus, minus), axis=1)


def interval(values):
    a = np.asarray(values, float)
    mean = float(np.mean(a))
    half = float(t.ppf(.975, len(a)-1) * sem(a)) if len(a) > 1 and np.std(a, ddof=1) else 0.0
    return mean, mean-half, mean+half


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from three_phase_trap.v15_dynamics import integrate_rk4_batch, dimensionless_angular_momentum
    from scripts.run_v15_chiral_pilot import field_voltage_peaks

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    field_hash = hashlib.sha256((ROOT / "src/three_phase_trap/v15_dynamics.py").read_bytes()).hexdigest()
    manifest = {"study": "V15 independent ideal chirality confirmation",
                "candidate_selected_from": "mass_ray_a9, mu=1.2, not this ensemble",
                "mu": MU, "alpha": ALPHA/MU, "beta": BETA/MU, "kappa": KAPPA/MU,
                "delta_rad": 0.0, "tau0_rad": 0.0, "seeds": SEEDS,
                "pairs_per_scramble": N_PAIRS, "starts_per_model": len(SEEDS)*N_PAIRS*2,
                "domain": "r<=7.5 mm area-uniform; physical transverse energy <=5 eV area-uniform; exact Lz-mirror pairs",
                "velocities": "dimensionless v scaled by 1/sqrt(mu) at fixed physical energy cutoff",
                "models": MODELS, "horizons_rf": [100, 300], "steps_per_rf": STEPS,
                "integrator": "vectorized fixed-step RK4; 300 RF by continuous extension from 100 RF",
                "voltage_limit_V": 250, "source_sha256": source_hash, "field_sha256": field_hash}
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=float,
                                              separators=(",", ":")).encode()).hexdigest()
    mp, cp = OUT / "manifest.json", OUT / "checkpoint.jsonl"
    if mp.exists() and json.loads(mp.read_text(encoding="utf8")) != json.loads(json.dumps(manifest, sort_keys=True, default=float)):
        raise RuntimeError("protocol mismatch; use another output directory")
    if not mp.exists():
        mp.write_text(json.dumps(json.loads(json.dumps(manifest, sort_keys=True, default=float)),
                                 indent=2, ensure_ascii=False)+"\n", encoding="utf8")

    done = {}
    if cp.exists():
        for line in cp.read_text(encoding="utf8").splitlines():
            if not line: continue
            row = json.loads(line)
            if row["manifest_hash"] != manifest_hash: raise RuntimeError("checkpoint hash mismatch")
            if row["key"] in done: raise RuntimeError("duplicate trajectory key")
            done[row["key"]] = row

    base_starts, meta = [], []
    for rep, seed in enumerate(SEEDS):
        pairset = starts(seed)
        for pair in range(N_PAIRS):
            for idx, sign in ((0, 1), (1, -1)):
                state = pairset[pair, idx].copy()
                state[2:] /= math.sqrt(MU)
                base_starts.append(state)
                meta.append((rep, seed, pair, sign))
    initial = np.asarray(base_starts)
    peak_base = {name: float(field_voltage_peaks(ALPHA, BETA, KAPPA*mult, name, 0.0).max())
                 for name, _, mult in MODELS}

    with cp.open("a", encoding="utf8") as f:
        for name, model, mult in MODELS:
            if all("|".join(map(str, (name, horizon, rep, pair, sign))) in done
                   for horizon in (100, 300) for rep in range(len(SEEDS))
                   for pair in range(N_PAIRS) for sign in (-1, 1)):
                continue
            kap = KAPPA * mult / MU
            first = integrate_rk4_batch(initial, ALPHA/MU, BETA/MU, kap, model,
                delta=0.0, tau0=0.0, horizon_rf=100, steps_per_rf=STEPS, aperture=1.0)
            status300 = first["status"].copy()
            exits300 = first["exit_rf_step_estimate"].copy()
            state300 = first["final_state"].copy()
            rho300 = first["rho_max_stage_sampled"].copy()
            alive = np.flatnonzero(first["status"] == "retained")
            if len(alive):
                second = integrate_rk4_batch(state300[alive], ALPHA/MU, BETA/MU, kap, model,
                    delta=0.0, tau0=0.0, horizon_rf=200, steps_per_rf=STEPS,
                    aperture=1.0, tau_start_cycles=100.0)
                status300[alive] = second["status"]
                exits300[alive] = second["exit_rf_step_estimate"]
                rho300[alive] = np.maximum(rho300[alive], second["rho_max_stage_sampled"])
                state300[alive] = second["final_state"]
            records = []
            for i, (rep, seed, pair, sign) in enumerate(meta):
                for horizon, result, status, exit_rf, rho_max in (
                    (100, first, first["status"], first["exit_rf_step_estimate"], first["rho_max_stage_sampled"]),
                    (300, first, status300, exits300, rho300)):
                    key = "|".join(map(str, (name, horizon, rep, pair, sign)))
                    if key in done: continue
                    rec = {"key": key, "config": name, "model": model, "mu": MU,
                           "alpha": ALPHA/MU, "beta": BETA/MU, "kappa": kap,
                           "delta_rad": 0.0, "tau0_rad": 0.0, "replica": rep,
                           "seed": seed, "pair_index": pair, "Lz_sign": sign,
                           "horizon_rf": horizon, "status": str(status[i]),
                           "exit_rf": float(exit_rf[i]) if np.isfinite(exit_rf[i]) else None,
                           "rho_max": float(rho_max[i]), "x0": float(initial[i, 0]),
                           "y0": float(initial[i, 1]), "vx0": float(initial[i, 2]),
                           "vy0": float(initial[i, 3]), "peak_electrode_V": peak_base[name],
                           "integrator": "RK4", "steps_per_rf": STEPS,
                           "manifest_hash": manifest_hash}
                    rec["key"] = key
                    records.append(rec)
            for rec in records:
                f.write(json.dumps(rec, separators=(",", ":"))+"\n")
                done[rec["key"]] = rec
            f.flush(); os.fsync(f.fileno())
            print(f"{name} completed; unique_rows={len(done)}", flush=True)

    rows = list(done.values())
    summary = []
    for config, _, _ in MODELS:
        for horizon in (100, 300):
            for metric in ("P_plus", "P_minus", "P_total", "Delta_chi"):
                vals = []
                for rep in range(len(SEEDS)):
                    rr = [r for r in rows if r["config"] == config and int(r["horizon_rf"]) == horizon and int(r["replica"]) == rep]
                    plus = [r for r in rr if int(r["Lz_sign"]) > 0]
                    minus = [r for r in rr if int(r["Lz_sign"]) < 0]
                    pp = np.mean([r["status"] == "retained" for r in plus])
                    pm = np.mean([r["status"] == "retained" for r in minus])
                    total = np.mean([r["status"] == "retained" for r in rr])
                    vals.append({"P_plus": pp, "P_minus": pm, "P_total": total, "Delta_chi": pp-pm}[metric])
                mean, low, high = interval(vals)
                summary.append({"config": config, "horizon_rf": horizon, "metric": metric,
                                "mean": mean, "ci95_low": low, "ci95_high": high,
                                "n_scrambles": len(vals), "replicate_values": json.dumps(vals)})
    with (OUT / "summary.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    # Paired model differences by scramble.
    paired = []
    for horizon in (100, 300):
        for a, b, label in (("RO_PLUS", "Q", "RO_PLUS-Q"),
                            ("RO_PLUS", "SO_equal_harmonic", "RO_PLUS-SO_equal_harmonic"),
                            ("RO_PLUS", "SO_equal_rms", "RO_PLUS-SO_equal_rms")):
            vals = []
            for rep in range(len(SEEDS)):
                aa = {(int(r["pair_index"]), int(r["Lz_sign"])): r for r in rows if r["config"] == a and int(r["replica"]) == rep and int(r["horizon_rf"]) == horizon}
                bb = {(int(r["pair_index"]), int(r["Lz_sign"])): r for r in rows if r["config"] == b and int(r["replica"]) == rep and int(r["horizon_rf"]) == horizon}
                vals.append(float(np.mean([(aa[k]["status"] == "retained") - (bb[k]["status"] == "retained") for k in aa])))
            mean, low, high = interval(vals)
            paired.append({"horizon_rf": horizon, "contrast": label, "mean_difference_pp": 100*mean,
                           "ci95_low_pp": 100*low, "ci95_high_pp": 100*high,
                           "n_scrambles": len(vals), "replicate_differences_pp": json.dumps([100*x for x in vals])})
    with (OUT / "paired_differences.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    keys = [r["key"] for r in rows]
    validation = {"rows": len(rows), "unique_keys": len(set(keys)), "expected_rows": len(MODELS)*2*len(SEEDS)*N_PAIRS*2,
                  "unresolved": sum(r["status"] == "unresolved" for r in rows),
                  "over_250V": sum(float(r["peak_electrode_V"]) > 250 for r in rows),
                  "seeds": SEEDS, "selected_peak_electrode_V": peak_base}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2)+"\n", encoding="utf8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
