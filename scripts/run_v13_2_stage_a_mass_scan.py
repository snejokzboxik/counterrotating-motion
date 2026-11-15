"""V13.2 Stage A ideal-field mass scan; paired, resumable, and phase-aware."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability"
OLD = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1/ideal_pilot"
MU = np.round(np.arange(0.97, 1.1600001, 0.01), 8)
PHASES = (0.0, math.pi / 2)
MODELS = (("Q", 0.0), ("SO", -0.05), ("RO", -0.05))
ALPHA0 = 0.027000013157894725
BETA0 = 0.175
MAX_STEP = 2 * math.pi / 64
RTOL, ATOL = 2e-10, 2e-12


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def stable_key(row: dict) -> tuple:
    return (round(float(row["mu"]), 8), round(float(row["phase_rad"]), 12),
            row["model"], round(float(row["eta"]), 8), int(row["index"]))


def advance(task: tuple) -> dict:
    mu, phase, model, eta, index, state = task
    from three_phase_trap.mass_filter_v13 import acceleration

    alpha, beta = ALPHA0 / mu, BETA0 / mu
    kappa = 0.0 if model == "Q" else eta * beta
    standing = model == "SO"
    y0 = np.asarray(state, dtype=float)

    def rhs(tau: float, y: np.ndarray) -> np.ndarray:
        return np.r_[y[2:], acceleration(y[0], y[1], tau + phase, kappa, beta, alpha,
                                         standing_octupole=standing)]

    def aperture(tau: float, y: np.ndarray) -> float:
        return float(np.hypot(y[0], y[1]) - 1.0)

    aperture.terminal = True
    aperture.direction = 1
    try:
        sol = solve_ivp(rhs, (0.0, 2 * math.pi * 100), y0, method="DOP853",
                        rtol=RTOL, atol=ATOL, max_step=MAX_STEP, events=aperture,
                        dense_output=True)
        if not sol.success:
            status = "numerical_error"
            exit_period = None
        elif len(sol.t_events[0]):
            status = "escaped"
            exit_period = float(sol.t_events[0][0] / (2 * math.pi))
        else:
            status = "retained"
            exit_period = None
        radius = np.hypot(sol.y[0], sol.y[1])
        # Midpoint samples of each accepted dense step reduce endpoint-only
        # underestimation of max radius without changing escape classification.
        if len(sol.t) > 1:
            mid = sol.sol((sol.t[:-1] + sol.t[1:]) / 2)
            max_rho = float(max(np.max(radius), np.max(np.hypot(mid[0], mid[1]))))
        else:
            max_rho = float(np.max(radius))
        endpoint = [float(v) for v in sol.y[:, -1]]
        return {
            "mu": float(mu), "phase_rad": float(phase), "phase_deg": float(np.degrees(phase)),
            "model": model, "eta": float(eta), "index": int(index),
            "alpha": alpha, "beta": beta, "kappa": kappa,
            "status_100": status, "exit_periods_100": exit_period,
            "max_rho_100": max(max_rho, 1.0 if status == "escaped" else max_rho),
            "state_100": endpoint, "x0": float(y0[0]), "y0": float(y0[1]),
            "vx0": float(y0[2]), "vy0": float(y0[3]),
            "field": "ideal_harmonic_V13.2", "integrator": "DOP853",
            "rtol": RTOL, "atol": ATOL, "max_step_tau": MAX_STEP,
            "nfev": int(sol.nfev), "source": "new_V13.2",
        }
    except Exception as exc:  # retain a machine-readable unresolved outcome
        return {
            "mu": float(mu), "phase_rad": float(phase), "phase_deg": float(np.degrees(phase)),
            "model": model, "eta": float(eta), "index": int(index),
            "alpha": alpha, "beta": beta, "kappa": kappa,
            "status_100": "numerical_error", "exit_periods_100": None,
            "max_rho_100": float("nan"), "state_100": None,
            "x0": float(y0[0]), "y0": float(y0[1]), "vx0": float(y0[2]), "vy0": float(y0[3]),
            "field": "ideal_harmonic_V13.2", "integrator": "DOP853",
            "rtol": RTOL, "atol": ATOL, "max_step_tau": MAX_STEP,
            "nfev": None, "source": f"numerical_error:{type(exc).__name__}:{exc}",
        }


def load_initials() -> np.ndarray:
    p = OLD / "initial_conditions.csv"
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.asarray([[float(r[k]) for k in
                        ("x0_rho", "y0_rho", "vx0_rho_per_tau", "vy0_rho_per_tau")]
                       for r in rows], dtype=float)


def load_reused() -> list[dict]:
    p = OLD / "trajectory_results.csv"
    selected = []
    with p.open(newline="", encoding="utf-8") as f:
        for src in csv.DictReader(f):
            mu, eta = round(float(src["mu"]), 8), round(float(src["eta"]), 8)
            model = src["model"]
            if mu not in {1.0, 1.05, 1.1}:
                continue
            if (model, eta) not in MODELS:
                continue
            row = dict(src)
            row.update({"mu": mu, "eta": eta, "phase_rad": float(src["phase_rad"]),
                        "phase_deg": float(src["phase_deg"]), "index": int(src["index"]),
                        "status_100": src["status_100"], "state_100": None,
                        "field": "ideal_harmonic_V13.1_exact_match", "source": "reused_V13.1"})
            selected.append(row)
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def summarize(rows: list[dict]) -> list[dict]:
    from three_phase_trap.mass_filter_v13 import linear_status

    output = []
    masses = sorted({round(float(r["mu"]), 8) for r in rows})
    for mu in masses:
        floquet = linear_status(ALPHA0 / mu, BETA0 / mu, tolerance=1e-8)
        for phase in PHASES:
            group = [r for r in rows if round(float(r["mu"]), 8) == mu and
                     abs(float(r["phase_rad"]) - phase) < 1e-9]
            by_model = {(r["model"], round(float(r["eta"]), 8), int(r["index"])): r for r in group}
            for model, eta in MODELS:
                rs = [r for r in group if r["model"] == model and round(float(r["eta"]), 8) == eta]
                resolved = [r for r in rs if r["status_100"] in ("retained", "escaped")]
                n_all, n = len(rs), len(resolved)
                k = sum(r["status_100"] == "retained" for r in resolved)
                lo, hi = wilson(k, n)
                all_lo = k / n_all if n_all else float("nan")
                all_hi = (k + n_all - n) / n_all if n_all else float("nan")
                q = [r for r in group if r["model"] == "Q"]
                q_by_idx = {int(r["index"]): r for r in q}
                pair = [(int(r["status_100"] == "retained") -
                         int(q_by_idx[int(r["index"])]["status_100"] == "retained"), r)
                        for r in resolved if int(r["index"]) in q_by_idx and
                        r["status_100"] in ("retained", "escaped") and
                        q_by_idx[int(r["index"])]["status_100"] in ("retained", "escaped")]
                diffs = np.asarray([v for v, _ in pair], dtype=float)
                pd = float(diffs.mean()) if len(diffs) else float("nan")
                se = float(diffs.std(ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
                out = {
                    "mu": mu, "phase_deg": float(np.degrees(phase)), "horizon_rf": 100,
                    "model": model, "eta": eta, "n_total": n_all, "n_resolved": n,
                    "n_unresolved": n_all - n, "n_retained": k,
                    "retention_resolved": k / n if n else float("nan"),
                    "retention_wilson95_low": lo, "retention_wilson95_high": hi,
                    "retention_lower_all_n": all_lo, "retention_upper_all_n": all_hi,
                    "paired_n_vs_Q": len(diffs), "paired_unresolved_vs_Q": n_all - len(diffs),
                    "paired_diff_vs_Q": pd, "paired_diff_se": se,
                    "paired_diff_ci95_low": max(-1., pd - 1.96 * se) if np.isfinite(se) else float("nan"),
                    "paired_diff_ci95_high": min(1., pd + 1.96 * se) if np.isfinite(se) else float("nan"),
                    "paired_gains_vs_Q": int(np.sum(diffs == 1)),
                    "paired_losses_vs_Q": int(np.sum(diffs == -1)),
                    "median_exit_periods": float(np.median([float(r["exit_periods_100"]) for r in rs
                        if r["status_100"] == "escaped" and r.get("exit_periods_100") not in (None, "")]))
                        if any(r["status_100"] == "escaped" and r.get("exit_periods_100") not in (None, "") for r in rs)
                        else float("nan"),
                    "linear_floquet_center_status": floquet["status"],
                    "linear_trace_x": floquet["trace_x"], "linear_trace_y": floquet["trace_y"],
                    "linear_margin_x": floquet["margin_x"], "linear_margin_y": floquet["margin_y"],
                }
                output.append(out)
    # Add directly paired RO-SO effects to rows for RO.
    index = {(round(float(r["mu"]), 8), float(r["phase_deg"]), r["model"]): r for r in output}
    for r in output:
        if r["model"] != "RO":
            r.update({"paired_diff_RO_minus_SO": float("nan"), "RO_SO_gains": None,
                      "RO_SO_losses": None, "RO_SO_n": 0})
            continue
        key = (round(float(r["mu"]), 8), float(r["phase_deg"]))
        source = [x for x in rows if round(float(x["mu"]), 8) == key[0] and
                  abs(float(x["phase_deg"]) - key[1]) < 1e-8 and x["model"] == "RO"]
        so = {int(x["index"]): x for x in rows if round(float(x["mu"]), 8) == key[0] and
              abs(float(x["phase_deg"]) - key[1]) < 1e-8 and x["model"] == "SO"}
        ds = [int(x["status_100"] == "retained") - int(so[int(x["index"])]["status_100"] == "retained")
              for x in source if int(x["index"]) in so and x["status_100"] in ("retained", "escaped")
              and so[int(x["index"])]["status_100"] in ("retained", "escaped")]
        r["RO_SO_n"] = len(ds)
        diff = float(np.mean(ds)) if ds else float("nan")
        se = float(np.std(ds, ddof=1) / math.sqrt(len(ds))) if len(ds) > 1 else float("nan")
        r["paired_diff_RO_minus_SO"] = diff
        r["paired_diff_RO_minus_SO_se"] = se
        r["paired_diff_RO_minus_SO_ci95_low"] = max(-1., diff - 1.96*se) if np.isfinite(se) else float("nan")
        r["paired_diff_RO_minus_SO_ci95_high"] = min(1., diff + 1.96*se) if np.isfinite(se) else float("nan")
        r["RO_SO_gains"] = sum(d == 1 for d in ds)
        r["RO_SO_losses"] = sum(d == -1 for d in ds)
    return output


def plot_summary(summary: list[dict]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    colors = {"Q": "#202020", "SO": "#d97706", "RO": "#2563eb"}
    labels = {"Q": "Q", "SO": "Q + SO, η = −0.05", "RO": "Q + RO, η = −0.05"}
    for ax, phase in zip(axes, (0., 90.)):
        for model in ("Q", "SO", "RO"):
            rs = sorted((r for r in summary if r["model"] == model and r["phase_deg"] == phase),
                        key=lambda r: r["mu"])
            x = np.array([r["mu"] for r in rs])
            p = np.array([r["retention_resolved"] for r in rs])
            low = np.array([r["retention_wilson95_low"] for r in rs])
            high = np.array([r["retention_wilson95_high"] for r in rs])
            ax.errorbar(x, p, yerr=np.vstack((p-low, high-p)), fmt="o", ms=4,
                        capsize=2.4, lw=1, color=colors[model], label=labels[model])
        ax.set_title(f"Начальная RF-фаза {phase:.0f}°")
        ax.set_xlabel("μ = (M/Q)/(M₀/Q₀)")
        ax.grid(alpha=.25)
        ax.set_ylim(-.04, 1.04)
    axes[0].set_ylabel("Доля удержания за 100 RF периодов")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("V13.2 Stage A: идеальная гармоническая модель, N=32 общих Sobol стартов")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"stage_a_mass_scan.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    initials = load_initials()
    write_csv(OUT / "stage_a_initial_conditions.csv", [
        {"index": i, "x0_rho": x[0], "y0_rho": x[1],
         "vx0_rho_per_tau": x[2], "vy0_rho_per_tau": x[3],
         "rho0": float(np.hypot(x[0], x[1])),
         "angular_momentum_z": float(x[0]*x[3]-x[1]*x[2])}
        for i, x in enumerate(initials)])

    # Persist the full protocol before starting any long-running workers.
    manifest = {
        "stage": "V13.2-A-ideal-mass-scan", "field": "V13 ideal leading harmonics; no FEM higher harmonics",
        "mass_grid": MU.tolist(), "phases_rad": list(PHASES), "models": [[m, e] for m, e in MODELS],
        "parameters_at_mu1": {"alpha": ALPHA0, "beta": BETA0, "eta_octupole": -0.05},
        "mass_scaling": "alpha,beta,kappa all scale as 1/mu; kappa=eta*beta; Q has kappa=0",
        "initial_conditions_source": "V13.1 exact common seed-131 scrambled Sobol sample; 32 states",
        "initial_domain": {"x0_rho": [-.25, .25], "y0_rho": [-.25, .25],
                           "vx0_rho_per_tau": [-.04, .04], "vy0_rho_per_tau": [-.04, .04]},
        "RF_phase_convention": "tau+phase at injection; phase 0 or pi/2; same states paired across models",
        "aperture_rho": 1.0, "horizon_rf_periods": 100,
        "integrator": {"method": "DOP853", "rtol": RTOL, "atol": ATOL,
                       "max_step_tau": MAX_STEP, "max_step_rf_period_fraction": 1/64},
        "exact_reuse": "V13.1 ideal-pilot trajectories at mu=1.00,1.05,1.10 for matching phase/model/state; no reintegration",
        "reused_v13_1_trajectory_csv_sha256": hashlib.sha256((OLD / "trajectory_results.csv").read_bytes()).hexdigest(),
        "interpretation": "finite-time transverse retention only; not finite-length filter transmission",
        "workers": args.workers,
    }
    (OUT / "stage_a_parameters.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checkpoint = OUT / "stage_a_checkpoint.jsonl"
    done = []
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.append(json.loads(line))
    completed = {stable_key(r) for r in done}
    reused = load_reused()
    reused_keys = {stable_key(r) for r in reused}
    tasks = []
    for mu in MU:
        if round(float(mu), 8) in {1.0, 1.05, 1.1}:
            continue
        for phase in PHASES:
            for model, eta in MODELS:
                for index, state in enumerate(initials):
                    key = (round(float(mu), 8), round(float(phase), 12), model, round(float(eta), 8), index)
                    if key not in completed:
                        tasks.append((float(mu), float(phase), model, eta, index, state.tolist()))
    print(f"Stage A tasks remaining: {len(tasks)}; reused V13.1 exact trajectories: {len(reused_keys)}; workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, checkpoint.open("a", encoding="utf-8") as ck:
        futures = [pool.submit(advance, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            row = future.result()
            done.append(row)
            ck.write(json.dumps(row, separators=(",", ":")) + "\n")
            ck.flush()
            if count % 256 == 0 or count == len(futures):
                print(f"completed {count}/{len(futures)}", flush=True)

    rows = reused + done
    # Select exactly the Stage A mass/phase/model cells, keeping unresolved outcomes explicit.
    allowed = {(round(float(mu), 8), round(float(phase), 12), model, round(float(eta), 8))
               for mu in MU for phase in PHASES for model, eta in MODELS}
    rows = [r for r in rows if (round(float(r["mu"]), 8), round(float(r["phase_rad"]), 12),
                                 r["model"], round(float(r["eta"]), 8)) in allowed]
    rows.sort(key=lambda r: (float(r["mu"]), float(r["phase_rad"]), r["model"], int(r["index"])))
    write_csv(OUT / "stage_a_trajectory_results.csv", rows)
    summary = summarize(rows)
    write_csv(OUT / "stage_a_retention_summary.csv", summary)
    plot_summary(summary)
    model_sha = hashlib.sha256((ROOT / "src/three_phase_trap/mass_filter_v13.py").read_bytes()).hexdigest()
    source_sha = hashlib.sha256((OLD / "trajectory_results.csv").read_bytes()).hexdigest()
    config = {
        "stage": "V13.2-A-ideal-mass-scan", "field": "V13 ideal leading harmonics; no FEM higher harmonics",
        "mass_grid": MU.tolist(), "phases_rad": list(PHASES), "models": [[m, e] for m, e in MODELS],
        "parameters_at_mu1": {"alpha": ALPHA0, "beta": BETA0, "eta_octupole": -0.05},
        "mass_scaling": "alpha,beta,kappa all scale as 1/mu; kappa=eta*beta; Q has kappa=0",
        "initial_conditions_source": "V13.1 exact common seed-131 scrambled Sobol sample; 32 states",
        "initial_domain": {"x0_rho": [-.25, .25], "y0_rho": [-.25, .25],
                           "vx0_rho_per_tau": [-.04, .04], "vy0_rho_per_tau": [-.04, .04]},
        "RF_phase_convention": "tau+phase at injection; phase 0 or pi/2; same states paired across models",
        "aperture_rho": 1.0, "horizon_rf_periods": 100,
        "integrator": {"method": "DOP853", "rtol": RTOL, "atol": ATOL,
                       "max_step_tau": MAX_STEP, "max_step_rf_period_fraction": 1/64},
        "exact_reuse": "V13.1 ideal-pilot trajectories at mu=1.00,1.05,1.10 for matching phase/model/state; no reintegration",
        "mass_filter_model_sha256": model_sha, "reused_v13_1_trajectory_csv_sha256": source_sha,
        "interpretation": "finite-time transverse retention only; not finite-length filter transmission",
        "workers": args.workers,
    }
    (OUT / "stage_a_parameters.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} trajectory rows and {len(summary)} summary rows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
