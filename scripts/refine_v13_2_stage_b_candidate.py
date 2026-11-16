"""Independent paired refinement of the only coarse RO-over-SO candidate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability/local_candidate_refinement"
ALPHA_CENTER = 0.026571428571428572
BETA_CENTER = 0.1572222222222222
ALPHA_STEP = 0.001428571428571427
BETA_STEP = 0.00305555555555556
ALPHAS = ALPHA_CENTER + np.arange(-1, 2) * ALPHA_STEP
BETAS = BETA_CENTER + np.arange(-1, 2) * BETA_STEP
PHASES = (0.0, math.pi / 2)
ETA = -0.05
N_INITIAL = 32
SEED = 20261116
N_PERIODS = 100
MAX_STEP = 2 * math.pi / 64
RTOL, ATOL = 2e-10, 2e-12


def make_initials() -> np.ndarray:
    u = qmc.Sobol(d=4, scramble=True, seed=SEED).random_base2(5)
    return np.column_stack((0.5 * (u[:, 0] - 0.5), 0.5 * (u[:, 1] - 0.5),
                            0.08 * (u[:, 2] - 0.5), 0.08 * (u[:, 3] - 0.5)))


def run_task(task: tuple) -> dict:
    alpha, beta, phase, model, index, state = task
    from three_phase_trap.mass_filter_v13 import acceleration

    kappa = 0.0 if model == "Q" else ETA * beta
    standing = model == "SO"
    y0 = np.asarray(state, float)

    def rhs(t, y):
        return np.r_[y[2:], acceleration(y[0], y[1], t + phase, kappa, beta, alpha,
                                          standing_octupole=standing)]

    def aperture(t, y):
        return float(np.hypot(y[0], y[1]) - 1.0)
    aperture.terminal = True
    aperture.direction = 1
    sol = solve_ivp(rhs, (0., 2 * math.pi * N_PERIODS), y0, method="DOP853",
                    rtol=RTOL, atol=ATOL, max_step=MAX_STEP, dense_output=True,
                    events=aperture)
    if not sol.success:
        status, exit_periods = "numerical_error", None
    elif len(sol.t_events[0]):
        status, exit_periods = "escaped", float(sol.t_events[0][0] / (2 * math.pi))
    else:
        status, exit_periods = "retained", None
    radius = np.hypot(sol.y[0], sol.y[1])
    if len(sol.t) > 1:
        mid = sol.sol((sol.t[:-1] + sol.t[1:]) / 2)
        rmax = float(max(np.max(radius), np.max(np.hypot(mid[0], mid[1]))))
    else:
        rmax = float(np.max(radius))
    if status == "escaped":
        rmax = max(rmax, 1.0)
    return {"alpha": alpha, "beta": beta, "kappa": kappa, "eta": ETA,
            "phase_rad": phase, "phase_deg": float(np.degrees(phase)),
            "model": model, "index": index, "status_100": status,
            "exit_periods_100": exit_periods, "max_rho_100": rmax,
            "x0": float(y0[0]), "y0": float(y0[1]), "vx0": float(y0[2]), "vy0": float(y0[3]),
            "nfev": int(sol.nfev), "integrator": "DOP853", "rtol": RTOL,
            "atol": ATOL, "max_step_tau": MAX_STEP}


def key(row: dict) -> tuple:
    return (round(float(row["alpha"]), 12), round(float(row["beta"]), 12),
            round(float(row["phase_rad"]), 12), row["model"], int(row["index"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    groups = sorted({(float(r["alpha"]), float(r["beta"]), float(r["phase_rad"])) for r in rows})
    for alpha, beta, phase in groups:
        cell = [r for r in rows if float(r["alpha"]) == alpha and float(r["beta"]) == beta
                and float(r["phase_rad"]) == phase]
        by = {(r["model"], int(r["index"])): r for r in cell}
        status = {m: {i: by[(m, i)]["status_100"] for i in range(N_INITIAL)} for m in ("Q", "SO", "RO")}
        retained = {m: sum(v == "retained" for v in status[m].values()) for m in ("Q", "SO", "RO")}
        row = {"alpha": alpha, "beta": beta, "phase_deg": float(np.degrees(phase)), "eta": ETA,
               "n": N_INITIAL, "floquet_status": "", "mean_max_rho_RO": float(np.mean([r["max_rho_100"] for r in cell if r["model"] == "RO"]))}
        from three_phase_trap.mass_filter_v13 import linear_status
        row["floquet_status"] = linear_status(alpha, beta)["status"]
        for model in ("Q", "SO", "RO"):
            row[f"n_retained_{model}"] = retained[model]
            row[f"P_{model}"] = retained[model] / N_INITIAL
        for name, other in (("RO_minus_Q", "Q"), ("RO_minus_SO", "SO")):
            diffs = np.asarray([int(status["RO"][i] == "retained") - int(status[other][i] == "retained")
                                for i in range(N_INITIAL)], dtype=float)
            mean = float(np.mean(diffs))
            se = float(np.std(diffs, ddof=1) / math.sqrt(N_INITIAL))
            row[f"paired_{name}"] = mean
            row[f"paired_{name}_normal95_low"] = max(-1., mean - 1.96 * se)
            row[f"paired_{name}_normal95_high"] = min(1., mean + 1.96 * se)
            row[f"gains_{name}"] = int(np.sum(diffs == 1))
            row[f"losses_{name}"] = int(np.sum(diffs == -1))
        output.append(row)
    return output


def plot(summary: list[dict]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ri, phase in enumerate((0., 90.)):
        for ci, model in enumerate(("Q", "SO", "RO")):
            ax = axes[ri, ci]
            rr = [r for r in summary if r["phase_deg"] == phase]
            sc = ax.scatter([r["beta"] for r in rr], [r["alpha"] for r in rr],
                            c=[r[f"P_{model}"] for r in rr], vmin=0, vmax=1,
                            cmap="viridis", marker="s", s=230, edgecolors="white", linewidths=.6)
            ax.set_title(f"{model}, RF фаза {phase:.0f}°")
            ax.set_xlabel("β")
            ax.set_ylabel("α")
            ax.grid(alpha=.2)
            fig.colorbar(sc, ax=ax, label="P(100 периодов)")
    fig.suptitle("Локальное независимое уточнение кандидата RO−SO; η=−0.05, N=32")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"retention_local_refinement.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    initials = make_initials()
    with (OUT / "initial_conditions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "x0_rho", "y0_rho", "vx0_rho_per_tau", "vy0_rho_per_tau"])
        w.writerows([[i, *x] for i, x in enumerate(initials)])
    params = {"purpose": "Independent local refinement of the only coarse positive RO-SO candidate",
              "center": {"alpha": ALPHA_CENTER, "beta": BETA_CENTER},
              "alpha_grid": ALPHAS.tolist(), "beta_grid": BETAS.tolist(), "eta": ETA,
              "initial_conditions": "Independent scrambled Sobol N=32; x,y uniform[-0.25,0.25], vx,vy uniform[-0.04,0.04]",
              "seed": SEED, "phases_rad": list(PHASES), "models": ["Q", "SO", "RO"],
              "horizon_periods": N_PERIODS, "aperture_rho": 1.0,
              "integrator": {"method": "DOP853", "rtol": RTOL, "atol": ATOL, "max_step_tau": MAX_STEP},
              "workers": args.workers, "physical_interpretation": "finite-time transverse retention only"}
    (OUT / "parameters.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    tasks = [(float(a), float(b), float(phase), model, i, initials[i].tolist())
             for a in ALPHAS for b in BETAS for phase in PHASES
             for model in ("Q", "SO", "RO") for i in range(N_INITIAL)]
    checkpoint = OUT / "checkpoint.jsonl"
    done = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines() if line.strip()] if checkpoint.exists() else []
    seen = {key(r) for r in done}
    pending = [t for t in tasks if (round(t[0], 12), round(t[1], 12), round(t[2], 12), t[3], t[4]) not in seen]
    print(f"Local refinement: {len(pending)} remaining of {len(tasks)}; workers={args.workers}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool, checkpoint.open("a", encoding="utf-8") as f:
            futures = [pool.submit(run_task, t) for t in pending]
            for n, future in enumerate(as_completed(futures), 1):
                result = future.result()
                done.append(result)
                f.write(json.dumps(result, separators=(",", ":")) + "\n")
                f.flush()
                if n % 300 == 0 or n == len(futures):
                    print(f"Local refinement completed {n}/{len(futures)}", flush=True)
    raw = sorted(done, key=key)
    write_csv(OUT / "trajectory_results.csv", raw)
    summary = summarize(raw)
    write_csv(OUT / "retention_summary.csv", summary)
    plot(summary)


if __name__ == "__main__":
    main()
