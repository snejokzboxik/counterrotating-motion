"""Small, paired 2D first-exit mass pilot; no axial/transmission claim."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc

from three_phase_trap.mass_filter_v13 import acceleration, linear_status


def integrate(initial, kappa, beta, alpha, horizon, max_step_rf, standing):
    def rhs(tau, state):
        return np.r_[state[2:], acceleration(state[0], state[1], tau,
                                              kappa, beta, alpha,
                                              standing_octupole=standing)]

    def exit_event(tau, state):
        return np.hypot(state[0], state[1]) - 1.0

    exit_event.terminal = True
    exit_event.direction = 1
    result = solve_ivp(rhs, (0, 2*np.pi*horizon), initial,
                       method="DOP853", rtol=2e-10, atol=2e-12,
                       max_step=2*np.pi*max_step_rf, events=exit_event)
    if not result.success:
        return "numerical_error", np.nan
    if len(result.t_events[0]):
        return "escaped", float(result.t_events[0][0] / (2*np.pi))
    return "retained", float(horizon)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sobol-power", type=int, default=4)
    parser.add_argument("--horizon", type=float, default=20.)
    parser.add_argument("--max-step-rf", type=float, default=1/64)
    parser.add_argument("--seed", type=int, default=1307)
    parser.add_argument("--alpha0", type=float, default=0.027)
    parser.add_argument("--beta0", type=float, default=0.175)
    parser.add_argument("--mass-ratios", type=float, nargs="+",
                        default=[0.88, 0.91, 0.94, 0.97, 1., 1.03, 1.06, 1.09, 1.12])
    parser.add_argument("--models", nargs="+", choices=["Q", "Q+SO", "Q+RO"],
                        default=["Q", "Q+SO", "Q+RO"])
    parser.add_argument("--eta-values", type=float, nargs="+", default=[-0.2, 0.2])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Near the conventional first-zone apex (a ≈ 0.237, |q_M| ≈ 0.706).
    alpha0, beta0 = args.alpha0, args.beta0
    mass_ratios = args.mass_ratios
    eta_values = args.eta_values
    # |F_oct/F_quad| = 2|eta| rho^2; at rho=.5 this is 0.1.
    raw = qmc.Sobol(d=4, scramble=True, seed=args.seed).random_base2(args.sobol_power)
    initials = (raw - 0.5) * np.array([0.5, 0.5, 0.08, 0.08])
    config = dict(model="ideal_dimensionless_v13", alpha0=alpha0, beta0=beta0,
                  mass_ratios=mass_ratios, eta_values=eta_values, models=args.models,
                  n_initial=len(initials), initial_box=[[-.25,.25],[-.25,.25],[-.04,.04],[-.04,.04]],
                  sobol_scrambled=True, seed=args.seed, rf_phase=0.,
                  aperture_rho=1., horizon_rf_periods=args.horizon,
                  integrator="solve_ivp DOP853", rtol=2e-10, atol=2e-12,
                  max_step_rf=args.max_step_rf,
                  interpretation="2D finite-time first-exit retention, not physical transmission")
    config_path = args.out / "parameters.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("checkpoint directory has incompatible pilot parameters")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    checkpoint = args.out / "checkpoint.jsonl"
    rows = []
    if checkpoint.exists():
        rows = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines() if line]
    done = {(r["mass_ratio"], r["eta"], r["model"], r["index"]) for r in rows}
    for mass_ratio in mass_ratios:
        alpha, beta = alpha0/mass_ratio, beta0/mass_ratio
        linear = linear_status(alpha, beta)
        for eta in eta_values:
            for model in args.models:
                kappa = 0. if model == "Q" else eta * beta
                for index, initial in enumerate(initials):
                    if (mass_ratio, eta, model, index) in done:
                        continue
                    status, reached = integrate(initial, kappa, beta, alpha,
                                                args.horizon, args.max_step_rf,
                                                model == "Q+SO")
                    row = dict(mass_ratio=mass_ratio, eta=eta, model=model,
                                     index=index, alpha=alpha, beta=beta, kappa=kappa,
                                     linear_status=linear["status"], status=status,
                                     reached_rf_periods=reached,
                                     x0=initial[0], y0=initial[1],
                                     ux0=initial[2], uy0=initial[3])
                    rows.append(row)
                    with checkpoint.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
    with (args.out / "trajectory_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
