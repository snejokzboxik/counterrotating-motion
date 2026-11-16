"""Selected full h50 P4 checks of the V13.2 nonlinear-map candidate and neighbor."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_v13_1_fem_pilot import initialize  # noqa: E402
import run_v13_1_fem_pilot as fem_pilot  # noqa: E402

BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability"
LOCAL = BASE / "local_candidate_refinement"
OUT = BASE / "stage_b_full_fem_h50_p4_controls"
R = 0.008
FREQUENCY = 100e3
OMEGA = 2 * math.pi * FREQUENCY
Q_E = 1.602176634e-19
C4, C2 = 0.5413395699439091, 0.7718255558274
ETA = -0.05
ALPHA_BETA = ((0.025142857142857144, 0.1572222222222222),
              (0.026571428571428572, 0.1572222222222222))
PHASES = (0.0, math.pi / 2)
MODELS = ("Q", "SO", "RO")
N_PERIODS = 100
RTOL, ATOL, STEP_RF = 3e-9, 3e-11, 1 / 64


def simulate(task: tuple) -> dict:
    alpha, beta, phase, model, index, state, u2, vdc, u4 = task
    field4, field2, scale = fem_pilot._F4, fem_pilot._F2, fem_pilot._SCALE

    def rhs(t, z):
        x, y = z[:2]
        xm, ym = R * x, R * y
        i, u, v = field4.find(xm, ym)
        ecx, ecy, esx, esy = field4.local_quadratures(i, u, v)
        gx4c, gy4c = -esx, -esy
        gx4s, gy4s = ecx, ecy
        j, a, b = field2.find(xm, ym)
        _, _, qsx, qsy = field2.local_quadratures(j, a, b)
        gx2, gy2 = -qsx, -qsy
        angle = t + phase
        if model == "SO":
            gx4, gy4 = gx4c * math.cos(angle), gy4c * math.cos(angle)
        elif model == "RO":
            gx4 = gx4c * math.cos(angle) + gx4s * math.sin(angle)
            gy4 = gy4c * math.cos(angle) + gy4s * math.sin(angle)
        else:
            gx4 = gy4 = 0.0
        qdrive = u2 * math.cos(angle) + vdc
        return np.array([z[2], z[3], -scale * (u4 * gx4 + qdrive * gx2),
                         -scale * (u4 * gy4 + qdrive * gy2)])

    def aperture(t, z): return math.hypot(z[0], z[1]) - 1.0
    aperture.terminal = True
    aperture.direction = 1
    y0 = np.asarray(state, float)
    sol = solve_ivp(rhs, (0., 2 * math.pi * N_PERIODS), y0, method="DOP853",
                    rtol=RTOL, atol=ATOL, max_step=2 * math.pi * STEP_RF,
                    events=aperture, dense_output=True)
    if not sol.success:
        status, exitp = "unresolved", None
    elif len(sol.t_events[0]):
        status, exitp = "escaped", float(sol.t_events[0][0] / (2 * math.pi))
    else:
        status, exitp = "retained", None
    rho = np.hypot(sol.y[0], sol.y[1])
    if len(sol.t) > 1:
        mid = sol.sol((sol.t[:-1] + sol.t[1:]) / 2)
        rmax = float(max(np.max(rho), np.max(np.hypot(mid[0], mid[1]))))
    else:
        rmax = float(np.max(rho))
    return {"alpha": alpha, "beta": beta, "eta": ETA, "phase_rad": phase,
            "phase_deg": float(np.degrees(phase)), "model": model, "index": index,
            "field": "full_native_h50_P4_m4_plus_m2", "status_100": status,
            "exit_periods_100": exitp, "max_rho_100": max(1., rmax) if status == "escaped" else rmax,
            "U2_V": u2, "VDC_V": vdc, "U4_V": u4, "integrator": "DOP853",
            "rtol": RTOL, "atol": ATOL, "max_step_rf": STEP_RF, "nfev": int(sol.nfev),
            "x0": float(y0[0]), "y0": float(y0[1]), "vx0": float(y0[2]), "vy0": float(y0[3])}


def key(row: dict) -> tuple:
    return (round(float(row["alpha"]), 12), round(float(row["beta"]), 12),
            round(float(row["phase_rad"]), 12), row["model"], int(row["index"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    initials = list(csv.DictReader((LOCAL / "initial_conditions.csv").open(newline="", encoding="utf-8")))
    states = {int(r["index"]): [float(r[k]) for k in ("x0_rho", "y0_rho", "vx0_rho_per_tau", "vy0_rho_per_tau")] for r in initials}
    params_file = json.loads((ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1/physical_parameters.json").read_text(encoding="utf-8"))
    mass = params_file["M0_Th"] * 1.66053906660e-27
    tasks = []
    for alpha, beta in ALPHA_BETA:
        u2 = 190. * beta / .175
        vdc = 29.3143 * alpha / 0.027000013157894725
        u4 = ETA * u2 * C2 / C4
        for phase in PHASES:
            for model in MODELS:
                for index, state in states.items():
                    tasks.append((alpha, beta, phase, model, index, state, u2, vdc,
                                  u4 if model != "Q" else 0.))
    m4_path = ROOT / "outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz"
    m2_path = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1/fem/quadrupole_h50_P4_sector_solution.npz"
    config = {"field": "full native h50 P4 FEM, reused m=4 and m=2 unit bases on shared mesh",
              "mesh_sha256": "8ea670e6bf5bada2c86756bc98718063fe38cbfd1692c872c4a4e77f59b1a284",
              "basis_hashes": {"m4": hashlib.sha256(m4_path.read_bytes()).hexdigest(),
                               "m2": hashlib.sha256(m2_path.read_bytes()).hexdigest()},
              "reference_mass_Th": params_file["M0_Th"], "R_m": R, "frequency_Hz": FREQUENCY,
              "coefficients": {"c4": C4, "c2": C2, "d2": C2}, "eta": ETA,
              "alpha_beta_points": [list(x) for x in ALPHA_BETA], "models": list(MODELS),
              "phase_rad": list(PHASES), "N_initial": len(states),
              "initial_source": "independent scrambled Sobol seed 20261116 from local_candidate_refinement/initial_conditions.csv",
              "initial_bounds": {"x_y_rho": [-.25, .25], "vx_vy_rho_per_tau": [-.04, .04]},
              "aperture_rho": 1., "horizon_periods": N_PERIODS,
              "integrator": {"method": "DOP853", "rtol": RTOL, "atol": ATOL, "max_step_rf": STEP_RF},
              "workers": args.workers,
              "drive_conversion": "U2=190*beta/0.175; VDC=29.3143*alpha/alpha0; U4=eta*U2*c2/c4; Q=0 V4",
              "interpretation": "selected finite-time transverse FEM checks; not finite-length transmission"}
    (OUT / "parameters.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    ck = OUT / "checkpoint.jsonl"
    done = [json.loads(line) for line in ck.read_text(encoding="utf-8").splitlines() if line.strip()] if ck.exists() else []
    seen = {key(row) for row in done}
    pending = [t for t in tasks if (round(t[0], 12), round(t[1], 12), round(t[2], 12), t[3], t[4]) not in seen]
    print(f"Full FEM controls: {len(pending)} remaining of {len(tasks)}; workers={args.workers}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool, ck.open("a", encoding="utf-8") as f:
            futures = [pool.submit(simulate, task) for task in pending]
            for n, future in enumerate(as_completed(futures), 1):
                row = future.result(); done.append(row)
                f.write(json.dumps(row, separators=(",", ":")) + "\n"); f.flush()
                if n % 48 == 0 or n == len(futures): print(f"FEM controls completed {n}/{len(futures)}", flush=True)
    raw = sorted(done, key=key)
    write_csv(OUT / "trajectory_results.csv", raw)
    summary = []
    for alpha, beta in ALPHA_BETA:
        for phase in PHASES:
            group = [r for r in raw if abs(float(r["alpha"]) - alpha) < 1e-12 and abs(float(r["beta"]) - beta) < 1e-12 and abs(float(r["phase_rad"]) - phase) < 1e-12]
            by = {(r["model"], int(r["index"])): r for r in group}
            for model in MODELS:
                rr = [r for r in group if r["model"] == model]
                k = sum(r["status_100"] == "retained" for r in rr)
                row = {"alpha": alpha, "beta": beta, "eta": ETA, "phase_deg": float(np.degrees(phase)),
                       "model": model, "n": len(rr), "n_retained": k, "P_retention": k / len(rr) if rr else math.nan,
                       "n_unresolved": sum(r["status_100"] not in ("retained", "escaped") for r in rr),
                       "U2_V": rr[0]["U2_V"], "VDC_V": rr[0]["VDC_V"], "U4_V": rr[0]["U4_V"]}
                if model == "RO":
                    for other, label in (("Q", "RO_minus_Q"), ("SO", "RO_minus_SO")):
                        d = [int(by[("RO", i)]["status_100"] == "retained") - int(by[(other, i)]["status_100"] == "retained") for i in range(len(states))]
                        row[f"paired_{label}"] = float(np.mean(d))
                        row[f"gains_{label}"] = sum(x == 1 for x in d)
                        row[f"losses_{label}"] = sum(x == -1 for x in d)
                summary.append(row)
    write_csv(OUT / "retention_summary.csv", summary)


if __name__ == "__main__": main()
