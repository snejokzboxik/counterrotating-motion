"""Conditional mechanical Lyapunov spectrum along exact Local15 RF orbits.

Two frozen S01 starts are integrated with RK4/96 together with the exact
4x4 tangent system. QR is accumulated each primary period. RF phases are held
fixed, so the result is a conditional spectrum and excludes phase directions.
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "results/frozen_local15_single_particle_panel.csv"
OUT = ROOT / "results/nonautonomous_stability"
P = {
    "A": 0.03716194907442204,
    "B": 0.1626938199343468,
    "nu": 3.1118029452860356,
    "beta": 0.017273303124515367,
    "species": {
        "species1": {"alpha": 1.0, "lambda": 0.20975541409005094},
        "species2": {"alpha": 0.34126014359705537,
                     "lambda": 0.12617311138613094},
    },
}
PERIODS = 15_000
STEPS_PER_PERIOD = 96
QR_BLOCK_PERIODS = 500


def load_start() -> dict:
    with PANEL.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["start_id"] == "S01")
    return {key: float(row[key]) for key in
            ("x0", "y0", "vx0", "vy0", "phase1", "phase2")}


def rhs_and_jacobian(tau: float, y: np.ndarray, alpha: float, lam: float,
                     phase1: float, phase2: float) -> tuple[np.ndarray, np.ndarray]:
    x, yc, vx, vy = map(float, y)
    z = complex(x, yc)
    c2 = P["A"] * np.exp(1j*phase1-1j*tau)
    c4 = P["B"] * np.exp(1j*phase2-1j*P["nu"]*tau)
    g0 = 2.0*c2*z
    g1 = 4.0*c4*np.conjugate(z)**3
    grad = np.array([np.real(g0+g1), np.real(1j*g0-1j*g1)])
    acc = -alpha*grad + alpha*P["beta"]*np.array([x, yc]) - lam*np.array([vx, vy])

    hxx = 2.0*c2 + 12.0*c4*np.conjugate(z)**2
    hxy = 2.0j*c2 - 12.0j*c4*np.conjugate(z)**2
    hessian = np.array([[np.real(hxx), np.real(hxy)],
                        [np.real(hxy), -np.real(hxx)]])
    jac = np.zeros((4, 4))
    jac[:2, 2:] = np.eye(2)
    jac[2:, :2] = -alpha*hessian + alpha*P["beta"]*np.eye(2)
    jac[2:, 2:] = -lam*np.eye(2)
    return np.array([vx, vy, *acc]), jac


def integrate_species(name: str, initial: dict) -> dict:
    alpha = P["species"][name]["alpha"]
    lam = P["species"][name]["lambda"]
    y = np.array([initial["x0"], initial["y0"],
                  initial["vx0"], initial["vy0"]], dtype=float)
    M = np.eye(4)
    sums = np.zeros(4)
    h = 2*np.pi/STEPS_PER_PERIOD
    panel_sha = hashlib.sha256(PANEL.read_bytes()).hexdigest()
    checkpoint = OUT/f"{name}_s01_qr_checkpoint.npz"
    records = []
    start_step = 0
    t0 = time.perf_counter()
    maxrho = float(np.hypot(*y[:2]))
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["panel_sha256"].item()) != panel_sha or \
               str(saved["species"].item()) != name:
                raise ValueError(f"checkpoint metadata mismatch: {checkpoint}")
            start_step = int(saved["step"])
            y = saved["y"].copy()
            M = saved["M"].copy()
            sums = saved["sums"].copy()
            maxrho = float(saved["maxrho"])
            periods = saved["record_periods"].copy()
            rhos = saved["record_rhos"].copy()
            maxrhos = saved["record_max_rhos"].copy()
            thetas = saved["record_thetas"].copy()
            exponents = saved["record_exponents"].copy()
            if start_step % STEPS_PER_PERIOD != 0 or start_step > PERIODS*STEPS_PER_PERIOD:
                raise ValueError(f"invalid checkpoint step: {start_step}")
        for p, r, mr, theta, ev in zip(periods, rhos, maxrhos, thetas, exponents):
            ex_tau = ev
            records.append({
                "period": int(p), "rho": float(r), "theta_wrapped": float(theta),
                "max_rho": float(mr),
                "conditional_lyapunov_per_tau": ex_tau.tolist(),
                "conditional_lyapunov_per_primary_period": (2*np.pi*ex_tau).tolist(),
            })
    else:
        periods = np.array([], dtype=np.int64)
        rhos = np.array([], dtype=float)
        maxrhos = np.array([], dtype=float)
        thetas = np.array([], dtype=float)
        exponents = np.empty((0, 4), dtype=float)

    def derivative(t: float, state: np.ndarray, tangent: np.ndarray):
        f, J = rhs_and_jacobian(t, state, alpha, lam,
                                initial["phase1"], initial["phase2"])
        return f, J@tangent

    for step in range(start_step+1, PERIODS*STEPS_PER_PERIOD+1):
        tau = (step-1)*h
        k1y, k1m = derivative(tau, y, M)
        k2y, k2m = derivative(tau+h/2, y+h*k1y/2, M+h*k1m/2)
        k3y, k3m = derivative(tau+h/2, y+h*k2y/2, M+h*k2m/2)
        k4y, k4m = derivative(tau+h, y+h*k3y, M+h*k3m)
        y += (h/6)*(k1y+2*k2y+2*k3y+k4y)
        M += (h/6)*(k1m+2*k2m+2*k3m+k4m)
        rho = float(np.hypot(*y[:2]))
        maxrho = max(maxrho, rho)
        if step % STEPS_PER_PERIOD == 0:
            period = step//STEPS_PER_PERIOD
            Q, R = np.linalg.qr(M)
            diagonal = np.abs(np.diag(R))
            if np.any(diagonal <= 0) or np.any(~np.isfinite(diagonal)):
                raise FloatingPointError(f"invalid QR diagonal at period {period}")
            sums += np.log(diagonal)
            M = Q
            if period % QR_BLOCK_PERIODS == 0:
                exponents_tau = sums/(2*np.pi*period)
                records.append({
                    "period": period,
                    "rho": rho,
                    "theta_wrapped": float(np.arctan2(y[1], y[0])),
                    "max_rho": maxrho,
                    "conditional_lyapunov_per_tau": exponents_tau.tolist(),
                    "conditional_lyapunov_per_primary_period":
                        (2*np.pi*exponents_tau).tolist(),
                })
                periods = np.append(periods, period)
                rhos = np.append(rhos, rho)
                maxrhos = np.append(maxrhos, maxrho)
                thetas = np.append(thetas, float(np.arctan2(y[1], y[0])))
                exponents = np.vstack((exponents, exponents_tau))
                temporary = checkpoint.with_name(checkpoint.stem+".tmp.npz")
                with temporary.open("wb") as file:
                    np.savez_compressed(
                        file, step=step, y=y, M=M, sums=sums, maxrho=maxrho,
                        record_periods=periods, record_rhos=rhos,
                        record_max_rhos=maxrhos, record_exponents=exponents,
                        record_thetas=thetas,
                        panel_sha256=panel_sha, species=name)
                os.replace(temporary, checkpoint)
    elapsed = time.perf_counter()-t0
    windows = {}
    for horizon in (5000, 10000, 15000):
        row = next(q for q in records if q["period"] == horizon)
        windows[str(horizon)] = row["conditional_lyapunov_per_tau"]
    return {
        "species": name,
        "initial_start_id": "S01",
        "initial_condition": initial,
        "alpha": alpha,
        "lambda": lam,
        "periods": PERIODS,
        "steps_per_period": STEPS_PER_PERIOD,
        "step_count": PERIODS*STEPS_PER_PERIOD,
        "computed_steps_this_run": (PERIODS*STEPS_PER_PERIOD-start_step),
        "elapsed_seconds": elapsed,
        "max_rho": maxrho,
        "windows_lyapunov_exponents_per_tau": windows,
        "periodic_diagnostics": records,
        "checkpoint_file": checkpoint.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/"local15_s01_conditional_qr_15000.json"
    if target.exists():
        archived = json.loads(target.read_text(encoding="utf-8"))
        if archived.get("status") == "completed":
            print(f"already completed: {target.relative_to(ROOT).as_posix()}")
            return
    initial = load_start()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(integrate_species, name, initial)
                   for name in ("species1", "species2")]
        species = {}
        for future in futures:
            result = future.result()
            species[result["species"]] = result
    result = {
        "status": "completed",
        "method": "RK4/96 exact ideal two-mode Local15 RF plus analytic 4x4 mechanical tangent Jacobian; QR every primary period",
        "conditional_spectrum": True,
        "rf_phase_variations_included": False,
        "coulomb_and_mfs_higher_harmonics_included": False,
        "parameters": P,
        "panel_sha256": hashlib.sha256(PANEL.read_bytes()).hexdigest(),
        "initial_panel_row": initial,
        "periods": PERIODS,
        "steps_per_period": STEPS_PER_PERIOD,
        "total_steps_per_species": PERIODS*STEPS_PER_PERIOD,
        "total_particle_periods": 2*PERIODS,
        "qr_block_periods": QR_BLOCK_PERIODS,
        "species": species,
        "limitations": [
            "conditional mechanical exponents use fixed externally prescribed RF phases and omit RF phase tangent directions",
            "one retained S01 orbit per species; this does not characterize all starts or basin measures",
            "finite-time exponents through 15,000 periods do not prove asymptotic stability",
            "ideal k=+2,-4 field only; no Coulomb, MFS harmonics, or physical drag correction",
        ],
    }
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({name: {
        "exponents_per_tau": result["species"][name]["windows_lyapunov_exponents_per_tau"],
        "max_rho": result["species"][name]["max_rho"],
        "elapsed_seconds": result["species"][name]["elapsed_seconds"],
    } for name in result["species"]}, indent=2))


if __name__ == "__main__":
    main()
