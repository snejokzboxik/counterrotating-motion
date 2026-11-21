"""Paired exact-RF screen of the Local15 species-1 sign transition.

Each worker integrates the same frozen 24-state panel for one frequency and
species. Checkpoints are written every 500 primary periods. The field is the
ideal two-mode model only; no Coulomb or full MFS spectrum is included.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "results/frozen_local15_single_particle_panel.csv"
OUT = ROOT / "results/rotation_sign_scan"
NU_VALUES = [2.1, 2.175, 2.25, 2.325, 2.4]
PERIODS = 3000
STEPS = 96
WINDOW = 500
PARAM = {
    "A": 0.03716194907442204,
    "B": 0.1626938199343468,
    "beta": 0.017273303124515367,
    "species": {
        "species1": {"alpha": 1.0, "lambda": 0.20975541409005094},
        "species2": {"alpha": 0.34126014359705537, "lambda": 0.12617311138613094},
    },
}


def force(t, y, phase1, phase2, nu, alpha, lam):
    x, z, vx, vz = y.T
    r = x + 1j * z
    g1 = 2.0 * PARAM["A"] * np.exp(1j * phase1) * r * np.exp(-1j * t)
    g2 = 4.0 * PARAM["B"] * np.exp(1j * phase2) * np.conjugate(r) ** 3 * np.exp(-1j * nu * t)
    gx = np.real(g1 + g2)
    gy = np.real(1j * g1 - 1j * g2)
    return np.column_stack((vx, vz,
        -alpha * gx + alpha * PARAM["beta"] * x - lam * vx,
        -alpha * gy + alpha * PARAM["beta"] * z - lam * vz))


def reduced_roots(nu, species):
    alpha = PARAM["species"][species]["alpha"]
    lam = PARAM["species"][species]["lambda"]
    k1 = 4 * alpha**2 * PARAM["A"]**2 / (1 + lam**2)
    k2 = 48 * alpha**2 * PARAM["B"]**2 / (nu**2 + lam**2)
    poly = [(k2 / nu)**2, -2 * k1 * k2 / nu - k2,
            k1**2 + alpha * PARAM["beta"] - k1]
    out = []
    for x in np.roots(poly):
        if abs(x.imag) > 1e-10 or not (0 < x.real < 1):
            continue
        rho = float(x.real ** 0.25)
        omega = float(k1 - (k2 / nu) * x.real)
        state = np.array([rho, 0.0, omega])

        def rhs(s):
            rr, vr, om = s
            fr = -k1 * rr - k2 * rr**5
            ft = lam * (k1 * rr - (k2 / nu) * rr**5)
            return np.array([vr, rr * om**2 + fr + alpha * PARAM["beta"] * rr - lam * vr,
                             ft / rr - 2 * vr * om / rr - lam * om])

        h = 1e-6
        jac = np.column_stack([(rhs(state + np.eye(3)[j] * h)
                                - rhs(state - np.eye(3)[j] * h)) / (2 * h)
                               for j in range(3)])
        eig = np.linalg.eigvals(jac)
        out.append({"rho": rho, "omega": omega,
                    "stable_reduced": bool(np.max(eig.real) < 1e-7),
                    "eigenvalues": [[float(v.real), float(v.imag)] for v in eig]})
    return out


def run_one(task):
    nu, species, rows, state0, phase1, phase2 = task
    cfg = PARAM["species"][species]
    name = f"nu_{nu:.3f}_{species}.npz"
    ckpt = OUT / name
    n = len(rows)
    y = state0.copy()
    active = np.ones(n, dtype=bool)
    escaped = np.hypot(y[:, 0], y[:, 1]) >= 1
    active &= ~escaped
    escape_period = np.where(escaped, 0.0, np.nan)
    maxrho = np.hypot(y[:, 0], y[:, 1])
    raw = np.arctan2(y[:, 1], y[:, 0])
    angle = raw.copy()
    rho_samples = [maxrho.copy()]
    angle_samples = [angle.copy()]
    step0 = 0
    if ckpt.exists():
        tmp = ckpt.with_suffix(".tmp.npz")
        source = ckpt
        if tmp.exists():
            with np.load(ckpt, allow_pickle=False) as old_z:
                old_step = int(old_z["step"])
            with np.load(tmp, allow_pickle=False) as tmp_z:
                tmp_step = int(tmp_z["step"])
            if tmp_step > old_step:
                source = tmp
        with np.load(source, allow_pickle=False) as z:
            step0 = int(z["step"])
            y = z["state"].copy()
            active = z["active"].copy()
            escaped = z["escaped"].copy()
            escape_period = z["escape_period"].copy()
            maxrho = z["maxrho"].copy()
            angle_samples = list(z["angle_samples"])
            rho_samples = list(z["rho_samples"])
        if source == tmp:
            tmp.replace(ckpt)
        if step0 > PERIODS * STEPS:
            raise RuntimeError(f"checkpoint exceeds requested horizon: {ckpt}")
        raw = np.arctan2(y[:, 1], y[:, 0])
        angle = angle_samples[-1].copy()
    dt = 2 * math.pi / STEPS
    started = time.perf_counter()
    for step in range(step0 + 1, PERIODS * STEPS + 1):
        t = (step - 1) * dt
        def acc(tt, yy):
            a = force(tt, yy, phase1, phase2, nu, cfg["alpha"], cfg["lambda"])
            a[~active] = 0
            return a
        k1 = acc(t, y)
        k2 = acc(t + dt / 2, y + dt * k1 / 2)
        k3 = acc(t + dt / 2, y + dt * k2 / 2)
        k4 = acc(t + dt, y + dt * k3)
        y += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        rho = np.hypot(y[:, 0], y[:, 1])
        maxrho = np.maximum(maxrho, rho)
        new = active & (rho >= 1)
        escape_period[new] = step / STEPS
        active[new] = False
        if step % STEPS == 0:
            now = np.arctan2(y[:, 1], y[:, 0])
            angle += np.arctan2(np.sin(now - raw), np.cos(now - raw))
            raw = now
            rho_samples.append(rho.copy())
            angle_samples.append(angle.copy())
        if step % (500 * STEPS) == 0:
            tmp = ckpt.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, step=step, state=y, active=active,
                escaped=~active, escape_period=escape_period, maxrho=maxrho,
                rho_samples=np.asarray(rho_samples), angle_samples=np.asarray(angle_samples),
                nu=nu, species=species, periods=PERIODS, steps_per_period=STEPS)
            tmp.replace(ckpt)
    rho_hist = np.asarray(rho_samples)
    angle_hist = np.asarray(angle_samples)
    result = []
    for i, row in enumerate(rows):
        om = float((angle_hist[PERIODS, i] - angle_hist[PERIODS - WINDOW, i])
                   / (2 * math.pi * WINDOW))
        lost = bool(np.isfinite(escape_period[i]) and escape_period[i] <= PERIODS)
        result.append({"species": species, "nu": nu, "start_id": row["start_id"],
            "rho_mean_last_500": float(np.mean(rho_hist[PERIODS-WINDOW:PERIODS+1, i])),
            "omega_last_500": om, "max_rho": float(maxrho[i]),
            "escaped": lost, "escape_period": None if not lost else float(escape_period[i]),
            "sign": 0 if lost else int(np.sign(om))})
    return {"species": species, "nu": nu, "rows": result,
            "leading_roots": reduced_roots(nu, species),
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint": name}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with PANEL.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    panel_sha = hashlib.sha256(PANEL.read_bytes()).hexdigest()
    state0 = np.array([[float(r[k]) for k in ("x0", "y0", "vx0", "vy0")] for r in rows])
    phase1 = np.array([float(r["phase1"]) for r in rows])
    phase2 = np.array([float(r["phase2"]) for r in rows])
    tasks = [(nu, species, rows, state0, phase1, phase2)
             for nu in NU_VALUES for species in ("species1", "species2")]
    started = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(run_one, tasks))
    results.sort(key=lambda r: (r["nu"], r["species"]))
    flat = [row for result in results for row in result["rows"]]
    doc = {"status": "completed", "model_scope": "exact ideal k=+2,-4 Local15; C=0; linear drag; no full MFS",
        "parameters": PARAM, "nu_values": NU_VALUES, "periods": PERIODS,
        "steps_per_primary_period": STEPS, "measurement_window_periods": WINDOW,
        "panel_seed": 15152026, "panel_sha256": panel_sha,
        "panel_is_iid_probability_sample": False, "worker_count": 10,
        "wall_seconds": time.perf_counter() - started,
        "results": results, "rows": flat}
    (OUT / "local15_sign_screen.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with (OUT / "local15_sign_screen.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    print(json.dumps({"status": doc["status"], "tasks": len(results),
                      "trajectories": len(flat), "wall_seconds": doc["wall_seconds"]}))


if __name__ == "__main__":
    main()
