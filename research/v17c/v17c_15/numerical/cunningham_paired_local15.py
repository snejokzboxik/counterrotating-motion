"""Paired old/new Cunningham convention check on frozen Local15 starts.

This deliberately isolates the Cunningham multiplier in the exact two-leading-
RF-mode model. It does not re-run the V17C.12B full MFS/nonlinear-drag campaign.
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
REPO = Path(__file__).resolve().parents[4]
OUT = ROOT / "results" / "cunningham_pairs"
PERIODS = 3000
STEPS = 96
CC_META = REPO / "research/v17c/v17c_12b/results/rescaled_legacy_candidate_validation/campaign_meta.json"
PARAM = {
    "A": 0.03716194907442204,
    "B": 0.1626938199343468,
    "nu": 3.1118029452860356,
    "beta": 0.017273303124515367,
    "species": {
        "species1": {"alpha": 1.0, "lambda_stokes": 0.20975541409005094},
        "species2": {"alpha": 0.34126014359705537, "lambda_stokes": 0.12617311138613094},
    },
}


def acceleration(t, y, phase1, phase2, alpha, lam):
    x, yy, vx, vy = y.T
    z = x + 1j * yy
    g0 = 2.0 * PARAM["A"] * np.exp(1j * phase1) * z * np.exp(-1j * t)
    g1 = 4.0 * PARAM["B"] * np.exp(1j * phase2) * np.conjugate(z) ** 3 * np.exp(-1j * PARAM["nu"] * t)
    gx = np.real(g0 + g1)
    gy = np.real(1j * g0 - 1j * g1)
    return np.column_stack((vx, vy,
        -alpha * gx + alpha * PARAM["beta"] * x - lam * vx,
        -alpha * gy + alpha * PARAM["beta"] * yy - lam * vy))


def integrate(task):
    species, convention, cc, initials, phases = task
    alpha = PARAM["species"][species]["alpha"]
    lam0 = PARAM["species"][species]["lambda_stokes"]
    lam = lam0 * cc if convention == "legacy_multiply" else lam0 / cc
    n = len(initials)
    y = initials.copy()
    active = np.ones(n, dtype=bool)
    escaped = np.hypot(y[:, 0], y[:, 1]) >= 1.0
    active &= ~escaped
    escape_period = np.where(escaped, 0.0, np.nan)
    maxrho = np.hypot(y[:, 0], y[:, 1])
    raw = np.arctan2(y[:, 1], y[:, 0])
    angle = raw.copy()
    rho_samples = [maxrho.copy()]
    angle_samples = [angle.copy()]
    dt = 2 * math.pi / STEPS
    t0 = time.perf_counter()
    ckpt = OUT / f"{species}_{convention}_checkpoint.npz"
    for step in range(1, PERIODS * STEPS + 1):
        t = (step - 1) * dt
        k1 = acceleration(t, y, phases[:, 0], phases[:, 1], alpha, lam)
        k1[~active] = 0
        k2 = acceleration(t + dt/2, y + dt*k1/2, phases[:, 0], phases[:, 1], alpha, lam)
        k2[~active] = 0
        k3 = acceleration(t + dt/2, y + dt*k2/2, phases[:, 0], phases[:, 1], alpha, lam)
        k3[~active] = 0
        k4 = acceleration(t + dt, y + dt*k3, phases[:, 0], phases[:, 1], alpha, lam)
        k4[~active] = 0
        y += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        rho = np.hypot(y[:, 0], y[:, 1])
        maxrho = np.maximum(maxrho, rho)
        new = active & (rho >= 1.0)
        escape_period[new] = step / STEPS
        active[new] = False
        if step % STEPS == 0:
            now = np.arctan2(y[:, 1], y[:, 0])
            angle += np.arctan2(np.sin(now - raw), np.cos(now - raw))
            raw = now
            rho_samples.append(rho.copy())
            angle_samples.append(angle.copy())
        if step % (500 * STEPS) == 0:
            np.savez_compressed(ckpt, step=step, state=y, active=active,
                escaped=~active, escape_period=escape_period, maxrho=maxrho,
                rho_samples=np.asarray(rho_samples), angle_samples=np.asarray(angle_samples))
    rh = np.asarray(rho_samples)
    ah = np.asarray(angle_samples)
    lo = PERIODS - 500
    out = []
    for i in range(n):
        omega = float((ah[PERIODS, i] - ah[lo, i]) / (2*math.pi*500))
        lost = bool(np.isfinite(escape_period[i]) and escape_period[i] <= PERIODS)
        out.append({
            "species": species, "convention": convention, "start_index": i,
            "lambda_stokes": lam0, "Cc_archived_mean": cc,
            "lambda_used": lam, "rho_mean_last_500": float(np.mean(rh[lo:PERIODS+1, i])),
            "rho_sd_last_500": float(np.std(rh[lo:PERIODS+1, i])),
            "omega_last_500": omega, "max_rho": float(maxrho[i]),
            "escaped": lost, "escape_period": None if not lost else float(escape_period[i]),
            "sign": int(np.sign(omega)) if not lost else 0,
        })
    return {"species": species, "convention": convention, "rows": out,
            "elapsed_seconds": time.perf_counter() - t0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = ROOT / "results/frozen_local15_single_particle_panel.csv"
    panel_sha = hashlib.sha256(panel.read_bytes()).hexdigest()
    with panel.open(newline="", encoding="utf-8") as f:
        start_rows = list(csv.DictReader(f))
    initials = np.array([[float(r[k]) for k in ("x0", "y0", "vx0", "vy0")] for r in start_rows])
    phases = np.array([[float(r[k]) for k in ("phase1", "phase2")] for r in start_rows])
    meta = json.loads(CC_META.read_text(encoding="utf-8"))
    local = next(x for x in meta["candidates"] if x["legacy_candidate"] == "local15")
    cc = float(local["physical"]["Cunningham_mean"])
    tasks = [(s, c, cc, initials, phases) for s in PARAM["species"] for c in ("legacy_multiply", "corrected_divide")]
    started = time.perf_counter()
    results = []
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(integrate, t) for t in tasks]
        for f in cf.as_completed(futures):
            res = f.result()
            results.append(res)
            checkpoint = {"status": "partial", "periods": PERIODS, "steps_per_period": STEPS,
                "panel_seed": 15152026, "panel_sha256": panel_sha, "Cc_source": str(CC_META),
                "Cc_meta_sha256": hashlib.sha256(CC_META.read_bytes()).hexdigest(),
                "cunningham_mean": cc, "parameters": PARAM, "completed": results,
                "wall_seconds": time.perf_counter() - started}
            (OUT / "local15_pair_check.json").write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"completed": (res["species"], res["convention"]),
                "seconds": res["elapsed_seconds"], "tasks": len(results)}), flush=True)
    results.sort(key=lambda x: (x["species"], x["convention"]))
    by = {(x["species"], x["convention"]): x["rows"] for x in results}
    paired = []
    for species in PARAM["species"]:
        old = {r["start_index"]: r for r in by[species, "legacy_multiply"]}
        new = {r["start_index"]: r for r in by[species, "corrected_divide"]}
        for i in range(len(start_rows)):
            a, b = old[i], new[i]
            paired.append({"species": species, "start_id": start_rows[i]["start_id"],
                "legacy_retained": not a["escaped"], "corrected_retained": not b["escaped"],
                "legacy_rho": a["rho_mean_last_500"], "corrected_rho": b["rho_mean_last_500"],
                "delta_rho": b["rho_mean_last_500"]-a["rho_mean_last_500"],
                "legacy_omega": a["omega_last_500"], "corrected_omega": b["omega_last_500"],
                "delta_omega": b["omega_last_500"]-a["omega_last_500"],
                "same_sign": a["sign"] == b["sign"]})
    doc = {"status": "completed", "model_scope": "exact ideal two-mode Local15; Cunningham factor isolated; no Coulomb/MFS/Oseen/hydrodynamic correction/gravity",
        "periods": PERIODS, "steps_per_primary_period": STEPS,
        "comparison_window": "last 500 primary RF periods", "panel_seed": 15152026,
        "panel_sha256": panel_sha, "Cc_source": str(CC_META),
        "Cc_meta_sha256": hashlib.sha256(CC_META.read_bytes()).hexdigest(),
        "cunningham_mean": cc, "relative_drag_old_to_corrected": cc**2,
        "parameters": PARAM, "method_results": results, "paired_rows": paired,
        "wall_seconds": time.perf_counter()-started}
    (OUT / "local15_pair_check.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with (OUT / "local15_pair_check.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    print(json.dumps({"status": doc["status"], "rows": len(paired), "wall_seconds": doc["wall_seconds"]}))


if __name__ == "__main__":
    main()
