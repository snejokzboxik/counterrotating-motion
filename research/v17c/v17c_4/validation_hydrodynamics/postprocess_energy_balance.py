"""Correct full-energy residuals from saved matched hydrodynamic NPZ files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CSV = OUT / "matched_hydrodynamics.csv"


def total_energy(state: np.ndarray, beta: np.ndarray, mass: np.ndarray,
                 charge_ratio: np.ndarray, chi: float) -> float:
    positions, velocity = state[:, :3], state[:, 3:6]
    energy = 0.0
    for i in range(2):
        x, y, z = positions[i]
        energy += mass[i] * (0.5 * float(velocity[i] @ velocity[i])
                             + beta[i] * (z*z - 0.5*(x*x+y*y)))
    distance = float(np.linalg.norm(positions[0] - positions[1]))
    return energy + chi * charge_ratio[0] * charge_ratio[1] / distance


with CSV.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
    fields = list(rows[0].keys()) if rows else []

if rows:
    for row in rows:
        path = OUT / f"{row['checkpoint']}__{row['model']}.npz"
        with np.load(path) as z:
            state, p = z["state"], z["parameters"]
        beta = np.asarray([p[2], p[3]], float)
        mass = np.asarray([1.0, p[6]], float)
        charge = np.asarray([1.0, p[7]], float)
        chi = float(p[8])
        e0 = total_energy(state[0], beta, mass, charge, chi)
        e1 = total_energy(state[-1], beta, mass, charge, chi)
        wrf = float(sum(mass[i] * state[-1, i, 6] for i in range(2)))
        wdrag = float(sum(mass[i] * state[-1, i, 9] for i in range(2)))
        wdc = float(sum(mass[i] * state[-1, i, 7] for i in range(2)))
        wc = float(sum(mass[i] * state[-1, i, 8] for i in range(2)))
        row["energy_balance_residual_double_counted"] = row["energy_balance_residual"]
        corrected = e1 - e0 - wrf - wdrag
        row["energy_balance_residual"] = f"{corrected:.17g}"
        row["relative_energy_balance_residual"] = f"{abs(corrected)/max(abs(e0), abs(e1), 1e-30):.17g}"
        row["work_dc_check"] = f"{wdc:.17g}"
        row["work_coulomb_check"] = f"{wc:.17g}"
        row["energy_balance_definition"] = "Delta(total K+DC+Coulomb potential)-RF_work-drag_work"
        windows = np.asarray(json.loads(row["late_window_turns"]), dtype=float)
        signs = np.sign(windows)
        row["late_signs_opposite"] = str(bool((np.all(signs[:, 0] > 0) and np.all(signs[:, 1] < 0))
                                               or (np.all(signs[:, 0] < 0) and np.all(signs[:, 1] > 0))))
        active = bool(np.all(np.abs(windows) >= 0.1))
        row["late_window_activity_ge_0p1_turns"] = str(active)
        counter = (row["late_signs_opposite"] == "True" and active and row["escaped"] == "False"
                   and row["hard_sphere_overlap"] == "False")
        row["counter_rotating_late"] = str(counter)
        row["counter_rotating_10turn_after_relax"] = str(counter and
            abs(float(row["turns1_after_relax"])) >= 10 and abs(float(row["turns2_after_relax"])) >= 10)
    for key in ("energy_balance_residual_double_counted", "relative_energy_balance_residual",
                "work_dc_check", "work_coulomb_check", "energy_balance_definition",
                "late_signs_opposite", "late_window_activity_ge_0p1_turns"):
        if key not in fields:
            fields.append(key)
    with CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Corrected {len(rows)} saved energy balances; no integration was run.")
