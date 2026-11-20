"""Quantify sampled Coulomb energy, RF-force ratio, torque and group power."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import nbody_dynamics as nb

HERE = Path(__file__).resolve().parent
DATA = HERE / "p1_coulomb_scan"
OUT = DATA / "coulomb_work_analysis.json"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
TAU = 2.0 * math.pi


def analyze_one(logs: np.ndarray, skip: int, p: np.ndarray, family: np.ndarray,
                modes: np.ndarray, coeff: np.ndarray, qratio: float,
                phases: np.ndarray, sample_stride: int = 5) -> dict:
    samples = logs[skip::sample_stride]
    n = len(samples)
    c_energy = np.zeros((n, 3), dtype=np.float64)  # intra1, cross, intra2
    torque = np.zeros((n, 2), dtype=np.float64)
    coulomb_power = np.zeros((n, 2), dtype=np.float64)
    ratio = np.zeros((n, 6), dtype=np.float64)
    min_d = np.full(15, np.inf)
    pair_index = 0
    mass = np.where(GROUPS == 0, 1.0, 1.0 / p[4])
    charge = np.where(GROUPS == 0, 1.0, qratio)
    inv_mass = 1.0 / mass
    for k, sample in enumerate(samples):
        t = float(sample[0, 0])
        state = sample[:, 1:7]
        pos = state[:, :3]
        vel = state[:, 3:]
        ac = np.zeros_like(pos)
        rf_ac = np.zeros_like(pos)
        u = pos[:, 0] + 1j * pos[:, 1]
        ca = p[0] * np.exp(-1j * (t + phases[0]))
        cb = p[1] * np.exp(-1j * (p[2] * t + phases[1]))
        for i in range(6):
            ex = ey = 0.0
            for h, mode in enumerate(modes):
                degree = abs(int(mode))
                base = u[i] if mode > 0 else np.conjugate(u[i])
                carrier = ca if family[h] == 0 else cb
                grad = degree * carrier * coeff[h] * base ** (degree - 1)
                ex -= grad.real
                ey += grad.imag if mode > 0 else -grad.imag
            rf_ac[i, 0] = charge[i] * inv_mass[i] * ex
            rf_ac[i, 1] = charge[i] * inv_mass[i] * ey
        idx = 0
        for i in range(5):
            for j in range(i + 1, 6):
                dr = pos[i] - pos[j]
                d2 = float(np.dot(dr, dr))
                d = math.sqrt(d2)
                min_d[idx] = min(min_d[idx], d)
                idx += 1
                if d2 <= 1.0e-28:
                    continue
                eidx = 0 if GROUPS[i] == GROUPS[j] == 0 else (2 if GROUPS[i] == GROUPS[j] == 1 else 1)
                e = p[5] * charge[i] * charge[j] / qratio / d
                c_energy[k, eidx] += e
                force = p[5] * (charge[i] * charge[j] / qratio) * dr / (d2 * d)
                ac[i] += inv_mass[i] * force
                ac[j] -= inv_mass[j] * force
        force_c = mass[:, None] * ac
        for g in (0, 1):
            ix = GROUPS == g
            torque[k, g] = np.sum(pos[ix, 0] * force_c[ix, 1] - pos[ix, 1] * force_c[ix, 0])
            coulomb_power[k, g] = float(np.sum(force_c[ix] * vel[ix]))
        ratio[k] = np.linalg.norm(ac, axis=1) / np.maximum(np.linalg.norm(rf_ac, axis=1), 1e-14)
    sample_time = samples[:, 0, 0]
    cross_torque_sum = torque[:, 0] + torque[:, 1]
    return {
        "samples_every_primary_period": sample_stride,
        "sample_count": int(n),
        "mean_Coulomb_potential_energy_normalized": {
            "within_group1": float(np.mean(c_energy[:, 0])),
            "between_groups": float(np.mean(c_energy[:, 1])),
            "within_group2": float(np.mean(c_energy[:, 2])),
            "total": float(np.mean(np.sum(c_energy, axis=1))),
        },
        "Coulomb_potential_energy_range_total": [float(np.min(np.sum(c_energy, axis=1))),
                                                 float(np.max(np.sum(c_energy, axis=1)))],
        "mean_Coulomb_acceleration_over_RF_acceleration_ratio": float(np.mean(ratio)),
        "p95_Coulomb_acceleration_over_RF_acceleration_ratio": float(np.quantile(ratio, 0.95)),
        "median_ratio_each_particle": np.median(ratio, axis=0).tolist(),
        "mean_group_Coulomb_torque_normalized": np.mean(torque, axis=0).tolist(),
        "rms_group_Coulomb_torque_normalized": np.sqrt(np.mean(torque ** 2, axis=0)).tolist(),
        "max_abs_pairwise_group_torque_closure": float(np.max(np.abs(cross_torque_sum))),
        "integrated_group1_Coulomb_torque_trapezoid": float(np.trapezoid(torque[:, 0], sample_time)),
        "integrated_group1_Coulomb_power_trapezoid": float(np.trapezoid(coulomb_power[:, 0], sample_time)),
        "integrated_group2_Coulomb_power_trapezoid": float(np.trapezoid(coulomb_power[:, 1], sample_time)),
        "mean_group_Coulomb_power": np.mean(coulomb_power, axis=0).tolist(),
        "minimum_pair_distances_R_i_lt_j": min_d.tolist(),
    }


def main() -> None:
    p0, family, modes, coeff, qratio = nb.load_local15()
    rows = [json.loads(x) for x in (DATA / "coulomb_runs.jsonl").read_text().splitlines() if x.strip()]
    output = {}
    for row in sorted(rows, key=lambda x: x["C_multiplier"]):
        p = p0.copy()
        p[5] = p0[5] * row["C_multiplier"]
        with np.load(DATA / f"{row['task_id']}_trajectory.npz") as archive:
            logs = archive["trajectory"]
            skip = int(archive["post_load_skip"])
            phases = archive["phases"].astype(np.float64)
        output[row["task_id"]] = analyze_one(logs, skip, p, family, modes, coeff, qratio, phases)
    result = {
        "status": "completed",
        "normalization": "dimensionless energy m1*R^2*Omega1^2, torque m1*R^2*Omega1^2, power per tau",
        "note": "Force and energy summaries use stroboscopic samples every five primary RF periods; between-sample peak forces are not represented.",
        "runs": output,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("saved", OUT)
    for key, value in output.items():
        print(key, "Uc", value["mean_Coulomb_potential_energy_normalized"],
              "mean/p95 aC/aRF", round(value["mean_Coulomb_acceleration_over_RF_acceleration_ratio"], 6),
              round(value["p95_Coulomb_acceleration_over_RF_acceleration_ratio"], 6),
              "torque rms", [round(v, 8) for v in value["rms_group_Coulomb_torque_normalized"]],
              "pair closure", value["max_abs_pairwise_group_torque_closure"])


if __name__ == "__main__":
    main()
