"""Postprocess saved DOP853 trajectories for Coulomb energy/force/torque diagnostics."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "stage3" / "runs"
CFG = json.loads((HERE.parent.parent / "v17c_5" / "input" / "config.json").read_text(encoding="utf-8-sig"))
K = np.asarray(CFG["k"], dtype=int)
FAMILY = np.asarray(CFG["family"], dtype=int)
COEFF = np.asarray(CFG["coeff"], dtype=float)
COEFF = COEFF[:, 0] + 1j * COEFF[:, 1]
TAU = 2 * np.pi


def electric_field(xy, times, p, phase):
    """Evaluate dimensionless 2-D RF field from the full saved signed MFS spectrum."""
    A, B, nu = p[:3]
    u = xy[:, 0] + 1j * xy[:, 1]
    ex = np.zeros(len(times))
    ey = np.zeros(len(times))
    carrier1 = A * np.exp(-1j * (times + phase[0]))
    carrier2 = B * np.exp(-1j * (nu * times + phase[1]))
    for mode, family, coeff in zip(K, FAMILY, COEFF):
        base = u if mode > 0 else np.conjugate(u)
        carrier = carrier1 if family == 0 else carrier2
        grad = carrier * coeff * abs(mode) * base ** (abs(mode) - 1)
        ex -= grad.real
        ey += grad.imag if mode > 0 else -grad.imag
    return np.column_stack((ex, ey))


def main():
    out = {"candidate_id": 48, "source": "saved independent DOP853 trajectories; stroboscopic samples",
           "window_start_cycle": 200, "trajectories": []}
    for group in ("rest", "weak", "wide"):
        results = json.loads((RUNS / f"dop_cfg48_{group}_results.json").read_text(encoding="utf-8"))
        for diag in results:
            # Keep the representative key already used in published validation.
            if diag.get("key") != f"48:{group}:{0 if group=='rest' else 1 if group=='weak' else 2}":
                continue
            idx = 0 if group == "rest" else 1 if group == "weak" else 2
            z = np.load(RUNS / f"dop_cfg48_{group}_{idx}.npz")
            times = np.asarray(z["tau"], dtype=float)
            state = np.asarray(z["state"], dtype=float)
            work = np.asarray(z["work"], dtype=float)
            pos = state[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3)
            p = np.asarray(diag["parameters"], dtype=float)
            phase = np.asarray(diag["rf_phase"], dtype=float)
            qratio, s, C = float(diag["charge_ratio"]), p[4], p[5]
            delta = pos[:, 0, :] - pos[:, 1, :]
            dist = np.linalg.norm(delta, axis=1)
            electric = np.stack([electric_field(pos[:, i, :2], times, p, phase) for i in range(2)], axis=1)
            e_mag = np.linalg.norm(electric, axis=2)
            rf_acc = np.stack((e_mag[:, 0], s * qratio * e_mag[:, 1]), axis=1)
            c_acc = np.stack((C / dist**2, s * C / dist**2), axis=1)
            ratio = c_acc / np.maximum(rf_acc, 1e-300)
            torque1 = C * (pos[:, 0, 0] * delta[:, 1] - pos[:, 0, 1] * delta[:, 0]) / dist**3
            torque2 = -C * (pos[:, 1, 0] * delta[:, 1] - pos[:, 1, 1] * delta[:, 0]) / dist**3
            i0 = 200
            cwork = work[-1, :, 3] - work[i0, :, 3]
            rfwork = float((work[-1, :, :2] - work[i0, :, :2]).sum())
            out["trajectories"].append({
                "key": diag["key"],
                "frequency_Hz": diag["frequency_Hz"],
                "turns_after_transient": diag["turns_after_transient"],
                "mean_radius_R": diag["mean_radius_stroboscopic"],
                "separation_R_mean_min_max_after_transient": [float(dist[i0:].mean()),
                    float(dist[i0:].min()), float(dist[i0:].max())],
                "Coulomb_potential_C_over_d_mean_min_max": [float((C/dist[i0:]).mean()),
                    float((C/dist[i0:]).min()), float((C/dist[i0:]).max())],
                "Coulomb_to_RF_force_ratio_median_p95_max_each": np.column_stack((
                    np.median(ratio[i0:], axis=0), np.quantile(ratio[i0:], .95, axis=0),
                    np.max(ratio[i0:], axis=0))).tolist(),
                "Coulomb_torque_z_mean_each": [float(torque1[i0:].mean()), float(torque2[i0:].mean())],
                "Coulomb_torque_z_integral_each_stroboscopic": [float(np.trapezoid(torque1[i0:], times[i0:])),
                    float(np.trapezoid(torque2[i0:], times[i0:]))],
                "Coulomb_work_each_over_late_window": cwork.tolist(),
                "Coulomb_work_sum_abs": float(abs(cwork.sum())),
                "RF_work_sum_late_window": rfwork,
                "Coulomb_work_sum_over_RF_work": float(abs(cwork.sum()) / max(abs(rfwork), 1e-300)),
                "stroboscopic_force_ratio_caveat": "RF-force ratios sampled once per primary cycle; RF micro-motion extrema are not resolved.",
                "torque_integral_caveat": "stroboscopic quadrature; equal/opposite exchange is exact instantaneously and approximate here.",
            })
    dst = HERE / "stage3" / "coulomb_diagnostics_dop853.json"
    dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(dst)
    for x in out["trajectories"]:
        print(x["key"], x["separation_R_mean_min_max_after_transient"],
              x["Coulomb_to_RF_force_ratio_median_p95_max_each"])


if __name__ == "__main__":
    main()
