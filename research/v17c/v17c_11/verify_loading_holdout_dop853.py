"""Independent DOP853 and energy/angular-momentum audit of holdout leaders."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
OUT = HERE / "strict_loading_holdout"
BASE = HERE.parent / "v17c_9" / "baseline"
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HERE))
import basin_scan
import loading_dynamics as ld
import verify_strict_leaders_dop853 as v

GROUP = "edge"
PROTOCOL_IDS = (1, 2, 4)  # #505, #353, #292
F1_HZ = 100_000.0
RTOL, ATOL = 2e-10, 2e-12
MAX_STEP = 2.0 * math.pi / 48.0
POST_LOAD = 15_000


def energy_ledger(tau, y, skip, p, protocol, phase, family, modes, coeff, qratio):
    istart = min(int(skip), len(tau) - 1)
    work = y[12:22].T.reshape(-1, 2, 5)
    explicit = y[24:27].T
    torque = y[27:37].T.reshape(-1, 2, 5)
    states = y[:12].T
    mass = np.asarray([1.0, 1.0 / p[4]])
    vel = states[:, [3, 4, 5, 9, 10, 11]].reshape(-1, 2, 3)
    kin = 0.5 * mass[None, :] * np.sum(vel * vel, axis=2)
    def interval(i0):
        per = {}
        for i in range(2):
            dK = float(kin[-1, i] - kin[i0, i])
            dw = work[-1, i] - work[i0, i]
            L0 = mass[i] * (states[i0, 6*i] * states[i0, 6*i+4] - states[i0, 6*i+1] * states[i0, 6*i+3])
            L1 = mass[i] * (states[-1, 6*i] * states[-1, 6*i+4] - states[-1, 6*i+1] * states[-1, 6*i+3])
            per[str(i+1)] = {
                "delta_kinetic": dK,
                "work_RF_A_RF_B_DC_Coulomb_drag": dw.tolist(),
                "kinetic_balance_residual": float(dK - dw.sum()),
                "delta_Lz": float(L1-L0),
                "integrated_torque_RF_A_RF_B_DC_Coulomb_drag": (torque[-1, i]-torque[i0, i]).tolist(),
            }
        H0 = v.energy_at(float(tau[i0]), y[:, i0], p, protocol, phase, family, modes, coeff, qratio)
        H1 = v.energy_at(float(tau[-1]), y[:, -1], p, protocol, phase, family, modes, coeff, qratio)
        dH = H1["H_total"] - H0["H_total"]
        explicit_work = explicit[-1] - explicit[i0]
        drag = float((work[-1, :, 4] - work[i0, :, 4]).sum())
        system_torque = (torque[-1] - torque[i0]).sum(axis=0)
        Lz0 = np.sum(mass * (states[i0, [0, 6]] * states[i0, [4, 10]] - states[i0, [1, 7]] * states[i0, [3, 9]]))
        Lz1 = np.sum(mass * (states[-1, [0, 6]] * states[-1, [4, 10]] - states[-1, [1, 7]] * states[-1, [3, 9]]))
        return {
            "per_particle": per,
            "system_energy": {
                "H_start": H0, "H_end": H1, "delta_H": float(dH),
                "explicit_work_RF_A_RF_B_DC": explicit_work.tolist(),
                "drag_work": drag,
                "balance_residual": float(dH - explicit_work.sum() - drag),
            },
            "system_angular_momentum": {
                "delta_Lz": float(Lz1-Lz0),
                "integrated_torque_RF_A_RF_B_DC_Coulomb_drag": system_torque.tolist(),
                "balance_residual": float((Lz1-Lz0)-system_torque.sum()),
            },
        }
    return {"full_interval": interval(0), "post_relaxation": interval(istart)}


def main():
    holdout = [json.loads(s) for s in (OUT / "holdout_runs.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    protocols = json.loads((OUT / "holdout_protocols.json").read_text(encoding="utf-8"))
    meta = json.loads((OUT / "holdout_metadata.json").read_text(encoding="utf-8"))
    pdoc = json.loads((HERE / "local15_parameters.json").read_text(encoding="utf-8"))
    p = np.asarray(pdoc["candidate"]["p"], dtype=np.float64)
    qratio = float(pdoc["candidate"]["q2_over_q1"])
    _, family, modes, coeff = basin_scan.load_model()
    reps = []
    for pid in PROTOCOL_IDS:
        rows = [r for r in holdout if r["protocol_id"] == pid and r["ensemble"] == GROUP and r["category"] == "CR_screen"]
        if not rows:
            raise RuntimeError(f"No strict-success representative for protocol {pid}")
        row = min(rows, key=lambda r: r["start_id"])
        reps.append(row)

    records = []
    for row in reps:
        pid = row["protocol_id"]
        protocol = np.asarray(protocols[pid]["protocol_vector_19"], dtype=np.float64)
        phase = np.asarray(row["initial_rf_phase"], dtype=np.float64)
        y0 = np.zeros(37, dtype=np.float64); y0[:12] = np.asarray(row["initial_state"], dtype=np.float64)
        trans = int(math.ceil(ld.protocol_duration_cycles(protocol)))
        skip = trans + int(meta["relax_periods"])
        total = skip + POST_LOAD
        t0 = time.perf_counter()
        sol = solve_ivp(
            lambda t, y: v.dop_rhs(t, y, p, protocol, phase, family, modes, coeff, qratio),
            (0.0, total * 2.0 * math.pi), y0, method="DOP853", rtol=RTOL, atol=ATOL,
            max_step=MAX_STEP, t_eval=np.arange(total + 1, dtype=np.float64) * 2.0 * math.pi,
            events=v.build_events(p),
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        states = sol.y[:12].T.copy()
        diag = v.strict_classification(sol.t, states, skip)
        turns = np.asarray(diag["turns_after_relaxation"], dtype=float)
        measured_cycles = max(1, diag["completed_periods"] - skip)
        frequencies = turns * F1_HZ / measured_cycles
        rk_delta = float(np.max(np.abs(states[-1] - np.asarray(row["final_state"], dtype=float))))
        ledger = energy_ledger(sol.t, sol.y, skip, p, protocol, phase, family, modes, coeff, qratio)
        rec = {
            "protocol_id": pid, "protocol_label": row["protocol_label"],
            "ensemble": GROUP, "start_id": row["start_id"],
            "independent_solver": "SciPy DOP853 with separately coded polar RF gradient and envelope derivatives",
            "rtol": RTOL, "atol": ATOL, "max_step_tau": MAX_STEP,
            "nfev": int(sol.nfev), "elapsed_seconds": time.perf_counter() - t0,
            "strict_dop853": diag, "frequency_Hz": frequencies.tolist(),
            "rk4_category": row["category"], "rk4_frequency_Hz": row["frequency_Hz"],
            "max_abs_final_state_delta_vs_RK4_96": rk_delta,
            "rk4_minimum_separation_R": row["minimum_separation"],
            "rk4_min_radius_R": row["r_min"],
            **ledger,
        }
        traj_path = OUT / f"dop853_{row['protocol_label']}_edge_{row['start_id']}.npz"
        np.savez_compressed(traj_path, tau=sol.t, state=states,
                            work=sol.y[12:22].T.reshape(-1, 2, 5),
                            explicit_work=sol.y[24:27].T,
                            torque=sol.y[27:37].T.reshape(-1, 2, 5),
                            initial=np.asarray(row["initial_state"]), phase=phase,
                            protocol=protocol)
        rec["trajectory_file"] = traj_path.name
        records.append(rec)
        print(row["protocol_label"], diag["category"],
              "freq", [round(float(x), 6) for x in frequencies],
              "RK4_delta", rk_delta,
              "energy_res", ledger["post_relaxation"]["system_energy"]["balance_residual"],
              "elapsed_s", round(rec["elapsed_seconds"], 1), flush=True)
        with (OUT / "dop853_representatives.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n"); f.flush(); os.fsync(f.fileno())
    (OUT / "dop853_representatives.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(records)} DOP853 holdout representatives")


if __name__ == "__main__":
    main()
