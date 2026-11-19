"""High-rate DOP853 audit window from a saved long-run state."""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import ode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_dop853_local as dop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-cycle", type=int, default=100000)
    ap.add_argument("--span-cycles", type=int, default=1000)
    a = ap.parse_args()
    config_id = 52
    start_cycle = a.start_cycle
    span_cycles = a.span_cycles
    samples_per_cycle = 24
    pz = np.load(HERE / "stage4_candidate48" / "configurations.npz")
    p = np.asarray(pz["p"][config_id], float)
    qratio = float(pz["qratio"][config_id])
    source = HERE / "stage4_candidate48" / "holdout_runs" / "dop_cfg52_rest_0.npz"
    data = np.load(source)
    sl = slice(start_cycle, start_cycle + 1)
    y0 = np.concatenate((data["state"][sl][0], data["work"][sl][0].reshape(-1),
                         data["angles_integral"][sl][0], data["torque"][sl][0].reshape(-1)))
    saved_diag = json.loads((HERE / "stage4_candidate48" / "holdout_runs" /
                             "dop_cfg52_rest_results.json").read_text(encoding="utf-8"))[0]
    phase = np.asarray(saved_diag["rf_phase"], float)
    t0 = start_cycle * dop.T
    n = span_cycles * samples_per_cycle
    times = t0 + np.arange(n + 1) * dop.T / samples_per_cycle
    state = np.empty((n + 1, 34), float)
    state[0] = y0
    driver = ode(dop.derivative).set_integrator("dop853", rtol=2e-10, atol=2e-12,
                                                max_step=dop.T / samples_per_cycle,
                                                nsteps=512)
    driver.set_f_params(p, qratio, phase)
    driver.set_initial_value(y0, t0)
    for i, t in enumerate(times[1:], 1):
        state[i] = driver.integrate(t)
        if not driver.successful() or not np.isfinite(state[i]).all():
            raise RuntimeError(f"DOP853 failed at dense sample {i}/{n}")

    pos = state[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3)
    vel = state[:, [3, 4, 5, 9, 10, 11]].reshape(-1, 2, 3)
    radii = np.linalg.norm(pos[:, :, :2], axis=2)
    sep = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
    angles = np.unwrap(np.arctan2(pos[:, :, 1], pos[:, :, 0]), axis=0)
    turns = (angles[-1] - angles[0]) / dop.T
    angle_integral_turns = (state[-1, 22:24] - state[0, 22:24]) / dop.T
    work = state[:, 12:22].reshape(-1, 2, 5)
    torque = state[:, 24:34].reshape(-1, 2, 5)
    r2 = radii ** 2
    ke = 0.5 * np.sum(np.array([1., 1./p[4]])[None, :, None] * vel**2, axis=(1, 2))
    dc = p[3] * np.sum(np.array([1., qratio])[None, :] *
                       (pos[:, :, 2]**2 - 0.5*r2), axis=1)
    uc = p[5] / sep
    ext = work[:, :, :2].sum(axis=(1, 2)) + work[:, :, 4].sum(axis=1)
    energy_residual = (ke[-1] + dc[-1] + uc[-1] - ke[0] - dc[0] - uc[0]) - (ext[-1] - ext[0])
    mass = np.array([1., 1./p[4]])
    lz = mass[None, :] * (pos[:, :, 0]*vel[:, :, 1] - pos[:, :, 1]*vel[:, :, 0])
    torque_delta = torque[-1] - torque[0]
    lz_change = lz[-1] - lz[0]
    summary = {
        "candidate_id": config_id, "solver": "SciPy DOP853 independent RHS",
        "source_long_run": source.name, "start_cycle": start_cycle,
        "span_cycles": span_cycles, "samples_per_primary_period": samples_per_cycle,
        "rtol": 2e-10, "atol": 2e-12, "max_step_tau": dop.T/samples_per_cycle,
        "frequency_Hz": (turns/span_cycles*float(dop.F1)).tolist(),
        "turns": turns.tolist(), "turns_by_omega_integral": angle_integral_turns.tolist(),
        "max_angle_integral_difference_turns": float(np.max(np.abs(
            (angles-angles[0])/dop.T - (state[:,22:24]-state[0,22:24])/dop.T))),
        "radius_min_R": radii.min(axis=0).tolist(), "radius_max_R": radii.max(axis=0).tolist(),
        "max_abs_z_R": np.abs(pos[:,:,2]).max(axis=0).tolist(),
        "minimum_center_distance_R": float(sep.min()),
        "minimum_surface_gap_R": float(sep.min()-0.0002-0.00189),
        "wall_clearance_R": [float(1-radii[:,0].max()-0.0002),
                              float(1-radii[:,1].max()-0.00189)],
        "energy_residual": float(energy_residual),
        "energy_residual_over_RF_work": float(abs(energy_residual)/abs(
            (work[-1,:,:2]-work[0,:,:2]).sum())),
        "angular_momentum_change": lz_change.tolist(),
        "angular_momentum_balance_residual": (lz_change-torque_delta.sum(axis=1)).tolist(),
        "late_window_turns": np.diff(angles[np.linspace(0,n,4,dtype=int)],axis=0).T.tolist()
    }
    outdir = HERE / "stage4_candidate48" / "long_runs"
    outdir.mkdir(exist_ok=True)
    stem = f"dop_cfg52_rest0_dense_{start_cycle}_{span_cycles}p_24s"
    (outdir / f"{stem}_diagnostics.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    np.savez_compressed(outdir / f"{stem}.npz", tau=times-t0, state=state[:,:12],
                        work=work, torque=torque, angles_integral=state[:,22:24])
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
