"""Checkpointable 120k-cycle RK4 trajectory for the validated candidate-48 rest start."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
RUNS = HERE / "stage3" / "runs"
TAU = 2.0 * np.pi
dop_derivative = None


def load_dependencies():
    global dop_derivative
    import sys
    sys.path.insert(0, str(HERE))
    import verify_dop853_local as dop
    dop_derivative = dop.derivative
    return dop


@njit(cache=True, nogil=True)
def rk4_chunk(cycle_start, y_start, p, qratio, phase, ncycles, steps):
    dt = TAU / steps
    states = np.empty((ncycles + 1, 34), dtype=np.float64)
    states[0] = y_start
    minimum_distance = 1e300
    min_radius = np.array([1e300, 1e300])
    max_radius = np.zeros(2)
    max_abs_z = np.zeros(2)
    y = y_start.copy()
    for cycle in range(ncycles):
        global_cycle = cycle_start + cycle
        for j in range(steps):
            t = global_cycle * TAU + j * dt
            k1 = dop_derivative(t, y, p, qratio, phase)
            y2 = y + 0.5 * dt * k1
            k2 = dop_derivative(t + 0.5 * dt, y2, p, qratio, phase)
            y3 = y + 0.5 * dt * k2
            k3 = dop_derivative(t + 0.5 * dt, y3, p, qratio, phase)
            y4 = y + dt * k3
            k4 = dop_derivative(t + dt, y4, p, qratio, phase)
            yn = y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            for sample in (y, y2, y3, y4, yn):
                for i, b in enumerate((0, 6)):
                    rr = np.sqrt(sample[b]*sample[b] + sample[b+1]*sample[b+1])
                    min_radius[i] = min(min_radius[i], rr)
                    max_radius[i] = max(max_radius[i], rr)
                    max_abs_z[i] = max(max_abs_z[i], abs(sample[b+2]))
                dx = sample[0]-sample[6]
                dy = sample[1]-sample[7]
                dz = sample[2]-sample[8]
                minimum_distance = min(minimum_distance, np.sqrt(dx*dx+dy*dy+dz*dz))
            y = yn
        states[cycle+1] = y
    extrema = np.array([minimum_distance, min_radius[0], min_radius[1],
                        max_radius[0], max_radius[1], max_abs_z[0], max_abs_z[1]])
    return states, extrema


def energy_terms(y, p, qratio):
    pos = y[[0, 1, 2, 6, 7, 8]]
    v = y[[3, 4, 5, 9, 10, 11]].reshape(2, 3)
    radii2 = np.array([pos[0]**2+pos[1]**2, pos[3]**2+pos[4]**2])
    mass = np.array([1.0, 1.0/p[4]])
    ke = 0.5*np.sum(mass[:, None]*v*v)
    dc = p[3]*(pos[2]**2-0.5*radii2[0] + qratio*(pos[5]**2-0.5*radii2[1]))
    d = np.linalg.norm(pos[:3]-pos[3:])
    uc = p[5]/d
    return float(ke), float(dc), float(uc)


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", type=int, default=48)
    ap.add_argument("--candidate-source", choices=("stage2_speed", "stage4_candidate48"), default="stage2_speed")
    ap.add_argument("--ensemble", choices=("rest", "weak", "wide"), default="rest")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--source-jsonl", type=Path, default=None,
                    help="explicit source row file; required for tagged candidate52 holdout")
    ap.add_argument("--cycles", type=int, default=120000)
    ap.add_argument("--checkpoint-cycles", type=int, default=5000)
    ap.add_argument("--steps", type=int, choices=(64, 128), default=64)
    ap.add_argument("--a1-over-R", type=float, default=0.0002)
    ap.add_argument("--a2-over-R", type=float, default=0.00189)
    ap.add_argument("--coupling-scale", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dop = load_dependencies()
    z = np.load(HERE / a.candidate_source / "configurations.npz")
    p = np.asarray(z["p"][a.config_id], dtype=float)
    p[5] *= a.coupling_scale
    qratio = float(z["qratio"][a.config_id])
    if a.candidate_source == "stage4_candidate48":
        output_dir = HERE / "stage4_candidate48" / "long_runs"
        default_source = HERE / "stage4_candidate48" / "holdout_runs" / (
            f"cfg{a.config_id}_{a.ensemble}_matched_holdout_none_chi0-0_a0.0002-0.00189_"
            f"pairedholdout20261150_128_15000p_64s.jsonl")
    else:
        output_dir = RUNS
        default_source = RUNS / f"cfg{a.config_id}_{a.ensemble}_holdout_128_15000p_64s.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = a.source_jsonl if a.source_jsonl is not None else default_source
    row = next(json.loads(s) for s in source.read_text(encoding="utf-8").splitlines()
               if s.strip() and json.loads(s)["index"] == a.index)
    phase = np.asarray(row["phase"], dtype=float)
    y0 = np.zeros(34)
    y0[:12] = np.asarray(row["initial"], dtype=float)

    tag = f"long_cfg{a.config_id}_{a.ensemble}{a.index}_{a.cycles}p_{a.steps}s"
    if a.coupling_scale != 1.0:
        tag += f"_Cscale{a.coupling_scale:g}"
    mem_path = output_dir / f"{tag}_checkpoint.npy"
    checkpoint = output_dir / f"{tag}_checkpoint.json"
    metadata = {"candidate": a.config_id, "candidate_source": a.candidate_source,
                "ensemble": a.ensemble, "index": a.index, "cycles": a.cycles,
                "steps_per_RF_period": a.steps, "checkpoint_cycles": a.checkpoint_cycles,
                "p": p.tolist(), "qratio": qratio, "phase": phase.tolist(),
                "source": str(source), "coupling_scale": a.coupling_scale,
                "solver": "RK4 full RF RHS from independent DOP853 formulation"}
    if a.resume:
        if not mem_path.exists() or not checkpoint.exists():
            raise FileNotFoundError("checkpoint files are absent")
        checkpoint_data = json.loads(checkpoint.read_text(encoding="utf-8"))
        if checkpoint_data["metadata"] != metadata:
            raise RuntimeError("long-run checkpoint parameters mismatch")
        states = np.lib.format.open_memmap(mem_path, mode="r+")
        done = int(checkpoint_data["completed_cycles"])
    else:
        if mem_path.exists() or checkpoint.exists():
            raise FileExistsError("checkpoint exists; pass --resume or move it aside")
        states = np.lib.format.open_memmap(mem_path, mode="w+", dtype=np.float64, shape=(a.cycles+1, 34))
        states[0] = y0
        states.flush()
        done = 0
        atomic_json(checkpoint, {"metadata": metadata, "completed_cycles": 0,
                                 "extrema": [1e300, 1e300, 1e300, 0, 0, 0, 0]})

    extrema = np.asarray(json.loads(checkpoint.read_text(encoding="utf-8"))["extrema"], dtype=float)
    while done < a.cycles:
        n = min(a.checkpoint_cycles, a.cycles-done)
        chunk, ext = rk4_chunk(done, np.asarray(states[done]).copy(), p, qratio, phase, n, a.steps)
        states[done+1:done+n+1] = chunk[1:]
        states.flush()
        extrema[0] = min(extrema[0], ext[0])
        extrema[1:3] = np.minimum(extrema[1:3], ext[1:3])
        extrema[3:7] = np.maximum(extrema[3:7], ext[3:7])
        done += n
        atomic_json(checkpoint, {"metadata": metadata, "completed_cycles": done,
                                 "extrema": extrema.tolist()})
        print(f"checkpoint {done}/{a.cycles} cycles", flush=True)

    state = np.asarray(states)
    times = np.arange(a.cycles+1, dtype=float)*TAU
    pos = state[:, [0, 1, 2, 6, 7, 8]].reshape(-1, 2, 3)
    angles = np.unwrap(np.arctan2(pos[:, :, 1], pos[:, :, 0]), axis=0)
    skip = 200
    turns = (angles[-1]-angles[skip])/TAU
    cuts = np.linspace(skip, a.cycles, 4, dtype=int)
    windows = np.diff(angles[cuts], axis=0)/TAU
    freq = turns/(a.cycles-skip)*float(dop.F1)
    radius = np.linalg.norm(pos[:, :, :2], axis=2)
    work = state[:, 12:22].reshape(-1, 2, 5)
    torque = state[:, 24:34].reshape(-1, 2, 5)
    work_delta = work[-1]-work[skip]
    torque_delta = torque[-1]-torque[skip]
    energy_initial = energy_terms(state[skip], p, qratio)
    energy_final = energy_terms(state[-1], p, qratio)
    ext_work = float(work_delta[:, :2].sum()+work_delta[:, 4].sum())
    energy_change = float(sum(energy_final)-sum(energy_initial))
    residual = energy_change-ext_work
    lz = (pos[:, :, 0]*state[:, [4, 10]] - pos[:, :, 1]*state[:, [3, 9]])
    lz[:, 1] /= p[4]
    diagnostics = {
        "candidate_id": a.config_id, "candidate_source": a.candidate_source,
        "start": f"{a.ensemble}:{a.index} from N=128 holdout", "cycles": a.cycles,
        "steps_per_period": a.steps, "solver": f"RK4, {a.steps} uniform substeps per primary RF cycle",
        "initial_state": row["initial"], "rf_phase": phase.tolist(), "parameters": p.tolist(),
        "frequency_Hz_after_200_cycle_transient": freq.tolist(), "turns_after_transient": turns.tolist(),
        "late_window_turns": windows.tolist(), "mean_radius_R": radius[skip:].mean(axis=0).tolist(),
        "r_min_R_substep_extrema": extrema[1:3].tolist(), "r_max_R_substep_extrema": extrema[3:5].tolist(),
        "max_abs_z_R_substep_extrema": extrema[5:7].tolist(),
        "minimum_center_distance_R_substep_extrema": float(extrema[0]),
        "sphere_radii_over_R": [a.a1_over_R, a.a2_over_R],
        "minimum_surface_gap_R": float(extrema[0]-a.a1_over_R-a.a2_over_R),
        "work_RF1_RF2_DC_Coulomb_drag_per_particle": work_delta.tolist(),
        "torque_RF1_RF2_DC_Coulomb_drag_per_particle": torque_delta.tolist(),
        "angular_momentum_change": (lz[-1]-lz[skip]).tolist(),
        "angular_momentum_balance_residual": ((lz[-1]-lz[skip])-torque_delta.sum(axis=1)).tolist(),
        "mechanical_energy_change": energy_change, "external_RF_plus_drag_work": ext_work,
        "energy_residual": residual,
        "energy_residual_over_abs_RF_work": abs(residual)/max(abs(float(work_delta[:, :2].sum())),1e-300),
        "average_RF_power_plus_drag_power_per_tau": ext_work/(TAU*(a.cycles-skip)),
        "classification": "CR_long_horizon" if (
            turns[0]*turns[1] < 0 and np.all(windows[:,0]*windows[:,1] < 0)
            and np.all(np.abs(turns)>=10) and extrema[3] + a.a1_over_R <= 1.0
            and extrema[4] + a.a2_over_R <= 1.0 and extrema[5] + a.a1_over_R <= 2.0
            and extrema[6] + a.a2_over_R <= 2.0 and np.all(extrema[1:3] > 0.02)
            and extrema[0] >= a.a1_over_R+a.a2_over_R
        ) else "review_required",
        "checkpoint_file": mem_path.name,
    }
    np.savez_compressed(output_dir / f"{tag}.npz", tau=times, state=state[:, :12],
                        work=work, torque=torque, angles_integral=state[:, 22:24])
    diagnostics_path = output_dir / f"{tag}_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2)+"\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
        stride = max(1, a.cycles//50000)
        for i, label in enumerate(("particle 1", "particle 2")):
            ax[0].plot(pos[skip::stride, i, 0], pos[skip::stride, i, 1], lw=.45, label=label)
            ax[2].plot(times[skip::stride]/TAU, radius[skip::stride, i], lw=.6, label=label)
        ax[0].set(xlabel="x/R", ylabel="y/R", aspect="equal", title="XY trajectories")
        ax[1].plot(times[skip::stride]/TAU, (angles[skip::stride]-angles[skip])/TAU)
        ax[1].set(xlabel="RF cycles", ylabel="accumulated turns", title="Unwrapped azimuth")
        ax[2].set(xlabel="RF cycles", ylabel="r/R", title="Orbital radii")
        fig.legend(*ax[2].get_legend_handles_labels(), loc="upper center", ncol=2,
                   bbox_to_anchor=(.5, 1.02), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(output_dir / f"{tag}.png", dpi=170)
        plt.close(fig)
    except Exception as exc:
        diagnostics["plot_error"] = repr(exc)
        diagnostics_path.write_text(
            json.dumps(diagnostics, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
