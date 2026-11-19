"""Independent DOP853 continuation for candidate 48 at twice the fitted C."""
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import ode

import verify_dop853_local as model

HERE = Path(__file__).resolve().parent
RUNS = HERE / "stage3" / "runs"
T = 2.0*np.pi


def main():
    z = np.load(HERE / "stage2_speed" / "configurations.npz")
    p = np.asarray(z["p"][48], dtype=float).copy()
    p[5] *= 2.0
    qratio = float(z["qratio"][48])
    source = RUNS / "cfg48_rest_matched_holdout_none_chi0-0_a0.0002-0.00189_Cscale2_64_15000p_64s.jsonl"
    old = next(json.loads(s) for s in source.read_text(encoding="utf-8").splitlines()
               if s.strip() and json.loads(s)["index"] == 0)
    initial = np.asarray(old["final_state"], dtype=float)
    phase = np.asarray(old["phase"], dtype=float).copy()
    phase[0] += T*15000
    phase[1] += T*p[2]*15000
    phase = np.mod(phase, T)
    periods = 25000
    times = np.arange(periods+1, dtype=float)*T
    y0 = np.zeros(34)
    y0[:12] = initial
    driver = ode(model.derivative).set_integrator(
        "dop853", rtol=2e-10, atol=2e-12, max_step=T/24, nsteps=512)
    driver.set_f_params(p, qratio, phase)
    driver.set_initial_value(y0, 0.0)
    state = np.empty((periods+1, 34), dtype=float)
    state[0] = y0
    start = time.perf_counter()
    for i, t in enumerate(times[1:], 1):
        state[i] = driver.integrate(t)
        if not driver.successful() or not np.isfinite(state[i]).all():
            raise RuntimeError(f"DOP853 failed at extension cycle {i}")
        if i % 5000 == 0:
            print(f"DOP853 continuation {i}/{periods}", flush=True)
    diag = model.evaluate(times, state, p, qratio, 0)
    diag.update({
        "key": "48:rest:0:Cscale2:continuation",
        "previous_turns_after_original_transient": old["turns"],
        "combined_turns_after_original_transient": (
            np.asarray(old["turns"], dtype=float)+np.asarray(diag["turns_after_transient"], dtype=float)
        ).tolist(),
        "previous_state_cycle": 15000,
        "extension_periods": periods,
        "total_periods_from_original_start": 15000+periods,
        "solver": "SciPy DOP853, independent polar-gradient RHS",
        "rtol": 2e-10,
        "atol": 2e-12,
        "max_step_tau": T/24,
        "scipy": scipy.__version__,
        "wall_s": time.perf_counter()-start,
        "parameters": p.tolist(),
        "charge_ratio": qratio,
        "phase_at_extension_start": phase.tolist(),
        "initial_state_from_saved_C2_cycle15000": initial.tolist(),
    })
    np.savez_compressed(RUNS / "dop_cfg48_rest0_Cscale2_extension_25000p.npz",
        tau=times, state=state[:, :12], work=state[:, 12:22].reshape(-1,2,5),
        angles_integral=state[:, 22:24], torque=state[:, 24:34].reshape(-1,2,5))
    dest = HERE / "stage3" / "dop853_c2_continuation.json"
    dest.write_text(json.dumps(diag, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(diag, indent=2), flush=True)


if __name__ == "__main__":
    main()
