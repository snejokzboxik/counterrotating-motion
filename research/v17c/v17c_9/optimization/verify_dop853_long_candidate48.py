"""Full-horizon independent DOP853 verification for candidate 48 rest:0."""
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
    cycles = 120000
    z = np.load(HERE / "stage2_speed" / "configurations.npz")
    p = np.asarray(z["p"][48], dtype=float)
    qratio = float(z["qratio"][48])
    src = RUNS / "cfg48_rest_holdout_128_15000p_64s.jsonl"
    start = next(json.loads(s) for s in src.read_text(encoding="utf-8").splitlines()
                 if s.strip() and json.loads(s)["index"] == 0)
    initial = np.asarray(start["initial"], dtype=float)
    phase = np.asarray(start["phase"], dtype=float)
    times = np.arange(cycles+1, dtype=float)*T
    y0 = np.zeros(34)
    y0[:12] = initial
    driver = ode(model.derivative).set_integrator(
        "dop853", rtol=2e-10, atol=2e-12, max_step=T/24, nsteps=512)
    driver.set_f_params(p, qratio, phase)
    driver.set_initial_value(y0, 0.0)
    state = np.empty((cycles+1, 34), dtype=float)
    state[0] = y0
    tic = time.perf_counter()
    for i, t in enumerate(times[1:], 1):
        state[i] = driver.integrate(t)
        if not driver.successful() or not np.isfinite(state[i]).all():
            raise RuntimeError(f"DOP853 failed at cycle {i}")
        if i % 20000 == 0:
            print(f"DOP853 long validation {i}/{cycles}", flush=True)
    diag = model.evaluate(times, state, p, qratio, 200)
    diag.update({"key": "48:rest:0:long120k", "config_id": 48, "candidate_source": "stage2_speed",
        "solver": "SciPy DOP853, independent polar-gradient RF implementation",
        "rtol": 2e-10, "atol": 2e-12, "max_step_tau": T/24, "scipy": scipy.__version__,
        "wall_s": time.perf_counter()-tic, "parameters": p.tolist(), "charge_ratio": qratio,
        "initial": initial.tolist(), "rf_phase": phase.tolist(), "cycles": cycles})
    np.savez_compressed(RUNS / "dop_long_cfg48_rest0_120000p.npz", tau=times,
        state=state[:, :12], work=state[:, 12:22].reshape(-1,2,5),
        angles_integral=state[:, 22:24], torque=state[:, 24:34].reshape(-1,2,5))
    dest = HERE / "stage3" / "dop853_long_candidate48.json"
    dest.write_text(json.dumps(diag, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(diag, indent=2), flush=True)


if __name__ == "__main__":
    main()
