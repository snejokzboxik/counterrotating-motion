"""Small-coupling full RF versus leading averaged dynamics control."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from three_phase_trap.mass_filter_v13 import acceleration, effective_potential

ROOT = Path(__file__).resolve().parents[1] / "outputs/octupole_comparison_home/mass_filter_v13"


def main():
    k, b, a = .01, .02, .0001
    q0 = np.array([.25, .15])
    h = 1e-6

    def grad_c(q):
        return -acceleration(*q, 0, k, b, 0)

    def grad_s(q):
        return -acceleration(*q, np.pi/2, k, b, 0)

    def grad_eff(q):
        return np.array([(effective_potential(*(q+np.array([h,0])),k,b,a)-effective_potential(*(q-np.array([h,0])),k,b,a))/(2*h),
                         (effective_potential(*(q+np.array([0,h])),k,b,a)-effective_potential(*(q-np.array([0,h])),k,b,a))/(2*h)])

    full0 = np.r_[q0+grad_c(q0), grad_s(q0)]
    avg0 = np.r_[q0, [0.,0.]]
    times = 2*np.pi*np.arange(21)
    full = solve_ivp(lambda t, y: np.r_[y[2:], acceleration(*y[:2],t,k,b,a)],
                     (0,times[-1]), full0, t_eval=times, method="DOP853", rtol=1e-11, atol=1e-13)
    avg = solve_ivp(lambda t, y: np.r_[y[2:], -grad_eff(y[:2])],
                    (0,times[-1]), avg0, t_eval=times, method="DOP853", rtol=1e-11, atol=1e-13)
    if not full.success or not avg.success:
        raise RuntimeError(f"integration failure: {full.message}; {avg.message}")
    errors = []
    with (ROOT / "averaging_control.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_rf", "full_x", "full_y", "avg_plus_micro_x", "avg_plus_micro_y", "position_error"])
        for n in range(len(times)):
            target = avg.y[:2,n] + grad_c(avg.y[:2,n])
            error = float(np.linalg.norm(full.y[:2,n]-target))
            errors.append(error)
            w.writerow([n, *full.y[:2,n], *target, error])
    (ROOT / "averaging_control.json").write_text(json.dumps({
        "kappa": k, "beta": b, "alpha": a, "q0": q0.tolist(), "slow_velocity0": [0,0],
        "initial_micromotion": "r=q+grad C, v=grad S", "n_rf": 20,
        "rtol": 1e-11, "atol": 1e-13, "max_stroboscopic_position_error": max(errors),
        "last_position_error": errors[-1], "model": "ideal leading averaged potential"
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
