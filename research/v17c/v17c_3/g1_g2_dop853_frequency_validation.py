"""Independent DOP853 validation of full-P1-FEM translation Floquet modes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from g1_g2_translation_spectrum import hessian
from rf_dc_core import RESULTS, _rf_accel

HERE = Path(__file__).resolve().parent
ROOT = RESULTS / "g1_g2_joint"
FEM = ROOT / "independent_fem_spectrum" / "G1_G2_full_P1_FEM_spectrum.json"
OUT = ROOT / "G1_G2_DOP853_translation_frequency_validation.json"
PERIOD = 2.0 * np.pi


def make_map(modes, coeff, alpha, beta, lam):
    def rhs(tau, s):
        x, y, vx, vy = s
        arx, ary = _rf_accel(x, y, tau, modes, coeff, alpha)
        return (vx, vy, arx + beta*x - lam*vx, ary + beta*y - lam*vy)

    def integrate(y0, sample=False):
        t_eval = np.linspace(0.0, PERIOD, 257) if sample else [PERIOD]
        sol = solve_ivp(rhs, (0.0, PERIOD), np.asarray(y0, float), method="DOP853",
                        rtol=2e-13, atol=2e-15, max_step=PERIOD/256,
                        t_eval=t_eval)
        if not sol.success:
            raise RuntimeError(sol.message)
        if sample:
            return sol.t, sol.y.T
        return sol.y[:, -1]
    return integrate


def main() -> None:
    fem_data = json.loads(FEM.read_text(encoding="utf-8"))
    old_orbits = {}
    for name in ("G1", "G2"):
        arr = np.load(ROOT / f"{name}_rf_periodic_orbit_monodromy.npz")
        old_orbits[name] = arr["initial"]

    results = {}
    for name, doc in fem_data.items():
        fit = doc["fine_mesh_full_spectrum_floquet_fit"]
        entries = doc["fine_mesh_coefficients"]
        modes = np.asarray([row[0] for row in entries], dtype=np.int64)
        coeff = np.asarray([complex(row[1], row[2]) for row in entries], dtype=np.complex128)
        alpha = .01
        beta = float(fit["beta"])
        lam = 1e-5
        one_period = make_map(modes, coeff, alpha, beta, lam)
        sol = root(lambda y: one_period(y) - y, old_orbits[name], method="hybr",
                   options={"xtol": 1e-11, "maxfev": 300})
        fixed = np.asarray(sol.x, float)
        end = one_period(fixed)
        t, orbit = one_period(fixed, sample=True)
        center = np.mean(orbit[:, :2], axis=0)
        H = hessian(center, fit["B"], modes, coeff)
        pseudo = alpha*np.sqrt(np.maximum(np.linalg.eigvalsh(H), 0.0))
        scales = np.asarray([1e-5, 1e-5, 1e-6, 1e-6])
        monodromy = np.empty((4, 4))
        for j, step in enumerate(scales):
            delta = np.zeros(4)
            delta[j] = step
            monodromy[:, j] = (one_period(fixed+delta)-one_period(fixed-delta))/(2*step)
        multipliers = np.linalg.eigvals(monodromy)
        frequencies = np.sort(np.abs(np.angle(multipliers))/(2*np.pi))
        unique = []
        for freq in frequencies:
            if not unique or abs(float(freq)-unique[-1]) > 2e-6:
                unique.append(float(freq))
            else:
                unique[-1] = .5*(unique[-1]+float(freq))
        full = np.asarray(sorted(unique)[:2])
        match = []
        for pred in pseudo:
            exact = float(full[np.argmin(abs(full-pred))])
            match.append({"pseudopotential_omega_tau": float(pred),
                          "DOP853_Floquet_omega_tau": exact,
                          "relative_error": float((exact-pred)/pred)})
        old = np.asarray(fit["floquet_frequencies_omega_tau"])
        result = {
            "geometry": name, "method": "independent solve_ivp DOP853, full fine-mesh P1-FEM spectrum",
            "rtol": 2e-13, "atol": 2e-15, "max_step_over_RF_period": 1/256,
            "alpha": alpha, "beta": beta, "lambda": lam, "modes": modes.tolist(),
            "poincare_root_success": bool(sol.success),
            "poincare_residual_norm": float(np.linalg.norm(end-fixed)),
            "guiding_center_xy": center.tolist(), "pseudopotential_hessian": H.tolist(),
            "pseudopotential_frequencies_omega_tau": pseudo.tolist(),
            "DOP853_Floquet_multipliers_abs": np.abs(multipliers).tolist(),
            "DOP853_Floquet_frequencies_omega_tau": full.tolist(),
            "pseudo_to_DOP853_matches": match,
            "DOP853_to_RK4_full_spectrum_frequency_relative_delta": [
                float((a-b)/b) for a, b in zip(full, old)],
            "orbit_samples": len(t),
        }
        results[name] = result
        np.savez_compressed(ROOT / f"{name}_DOP853_translation_orbit_monodromy.npz",
                            initial=fixed, orbit_times=t, orbit_states=orbit,
                            monodromy=monodromy, multipliers=multipliers,
                            mean_xy=center)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
