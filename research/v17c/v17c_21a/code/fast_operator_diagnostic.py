"""Locate close-to-resonant fast Fourier denominators and mask boundaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "v17c" / "v17c_21" / "code"))
import analytical_model as legacy  # noqa: E402


def main():
    data = legacy.load_inputs()
    campaign = json.loads((BASE / "results" / "full_grid_campaign.json").read_text(encoding="utf-8"))
    stage = next(x for x in campaign["stages"] if x["grid"] == [192, 16, 16])
    root = stage["root"]
    ntheta, nphase = 192, 16
    kt = np.fft.fftfreq(ntheta) * ntheta
    kp = np.fft.fftfreq(nphase) * nphase
    kt, kp1, kp2 = np.meshgrid(kt, kp, kp, indexing="ij")
    rows = []
    for i in range(2):
        omega = root["omega_over_Omega1"][i]
        lam = data["lambda"][i]
        alpha = data["alpha"][i]
        f1 = next(m.frequency for m in data["modes"] if m.family == 0)
        f2 = next(m.frequency for m in data["modes"] if m.family == 1)
        kt_eff = np.where(kt == -ntheta // 2, 0.0, kt)
        kp1_eff = np.where(kp1 == -nphase // 2, 0.0, kp1)
        kp2_eff = np.where(kp2 == -nphase // 2, 0.0, kp2)
        kappa = kt_eff * omega + kp1_eff * f1 + kp2_eff * f2
        d0 = -kappa**2 + 1j * kappa * lam - omega**2 - alpha * data["beta"]
        dj = 2j * omega * kappa + lam * omega
        den = d0 * d0 + dj * dj
        rf_combo = (kp1 != 0) | (kp2 != 0)
        for cutoff in (0.02, 0.035, 0.05, 0.075, 0.10):
            fast = rf_combo & (np.abs(kappa) >= cutoff)
            excluded = rf_combo & ~fast
            flat_idx = np.flatnonzero(fast)
            arg = int(flat_idx[np.argmin(np.abs(den.ravel()[flat_idx]))])
            index = np.unravel_index(arg, den.shape)
            min_fast_kappa = int(np.argmin(np.where(rf_combo, np.abs(kappa), np.inf).ravel()))
            min_rf_index = np.unravel_index(min_fast_kappa, kappa.shape)
            d = d0[index]
            j = dj[index]
            # Singular values of the complex 2x2 operator [[d0,-dJ],[dJ,d0]].
            s = abs(d) ** 2 + abs(j) ** 2
            off = -np.conjugate(d) * j + np.conjugate(j) * d
            smn = np.sqrt(max(s - abs(off), 0.0))
            smx = np.sqrt(max(s + abs(off), 0.0))
            rows.append({
                "particle": i + 1, "cutoff": cutoff,
                "number_fast_modes": int(np.count_nonzero(fast)),
                "number_excluded_rf_modes": int(np.count_nonzero(excluded)),
                "minimum_excluded_abs_kappa": (
                    float(np.min(np.abs(kappa[excluded]))) if np.any(excluded) else None
                ),
                "minimum_abs_kappa_any_rf_mode": float(np.abs(kappa[min_rf_index])),
                "minimum_fast_abs_determinant": float(np.abs(den[index])),
                "min_determinant_mode": [int(kt[index]), int(kp1[index]), int(kp2[index])],
                "min_determinant_kappa": float(kappa[index]),
                "min_determinant_d0_real_imag": [float(d.real), float(d.imag)],
                "min_determinant_dJ_real_imag": [float(j.real), float(j.imag)],
                "min_mode_singular_values": [float(smn), float(smx)],
                "min_mode_inverse_gain": float(1.0 / max(smn, 1e-300)),
            })
    output = {
        "study": "V17C.21.1 fast-operator resonance and mask diagnostic",
        "grid": [192, 16, 16], "nu": data["modes"][-1].frequency,
        "cutoff_rows": rows,
        "scope": "The determinant is the 2x2 fast Fourier response denominator, not a nonlinear HB Jacobian condition number.",
    }
    (BASE / "results" / "fast_operator_diagnostic.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
