"""Full-spectrum helpers for V17C.4 six-phase RF--DC dynamics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v17c_3"))
from rf_dc_core import pseudopotential, radial_shape, run_rk4

RESULTS = HERE / "results"
FEM_SPECTRUM = RESULTS / "fem_sixphase" / "G1_G2_12electrode_6phase_P1_FEM.json"


def get_coefficients(name: str = "ADCFEBADCFEB", source: str = "mfs", level: int = -1):
    doc = json.loads(FEM_SPECTRUM.read_text(encoding="utf-8"))
    mesh = doc["geometries"][name]["mesh_levels"][level]
    rows = mesh["mfs_coefficients"] if source == "mfs" else mesh["coefficients"]
    coeff = {int(k): complex(float(re), float(im)) for k, re, im in rows}
    # The repeated phase sequence has exact pi-rotational symmetry: odd k
    # vanish analytically. Remove only FEM/MFS numerical round-off leakage.
    coeff = {k: (v if abs(k) % 2 == 0 else 0j) for k, v in coeff.items()}
    modes = np.asarray(sorted(coeff), dtype=np.int64)
    values = np.asarray([coeff[int(k)] for k in modes], np.complex128)
    return modes, values


def make_geometry(name: str = "ADCFEBADCFEB", source: str = "mfs", level: int = -1):
    modes, coeff = get_coefficients(name, source, level)
    return {"label": name, "N": 12 if name in ("ADCFEBADCFEB", "ABCDEFABCDEF", "ABCABCABCABC") else 6,
            "modes": modes, "coeff": coeff}


def pseudo(alpha: float, beta: float, x: float, y: float, modes, coeff, lam: float = 0.0):
    return pseudopotential(alpha, beta, x, y, modes, coeff, lam)


def pseudo_gradient(alpha: float, beta: float, xy, modes, coeff, h: float = 2e-6):
    x, y = map(float, xy)
    f = lambda a, b: pseudo(alpha, beta, a, b, modes, coeff)
    return np.asarray([(f(x+h, y)-f(x-h, y))/(2*h),
                       (f(x, y+h)-f(x, y-h))/(2*h)])


def pseudo_hessian(alpha: float, beta: float, xy, modes, coeff, h: float = 5e-5):
    x, y = map(float, xy)
    return np.column_stack([
        (pseudo_gradient(alpha, beta, [x+h, y], modes, coeff)
         - pseudo_gradient(alpha, beta, [x-h, y], modes, coeff))/(2*h),
        (pseudo_gradient(alpha, beta, [x, y+h], modes, coeff)
         - pseudo_gradient(alpha, beta, [x, y-h], modes, coeff))/(2*h),
    ])


def torque_shape(coeff: dict[int, complex], r: float):
    return float(sum(np.sign(k)*abs(k)**2*(abs(k)-1)*abs(c)**2*r**(2*abs(k)-2)
                     for k, c in coeff.items()))


def chirality(coeff: dict[int, complex], r: float):
    terms = [np.sign(k)*abs(k)**2*(abs(k)-1)*abs(c)**2*r**(2*abs(k)-2)
             for k, c in coeff.items()]
    return float(sum(terms)/max(sum(abs(x) for x in terms), 1e-300))


def angular_profile(alpha: float, r: float, modes, coeff, samples: int = 1440):
    theta = np.arange(samples)*2*np.pi/samples
    u = r*np.exp(1j*theta)
    fx = np.zeros(samples, complex); fy = np.zeros(samples, complex)
    for k, c in zip(modes, coeff):
        n = abs(int(k)); base = u if k > 0 else np.conjugate(u)
        f = -n*c*base**(n-1)
        fx += f
        fy += (-1j*f) if k > 0 else (1j*f)
    potential = alpha*alpha*(np.abs(fx)**2+np.abs(fy)**2)/4.0
    return theta, potential


def angular_barrier(alpha: float, r: float, modes, coeff):
    theta, profile = angular_profile(alpha, r, modes, coeff)
    imin, imax = int(np.argmin(profile)), int(np.argmax(profile))
    return {"r_over_R": r, "theta_min_rad": float(theta[imin]),
            "U_min": float(profile[imin]), "U_max": float(profile[imax]),
            "barrier_max_minus_min": float(profile[imax]-profile[imin]),
            "corrugation": float((profile.max()-profile.min())/profile.mean()),
            "profile_theta": theta.tolist(), "profile_U": profile.tolist()}


def barrier_crossing_omega(alpha: float, r: float, modes, coeff, factor: float = 1.0):
    b = angular_barrier(alpha, r, modes, coeff)["barrier_max_minus_min"]
    return float(np.sqrt(max(2*factor*b, 0.0))/r)


def full_torque_prefactor(r: float, modes, coeff, lam: float):
    # The azimuthally averaged RF tangential acceleration is this torque
    # polynomial divided by r, multiplied by lambda/(1+lambda^2).
    cdict = {int(k): complex(c) for k, c in zip(modes, coeff)}
    return float(lam/(1+lam*lam)*torque_shape(cdict, r)/max(r, 1e-30))

