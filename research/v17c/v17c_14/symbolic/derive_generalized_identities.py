"""Reproducible SymPy checks for V17C.14 identities; no trajectory fit."""
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

OUT = Path(__file__).with_name("derivation_checks.json")
x, y, z = sp.symbols("x y z", real=True)
rho2 = x**2 + y**2
beta2, beta4 = sp.symbols("beta_2 beta_4", real=True)
H2 = z**2 - rho2 / 2
H4 = z**4 - 3 * rho2 * z**2 + sp.Rational(3, 8) * rho2**2
lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, z, 2)

assert sp.simplify(lap(H2)) == 0
assert sp.simplify(lap(H4)) == 0
assert sp.simplify(sp.diff(beta2 * H2 + beta4 * H4, z, 2) -
                   (2 * beta2 + beta4 * (12 * z**2 - 6 * rho2))) == 0

n, nu, lam, alpha, amp, r = sp.symbols("n nu lambda alpha a rho", positive=True, real=True)
den_plus = -nu**2 + sp.I * lam * nu
f0 = -n * alpha * amp * r**(n - 1)
xi0 = f0 / den_plus
# For k>0, Fc=-n alpha a conjugate(z)^(n-1) exp(+i nu tau).
# Its linearized mean is dFc/dconjugate(z) times conjugate(xi0).
mean_Fc = sp.simplify(
    -n * (n - 1) * alpha * amp * r**(n - 2) * sp.conjugate(xi0)
)
expected_real = -alpha**2 * n**2 * (n - 1) * amp**2 * r**(2*n - 3) / (nu**2 + lam**2)
expected_imag = alpha**2 * n**2 * (n - 1) * amp**2 * r**(2*n - 3) * lam / (nu * (nu**2 + lam**2))
assert sp.simplify(sp.re(mean_Fc) - expected_real) == 0
assert sp.simplify(sp.im(mean_Fc) - expected_imag) == 0

# Exact moment/guide-center split in the polar phasor calculation.
Q, mu, sig = sp.symbols("Q mu sigma", nonzero=True, real=True)
Mgc = sig * Q**2 / mu * n**2 * (n - 1) * amp**2 * lam / (nu * (nu**2 + lam**2)) * r**(2*n - 2)
Lmic = sig * Q**2 / mu * n**2 * amp**2 / (nu * (nu**2 + lam**2)) * r**(2*n - 2)
Mrf = sig * Q**2 / mu * n**3 * amp**2 * lam / (nu * (nu**2 + lam**2)) * r**(2*n - 2)
assert sp.simplify(Mrf - Mgc - lam * Lmic) == 0

# Coulomb pair action-reaction and torque cancellation.
xi, yi, zi, xj, yj, zj = sp.symbols("x_i y_i z_i x_j y_j z_j", real=True)
d = sp.Matrix([xi-xj, yi-yj, zi-zj])
qprod, kap = sp.symbols("qprod kappa", real=True)
Fi = kap * qprod * d / (d.dot(d))**sp.Rational(3, 2)
Fj = -Fi
ri = sp.Matrix([xi, yi, zi])
rj = sp.Matrix([xj, yj, zj])
pair_torque = sp.simplify(ri.cross(Fi) + rj.cross(Fj))
assert all(v == 0 for v in pair_torque)

checks = {
    "laplace_H2": True,
    "laplace_H4": True,
    "H4_axial_curvature": sp.sstr(sp.diff(beta2 * H2 + beta4 * H4, z, 2)),
    "averaged_force_radial": sp.sstr(expected_real),
    "averaged_force_tangential_positive_chirality": sp.sstr(expected_imag),
    "moment_split_identity": True,
    "coulomb_pair_torque_cancellation": True,
    "sympy_version": sp.__version__,
    "scope": "symbolic identities; no nonlinear-orbit stability theorem",
}
OUT.write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(checks, indent=2, ensure_ascii=False))
