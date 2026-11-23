from __future__ import annotations

"""Independent algebraic/sign checks for the V17C.21 averaging implementation."""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analytical_model as model  # noqa: E402


def direct_rotating_field(w, theta, psi, mode, alpha=1.0):
    """Rebuild the V17C.17 lab-frame field, then rotate it to the orbit frame."""
    n = mode.n
    z_lab = np.exp(1j * theta) * w
    P = mode.amplitude * mode.coefficient
    c = P * np.exp(-1j * psi)
    if mode.k > 0:
        # Pair integrator: Ex+iEy = -conj(n*c*z**(n-1)).
        E_lab = -n * np.conjugate(c * z_lab ** (n - 1))
    else:
        # Pair integrator: Ex+iEy = -n*c*conj(z)**(n-1).
        E_lab = -n * c * np.conjugate(z_lab) ** (n - 1)
    return alpha * np.exp(-1j * theta) * E_lab


def test_full_rf_field_matches_pair_convention():
    data = model.load_inputs()
    modes = data["modes"]
    rng = np.random.default_rng(210021)
    for _ in range(40):
        theta = float(rng.uniform(-np.pi, np.pi))
        psi = (float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(-np.pi, np.pi)))
        w = complex(*rng.normal(size=2)) * 0.25
        for m in modes:
            actual = model._rf_field_rotating(np.asarray(w), np.asarray(theta),
                                              tuple(np.asarray(x) for x in psi),
                                              1.0, [m]).item()
            expected = direct_rotating_field(w, theta, psi[m.family], m)
            assert abs(actual - expected) < 2e-13 * max(1.0, abs(expected))


def test_complex_phase_projection():
    data = model.load_inputs()
    modes = data["modes"]
    # Need enough theta samples to resolve the largest within-family mode gap
    # (72); smaller grids alias distinct MFS modes and are not valid quadrature.
    for ntheta, nphase in ((96, 8), (128, 10)):
        theta, phases = model._phase_grid(ntheta, nphase)
        w = np.full(theta.shape, 0.57 + 0j)
        rf = model._rf_field_rotating(w, theta, phases, 1.0, modes)
        vec = np.stack((rf.real, rf.imag), axis=-1)
        for h, m in enumerate(modes):
            eta = np.exp(1j * (np.angle(m.coefficient) + m.k * theta - phases[m.family]))
            measured = 2 * np.mean(vec * np.conjugate(eta)[..., None], axis=(0, 1, 2))
            force = m.amplitude * abs(m.coefficient) * m.n * 0.57 ** (m.n - 1)
            expected = -force * np.array([1.0, 1j * m.chirality])
            assert np.max(np.abs(measured - expected)) < 5e-12 * max(1.0, force)


def test_coulomb_action_reaction_and_dimensionless_identities():
    C, s = sp.symbols("C s", nonzero=True, real=True)
    x1, y1, z1, x2, y2, z2 = sp.symbols("x1 y1 z1 x2 y2 z2", real=True)
    dvec = sp.Matrix([x1-x2, y1-y2, z1-z2])
    d3 = sp.sqrt(dvec.dot(dvec))**3
    acc1 = C*dvec/d3
    acc2 = -s*C*dvec/d3
    assert all(sp.simplify(v) == 0 for v in acc1 + acc2/s)
    # SI exponent bookkeeping in the basis (M,L,T,Q).
    add = lambda *items: tuple(sum(x[j] for x in items) for j in range(4))
    neg = lambda x: tuple(-v for v in x)
    M, L, T, Q = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
    eps = ( -1, -3, 2, 2)  # vacuum permittivity: Q^2 T^2 M^-1 L^-3
    volt = (1, 2, -2, -1)
    viscosity = (1, -1, -1, 0)
    C_dim = add(Q, Q, neg(eps), neg(M), tuple(-3*x for x in L), tuple(2*x for x in T))
    beta_dim = add(Q, volt, neg(M), tuple(-2*x for x in L), tuple(2*x for x in T))
    lambda_dim = add(viscosity, L, neg(M), neg(tuple(-1*x for x in T)))
    amplitude_dim = add(Q, volt, neg(M), tuple(-2*x for x in L), tuple(2*x for x in T))
    assert C_dim == beta_dim == lambda_dim == amplitude_dim == (0, 0, 0, 0)


def test_rotating_frame_and_laplace_identities():
    t, omega = sp.symbols("t omega", real=True)
    x = sp.Matrix([sp.Function("x")(t), sp.Function("y")(t)])
    Qrot = sp.Matrix([[sp.cos(omega*t), -sp.sin(omega*t)],
                      [sp.sin(omega*t), sp.cos(omega*t)]])
    J = sp.Matrix([[0, -1], [1, 0]])
    lab_acc = sp.diff(Qrot*x, t, 2)
    in_rotating = sp.simplify(Qrot.T*lab_acc)
    expected = sp.diff(x, t, 2) + 2*omega*J*sp.diff(x, t) - omega**2*x
    assert all(sp.simplify(a-b) == 0 for a, b in zip(in_rotating, expected))

    xx, yy = sp.symbols("x y", real=True)
    for n in (2, 4, 20, 46):
        harmonic = sp.re((xx + sp.I*yy)**n)
        assert sp.simplify(sp.diff(harmonic, xx, 2) + sp.diff(harmonic, yy, 2)) == 0


if __name__ == "__main__":
    test_full_rf_field_matches_pair_convention()
    test_complex_phase_projection()
    test_coulomb_action_reaction_and_dimensionless_identities()
    test_rotating_frame_and_laplace_identities()
    print("V17C.21 algebraic/sign audit: PASS")
