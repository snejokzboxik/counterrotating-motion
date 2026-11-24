import sympy as sp
import json
from pathlib import Path

from theory.three_parameter_criterion import (
    critical_radius, exact_ratio_s, leading_cr_condition, radial_root,
    simplified_s,
)
from theory.reynolds_audit import derivative_frequencies


def test_symbolic_radius_elimination_and_s_identity():
    A, B, nu, beta, alpha, lam, c2, c4 = sp.symbols(
        "A B nu beta alpha lam c2 c4", positive=True)
    k1, k2 = 1 + alpha * beta, nu**2 + alpha * beta
    d1, d2 = k1**2 + lam**2, k2**2 + (lam * nu)**2
    r4 = (alpha * beta - 4 * alpha**2 * A**2 * c2**2 * k1 / d1) * d2 / (
        48 * alpha**2 * B**2 * c4**2 * k2)
    rcr4 = A**2 * c2**2 * d2 / (12 * B**2 * c4**2 * nu * d1)
    expected = nu / k2 * (beta * d1 / (4 * alpha * A**2 * c2**2) - k1)
    assert sp.simplify(r4 / rcr4 - expected) == 0


def test_simplified_three_parameter_interval_is_equivalent():
    nu, alpha2, Pi = sp.symbols("nu alpha2 Pi", positive=True)
    s1, s2 = (Pi - 1) / nu, (Pi / alpha2 - 1) / nu
    # Under nu>0 and 0<alpha2<1, these inequalities reduce to the claimed band.
    assert sp.diff(s1, Pi) == 1 / nu
    assert sp.solve(sp.Eq(s1, 0), Pi)[0] == 1
    assert sp.solve(sp.Eq(s1, 1), Pi)[0] == 1 + nu
    assert sp.diff(s2, Pi) == 1 / (alpha2 * nu)
    assert sp.solve(sp.Eq(s2, 1), Pi)[0] == alpha2 * (1 + nu)


def test_mfs_coefficient_and_b_cancellation_in_ratio():
    values = (0.8, 1.7, 2.4, 0.8, 0.63, 0.2, 0.7 + 0.4j, 0.3 - 0.6j)
    A, B, nu, beta, alpha, lam, c2, c4 = values
    r = radial_root(*values)
    assert r is not None
    s = exact_ratio_s(*values)
    k1, k2 = 1 + alpha * beta, nu**2 + alpha * beta
    d1, d2 = k1**2 + lam**2, k2**2 + (lam * nu)**2
    expect = nu / k2 * (beta * d1 / (4 * alpha * A**2 * abs(c2)**2) - k1)
    assert abs(s - expect) < 1e-12
    assert critical_radius(*[A, B, nu, alpha, lam, beta], c2, c4) > 0
    assert exact_ratio_s(A, B * 4, nu, beta, alpha, lam, c2, c4) == s


def test_nominal_criterion_and_open_boundary():
    assert leading_cr_condition(3.15418, 3.2196773638, 0.37814609618)
    assert not leading_cr_condition(1.0, 3.2196773638, 0.37814609618)
    assert simplified_s(3.15418, 3.2196773638, 1.0) < 1


def test_new_machine_results_are_strict_json():
    root = Path(__file__).resolve().parents[1]
    for name in ("three_parameter_criterion_validation.json",
                 "reynolds_m3_grid_audit.json", "m3_schiller_naumann_vs_linear.json"):
        obj = json.loads((root / "data" / name).read_text(encoding="utf-8"))
        json.dumps(obj, allow_nan=False)


def test_reynolds_fourier_velocity_uses_orbit_rate_and_nyquist_rule():
    omega, nu = 0.031, 3.2
    deriv = derivative_frequencies(8, 4, omega, nu)
    # theta Fourier index +1 contributes omega; phase index +1 contributes 1.
    assert abs(deriv[1, 0, 0] - omega) < 1e-14
    assert abs(deriv[0, 1, 0] - 1.0) < 1e-14
    assert abs(deriv[0, 0, 1] - nu) < 1e-14
    # Self-conjugate Nyquist modes have zero derivative in the M3 convention.
    assert abs(deriv[4, 0, 0]) < 1e-14
    assert abs(deriv[0, 2, 0]) < 1e-14
    assert abs(deriv[0, 0, 2]) < 1e-14
