import math
import importlib.util
from pathlib import Path

import numpy as np

from three_phase_trap.mass_filter_v13 import acceleration as old_acceleration
from three_phase_trap.mass_filter_v13_3 import acceleration, linear_matrix, potential

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_v13_3_atlas.py"
_SPEC = importlib.util.spec_from_file_location("v13_3_runner", _RUNNER_PATH)
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def test_v13_3_delta_zero_matches_v13_ro():
    for x, y, tau, kappa, beta, alpha, tau0 in [
        (0.23, -0.17, 0.91, -0.011, 0.18, 0.027, 0.0),
        (-0.31, 0.09, 4.3, 0.022, 0.12, -0.015, math.pi / 2),
    ]:
        expected = old_acceleration(x, y, tau + tau0, kappa, beta, alpha)
        np.testing.assert_allclose(acceleration(x, y, tau, kappa, beta, alpha,
                                                delta=0.0, tau0=tau0), expected,
                                   rtol=2e-14, atol=2e-14)


def test_acceleration_is_negative_gradient_of_potential():
    x, y, tau, kappa, beta, alpha, delta, tau0 = 0.28, -0.16, 1.37, -0.017, 0.21, -0.031, 0.42, 0.7
    h = 2e-6
    dx = (potential(x+h, y, tau, kappa, beta, alpha, delta, tau0)
          - potential(x-h, y, tau, kappa, beta, alpha, delta, tau0)) / (2*h)
    dy = (potential(x, y+h, tau, kappa, beta, alpha, delta, tau0)
          - potential(x, y-h, tau, kappa, beta, alpha, delta, tau0)) / (2*h)
    np.testing.assert_allclose(acceleration(x, y, tau, kappa, beta, alpha, delta, tau0),
                               [-dx, -dy], rtol=2e-8, atol=2e-9)


def test_relative_phase_pi_flips_only_rotating_octupole():
    x, y, tau, kappa, beta, alpha = 0.19, 0.22, 2.1, 0.03, 0.14, 0.02
    full = acceleration(x, y, tau, kappa, beta, alpha, delta=0.0)
    quadrupole = acceleration(x, y, tau, 0.0, beta, alpha, delta=0.0)
    flipped = acceleration(x, y, tau, kappa, beta, alpha, delta=math.pi)
    np.testing.assert_allclose(flipped, 2*quadrupole-full, rtol=2e-14, atol=2e-14)


def test_linearization_is_independent_of_octupole():
    alpha, beta, tau, tau0 = 0.027, 0.175, 0.84, math.pi/2
    eps = 1e-7
    jx = acceleration(eps, 0, tau, 1.0, beta, alpha, delta=0.7, tau0=tau0)[0] / eps
    jy = acceleration(0, eps, tau, 1.0, beta, alpha, delta=0.7, tau0=tau0)[1] / eps
    expected = linear_matrix(alpha, beta, tau, tau0)
    np.testing.assert_allclose([jx, jy], [expected[0, 0], expected[1, 1]],
                               rtol=1e-11, atol=1e-11)


def test_atlas_worker_force_matches_validated_model():
    args = (0.17, -0.26, 2.41, -0.024, 0.19, 0.033)
    np.testing.assert_allclose(_RUNNER.acceleration(*args, delta=0.31, tau0=1.2),
                               acceleration(*args, delta=0.31, tau0=1.2),
                               rtol=2e-14, atol=2e-14)
