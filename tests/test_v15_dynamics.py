import math

import numpy as np
import pytest

from three_phase_trap.v15_dynamics import (
    acceleration_and_jacobian,
    dimensionless_angular_momentum,
    dimensionless_torque,
    integrate_rk4_batch,
    mirror_y,
    octupole_intensity_factor,
    potential,
)


MODELS = ("Q", "SO", "RO_PLUS", "RO_MINUS")


def test_force_and_jacobian_match_independent_potential_gradient():
    state = np.array([0.23, -0.17, 0.04, -0.02])
    alpha, beta, kappa, delta, tau0, tau = 0.0125, 0.1125, -0.009, 0.37, 0.81, 1.13
    h = 1e-6
    for model in MODELS:
        force, jac = acceleration_and_jacobian(state, tau, alpha, beta, kappa, model, delta, tau0)
        grad = []
        for axis in (0, 1):
            plus = state[:2].copy(); plus[axis] += h
            minus = state[:2].copy(); minus[axis] -= h
            grad.append((potential(plus, tau, alpha, beta, kappa, model, delta, tau0)
                         - potential(minus, tau, alpha, beta, kappa, model, delta, tau0)) / (2*h))
        np.testing.assert_allclose(force, -np.asarray(grad), rtol=2e-8, atol=2e-10)
        fd = np.column_stack([
            (acceleration_and_jacobian(state + np.eye(4)[axis]*h, tau, alpha, beta, kappa,
                                       model, delta, tau0)[0]
             - acceleration_and_jacobian(state - np.eye(4)[axis]*h, tau, alpha, beta, kappa,
                                         model, delta, tau0)[0]) / (2*h)
            for axis in (0, 1)
        ])
        np.testing.assert_allclose(jac, fd, rtol=2e-8, atol=2e-10)


def test_reflection_maps_ro_plus_to_ro_minus_and_preserves_q_so():
    states = np.array([[0.21, 0.13, 0.02, 0.03], [-0.31, 0.18, -0.01, 0.04]])
    reflected = mirror_y(states)
    tau, alpha, beta, kappa, delta, tau0 = 0.79, 0.0125, 0.1125, -0.009, 0.42, 1.07
    for model in ("Q", "SO"):
        f, _ = acceleration_and_jacobian(states, tau, alpha, beta, kappa, model, delta, tau0)
        fm, _ = acceleration_and_jacobian(reflected, tau, alpha, beta, kappa, model, delta, tau0)
        expected = f.copy(); expected[:, 1] *= -1.0
        np.testing.assert_allclose(fm, expected, atol=1e-14)
    fplus, _ = acceleration_and_jacobian(reflected, tau, alpha, beta, kappa, "RO_PLUS", delta, tau0)
    fminus, _ = acceleration_and_jacobian(states, tau, alpha, beta, kappa, "RO_MINUS", delta, tau0)
    expected = fminus.copy(); expected[:, 1] *= -1.0
    np.testing.assert_allclose(fplus, expected, atol=1e-14)
    np.testing.assert_allclose(potential(reflected, tau, alpha, beta, kappa, "RO_PLUS", delta, tau0),
                               potential(states, tau, alpha, beta, kappa, "RO_MINUS", delta, tau0), atol=1e-14)


def test_mirror_reverses_lz_and_rotating_direction_is_not_kappa_sign():
    state = np.array([0.3, 0.2, -0.4, 0.7])
    assert dimensionless_angular_momentum(mirror_y(state)) == pytest.approx(-dimensionless_angular_momentum(state))
    fplus, _ = acceleration_and_jacobian(state, 0.7, 0.01, 0.1, 0.02, "RO_PLUS", 0.2)
    fminus, _ = acceleration_and_jacobian(state, 0.7, 0.01, 0.1, 0.02, "RO_MINUS", 0.2)
    fnegative_k, _ = acceleration_and_jacobian(state, 0.7, 0.01, 0.1, -0.02, "RO_PLUS", 0.2)
    assert not np.allclose(fplus, fminus)
    assert not np.allclose(fminus, fnegative_k)


def test_torque_equals_negative_angular_derivative_of_potential():
    state = np.array([0.29, -0.24, 0.03, 0.02])
    tau, alpha, beta, kappa, delta, tau0 = 0.65, 0.0125, 0.1125, -0.009, 0.31, 0.47
    h = 1e-6
    for model in MODELS:
        r = np.linalg.norm(state[:2]); theta = math.atan2(state[1], state[0])
        plus = np.array([r*math.cos(theta+h), r*math.sin(theta+h), 0., 0.])
        minus = np.array([r*math.cos(theta-h), r*math.sin(theta-h), 0., 0.])
        dV_dtheta = (potential(plus, tau, alpha, beta, kappa, model, delta, tau0)
                     - potential(minus, tau, alpha, beta, kappa, model, delta, tau0))/(2*h)
        assert dimensionless_torque(state, tau, alpha, beta, kappa, model, delta, tau0) == pytest.approx(-dV_dtheta, abs=2e-10)


def test_standing_intensity_normalization_and_delta_pi_sign_relation():
    kappa = 0.017
    assert octupole_intensity_factor("SO", math.sqrt(2)*kappa) == pytest.approx(
        octupole_intensity_factor("RO_PLUS", kappa))
    state = np.array([0.24, 0.19, 0., 0.])
    for model in ("SO", "RO_PLUS", "RO_MINUS"):
        a, _ = acceleration_and_jacobian(state, 0.91, 0.01, 0.12, kappa, model, 0.4)
        b, _ = acceleration_and_jacobian(state, 0.91, 0.01, 0.12, -kappa, model, 0.4)
        c, _ = acceleration_and_jacobian(state, 0.91, 0.01, 0.12, kappa, model, 0.4+math.pi)
        np.testing.assert_allclose(c-a, b-a, atol=1e-14)


def test_rk4_torque_identity_and_mirror_classification():
    states = np.array([[0.08, 0.03, 0.02, 0.01], [0.11, -0.07, -0.03, 0.02]])
    for model in ("Q", "SO", "RO_PLUS"):
        result = integrate_rk4_batch(states, 0.02, 0.11, -0.006, model,
                                     delta=0.33, tau0=0.71, horizon_rf=2,
                                     steps_per_rf=256, aperture=2.0)
        assert np.max(np.abs(result["torque_identity_residual"])) < 2e-8
        mirrored = integrate_rk4_batch(mirror_y(states), 0.02, 0.11, -0.006,
                                       "RO_MINUS" if model == "RO_PLUS" else model,
                                       delta=0.33, tau0=0.71, horizon_rf=2,
                                       steps_per_rf=256, aperture=2.0)
        assert list(result["status"]) == list(mirrored["status"])
        np.testing.assert_allclose(mirrored["ell_end"], -result["ell_end"], atol=2e-10)
