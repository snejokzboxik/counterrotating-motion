import numpy as np
from scipy.integrate import solve_ivp

from three_phase_trap.dimensionless_dynamics_v12 import DimensionlessRFModel, integrate_segment


def rotating_mode(order=4, coupling=0.004, phase=0.0, handedness=1):
    c = np.zeros(order + 1, dtype=complex)
    s = np.zeros(order + 1, dtype=complex)
    c[order] = 1.0
    s[order] = -1j * handedness
    return DimensionlessRFModel(c, s, coupling, 1.0, phase, f"pure m={order}")


def test_complex_polynomial_gradient_matches_finite_difference():
    c = np.array([0j, 0.2 + 0.1j, -0.3 + 0.07j, 0.04 - 0.11j])
    s = np.array([0j, -0.1 + 0.05j, 0.12 + 0.2j, 0.08 + 0.03j])
    model = DimensionlessRFModel(c, s, 0.7, aperture_rho=1.0, phase_rad=0.31)
    point = np.array([0.21, -0.17])
    tau = 1.13
    _, gradient = model.potential_and_gradient(point, tau)
    step = 2e-7
    numerical = np.empty(2)
    for axis in range(2):
        plus, minus = point.copy(), point.copy()
        plus[axis] += step
        minus[axis] -= step
        fp = model.potential_and_gradient(plus, tau)[0]
        fm = model.potential_and_gradient(minus, tau)[0]
        numerical[axis] = (fp - fm) / (2 * step)
    assert np.allclose(gradient, numerical, rtol=2e-8, atol=2e-10)


def test_physical_scaling_produces_expected_dimensionless_coupling():
    e = 1.602176634e-19
    mass = 100_000 * 1.66053906660e-27
    omega = 2 * np.pi * 100_000
    radius = 8e-3
    cosine = np.array([0j, 0.0j, 0.0j, 0.0j, 0.5413596632303368 + 0j])
    sine = np.array([0j, 0.0j, 0.0j, 0.0j, -1j * 0.5413596632303368])
    model = DimensionlessRFModel.from_physical(
        cosine, sine, charge_c=e, mass_kg=mass, voltage_v=250.0,
        omega_rad_s=omega, length_m=radius, reference_radius_m=radius,
        aperture_m=radius,
    )
    expected = e * 250 / (mass * omega**2 * radius**2)
    assert model.coupling == expected
    assert np.isclose(model.aperture_rho, 1.0)


def test_pseudopotential_includes_both_traveling_rf_quadratures():
    model = rotating_mode(order=4, coupling=0.01)
    rho = 0.37
    expected = (0.01**2 * 4**2 / 2.0) * rho**6
    assert np.isclose(model.pseudopotential(np.array([rho, 0.0])), expected,
                      rtol=2e-14, atol=1e-18)


def test_rotating_pure_multipole_jacobi_integral_is_conserved():
    model = rotating_mode(order=4, coupling=0.004)
    state0 = np.array([0.22, 0.08, -0.003, 0.012])
    omega_pattern = 1.0 / 4.0
    angular_momentum = lambda s: s[0] * s[3] - s[1] * s[2]
    jacobi = lambda s, t: model.hamiltonian(s, t) - omega_pattern * angular_momentum(s)
    end = 80 * 2 * np.pi
    times = np.linspace(0.0, end, 1601)
    solution = solve_ivp(model.rhs, (0.0, end), state0, method="DOP853",
                         rtol=2e-12, atol=2e-14, max_step=2*np.pi/80,
                         t_eval=times)
    assert solution.success
    values = np.array([jacobi(solution.y[:, i], t) for i, t in enumerate(times)])
    assert np.ptp(values) < 2e-10 * max(1.0, abs(values[0]))


def test_segment_restart_preserves_rf_phase_and_state():
    model = rotating_mode(order=4, coupling=0.003, phase=0.27)
    state0 = np.array([0.16, 0.07, -0.001, 0.009])
    one = integrate_segment(model, state0, 0.0, 20.0, rtol=2e-11,
                            atol=2e-13, max_step_rf=1/64)
    a = integrate_segment(model, state0, 0.0, 10.0, rtol=2e-11,
                          atol=2e-13, max_step_rf=1/64)
    b = integrate_segment(model, a.final_state, 10.0, 20.0, rtol=2e-11,
                          atol=2e-13, max_step_rf=1/64)
    assert one.status == a.status == b.status == "retained"
    assert np.allclose(one.final_state, b.final_state, rtol=2e-9, atol=2e-11)


def test_trial_stage_field_query_is_clipped_to_aperture():
    model = rotating_mode(order=4, coupling=0.004)
    state = np.array([1.0001, 0.0, 0.0, 0.0])
    # A just-outside RK stage must remain evaluable on the FEM-valid disk.
    derivative = model.rhs(0.3, state)
    assert np.all(np.isfinite(derivative))


def test_reflection_reverses_traveling_field_and_angular_momentum():
    forward = rotating_mode(order=4, coupling=0.004, handedness=1)
    reflected = rotating_mode(order=4, coupling=0.004, handedness=-1)
    state = np.array([0.23, 0.12, -0.006, 0.011])
    mirror = np.array([state[0], -state[1], state[2], -state[3]])
    rhs = forward.rhs(0.73, state)
    rhs_mirror = reflected.rhs(0.73, mirror)
    assert np.allclose(rhs_mirror, [rhs[0], -rhs[1], rhs[2], -rhs[3]], rtol=2e-14, atol=2e-15)
    lz = state[0]*state[3]-state[1]*state[2]
    lz_mirror = mirror[0]*mirror[3]-mirror[1]*mirror[2]
    assert np.isclose(lz_mirror, -lz)


def test_first_aperture_crossing_is_a_separate_status():
    model = rotating_mode(order=4, coupling=1e-4)
    result = integrate_segment(model, [0.99, 0.0, 0.15, 0.0], 0.0, 1.0,
                               rtol=2e-10, atol=2e-12, max_step_rf=1/64)
    assert result.status == "escaped"
    assert result.escape_period is not None and 0 < result.escape_period < 1
    assert np.isclose(np.linalg.norm(result.final_state[:2]), model.aperture_rho, atol=1e-10)
