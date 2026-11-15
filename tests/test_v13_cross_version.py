"""Independent cross-version control: ideal rotating m=4 in V13 versus V12."""
import numpy as np
from scipy.integrate import solve_ivp

from three_phase_trap.dimensionless_dynamics_v12 import DimensionlessRFModel
from three_phase_trap.mass_filter_v13 import acceleration


def test_pure_rotating_octupole_v13_matches_v12_field_and_trajectory():
    kappa = 0.013
    cosine = np.zeros(5, dtype=complex)
    sine = np.zeros(5, dtype=complex)
    cosine[4] = 1.0
    sine[4] = -1j  # Re[-i z^4] = Im[z^4], hence cos(4 theta - tau).
    old = DimensionlessRFModel(cosine, sine, kappa, aperture_rho=1.0)

    rng = np.random.default_rng(1307)
    for x, y, tau in np.column_stack((rng.uniform(-0.4, 0.4, (32, 2)),
                                       rng.uniform(0, 2*np.pi, 32))):
        np.testing.assert_allclose(acceleration(x, y, tau, kappa, 0, 0),
                                   old.acceleration(np.array([x, y]), tau),
                                   rtol=1e-13, atol=1e-15)

    initial = np.array([0.22, -0.13, 0.012, -0.009])
    stop = 18 * 2*np.pi
    new_rhs = lambda tau, state: np.r_[state[2:],
                                       acceleration(state[0], state[1], tau, kappa, 0, 0)]
    settings = dict(method="DOP853", rtol=1e-11, atol=1e-13, max_step=2*np.pi/64)
    new = solve_ivp(new_rhs, (0, stop), initial, **settings)
    prior = solve_ivp(old.rhs, (0, stop), initial, **settings)
    assert new.success and prior.success
    np.testing.assert_allclose(new.y[:, -1], prior.y[:, -1], rtol=2e-10, atol=2e-11)
