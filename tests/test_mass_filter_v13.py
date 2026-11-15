import numpy as np

from three_phase_trap.mass_filter_v13 import acceleration, effective_potential, linear_status, monodromy, potential


def test_cartesian_gradient_and_limiting_fields():
    x, y, tau = 0.23, -0.17, 0.63
    cases = [(0, .2, .04), (.1, 0, 0), (0, .2, 0), (0, 0, .04), (.1, .2, .04)]
    h = 1e-6
    for k, b, a in cases:
        grad = np.array([(potential(x+h, y, tau, k,b,a)-potential(x-h,y,tau,k,b,a))/(2*h),
                         (potential(x,y+h,tau,k,b,a)-potential(x,y-h,tau,k,b,a))/(2*h)])
        np.testing.assert_allclose(acceleration(x,y,tau,k,b,a), -grad, atol=1e-10)
    np.testing.assert_allclose(acceleration(x,y,tau,0,0,.04), [-.08*x, .08*y])


def test_effective_potential_time_quadrature():
    x, y, k, b, a = .23, -.17, .13, .19, .03
    h = 1e-5
    taus = 2*np.pi*np.arange(1024)/1024
    rf = np.array([acceleration(x,y,t,k,b,0) for t in taus])
    direct = np.mean(np.sum(rf*rf, axis=1))/2 + a*(x*x-y*y)
    np.testing.assert_allclose(effective_potential(x,y,k,b,a), direct, rtol=1e-12)


def test_polar_cartesian_force_agreement():
    rho, theta, tau, k, b, a = .41, .73, 1.12, .09, .17, .03
    x, y = rho*np.cos(theta), rho*np.sin(theta)
    f = acceleration(x,y,tau,k,b,a)
    radial = -4*k*rho**3*np.cos(4*theta-tau) - 2*(b*np.cos(tau)+a)*rho*np.cos(2*theta)
    tangential = 4*k*rho**3*np.sin(4*theta-tau) + 2*(b*np.cos(tau)+a)*rho*np.sin(2*theta)
    np.testing.assert_allclose([f @ [np.cos(theta),np.sin(theta)],
                                f @ [-np.sin(theta),np.cos(theta)]],
                               [radial,tangential], atol=1e-14)


def test_monodromy_and_mathieu_symmetry():
    m = monodromy(.017, .19)
    np.testing.assert_allclose(np.linalg.det(m), 1, atol=1e-9)
    np.testing.assert_allclose(np.trace(m), np.trace(monodromy(-.017,.19,"y")), atol=1e-9)
    assert linear_status(0,.1)["status"] == "strictly_stable"
    assert linear_status(.05,0)["status"] == "unstable"
    assert linear_status(0,0)["status"] == "boundary_or_unresolved"
