from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"numerical"))

from physical_screen import build_screen  # noqa: E402
from two_particle_oseen import (  # noqa: E402
    drag_acceleration,
    drag_velocity_jacobian,
    local15_trial_model,
    pair_forces,
    rf_potential_jet,
    rhs,
)
from averaged_theory import (  # noqa: E402
    solve_branch,
    torus_oseen_average,
)
from mfs_two_particle_dynamics import MFSRFField  # noqa: E402


class DimensionalScreenTests(unittest.TestCase):
    def test_trial_pair_reconstructs_supplied_masses_and_qm_ratio(self):
        d = build_screen()
        p = d["particles"]
        self.assertAlmostEqual(p[0]["mass_kg"], 1.88574099e-10, delta=1e-17)
        self.assertAlmostEqual(p[1]["mass_kg"], 4.00788683e-10, delta=1e-17)
        self.assertAlmostEqual(d["ratios"]["(q2/m2)/(q1/m1)"], 0.3412603, places=6)
        self.assertAlmostEqual(p[0]["lambda_stokes_at_Omega1"], 0.2121740, places=6)
        self.assertAlmostEqual(p[1]["lambda_stokes_at_Omega1"], 0.1283522, places=6)
        self.assertAlmostEqual(p[0]["Cunningham_slip_factor"], 1.0021669,
                               places=7)
        self.assertAlmostEqual(p[1]["Cunningham_slip_factor"], 1.00168537,
                               places=7)

    def test_cunningham_radius_kn_convention_is_applied_as_separate_stokes_pair(self):
        d = build_screen()
        model = local15_trial_model(coulomb_scale=0.0, gravity=False)
        velocity = np.array([0.02, -0.01, 0.0])
        for i, particle in enumerate(d["particles"]):
            cc = particle["Cunningham_slip_factor"]
            base = drag_acceleration(velocity, i, model, oseen=False,
                                     cunningham_factor=1.0)
            slip = drag_acceleration(velocity, i, model, oseen=False,
                                     cunningham_factor=cc)
            np.testing.assert_allclose(slip, base/cc, rtol=1e-14, atol=0.0)


class ExactRhsTests(unittest.TestCase):
    def setUp(self):
        self.model = local15_trial_model(coulomb_scale=1.0, gravity=True)

    def test_each_ideal_potential_component_is_harmonic(self):
        pos = np.array([0.31, -0.22, 0.47])
        _, _, hess = rf_potential_jet(pos, 0.37, self.model.rf_modes,
                                      self.model.beta2)
        self.assertAlmostEqual(float(np.trace(hess)), 0.0, delta=2e-14)

    def test_coulomb_pair_force_is_equal_and_opposite(self):
        positions = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.1]])
        force = pair_forces(positions, self.model)
        np.testing.assert_allclose(force[0]+force[1], np.zeros(3), atol=1e-18)
        self.assertGreater(float(force[0] @ (positions[0]-positions[1])), 0.0)

    def test_oseen_drag_is_dissipative_and_has_correct_zero_speed_limit(self):
        u = np.array([0.0, -0.03, 0.04])
        stokes = drag_acceleration(u, 0, self.model, oseen=False)
        oseen = drag_acceleration(u, 0, self.model, oseen=True)
        self.assertLess(float(oseen @ u), 0.0)
        self.assertGreater(np.linalg.norm(oseen), np.linalg.norm(stokes))
        np.testing.assert_allclose(drag_acceleration(np.zeros(3), 0, self.model),
                                   np.zeros(3), atol=0.0)

    def test_oseen_drag_jacobian_matches_finite_difference(self):
        u = np.array([0.04, -0.02, 0.03])
        J = drag_velocity_jacobian(u, 1, self.model, oseen=True)
        h = 1e-7
        fd = np.column_stack([
            (drag_acceleration(u+np.eye(3)[k]*h, 1, self.model)
             - drag_acceleration(u-np.eye(3)[k]*h, 1, self.model))/(2*h)
            for k in range(3)
        ])
        np.testing.assert_allclose(J, fd, rtol=2e-7, atol=2e-9)

    def test_dop853_and_rk4_rhs_consistency_on_short_horizon(self):
        y0 = np.array([0.55, 0.0, 0.637, 0.0, 0.55*0.0012, 0.0,
                       0.766, 0.1, 1.868, 0.0001, -0.766*0.0010, 0.0])
        t_end = 2.0
        step = 2.0* np.pi/(64.0*3.1118)
        t_rk, y_rk = _rk4_local(self.model, y0, t_end, step)
        ref = solve_ivp(lambda t, y: rhs(t, y, self.model, oseen=True),
                        (0.0, t_end), y0, method="DOP853", rtol=1e-11,
                        atol=1e-13, max_step=step/2)
        self.assertTrue(ref.success)
        np.testing.assert_allclose(y_rk[-1], ref.y[:, -1], rtol=2e-6, atol=2e-8)

    def test_mfs_reconstructs_archived_leading_transfer_factors(self):
        field = MFSRFField(self.model, order_per_rod=24)
        check = field.leading_transfer_check()
        self.assertAlmostEqual(check["h_plus2_complex"][0],
                               check["expected_h_plus2"], delta=1e-6)
        self.assertAlmostEqual(check["h_minus4_complex"][0],
                               check["expected_h_minus4"], delta=1e-6)
        self.assertLess(check["boundary_residual"], 1e-12)

    def test_mfs_independent_rf_phase_superposition_matches_phasors(self):
        field = MFSRFField(self.model, order_per_rod=16)
        pos = np.array([0.54, -0.12, 0.0])
        phase1, phase2 = 0.73, -1.21
        g1, g2 = field.rf_gradient_phasors_dimensionless(pos)
        expected = np.real(g1*np.exp(-1j*phase1)+g2*np.exp(-1j*phase2))
        actual = field.rf_gradient_dimensionless_phases(pos, phase1, phase2)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)

    def test_mfs_rf_hessian_matches_gradient_difference_and_is_harmonic(self):
        field = MFSRFField(self.model, order_per_rod=16)
        pos = np.array([0.54, -0.12, 0.03])
        phases = (0.73, -1.21)
        _, H = field.rf_gradient_hessian_dimensionless_phases(pos, *phases)
        h = 1e-5
        fd = np.column_stack([
            (field.rf_gradient_dimensionless_phases(pos+np.eye(3)[j]*h, *phases)
             -field.rf_gradient_dimensionless_phases(pos-np.eye(3)[j]*h, *phases))/(2*h)
            for j in range(3)
        ])
        np.testing.assert_allclose(H, fd, rtol=2e-6, atol=2e-7)
        self.assertAlmostEqual(float(np.trace(H)), 0.0, delta=2e-12)

    def test_induced_dipole_force_is_gradient_of_field_energy(self):
        field = MFSRFField(self.model, order_per_rod=12)
        pos = np.array([0.43, 0.17, 0.0])
        phases = (0.31, -0.72)
        grad, hess = field.rf_gradient_hessian_dimensionless_phases(
            pos, *phases)
        analytic = hess @ grad
        step = 2.0e-6

        def energy(x):
            g = field.rf_gradient_dimensionless_phases(x, *phases)
            return 0.5*float(g @ g)

        numeric = np.zeros(3)
        for axis in range(2):
            delta = np.zeros(3)
            delta[axis] = step
            numeric[axis] = (energy(pos+delta)-energy(pos-delta))/(2*step)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-9)


class AveragingTests(unittest.TestCase):
    def test_torus_averaged_drag_quadrature_converges(self):
        model = local15_trial_model(coulomb_scale=0.0, gravity=False)
        lo = torus_oseen_average(0.55, 0.0012, 1.0,
                                 model.lambdas_stokes[0], model.rf_modes,
                                 model.beta2, 0, samples=32)
        hi = torus_oseen_average(0.55, 0.0012, 1.0,
                                 model.lambdas_stokes[0], model.rf_modes,
                                 model.beta2, 0, samples=64)
        self.assertAlmostEqual(lo["oseen_drag_theta_dimensionless"],
                               hi["oseen_drag_theta_dimensionless"], delta=2e-9)
        self.assertAlmostEqual(lo["diameter_Re_phase_rms"],
                               hi["diameter_Re_phase_rms"], delta=1e-10)

    def test_isolated_oseen_branch_keeps_counterrotation_signs(self):
        first = solve_branch(0, (0.5518, 0.0012), oseen=True, samples=32)
        second = solve_branch(1, (0.7663, -0.0010), oseen=True, samples=32)
        self.assertTrue(first["success"] and second["success"])
        self.assertGreater(first["omega"], 0.0)
        self.assertLess(second["omega"], 0.0)


def _rk4_local(model, y0, t_end, h):
    n = int(np.ceil(t_end/h))
    h = t_end/n
    y = np.asarray(y0, dtype=float).copy()
    rows = [y.copy()]
    for k in range(n):
        t = k*h
        k1 = rhs(t, y, model, oseen=True)
        k2 = rhs(t+h/2, y+h*k1/2, model, oseen=True)
        k3 = rhs(t+h/2, y+h*k2/2, model, oseen=True)
        k4 = rhs(t+h, y+h*k3, model, oseen=True)
        y += h*(k1+2*k2+2*k3+k4)/6
        rows.append(y.copy())
    return np.linspace(0.0, t_end, n+1), np.asarray(rows)


if __name__ == "__main__":
    unittest.main()
