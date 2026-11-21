import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerical"))
from nonautonomous_qr_local15 import rhs_and_jacobian


class ExactLocal15TangentCheck(unittest.TestCase):
    def test_analytic_tangent_jacobian_matches_finite_difference(self):
        y = np.array([0.31, -0.27, 0.04, 0.02], dtype=float)
        tau = 3.72
        alpha, lam = 0.71, 0.16
        phase1, phase2 = 0.43, -0.29
        _, analytic = rhs_and_jacobian(tau, y, alpha, lam, phase1, phase2)

        def state_rhs(state):
            return rhs_and_jacobian(tau, state, alpha, lam,
                                    phase1, phase2)[0]

        h = 1e-6
        numerical = np.column_stack([
            (state_rhs(y+np.eye(4)[i]*h)-state_rhs(y-np.eye(4)[i]*h))/(2*h)
            for i in range(4)
        ])
        np.testing.assert_allclose(analytic, numerical, rtol=2e-8, atol=2e-9)


if __name__ == "__main__":
    unittest.main()
