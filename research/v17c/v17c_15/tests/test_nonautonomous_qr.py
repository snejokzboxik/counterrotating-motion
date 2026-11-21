import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerical"))
from nonautonomous_qr_local15 import rhs_and_jacobian
import nonautonomous_qr_panel as qr_panel


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

    def test_panel_qr_runner_resumes_from_checkpoint(self):
        starts, panel_sha = qr_panel.load_panel()
        original_qr_block_periods = qr_panel.QR_BLOCK_PERIODS
        with tempfile.TemporaryDirectory(dir=qr_panel.OUT) as temp:
            try:
                qr_panel.QR_BLOCK_PERIODS = 1
                task_one = ("species1", "S01", starts["S01"], panel_sha,
                            1, temp)
                first = qr_panel.integrate_one(task_one)
                self.assertEqual(first["status"], "completed")
                self.assertEqual(first["periodic_diagnostics"][-1]["period"], 1)

                task_two = ("species1", "S01", starts["S01"], panel_sha,
                            2, temp)
                resumed = qr_panel.integrate_one(task_two)
                self.assertEqual(resumed["status"], "completed")
                self.assertEqual(resumed["computed_steps_this_run"],
                                 qr_panel.STEPS_PER_PERIOD)
                self.assertEqual(resumed["periodic_diagnostics"][-1]["period"], 2)
            finally:
                qr_panel.QR_BLOCK_PERIODS = original_qr_block_periods


if __name__ == "__main__":
    unittest.main()
