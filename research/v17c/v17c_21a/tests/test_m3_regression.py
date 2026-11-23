"""Regression checks for the V17C.21.1 M3 numerical gates."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"


class M3RegressionTests(unittest.TestCase):
    def test_even_grid_nyquist_derivative_is_real(self):
        rng = np.random.default_rng(2101)
        shape = (24, 12, 12)
        values = rng.standard_normal(shape)
        k0, k1, k2 = [np.fft.fftfreq(n)*n for n in shape]
        k0[k0 == -shape[0]//2] = 0.0
        k1[k1 == -shape[1]//2] = 0.0
        k2[k2 == -shape[2]//2] = 0.0
        grids = np.meshgrid(k0, k1, k2, indexing="ij")
        derivative_symbol = grids[0]*0.013 + grids[1]*0.71 + grids[2]*3.22
        derivative = np.fft.ifftn(1j*derivative_symbol*np.fft.fftn(values))
        self.assertLess(float(np.max(np.abs(derivative.imag))), 2e-13)

    def test_every_grid_is_a_verified_full_root(self):
        data = json.loads((RESULTS/"full_grid_campaign.json").read_text(encoding="utf-8"))
        self.assertEqual([s["grid"] for s in data["stages"]],
                         [[96,8,8],[96,12,12],[128,12,12],[192,16,16]])
        for stage in data["stages"]:
            self.assertEqual(stage["status"], "VERIFIED_ROOT")
            self.assertTrue(stage["inner_all_converged"])
            self.assertLessEqual(stage["scaled_balance_max_norm"], stage["scaled_balance_tolerance"])
            for part in stage["inner"]:
                self.assertLessEqual(part["relative_original_fast_equation_residual"], part["equation_tolerance"])

    def test_engineering_grid_convergence_targets(self):
        stages=json.loads((RESULTS/"full_grid_campaign.json").read_text(encoding="utf-8"))["stages"]
        ref=stages[-1]["root"]
        for stage in stages:
            root=stage["root"]
            for a,b in zip(root["radii_R"],ref["radii_R"]):
                self.assertLess(abs(100*(a/b-1)),0.05)
            for a,b in zip(root["frequency_Hz"],ref["frequency_Hz"]):
                self.assertLess(abs(100*(a/b-1)),0.1)

    def test_solver_never_accepts_failed_legacy_case(self):
        rows=json.loads((RESULTS/"solver_benchmark.json").read_text(encoding="utf-8"))["rows"]
        p1={r["method"]:r for r in rows if r["particle"]==1}
        self.assertFalse(p1["fixed_point"]["converged"])
        self.assertFalse(p1["adaptive"]["converged"])
        self.assertTrue(p1["newton_krylov"]["converged"])
        self.assertLess(p1["newton_krylov"]["relative_original_fast_equation_residual"],
                        p1["newton_krylov"]["equation_tolerance"])

    def test_cutoff_change_does_not_move_verified_root(self):
        stages=json.loads((RESULTS/"cutoff_sensitivity.json").read_text(encoding="utf-8"))["stages"]
        base=next(s["root"] for s in stages if abs(s["cutoff"]-.05)<1e-12)
        for stage in stages:
            self.assertEqual(stage["status"],"VERIFIED_ROOT")
            for a,b in zip(stage["root"]["radii_R"],base["radii_R"]):
                self.assertLess(abs(100*(a/b-1)),1e-6)
            for a,b in zip(stage["root"]["frequency_Hz"],base["frequency_Hz"]):
                self.assertLess(abs(100*(a/b-1)),1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
