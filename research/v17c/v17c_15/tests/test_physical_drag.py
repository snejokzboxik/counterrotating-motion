import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"numerical"))
from physical_drag import (stokes_gamma,legacy_gamma_with_reversed_slip,
                           dimensionless_lambda)

class CunninghamConventionChecks(unittest.TestCase):
    def test_old_to_corrected_ratio_is_cc_squared(self):
        for cc in (1.0,1.0007639056387618,1.0083661228712946):
            old=legacy_gamma_with_reversed_slip(2e-5,1.81e-5,cc)
            new=stokes_gamma(2e-5,1.81e-5,cc)
            self.assertAlmostEqual(old/new,cc**2,places=13)

    def test_slip_increases_mobility_and_lambda_is_dimensionless(self):
        base=stokes_gamma(2e-5,1.81e-5,1.0)
        slipped=stokes_gamma(2e-5,1.81e-5,1.01)
        self.assertAlmostEqual(slipped/base,1/1.01,places=14)
        lam=dimensionless_lambda(2e-5,1.81e-5,1e-10,300.0,1.01)
        self.assertAlmostEqual(lam,slipped/(1e-10*300.0),places=14)

if __name__=="__main__": unittest.main()
