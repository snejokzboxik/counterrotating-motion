import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"numerical"))
from collective_rings import (intraring_energy_direct,intraring_energy_formula,
 radial_coulomb_force_direct,radial_coulomb_force_formula,
 vertical_laplacian,vertical_laplacian_spectrum_formula,phase_metrics)

class CollectiveRingChecks(unittest.TestCase):
    def test_ring_energy_and_radial_force(self):
        for n in (2,3,6):
            with self.subTest(n=n):
                np.testing.assert_allclose(
                    intraring_energy_direct(n,0.73,1.4,0.21),
                    intraring_energy_formula(n,0.73,1.4,0.21),
                    rtol=2e-14,atol=2e-14)
                np.testing.assert_allclose(
                    radial_coulomb_force_direct(n,0.73,1.4,0.21),
                    radial_coulomb_force_formula(n,0.73,1.4,0.21),
                    rtol=2e-14,atol=2e-14)

    def test_vertical_fourier_spectrum(self):
        for n in (2,3,6):
            with self.subTest(n=n):
                got=np.linalg.eigvalsh(vertical_laplacian(n,0.8))
                expected=np.sort(vertical_laplacian_spectrum_formula(n,0.8))
                np.testing.assert_allclose(got,expected,rtol=2e-13,atol=2e-13)
                self.assertAlmostEqual(float(got[0]),0.0,places=12)

    def test_phase_metric_flags_cluster(self):
        regular=np.arange(5)*2*np.pi/5
        clustered=np.array([0.0,0.01,0.02,3.0,3.01])
        self.assertGreater(phase_metrics(regular)["psi_N"],0.999999)
        self.assertGreater(phase_metrics(clustered)["gap_cv"],0.5)

if __name__=="__main__": unittest.main()
