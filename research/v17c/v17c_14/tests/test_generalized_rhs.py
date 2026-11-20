from __future__ import annotations

import unittest

import numpy as np

from research.v17c.v17c_14.numerical.generalized_dynamics import (
    RFFamily, dc_value_gradient_hessian, potential_value_gradient_hessian,
    rhs, tangent_jacobian)


class GeneralizedDynamicsTests(unittest.TestCase):
    def test_harmonic_dc_laplacian(self):
        _, _, h2 = dc_value_gradient_hessian(np.array([0.21, -0.18, 0.33]), 0.7, 0.0)
        _, _, h4 = dc_value_gradient_hessian(np.array([0.21, -0.18, 0.33]), 0.0, 0.4)
        self.assertAlmostEqual(float(np.trace(h2)), 0.0, places=14)
        self.assertAlmostEqual(float(np.trace(h4)), 0.0, places=14)

    def test_pair_momentum_conservation(self):
        masses=np.array([1.0, 2.3]); charges=np.array([1.2, 2.1])
        y=np.array([0.2,0.1,-0.05,0,0,0, -0.3,0.4,0.12,0,0,0],float)
        acc=rhs(0.3,y,masses,charges,np.zeros(2),0.017,[])
        momentum_rate=masses[0]*acc[3:6]+masses[1]*acc[9:12]
        np.testing.assert_allclose(momentum_rate,0,atol=1e-14)

    def test_rf_gradient_and_hessian_match_finite_differences(self):
        families=[RFFamily(0.07+0.01j,1.3,(2,-4),(1.0+0j,0.2-0.03j))]
        pos=np.array([0.21,-0.15,0.03])
        value,grad,hess=potential_value_gradient_hessian(pos,0.37,families,0.014,0.008)
        h=1e-5
        basis=np.eye(3)
        grad_fd=np.array([(potential_value_gradient_hessian(pos+basis[j]*h,0.37,families,0.014,0.008)[0]-
                           potential_value_gradient_hessian(pos-basis[j]*h,0.37,families,0.014,0.008)[0])/(2*h)
                          for j in range(3)])
        hess_fd=np.column_stack([(potential_value_gradient_hessian(pos+basis[j]*h,0.37,families,0.014,0.008)[1]-
                                  potential_value_gradient_hessian(pos-basis[j]*h,0.37,families,0.014,0.008)[1])/(2*h)
                                 for j in range(3)])
        np.testing.assert_allclose(grad,grad_fd,rtol=1e-7,atol=1e-9)
        np.testing.assert_allclose(hess,hess_fd,rtol=2e-6,atol=2e-8)
        self.assertAlmostEqual(float(np.trace(hess)),0.0,places=12)

    def test_jacobian_matches_central_difference(self):
        families=[RFFamily(0.07+0.01j,1.3,(2,-4),(1.0+0j,0.2-0.03j))]
        masses=np.array([1.0,1.7]); charges=np.array([1.0,2.0])
        drag=np.array([0.12,0.08])
        y=np.array([0.21,-0.15,0.03,0.04,-0.02,0.01,
                    -0.31,0.27,-0.09,-0.01,0.03,0.02],float)
        args=(masses,charges,drag,2e-4,families,0.014,0.008)
        J=tangent_jacobian(0.37,y,*args)
        h=2e-6
        Jfd=np.column_stack([(rhs(0.37,y+np.eye(y.size)[j]*h,*args)-
                              rhs(0.37,y-np.eye(y.size)[j]*h,*args))/(2*h)
                             for j in range(y.size)])
        np.testing.assert_allclose(J,Jfd,rtol=3e-7,atol=2e-8)

    def test_permutation_equivariance_for_identical_particles(self):
        masses=np.ones(2); charges=np.ones(2); drag=np.full(2,0.1)
        families=[RFFamily(0.05+0j,1.0,(2,-4),(1+0j,0.2+0j))]
        y=np.array([0.21,0.17,0.0,0.02,-0.01,0.0,
                    -0.36,0.29,0.0,-0.03,0.01,0.0])
        args=(masses,charges,drag,1e-4,families,0.01,0.0)
        a=rhs(0.2,y,*args).reshape(2,6)
        b=rhs(0.2,y.reshape(2,6)[::-1].reshape(-1),*args).reshape(2,6)[::-1]
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)


if __name__ == "__main__":
    unittest.main()
