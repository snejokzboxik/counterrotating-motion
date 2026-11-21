import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerical"))
from general_nbody import (RFFamily, potential_value_gradient_hessian,
                           rhs, tangent_jacobian, coulomb_forces,
                           mechanical_energy, angular_momenta)


def fd_gradient(fun, x, h=2e-6):
    g=np.zeros_like(x, dtype=float)
    for k in range(len(x)):
        d=np.zeros_like(x); d[k]=h
        g[k]=(fun(x+d)-fun(x-d))/(2*h)
    return g


def fd_jacobian(fun, x, h=1e-6):
    J=np.zeros((len(x),len(x)))
    for k in range(len(x)):
        d=np.zeros_like(x); d[k]=h
        J[:,k]=(fun(x+d)-fun(x-d))/(2*h)
    return J


class ExactModelChecks(unittest.TestCase):
    def setUp(self):
        self.families=(
            RFFamily(0.037+0.004j, 1.0, (2,), (1.0+0.0j,)),
            RFFamily(0.16, 3.11, (-4,), (1.0+0.0j,)),
            RFFamily(0.008, 2.4, (3,-2), (0.7+0.2j, -0.3j)),
        )

    def test_gradient_and_hessian_finite_difference(self):
        x=np.array([0.31,-0.27,0.14]); t=0.713
        v,g,H=potential_value_gradient_hessian(x,t,self.families,0.017,0.002)
        def value(y):
            return potential_value_gradient_hessian(y,t,self.families,0.017,0.002)[0]
        np.testing.assert_allclose(g,fd_gradient(value,x),rtol=2e-7,atol=2e-8)
        for k in range(3):
            d=np.zeros(3); d[k]=2e-5
            gp=potential_value_gradient_hessian(x+d,t,self.families,0.017,0.002)[1]
            gm=potential_value_gradient_hessian(x-d,t,self.families,0.017,0.002)[1]
            np.testing.assert_allclose(H[:,k],(gp-gm)/(4e-5),rtol=2e-6,atol=2e-7)
        self.assertAlmostEqual(float(np.trace(H)),0.0,places=10)

    def test_full_rhs_jacobian(self):
        y=np.array([0.23,-0.17,0.08,0.02,-0.01,0.03,
                    -0.19,0.29,-0.05,-0.03,0.04,-0.02])
        args=([1.0,2.7],[1.0,-0.8],[0.2,0.13],0.04,self.families,0.017,0.002)
        fun=lambda z: rhs(0.37,z,*args)
        J=tangent_jacobian(0.37,y,*args)
        np.testing.assert_allclose(J,fd_jacobian(fun,y),rtol=3e-6,atol=3e-7)

    def test_pair_force_total_and_coulomb_torque(self):
        x=np.array([[0.3,0.0,0.1],[-0.2,0.4,-0.1],[0.1,-0.3,0.05]])
        q=np.array([1.3,-0.7,2.1])
        F=coulomb_forces(x,q,0.8)
        np.testing.assert_allclose(np.sum(F,axis=0),0.0,atol=2e-15)
        torque=np.sum(x[:,0]*F[:,1]-x[:,1]*F[:,0])
        self.assertAlmostEqual(float(torque),0.0,places=14)

    def test_mass_scaling_and_energy_identity_at_zero_drive(self):
        # Pair acceleration changes as 1/m_i; pair forces cancel in momentum.
        y=np.array([0.3,0.0,0.0,0.0,0.1,0.0,
                    -0.4,0.1,0.0,0.0,-0.2,0.0])
        masses=np.array([1.0,3.0]); charges=np.array([1.0,2.0])
        zero=()
        d=rhs(0.0,y,masses,charges,[0.0,0.0],0.07,zero)
        acc=d.reshape(2,6)[:,3:]
        np.testing.assert_allclose(np.sum(masses[:,None]*acc,axis=0),0.0,atol=2e-15)
        Ldot=np.sum(masses*(y.reshape(2,6)[:,0]*acc[:,1]-
                           y.reshape(2,6)[:,1]*acc[:,0]))
        self.assertAlmostEqual(float(Ldot),0.0,places=14)

    def test_energy_derivative_drag_only(self):
        fam=()
        masses=np.array([1.0,2.0]); charges=np.array([1.0,1.2])
        lam=np.array([0.15,0.08]); kap=0.03
        y=np.array([0.2,0.1,0.0,0.03,-0.02,0.01,
                    -0.3,-0.1,0.0,-0.01,0.04,-0.02])
        f=rhs(0.0,y,masses,charges,lam,kap,fam,0.02,0.0)
        h=2e-6
        yp=y+h*f
        ym=y-h*f
        dEdt=(mechanical_energy(0.0,yp,masses,charges,kap,fam,0.02,0.0)-
              mechanical_energy(0.0,ym,masses,charges,kap,fam,0.02,0.0))/(2*h)
        expected=-float(np.sum(masses[:,None]*lam[:,None]*y.reshape(-1,6)[:,3:]**2))
        self.assertAlmostEqual(dEdt,expected,places=8)

if __name__ == "__main__":
    unittest.main()
