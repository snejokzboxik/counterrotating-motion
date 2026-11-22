"""Full 2D MFS RF-field gradients coupled to the dimensionless pair RHS.

The electrode bases solve the 24-cylinder/grounded-shield boundary problem in
the infinite-rod cross section. Harmonics above k=+2 and k=-4 are retained by
evaluating the full MFS solution rather than a leading-multipole expansion.
The ideal DC quadrupole remains a separate analytic term.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve

from physical_screen import MFS, PAIR, build_screen
from mfs_breakdown import boundary_system, gradient_matrix, hessian_matrix
from two_particle_oseen import (TwoParticleModel, drag_acceleration,
                                pair_forces)


class MFSRFField:
    def __init__(self, model: TwoParticleModel, order_per_rod: int = 32):
        self.model = model
        self.order_per_rod = int(order_per_rod)
        G, sources, theta, V = boundary_system(self.order_per_rod)
        self.sources = sources
        self.strengths = solve(G, V, assume_a="gen", check_finite=False)
        self.boundary_residual = float(np.max(np.abs(G@self.strengths-V)))
        self.theta = theta
        screen = build_screen()
        drive = screen["field_and_electrodes"]
        self.U1 = drive["U1_electrode_peak_V"]
        self.U2 = drive["U2_electrode_peak_V"]
        self.D = drive["D_m"]
        self.R = PAIR["radial_scale_R_m"]
        self.r_over_D = self.R/self.D
        self.q1 = PAIR["charges_e"][0]*1.602176634e-19
        self.m1 = screen["particles"][0]["mass_kg"]
        self.Omega1 = 2.0*math.pi*PAIR["f1_Hz"]
        self.nu = PAIR["frequency_ratio_nu"]
        self.vscale = self.m1*self.R**2*self.Omega1**2/self.q1
        # Fourier coefficients at r=R/D reproduce the archived MFS transfers.
        ph = 2.0*math.pi*np.arange(4096)/4096
        rr = self.r_over_D*np.exp(1j*ph)
        Gp = self._potential_matrix(rr)
        value = Gp@self.strengths
        self.h2 = complex(np.mean(value[:, 0]*np.exp(-2j*ph)))
        self.hminus4 = complex(np.mean(value[:, 1]*np.exp(4j*ph)))

    def _potential_matrix(self, target: np.ndarray) -> np.ndarray:
        z = np.asarray(target, dtype=complex)[:, None]
        s = self.sources[None, :]
        from mfs_breakdown import B_D
        G = np.log(np.abs(B_D*B_D-z*np.conjugate(s))
                   /(B_D*np.abs(z-s)))
        return G

    def leading_transfer_check(self) -> dict:
        return {
            "h_plus2_complex": [float(self.h2.real), float(self.h2.imag)],
            "h_minus4_complex": [float(self.hminus4.real), float(self.hminus4.imag)],
            "expected_h_plus2": MFS["h_plus2"],
            "expected_h_minus4": MFS["h_minus4"],
            "boundary_residual": self.boundary_residual,
        }

    def rf_gradient_phasors_dimensionless(self, position: np.ndarray
                                         ) -> tuple[np.ndarray, np.ndarray]:
        g1, g2, _, _ = self.rf_gradient_hessian_phasors_dimensionless(position)
        return g1, g2

    def rf_gradient_hessian_phasors_dimensionless(self, position: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos = np.asarray(position, dtype=float)
        target = np.array([self.r_over_D*(pos[0]+1j*pos[1])], dtype=complex)
        gx, gy = gradient_matrix(target, self.sources)
        hxx, hxy, hyy = hessian_matrix(target, self.sources)
        g1x, g1y = complex((gx@self.strengths[:, 0])[0]), complex((gy@self.strengths[:, 0])[0])
        g2x, g2y = complex((gx@self.strengths[:, 1])[0]), complex((gy@self.strengths[:, 1])[0])
        h1xx = complex((hxx@self.strengths[:, 0])[0])
        h1xy = complex((hxy@self.strengths[:, 0])[0])
        h1yy = complex((hyy@self.strengths[:, 0])[0])
        h2xx = complex((hxx@self.strengths[:, 1])[0])
        h2xy = complex((hxy@self.strengths[:, 1])[0])
        h2yy = complex((hyy@self.strengths[:, 1])[0])
        # Convert derivative w.r.t. r/D to derivative w.r.t. x=r/R and scale
        # potential by q1/(m1 R^2 Omega1^2).
        factor = self.r_over_D/self.vscale
        p1 = self.U1*factor
        p2 = self.U2*factor
        # The Hessian requires a second derivative conversion d(r/D)/d(r/R).
        p1_hess = p1*self.r_over_D
        p2_hess = p2*self.r_over_D
        g1 = np.array([p1*g1x, p1*g1y, 0.0j])
        g2 = np.array([p2*g2x, p2*g2y, 0.0j])
        H1 = np.array([[p1_hess*h1xx, p1_hess*h1xy, 0.0j],
                       [p1_hess*h1xy, p1_hess*h1yy, 0.0j],
                       [0.0j, 0.0j, 0.0j]])
        H2 = np.array([[p2_hess*h2xx, p2_hess*h2xy, 0.0j],
                       [p2_hess*h2xy, p2_hess*h2yy, 0.0j],
                       [0.0j, 0.0j, 0.0j]])
        return g1, g2, H1, H2

    def rf_gradient_dimensionless_phases(self, position: np.ndarray,
                                         phase1: float,
                                         phase2: float) -> np.ndarray:
        g1, g2 = self.rf_gradient_phasors_dimensionless(position)
        return np.real(g1*np.exp(-1j*phase1)+g2*np.exp(-1j*phase2))

    def rf_gradient_hessian_dimensionless_phases(self, position: np.ndarray,
            phase1: float, phase2: float) -> tuple[np.ndarray, np.ndarray]:
        g1, g2, H1, H2 = self.rf_gradient_hessian_phasors_dimensionless(position)
        g = np.real(g1*np.exp(-1j*phase1)+g2*np.exp(-1j*phase2))
        H = np.real(H1*np.exp(-1j*phase1)+H2*np.exp(-1j*phase2))
        return g, H

    def rf_gradient_dimensionless(self, position: np.ndarray, tau: float) -> np.ndarray:
        return self.rf_gradient_dimensionless_phases(
            position, tau, self.nu*tau)


def rhs_mfs(tau: float, state, model: TwoParticleModel,
            field: MFSRFField, oseen: bool = True, coulomb: bool = True,
            cunningham_factors=(1.0, 1.0), buoyancy: bool = True,
            polarizabilities_SI=(0.0, 0.0)) -> np.ndarray:
    """Two-particle RHS using full MFS RF gradients plus ideal DC quadrupole."""
    model.validate()
    y = np.asarray(state, dtype=float).reshape((2, 6))
    out = np.zeros_like(y)
    Fpair = pair_forces(y[:, :3], model) if coulomb else np.zeros((2, 3))
    for i in range(2):
        pos = y[i, :3]
        phase1, phase2 = tau, field.nu*tau
        grad, hess = field.rf_gradient_hessian_dimensionless_phases(
            pos, phase1, phase2)
        grad += model.beta2*np.array([-pos[0], -pos[1], 2.0*pos[2]])
        hess += model.beta2*np.diag([-1.0, -1.0, 2.0])
        qi_mi = model.charges_relative[i]/model.masses_relative[i]
        acceleration = -qi_mi*grad + Fpair[i]/model.masses_relative[i]
        if polarizabilities_SI[i] != 0.0:
            mi = model.reference_mass_kg*model.masses_relative[i]
            coeff = polarizabilities_SI[i]*model.reference_mass_kg**2
            coeff *= model.omega1_rad_s**2
            coeff /= model.reference_charge_C**2*mi
            acceleration += coeff*(hess@grad)
        acceleration += drag_acceleration(y[i, 3:], i, model, oseen,
                                          cunningham_factors[i])
        if buoyancy and np.linalg.norm(model.gravity_hat) > 0.0:
            acceleration += (1.0-model.air_density_kg_m3
                             /model.densities_kg_m3[i])*np.asarray(model.gravity_hat)
        out[i, :3] = y[i, 3:]
        out[i, 3:] = acceleration
    return out.reshape(-1)
