"""Dimensionless exact N-body dynamics for arbitrary species and RF harmonics.

Coordinate convention: each body state is [x,y,z,vx,vy,vz], positions are in R,
time is tau=Omega_ref*t, masses and charges are relative to positive reference
scales, and RFFamily modes are signed planar harmonic indices.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class RFFamily:
    amplitude: complex
    frequency: float
    modes: tuple[int, ...]
    coefficients: tuple[complex, ...]


def _validate_particle_arrays(masses, charges, lambdas):
    m = np.asarray(masses, dtype=float)
    q = np.asarray(charges, dtype=float)
    lam = np.asarray(lambdas, dtype=float)
    if m.ndim != 1 or q.shape != m.shape or lam.shape != m.shape:
        raise ValueError("masses, charges, lambdas must be equal-length 1-D arrays")
    if np.any(~np.isfinite(m)) or np.any(m <= 0):
        raise ValueError("dimensionless masses must be finite and positive")
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(lam)) or np.any(lam < 0):
        raise ValueError("charges and nonnegative drags must be finite")
    return m, q, lam


def dc_value_gradient_hessian(pos, beta2=0.0, beta4=0.0):
    x, y, z = np.asarray(pos, dtype=float)
    r2 = x*x + y*y
    H2 = z*z - 0.5*r2
    H4 = z**4 - 3.0*r2*z*z + 0.375*r2*r2
    value = beta2*H2 + beta4*H4
    grad = beta2*np.array([-x, -y, 2*z], dtype=float)
    grad += beta4*np.array([-6*x*z*z + 1.5*x*r2,
                            -6*y*z*z + 1.5*y*r2,
                            4*z**3 - 6*r2*z], dtype=float)
    hess = beta2*np.diag([-1.0, -1.0, 2.0])
    hess = hess.astype(float)
    hess += beta4*np.array([
        [-6*z*z + 1.5*(r2 + 2*x*x), 3*x*y, -12*x*z],
        [3*x*y, -6*z*z + 1.5*(r2 + 2*y*y), -12*y*z],
        [-12*x*z, -12*y*z, 12*z*z - 6*r2],
    ])
    return float(value), grad, hess


def rf_value_gradient_hessian(pos, tau, families: Sequence[RFFamily]):
    x, y, _ = np.asarray(pos, dtype=float)
    zc = complex(x, y)
    value = 0.0
    grad = np.zeros(3, dtype=float)
    hess = np.zeros((3, 3), dtype=float)
    for fam in families:
        if not np.isfinite(fam.frequency):
            raise ValueError("RF frequency must be finite")
        if len(fam.modes) != len(fam.coefficients):
            raise ValueError("each RF mode needs one complex coefficient")
        carrier = np.exp(-1j*float(fam.frequency)*float(tau))
        for k, ck in zip(fam.modes, fam.coefficients):
            k = int(k)
            n = abs(k)
            if n == 0:
                raise ValueError("planar RF mode index must be nonzero")
            base = zc if k > 0 else zc.conjugate()
            term = fam.amplitude * ck * base**n
            gx = fam.amplitude * ck * n * base**(n-1)
            sign = 1.0 if k > 0 else -1.0
            gy = sign*1j*gx
            value += float(np.real(term*carrier))
            grad[0] += float(np.real(gx*carrier))
            grad[1] += float(np.real(gy*carrier))
            if n >= 2:
                hxx = fam.amplitude*ck*n*(n-1)*base**(n-2)
                hxy = sign*1j*hxx
                hess[0, 0] += float(np.real(hxx*carrier))
                hess[0, 1] += float(np.real(hxy*carrier))
                hess[1, 0] += float(np.real(hxy*carrier))
                hess[1, 1] -= float(np.real(hxx*carrier))
    return value, grad, hess


def potential_value_gradient_hessian(pos, tau, families,
                                     beta2=0.0, beta4=0.0):
    vr, gr, hr = rf_value_gradient_hessian(pos, tau, families)
    vd, gd, hd = dc_value_gradient_hessian(pos, beta2, beta4)
    return vr+vd, gr+gd, hr+hd


def coulomb_forces(positions, charges, kappa, min_separation=1e-14):
    """Return dimensionless pair forces before division by particle masses."""
    x = np.asarray(positions, dtype=float)
    q = np.asarray(charges, dtype=float)
    n = len(q)
    if x.shape != (n, 3):
        raise ValueError("positions must have shape (N,3)")
    out = np.zeros_like(x)
    for i in range(n):
        for j in range(i+1, n):
            d = x[i]-x[j]
            d2 = float(d@d)
            if d2 <= min_separation**2:
                raise FloatingPointError("Coulomb singularity/contact threshold")
            fij = float(kappa)*q[i]*q[j]*d/(d2*np.sqrt(d2))
            out[i] += fij
            out[j] -= fij
    return out


def rhs(tau, state, masses, charges, lambdas, kappa, families,
        beta2=0.0, beta4=0.0, gravity=None):
    m, q, lam = _validate_particle_arrays(masses, charges, lambdas)
    y = np.asarray(state, dtype=float).reshape((-1, 6))
    n = len(m)
    if y.shape != (n, 6):
        raise ValueError("state must contain six entries per particle")
    grav = np.zeros((n, 3), dtype=float) if gravity is None else np.asarray(gravity, dtype=float)
    if grav.shape == (3,):
        grav = np.broadcast_to(grav, (n, 3))
    if grav.shape != (n, 3):
        raise ValueError("gravity must be one acceleration vector or shape (N,3)")
    out = np.zeros_like(y)
    Fpair = coulomb_forces(y[:, :3], q, kappa) if n > 1 else np.zeros((n, 3))
    for i in range(n):
        _, grad, _ = potential_value_gradient_hessian(
            y[i, :3], tau, families, beta2, beta4)
        out[i, :3] = y[i, 3:]
        out[i, 3:] = -(q[i]/m[i])*grad - lam[i]*y[i, 3:] + Fpair[i]/m[i] + grav[i]
    return out.reshape(-1)


def tangent_jacobian(tau, state, masses, charges, lambdas, kappa,
                     families, beta2=0.0, beta4=0.0):
    m, q, lam = _validate_particle_arrays(masses, charges, lambdas)
    y = np.asarray(state, dtype=float).reshape((-1, 6))
    n = len(m)
    if y.shape != (n, 6):
        raise ValueError("state must contain six entries per particle")
    J = np.zeros((6*n, 6*n), dtype=float)
    eye = np.eye(3)
    for i in range(n):
        J[6*i:6*i+3, 6*i+3:6*i+6] = eye
        _, _, hess = potential_value_gradient_hessian(
            y[i, :3], tau, families, beta2, beta4)
        J[6*i+3:6*i+6, 6*i:6*i+3] = -(q[i]/m[i])*hess
        J[6*i+3:6*i+6, 6*i+3:6*i+6] = -lam[i]*eye
    for i in range(n):
        for j in range(i+1, n):
            d = y[i, :3]-y[j, :3]
            d2 = float(d@d)
            if d2 <= 1e-28:
                raise FloatingPointError("Coulomb singularity")
            T = float(kappa)*q[i]*q[j] * (
                eye/(d2*np.sqrt(d2)) - 3.0*np.outer(d, d)/(d2*d2*np.sqrt(d2)))
            J[6*i+3:6*i+6, 6*i:6*i+3] += T/m[i]
            J[6*i+3:6*i+6, 6*j:6*j+3] -= T/m[i]
            J[6*j+3:6*j+6, 6*j:6*j+3] += T/m[j]
            J[6*j+3:6*j+6, 6*i:6*i+3] -= T/m[j]
    return J


def mechanical_energy(tau, state, masses, charges, kappa, families,
                      beta2=0.0, beta4=0.0):
    m, q, _ = _validate_particle_arrays(masses, charges, np.zeros_like(masses))
    y = np.asarray(state, dtype=float).reshape((-1, 6))
    energy = 0.5*float(np.sum(m[:, None]*y[:, 3:]**2))
    for i in range(len(m)):
        value, _, _ = potential_value_gradient_hessian(
            y[i, :3], tau, families, beta2, beta4)
        energy += q[i]*value
    for i in range(len(m)):
        for j in range(i+1, len(m)):
            d = np.linalg.norm(y[i, :3]-y[j, :3])
            if d <= 1e-14:
                raise FloatingPointError("Coulomb singularity")
            energy += float(kappa)*q[i]*q[j]/d
    return float(energy)


def angular_momenta(state, masses):
    m = np.asarray(masses, dtype=float)
    y = np.asarray(state, dtype=float).reshape((-1, 6))
    return m*(y[:, 0]*y[:, 4]-y[:, 1]*y[:, 3])


def electrode_amplitude_to_harmonic(q_ref, U_electrode_peak, transfer,
                                    m_ref, R, omega_ref):
    """A=q_ref*H*U_e/(m_ref R^2 Omega^2); H maps electrode voltage to the
    leading spatial-potential coefficient, not to an electric field.
    """
    return q_ref*transfer*U_electrode_peak/(m_ref*R**2*omega_ref**2)
