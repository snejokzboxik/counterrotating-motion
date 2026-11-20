"""Reference dimensionless N-body RHS, potential derivatives, and tangent Jacobian."""
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


def dc_value_gradient_hessian(pos: np.ndarray, beta2: float = 0.0,
                              beta4: float = 0.0) -> tuple[float, np.ndarray, np.ndarray]:
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


def rf_value_gradient_hessian(pos: np.ndarray, tau: float,
                              families: Sequence[RFFamily]
                              ) -> tuple[float, np.ndarray, np.ndarray]:
    x, y, _ = np.asarray(pos, dtype=float)
    zc = complex(x, y)
    value = 0.0
    grad = np.zeros(3, dtype=float)
    hess = np.zeros((3, 3), dtype=float)
    for fam in families:
        carrier = np.exp(-1j*fam.frequency*tau)
        for k, ck in zip(fam.modes, fam.coefficients):
            n = abs(int(k))
            if n == 0:
                raise ValueError("mode index must be nonzero")
            base = zc if k > 0 else zc.conjugate()
            term = fam.amplitude * ck * base**n
            gx = fam.amplitude * ck * n * base**(n-1)
            gy = (1j if k > 0 else -1j) * gx
            value += float(np.real(term*carrier))
            grad[0] += float(np.real(gx*carrier))
            grad[1] += float(np.real(gy*carrier))
            if n >= 2:
                hxx = fam.amplitude*ck*n*(n-1)*base**(n-2)
                hxy = (1j if k > 0 else -1j)*hxx
                hess[0, 0] += np.real(hxx*carrier)
                hess[0, 1] += np.real(hxy*carrier)
                hess[1, 0] = hess[0, 1]
                hess[1, 1] -= np.real(hxx*carrier)
    return value, grad, hess


def potential_value_gradient_hessian(pos: np.ndarray, tau: float,
                                     families: Sequence[RFFamily],
                                     beta2: float = 0.0, beta4: float = 0.0
                                     ) -> tuple[float, np.ndarray, np.ndarray]:
    vr, gr, hr = rf_value_gradient_hessian(pos, tau, families)
    vd, gd, hd = dc_value_gradient_hessian(pos, beta2, beta4)
    return vr + vd, gr + gd, hr + hd


def rhs(tau: float, state: np.ndarray, masses: np.ndarray, charges: np.ndarray,
        lambdas: np.ndarray, kappa: float, families: Sequence[RFFamily],
        beta2: float = 0.0, beta4: float = 0.0,
        gravity: np.ndarray | None = None) -> np.ndarray:
    masses = np.asarray(masses, dtype=float)
    charges = np.asarray(charges, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    y = np.asarray(state, dtype=float).reshape((-1, 6))
    nbody = len(masses)
    if y.shape[0] != nbody or charges.shape != masses.shape or lambdas.shape != masses.shape:
        raise ValueError("state, mass, charge and drag arrays have inconsistent lengths")
    grav = np.zeros(3) if gravity is None else np.asarray(gravity, dtype=float)
    out = np.zeros_like(y)
    for i in range(nbody):
        pos, vel = y[i, :3], y[i, 3:]
        _, grad, _ = potential_value_gradient_hessian(pos, tau, families, beta2, beta4)
        acc = -(charges[i]/masses[i])*grad - lambdas[i]*vel + grav
        for j in range(nbody):
            if j == i:
                continue
            d = pos-y[j, :3]
            d2 = float(d@d)
            if d2 <= 1e-28:
                raise FloatingPointError("Coulomb singularity")
            acc += kappa*charges[i]*charges[j]/masses[i] * d/(d2*np.sqrt(d2))
        out[i, :3] = vel
        out[i, 3:] = acc
    return out.reshape(-1)


def tangent_jacobian(tau: float, state: np.ndarray, masses: np.ndarray,
                     charges: np.ndarray, lambdas: np.ndarray, kappa: float,
                     families: Sequence[RFFamily], beta2: float = 0.0,
                     beta4: float = 0.0) -> np.ndarray:
    masses = np.asarray(masses, dtype=float)
    charges = np.asarray(charges, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    y = np.asarray(state, dtype=float).reshape((-1, 6))
    nbody = len(masses)
    J = np.zeros((6*nbody, 6*nbody), dtype=float)
    eye = np.eye(3)
    for i in range(nbody):
        ii = slice(6*i, 6*i+6)
        J[6*i:6*i+3, 6*i+3:6*i+6] = eye
        _, _, hess = potential_value_gradient_hessian(
            y[i, :3], tau, families, beta2, beta4)
        J[6*i+3:6*i+6, 6*i:6*i+3] = -(charges[i]/masses[i])*hess
        J[6*i+3:6*i+6, 6*i+3:6*i+6] = -lambdas[i]*eye
    for i in range(nbody):
        for j in range(i+1, nbody):
            d = y[i, :3]-y[j, :3]
            d2 = float(d@d)
            if d2 <= 1e-28:
                raise FloatingPointError("Coulomb singularity")
            T = kappa*charges[i]*charges[j] * (
                eye/(d2*np.sqrt(d2)) - 3.0*np.outer(d, d)/(d2*d2*np.sqrt(d2)))
            J[6*i+3:6*i+6, 6*i:6*i+3] += T/masses[i]
            J[6*i+3:6*i+6, 6*j:6*j+3] -= T/masses[i]
            J[6*j+3:6*j+6, 6*j:6*j+3] += T/masses[j]
            J[6*j+3:6*j+6, 6*i:6*i+3] -= T/masses[j]
    return J
