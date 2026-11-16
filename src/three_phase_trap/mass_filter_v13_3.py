"""V13.3 ideal RF/DC quadrupole plus rotating-octupole equations."""
from __future__ import annotations

import math

import numpy as np


def potential(x: float, y: float, tau: float, kappa: float, beta: float,
              alpha: float, delta: float = 0.0, tau0: float = 0.0) -> float:
    z = complex(x, y)
    rf_phase = tau + tau0
    ro = (z ** 4 * np.exp(-1j * (rf_phase - delta))).real
    quadrupole = (z ** 2).real
    return float(kappa * ro + (beta * math.cos(rf_phase) + alpha) * quadrupole)


def acceleration(x: float, y: float, tau: float, kappa: float, beta: float,
                 alpha: float, delta: float = 0.0, tau0: float = 0.0) -> np.ndarray:
    """Return (x'',y'') for rho''=-grad(V); tau0 shifts both RF fields."""
    z = complex(x, y)
    phase = tau + tau0 - delta
    w = 4.0 * kappa * z ** 3 * np.exp(-1j * phase)
    w += 2.0 * (beta * math.cos(tau + tau0) + alpha) * z
    return np.array([-w.real, w.imag], dtype=float)


def linear_matrix(alpha: float, beta: float, tau: float,
                  tau0: float = 0.0) -> np.ndarray:
    """Jacobian of acceleration at the origin, independent of kappa and delta."""
    b = 2.0 * (alpha + beta * math.cos(tau + tau0))
    return np.array([[-b, 0.0], [0.0, b]], dtype=float)
