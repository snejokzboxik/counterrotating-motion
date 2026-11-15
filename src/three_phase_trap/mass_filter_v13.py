"""Ideal, dimensionless three-component RF/DC mass-filter model (V13)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def potential(x: float, y: float, tau: float, kappa: float, beta: float, alpha: float,
              *, standing_octupole: bool = False) -> float:
    z = complex(x, y)
    octupole = z ** 4
    oct_value = octupole.real * np.cos(tau)
    if not standing_octupole:
        oct_value += octupole.imag * np.sin(tau)
    return float(kappa * oct_value + (beta * np.cos(tau) + alpha) * (z ** 2).real)


def acceleration(x: float, y: float, tau: float, kappa: float, beta: float, alpha: float,
                 *, standing_octupole: bool = False) -> np.ndarray:
    z = complex(x, y)
    derivative = 4 * kappa * z ** 3 * np.cos(tau) + 2 * (beta * np.cos(tau) + alpha) * z
    if not standing_octupole:
        derivative -= 4j * kappa * z ** 3 * np.sin(tau)
    return np.array([-derivative.real, derivative.imag], dtype=float)


def effective_potential(x: float, y: float, kappa: float, beta: float, alpha: float) -> float:
    z = complex(x, y)
    r2 = x*x + y*y
    return float(8*kappa*kappa*r2**3 + 4*kappa*beta*r2*(z*z).real
                 + beta*beta*r2 + alpha*(z*z).real)


def monodromy(alpha: float, beta: float, direction: str = "x", *, rtol: float = 2e-11,
              atol: float = 2e-13) -> np.ndarray:
    """One-period fundamental matrix, with rows (position, velocity)."""
    if direction not in ("x", "y"):
        raise ValueError("direction must be x or y")
    sign = 1 if direction == "x" else -1

    def rhs(t: float, flat: np.ndarray) -> np.ndarray:
        mat = flat.reshape(2, 2)
        return (np.array([[0., 1.], [-2*sign*(alpha + beta*np.cos(t)), 0.]]) @ mat).ravel()

    sol = solve_ivp(rhs, (0., 2*np.pi), np.eye(2).ravel(), method="DOP853",
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1].reshape(2, 2)


def linear_status(alpha: float, beta: float, *, tolerance: float = 1e-8,
                  rtol: float = 2e-11, atol: float = 2e-13) -> dict:
    """Strict stability only; |trace|=2 and unresolved neighborhoods are boundaries."""
    mx = monodromy(alpha, beta, "x", rtol=rtol, atol=atol)
    my = monodromy(alpha, beta, "y", rtol=rtol, atol=atol)
    traces = [float(np.trace(m)) for m in (mx, my)]
    margins = [2. - abs(t) for t in traces]
    if all(m > tolerance for m in margins):
        state = "strictly_stable"
    elif any(m < -tolerance for m in margins):
        state = "unstable"
    else:
        state = "boundary_or_unresolved"
    return {"status": state, "trace_x": traces[0], "trace_y": traces[1],
            "det_x": float(np.linalg.det(mx)), "det_y": float(np.linalg.det(my)),
            "margin_x": margins[0], "margin_y": margins[1]}
