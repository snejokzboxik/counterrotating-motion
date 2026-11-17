"""Dimensionless V15 Q/SO/rotating-octupole dynamics and paired integrator.

Coordinates are (x, y, x', y') with x=r_x/R and tau=Omega*t.  The two
rotating models are distinct: reflection in y exchanges RO_PLUS and RO_MINUS;
changing kappa's sign does not reverse the rotation direction.
"""
from __future__ import annotations

import math
import numpy as np


MODELS = ("Q", "SO", "RO_PLUS", "RO_MINUS")


def potential(state, tau, alpha, beta, kappa=0.0, model="Q", delta=0.0, tau0=0.0):
    a = np.asarray(state)
    z = a[..., 0] + 1j * a[..., 1]
    phase = tau + tau0 - delta
    q = alpha + beta * math.cos(tau + tau0)
    value = q * (z * z).real
    if model == "Q":
        return value
    if model == "SO":
        return value + kappa * (z ** 4).real * math.cos(phase)
    if model == "RO_PLUS":
        return value + kappa * (z ** 4 * complex(math.cos(phase), -math.sin(phase))).real
    if model == "RO_MINUS":
        return value + kappa * (z ** 4 * complex(math.cos(phase), math.sin(phase))).real
    raise ValueError(f"unknown model: {model}")


def acceleration_and_jacobian(state, tau, alpha, beta, kappa=0.0, model="Q",
                              delta=0.0, tau0=0.0):
    """Return dimensionless transverse acceleration and position Jacobian."""
    a = np.asarray(state)
    z = a[..., 0] + 1j * a[..., 1]
    q = alpha + beta * math.cos(tau + tau0)
    w = 2.0 * q * z
    dw = 2.0 * q + np.zeros_like(z)
    if model == "Q":
        pass
    elif model == "SO":
        c = math.cos(tau + tau0 - delta)
        w = w + 4.0 * kappa * z ** 3 * c
        dw = dw + 12.0 * kappa * z ** 2 * c
    elif model == "RO_PLUS":
        phase = tau + tau0 - delta
        e = complex(math.cos(phase), -math.sin(phase))
        w = w + 4.0 * kappa * z ** 3 * e
        dw = dw + 12.0 * kappa * z ** 2 * e
    elif model == "RO_MINUS":
        phase = tau + tau0 - delta
        e = complex(math.cos(phase), math.sin(phase))
        w = w + 4.0 * kappa * z ** 3 * e
        dw = dw + 12.0 * kappa * z ** 2 * e
    else:
        raise ValueError(f"unknown model: {model}")
    force = np.stack((-w.real, w.imag), axis=-1)
    jac = np.stack((
        np.stack((-dw.real, dw.imag), axis=-1),
        np.stack((dw.imag, dw.real), axis=-1),
    ), axis=-2)
    return force, jac


def rhs(tau, state, alpha, beta, kappa=0.0, model="Q", delta=0.0, tau0=0.0):
    a = np.asarray(state)
    force, _ = acceleration_and_jacobian(a, tau, alpha, beta, kappa, model, delta, tau0)
    return np.concatenate((a[..., 2:4], force), axis=-1)


def dimensionless_angular_momentum(state):
    a = np.asarray(state)
    return a[..., 0] * a[..., 3] - a[..., 1] * a[..., 2]


def dimensionless_torque(state, tau, alpha, beta, kappa=0.0, model="Q",
                         delta=0.0, tau0=0.0):
    """d(x*y'-y*x')/d tau = x*y'' - y*x''."""
    a = np.asarray(state)
    force, _ = acceleration_and_jacobian(a, tau, alpha, beta, kappa, model, delta, tau0)
    return a[..., 0] * force[..., 1] - a[..., 1] * force[..., 0]


def mirror_y(state):
    """Reflect a state across x: (x,y,vx,vy) -> (x,-y,vx,-vy)."""
    a = np.array(state, dtype=float, copy=True)
    a[..., 1] *= -1.0
    a[..., 3] *= -1.0
    return a


def _derivative_and_torque(states, tau, alpha, beta, kappa, model, delta, tau0):
    force, _ = acceleration_and_jacobian(states, tau, alpha, beta, kappa, model, delta, tau0)
    derivative = np.concatenate((states[..., 2:4], force), axis=-1)
    torque = states[..., 0] * force[..., 1] - states[..., 1] * force[..., 0]
    return derivative, torque


def integrate_rk4_batch(initial, alpha, beta, kappa, model, delta=0.0, tau0=0.0,
                        horizon_rf=100, steps_per_rf=128, aperture=1.0,
                        tau_start_cycles=0.0):
    """Vectorized RK4 with aperture exit and integrated torque diagnostics.

    `torque_integral` is integrated over full last steps.  For escaped pilot
    cases, that differs from the integral exactly to the event by at most one
    step; final headline/event measurements should be refined with DOP853.
    """
    state = np.asarray(initial, dtype=float).copy()
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError("initial must have shape (n,4)")
    count = len(state)
    status = np.full(count, "retained", dtype=object)
    exit_rf = np.full(count, np.nan)
    rho_max = np.linalg.norm(state[:, :2], axis=1)
    ell0 = dimensionless_angular_momentum(state)
    torque_integral = np.zeros(count)
    alive = (rho_max < aperture) & np.all(np.isfinite(state), axis=1)
    status[~alive] = "unresolved"
    h = 2.0 * math.pi / int(steps_per_rf)
    nsteps = int(horizon_rf) * int(steps_per_rf)
    tau0_abs = 2.0 * math.pi * float(tau_start_cycles)

    for step in range(nsteps):
        ids = np.flatnonzero(alive)
        if ids.size == 0:
            break
        y0 = state[ids]
        t = tau0_abs + step * h
        k1, t1 = _derivative_and_torque(y0, t, alpha, beta, kappa, model, delta, tau0)
        y2 = y0 + 0.5 * h * k1
        k2, t2 = _derivative_and_torque(y2, t + 0.5 * h, alpha, beta, kappa, model, delta, tau0)
        y3 = y0 + 0.5 * h * k2
        k3, t3 = _derivative_and_torque(y3, t + 0.5 * h, alpha, beta, kappa, model, delta, tau0)
        y4 = y0 + h * k3
        k4, t4 = _derivative_and_torque(y4, t + h, alpha, beta, kappa, model, delta, tau0)
        ynew = y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        dint = (h / 6.0) * (t1 + 2.0 * t2 + 2.0 * t3 + t4)

        radii = np.stack((np.linalg.norm(y2[:, :2], axis=1),
                          np.linalg.norm(y3[:, :2], axis=1),
                          np.linalg.norm(y4[:, :2], axis=1),
                          np.linalg.norm(ynew[:, :2], axis=1)))
        local_max = np.max(radii, axis=0)
        rho_max[ids] = np.maximum(rho_max[ids], local_max)
        torque_integral[ids] += dint
        state[ids] = ynew
        finite = np.all(np.isfinite(ynew), axis=1) & np.isfinite(dint)
        escaped = (local_max >= aperture) & finite
        unresolved = ~finite
        if np.any(escaped):
            eids = ids[escaped]
            status[eids] = "escaped"
            exit_rf[eids] = tau0_abs / (2.0 * math.pi) + (step + 1.0) / steps_per_rf
            alive[eids] = False
        if np.any(unresolved):
            uids = ids[unresolved]
            status[uids] = "unresolved"
            alive[uids] = False

    ell_end = dimensionless_angular_momentum(state)
    residual = (ell_end - ell0) - torque_integral
    return {
        "final_state": state,
        "status": status,
        "exit_rf_step_estimate": exit_rf,
        "rho_max_stage_sampled": rho_max,
        "ell0": ell0,
        "ell_end": ell_end,
        "torque_integral": torque_integral,
        "torque_identity_residual": residual,
    }


def octupole_intensity_factor(model, kappa):
    """RF-period mean |grad V_oct|^2/r^6 for an ideal m=4 field."""
    if model == "Q":
        return 0.0
    if model == "SO":
        return 8.0 * kappa * kappa
    if model in ("RO_PLUS", "RO_MINUS"):
        return 16.0 * kappa * kappa
    raise ValueError(f"unknown model: {model}")
