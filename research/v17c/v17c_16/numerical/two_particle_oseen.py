"""Exact dimensional-to-dimensionless two-particle V17C.16 equations.

The integrator state is (x,y,z,vx,vy,vz) for each particle, with x=r/R,
v=dr/dtau/(R*Omega1), tau=Omega1*t. The two RF potential modes are the ideal
leading harmonics k=+2 and k=-4. Oseen drag is evaluated on the full
instantaneous particle/fluid relative velocity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RFMode:
    amplitude: float
    frequency: float
    k: int
    phase: float = 0.0


@dataclass(frozen=True)
class TwoParticleModel:
    reference_mass_kg: float
    reference_charge_C: float
    masses_relative: tuple[float, float]
    charges_relative: tuple[float, float]
    radii_m: tuple[float, float]
    densities_kg_m3: tuple[float, float]
    lambdas_stokes: tuple[float, float]
    kappa_base: float
    omega1_rad_s: float
    radial_scale_m: float
    air_density_kg_m3: float
    air_dynamic_viscosity_Pa_s: float
    beta2: float
    rf_modes: tuple[RFMode, ...]
    gravity_hat: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        if any(x <= 0.0 or not np.isfinite(x) for x in self.masses_relative):
            raise ValueError("relative masses must be finite and positive")
        if any(a <= 0.0 or not np.isfinite(a) for a in self.radii_m):
            raise ValueError("particle radii must be finite and positive")
        if any(x < 0.0 or not np.isfinite(x) for x in self.lambdas_stokes):
            raise ValueError("Stokes drag coefficients must be finite and nonnegative")
        if self.kappa_base < 0.0:
            raise ValueError("kappa_base must be nonnegative")
        if self.radial_scale_m <= 0.0 or self.omega1_rad_s <= 0.0:
            raise ValueError("reference length and frequency must be positive")
        if self.reference_mass_kg <= 0.0 or self.reference_charge_C == 0.0:
            raise ValueError("reference mass and charge must be physical")
        if any(mode.k == 0 for mode in self.rf_modes):
            raise ValueError("planar RF mode indices cannot be zero")


def local15_trial_model(oseen: bool = True,
                        coulomb_scale: float = 1.0,
                        gravity: bool = False) -> TwoParticleModel:
    """Construct one dimensionless model from the trial dimensional pair.

    `coulomb_scale=0` is an explicit uncoupled control. The RF/DC coefficients
    are frozen Local15 values and remain a screening input pending physical
    feasibility; no parameter is independently optimized here.
    """
    del oseen  # Drag selection is made explicitly in rhs/tangent_jacobian.
    from physical_screen import (PAIR, LOCAL15, EPS0, G, E_CHARGE)
    import math

    a = PAIR["radii_m"]
    density = PAIR["particle_densities_kg_m3"]
    q = [n * E_CHARGE for n in PAIR["charges_e"]]
    m = [4.0 * math.pi * density[i] * a[i]**3 / 3.0 for i in range(2)]
    eta = PAIR["air_dynamic_viscosity_Pa_s"]
    rho_f = PAIR["air_density_kg_m3"]
    R = PAIR["radial_scale_R_m"]
    omega = 2.0 * math.pi * PAIR["f1_Hz"]
    lam = tuple(6.0 * math.pi * eta * a[i] / (m[i] * omega) for i in range(2))
    qrel = tuple(qi / q[0] for qi in q)
    mrel = tuple(mi / m[0] for mi in m)
    kappa0 = q[0]**2 / (4.0 * math.pi * EPS0 * m[0] * R**3 * omega**2)
    grav_hat = (0.0, 0.0, -G / (R * omega**2)) if gravity else (0.0, 0.0, 0.0)
    modes = (RFMode(LOCAL15["A"], 1.0, +2),
             RFMode(LOCAL15["B"], LOCAL15["nu"], -4))
    return TwoParticleModel(
        reference_mass_kg=m[0],
        reference_charge_C=q[0],
        masses_relative=mrel,
        charges_relative=qrel,
        radii_m=tuple(a),
        densities_kg_m3=tuple(density),
        lambdas_stokes=lam,
        kappa_base=kappa0 * coulomb_scale,
        omega1_rad_s=omega,
        radial_scale_m=R,
        air_density_kg_m3=rho_f,
        air_dynamic_viscosity_Pa_s=eta,
        beta2=LOCAL15["beta"],
        rf_modes=modes,
        gravity_hat=grav_hat,
    )


def rf_potential_jet(position: Sequence[float], tau: float,
                     modes: Sequence[RFMode], beta2: float
                     ) -> tuple[float, np.ndarray, np.ndarray]:
    """Return dimensionless potential, gradient, and Hessian.

    For each mode phi=Re[a (x+i sgn(k)y)^|k| exp(-i*frequency*tau+i*phase)].
    The DC potential is beta2*(z^2-(x^2+y^2)/2); each component is harmonic.
    """
    x, y, z = np.asarray(position, dtype=float)
    zc = complex(x, y)
    value = 0.0
    grad = np.zeros(3, dtype=float)
    hess = np.zeros((3, 3), dtype=float)
    for mode in modes:
        n = abs(mode.k)
        sgn = 1.0 if mode.k > 0 else -1.0
        base = zc if mode.k > 0 else zc.conjugate()
        carrier = np.exp(-1j * mode.frequency * tau + 1j * mode.phase)
        a = mode.amplitude
        term = a * base**n
        gx = a * n * base**(n-1)
        gy = 1j * sgn * gx
        value += float(np.real(term * carrier))
        grad[0] += float(np.real(gx * carrier))
        grad[1] += float(np.real(gy * carrier))
        if n >= 2:
            hxx = a * n * (n-1) * base**(n-2)
            hxy = 1j * sgn * hxx
            hess[0, 0] += float(np.real(hxx * carrier))
            hess[0, 1] += float(np.real(hxy * carrier))
            hess[1, 0] += float(np.real(hxy * carrier))
            hess[1, 1] -= float(np.real(hxx * carrier))
    value += beta2 * (z*z - 0.5*(x*x + y*y))
    grad += beta2 * np.array([-x, -y, 2.0*z])
    hess += beta2 * np.diag([-1.0, -1.0, 2.0])
    return value, grad, hess


def pair_forces(positions: np.ndarray, model: TwoParticleModel,
                min_separation: float = 1e-12) -> np.ndarray:
    """Dimensionless Coulomb forces before division by relative masses."""
    p = np.asarray(positions, dtype=float)
    if p.shape != (2, 3):
        raise ValueError("positions must have shape (2,3)")
    d = p[0] - p[1]
    d2 = float(d @ d)
    if d2 <= min_separation**2:
        raise FloatingPointError("Coulomb singularity/contact threshold")
    amp = model.kappa_base * model.charges_relative[0] * model.charges_relative[1]
    f0 = amp * d / (d2 * np.sqrt(d2))
    return np.array([f0, -f0])


def diameter_re(velocity_dimensionless: Sequence[float], particle_index: int,
                model: TwoParticleModel) -> float:
    """Instantaneous relative-flow Reynolds number based on particle diameter."""
    speed = float(np.linalg.norm(velocity_dimensionless))
    a = model.radii_m[particle_index]
    Uscale = model.radial_scale_m * model.omega1_rad_s
    return 2.0 * model.air_density_kg_m3 * a * Uscale * speed / model.air_dynamic_viscosity_Pa_s


def drag_acceleration(velocity_dimensionless: Sequence[float], particle_index: int,
                      model: TwoParticleModel, oseen: bool = True,
                      cunningham_factor: float = 1.0) -> np.ndarray:
    """Dimensionless drag acceleration; the Oseen factor uses full speed.

    Cunningham slip, when explicitly passed, acts on the linear Stokes term.
    Multiplying both slip and Oseen corrections is only a selectable screening
    approximation, not a claimed uniformly valid combined drag law.
    """
    u = np.asarray(velocity_dimensionless, dtype=float)
    if cunningham_factor < 1.0 or not np.isfinite(cunningham_factor):
        raise ValueError("Cunningham factor must be finite and >= 1")
    lam = model.lambdas_stokes[particle_index] / cunningham_factor
    correction = 1.0 + (3.0/16.0)*diameter_re(u, particle_index, model) if oseen else 1.0
    return -lam * correction * u


def rhs(tau: float, state: Sequence[float], model: TwoParticleModel,
        oseen: bool = True, coulomb: bool = True,
        cunningham_factors: Sequence[float] = (1.0, 1.0),
        buoyancy: bool = True,
        polarizabilities_SI: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Exact 12-state RHS with optional Oseen, Coulomb, slip, and gravity."""
    model.validate()
    y = np.asarray(state, dtype=float).reshape((2, 6))
    out = np.zeros_like(y)
    Fpair = pair_forces(y[:, :3], model) if coulomb else np.zeros((2, 3))
    for i in range(2):
        _, grad, hess = rf_potential_jet(y[i, :3], tau, model.rf_modes, model.beta2)
        qi_mi = model.charges_relative[i] / model.masses_relative[i]
        acceleration = -qi_mi * grad
        if polarizabilities_SI[i] != 0.0:
            mi = model.reference_mass_kg*model.masses_relative[i]
            coeff = polarizabilities_SI[i]*model.reference_mass_kg**2
            coeff *= model.omega1_rad_s**2
            coeff /= model.reference_charge_C**2*mi
            acceleration += coeff*(hess@grad)
        acceleration += Fpair[i] / model.masses_relative[i]
        acceleration += drag_acceleration(y[i, 3:], i, model, oseen,
                                          cunningham_factors[i])
        if buoyancy and np.linalg.norm(model.gravity_hat) > 0.0:
            rho_p = model.densities_kg_m3[i]
            rho_f = model.air_density_kg_m3
            acceleration += (1.0 - rho_f/rho_p) * np.asarray(model.gravity_hat)
        out[i, :3] = y[i, 3:]
        out[i, 3:] = acceleration
    return out.reshape(-1)


def drag_velocity_jacobian(velocity_dimensionless: Sequence[float],
                           particle_index: int, model: TwoParticleModel,
                           oseen: bool = True,
                           cunningham_factor: float = 1.0) -> np.ndarray:
    """Jacobian of drag acceleration with respect to dimensionless velocity."""
    u = np.asarray(velocity_dimensionless, dtype=float)
    lam = model.lambdas_stokes[particle_index] / cunningham_factor
    J = -lam * np.eye(3)
    if not oseen:
        return J
    speed = float(np.linalg.norm(u))
    if speed == 0.0:
        return J
    K = 2.0 * model.air_density_kg_m3 * model.radii_m[particle_index]
    K *= model.radial_scale_m * model.omega1_rad_s / model.air_dynamic_viscosity_Pa_s
    J -= lam * (3.0/16.0) * K * (speed*np.eye(3) + np.outer(u, u)/speed)
    return J


def tangent_jacobian(tau: float, state: Sequence[float], model: TwoParticleModel,
                     oseen: bool = True, coulomb: bool = True) -> np.ndarray:
    """State Jacobian for local QR/Lyapunov and finite-time stability work."""
    y = np.asarray(state, dtype=float).reshape((2, 6))
    J = np.zeros((12, 12), dtype=float)
    I = np.eye(3)
    for i in range(2):
        J[6*i:6*i+3, 6*i+3:6*i+6] = I
        _, _, H = rf_potential_jet(y[i, :3], tau, model.rf_modes, model.beta2)
        qi_mi = model.charges_relative[i] / model.masses_relative[i]
        J[6*i+3:6*i+6, 6*i:6*i+3] = -qi_mi * H
        J[6*i+3:6*i+6, 6*i+3:6*i+6] = drag_velocity_jacobian(
            y[i, 3:], i, model, oseen)
    if coulomb:
        d = y[0, :3] - y[1, :3]
        d2 = float(d @ d)
        if d2 <= 1e-24:
            raise FloatingPointError("Coulomb singularity")
        qprod = model.kappa_base * model.charges_relative[0] * model.charges_relative[1]
        T = qprod * (I/(d2*np.sqrt(d2))
                     - 3.0*np.outer(d, d)/(d2*d2*np.sqrt(d2)))
        for i, j in ((0, 1), (1, 0)):
            mi = model.masses_relative[i]
            J[6*i+3:6*i+6, 6*i:6*i+3] += T/mi
            J[6*i+3:6*i+6, 6*j:6*j+3] -= T/mi
    return J


def rk4_integrate(model: TwoParticleModel, initial_state: Sequence[float],
                  tau_end: float, step: float, **rhs_options) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step RK4 used for step-refinement checks on short horizons."""
    if step <= 0 or tau_end <= 0:
        raise ValueError("step and final time must be positive")
    y = np.asarray(initial_state, dtype=float).reshape(12).copy()
    n = int(np.ceil(tau_end/step))
    h = tau_end/n
    ts = np.linspace(0.0, tau_end, n+1)
    ys = np.empty((n+1, 12), dtype=float)
    ys[0] = y
    for k in range(n):
        t = ts[k]
        k1 = rhs(t, y, model, **rhs_options)
        k2 = rhs(t+h/2, y+h*k1/2, model, **rhs_options)
        k3 = rhs(t+h/2, y+h*k2/2, model, **rhs_options)
        k4 = rhs(t+h, y+h*k3, model, **rhs_options)
        y = y + h*(k1+2*k2+2*k3+k4)/6
        ys[k+1] = y
    return ts, ys
