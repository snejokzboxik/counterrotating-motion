"""Exact V20B two-particle RHS and a particle-specific Reynolds coordinate map.

This module does not claim that the map removes physical parameters.  Its purpose
is to expose where each parameter moves and to test the transformation against
the independent archived V20B implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trap:
    p: np.ndarray  # A, B, nu, beta, s=m1/m2, C12, lambda1, lambda2
    qratio: float
    re_star: np.ndarray
    gravity_z: float
    family: np.ndarray
    modes: np.ndarray
    coeff: np.ndarray
    particle_radii_R: np.ndarray

    @classmethod
    def refine12(cls, root: Path) -> "Trap":
        path = root / "research/v17c/v17c_20b/results/candidate_parameters.json"
        registry = json.loads(path.read_text(encoding="utf-8"))
        case = next(x for x in registry["candidates"]
                    if x["legacy_candidate"] == "refine12_sobol17")
        field = registry["model_source"]
        coeff = np.asarray(field["harmonics_coeff"], dtype=float)
        return cls(
            p=np.asarray(case["p_corrected"], dtype=float),
            qratio=float(case["qratio"]),
            re_star=np.asarray([case["re1_per_speed"], case["re2_per_speed"]]),
            gravity_z=float(case["gravity_z"]),
            family=np.asarray(field["harmonics_family"], dtype=int),
            modes=np.asarray(field["harmonics_k"], dtype=int),
            coeff=coeff[:, 0] + 1j * coeff[:, 1],
            particle_radii_R=np.asarray([case["a1_over_R"], case["a2_over_R"]]),
        )

    @property
    def alpha(self) -> np.ndarray:
        return np.asarray([1.0, self.p[4] * self.qratio])

    @property
    def mass_ratio(self) -> np.ndarray:
        return np.asarray([1.0, 1.0 / self.p[4]])


def rf_values(tau: float, position: np.ndarray, trap: Trap,
              phases: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return E=-grad(phi), phi, and explicit partial_tau(phi) at fixed position."""
    x, y = float(position[0]), float(position[1])
    w = complex(x, y)
    e = np.zeros(3)
    phi = 0.0
    phi_t = 0.0
    for family, k, c in zip(trap.family, trap.modes, trap.coeff):
        n = abs(int(k))
        amplitude = trap.p[0] if family == 0 else trap.p[1]
        omega = 1.0 if family == 0 else trap.p[2]
        carrier = amplitude * np.exp(-1j * (omega * tau + phases[family]))
        base = w if k > 0 else w.conjugate()
        term = carrier * c * base**n
        gradient = carrier * c * n * base ** (n - 1)
        e[0] -= gradient.real
        e[1] += gradient.imag if k > 0 else -gradient.imag
        phi += term.real
        phi_t += (-1j * omega * term).real
    return e, float(phi), float(phi_t)


def original_rhs(tau: float, state: np.ndarray, trap: Trap,
                 phases: np.ndarray) -> np.ndarray:
    """Reconstructed V20B RHS in xi=r/R, tau=Omega1*t coordinates."""
    x = np.asarray(state, dtype=float).reshape(2, 6)
    out = np.zeros((2, 6))
    out[:, :3] = x[:, 3:]
    delta = x[0, :3] - x[1, :3]
    d3 = float(np.linalg.norm(delta) ** 3)
    if d3 == 0:
        raise ValueError("coincident particles")
    A, B, nu, beta, s, C, lam1, lam2 = trap.p
    for i in range(2):
        pos, vel = x[i, :3], x[i, 3:]
        field, _, _ = rf_values(tau, pos, trap, phases)
        dc = beta * np.asarray([pos[0], pos[1], -2 * pos[2]])
        coulomb = (1 if i == 0 else -s) * C * delta / d3
        lam = lam1 if i == 0 else lam2
        drag = -lam * (1 + 0.15 * (trap.re_star[i] * np.linalg.norm(vel)) ** 0.687) * vel
        out[i, 3:] = trap.alpha[i] * (field + dc) + coulomb + drag
        out[i, 5] += trap.gravity_z
    return out.ravel()


def to_reynolds_state(state: np.ndarray, trap: Trap) -> np.ndarray:
    x = np.asarray(state, dtype=float).reshape(2, 6)
    return (x * trap.re_star[:, None]).ravel()


def from_reynolds_state(zeta_state: np.ndarray, trap: Trap) -> np.ndarray:
    z = np.asarray(zeta_state, dtype=float).reshape(2, 6)
    return (z / trap.re_star[:, None]).ravel()


def reynolds_rhs(tau: float, zeta_state: np.ndarray, trap: Trap,
                 phases: np.ndarray) -> np.ndarray:
    """Independently written transformed RHS for zeta_i=Re_star_i*xi_i.

    In particular, Coulomb uses zeta_1/s_1-zeta_2/s_2, never zeta_1-zeta_2.
    """
    z = np.asarray(zeta_state, dtype=float).reshape(2, 6)
    out = np.zeros((2, 6))
    out[:, :3] = z[:, 3:]
    physical_delta = z[0, :3] / trap.re_star[0] - z[1, :3] / trap.re_star[1]
    d3 = float(np.linalg.norm(physical_delta) ** 3)
    if d3 == 0:
        raise ValueError("coincident particles")
    _, _, _, beta, mass_s, C, lam1, lam2 = trap.p
    for i in range(2):
        scale = trap.re_star[i]
        position = z[i, :3] / scale
        velocity = z[i, 3:]
        field, _, _ = rf_values(tau, position, trap, phases)
        dc = trap.alpha[i] * beta * np.asarray([z[i, 0], z[i, 1], -2 * z[i, 2]])
        coulomb = scale * (1 if i == 0 else -mass_s) * C * physical_delta / d3
        lam = lam1 if i == 0 else lam2
        drag = -lam * (1 + 0.15 * np.linalg.norm(velocity) ** 0.687) * velocity
        out[i, 3:] = scale * trap.alpha[i] * field + dc + coulomb + drag
        out[i, 5] += scale * trap.gravity_z
    return out.ravel()


def energy_and_angular_momentum(state: np.ndarray, trap: Trap,
                                tau: float, phases: np.ndarray) -> tuple[float, float]:
    """Dimensionless total energy and angular momentum in m1*R²*Omega1² units."""
    x = np.asarray(state, dtype=float).reshape(2, 6)
    beta, C, gz = trap.p[3], trap.p[5], trap.gravity_z
    energy = 0.0
    angular_momentum = 0.0
    for i in range(2):
        pos, vel = x[i, :3], x[i, 3:]
        mass, alpha = trap.mass_ratio[i], trap.alpha[i]
        _, phi, _ = rf_values(tau, pos, trap, phases)
        dc_phi = beta * (pos[2] ** 2 - (pos[0] ** 2 + pos[1] ** 2) / 2)
        energy += mass * (0.5 * float(vel @ vel) + alpha * (phi + dc_phi) - gz * pos[2])
        angular_momentum += mass * (pos[0] * vel[1] - pos[1] * vel[0])
    energy += C / float(np.linalg.norm(x[0, :3] - x[1, :3]))
    return energy, angular_momentum


def energy_and_angular_momentum_reynolds(zeta_state: np.ndarray, trap: Trap,
                                         tau: float, phases: np.ndarray) -> tuple[float, float]:
    """Same physical invariants expressed directly using particle-specific scales."""
    z = np.asarray(zeta_state, dtype=float).reshape(2, 6)
    energy = 0.0
    momentum = 0.0
    for i in range(2):
        scale = trap.re_star[i]
        pos, vel = z[i, :3], z[i, 3:]
        mass, alpha = trap.mass_ratio[i], trap.alpha[i]
        _, phi, _ = rf_values(tau, pos / scale, trap, phases)
        dc_phi = trap.p[3] * (pos[2] ** 2 - (pos[0] ** 2 + pos[1] ** 2) / 2) / scale**2
        energy += mass * (0.5 * float(vel @ vel) / scale**2 + alpha * (phi + dc_phi) - trap.gravity_z * pos[2] / scale)
        momentum += mass * (pos[0] * vel[1] - pos[1] * vel[0]) / scale**2
    separation = np.linalg.norm(z[0, :3] / trap.re_star[0] - z[1, :3] / trap.re_star[1])
    energy += trap.p[5] / float(separation)
    return energy, momentum


def power_and_torque_balance(state: np.ndarray, trap: Trap,
                             tau: float, phases: np.ndarray) -> tuple[float, float]:
    """Exact dH/dtau and dLz/dtau: explicit RF power plus drag; RF plus drag torque."""
    x = np.asarray(state, dtype=float).reshape(2, 6)
    power = torque = 0.0
    for i in range(2):
        pos, vel = x[i, :3], x[i, 3:]
        field, _, phi_t = rf_values(tau, pos, trap, phases)
        lam = trap.p[6 + i]
        drag = -lam * (1 + 0.15 * (trap.re_star[i] * np.linalg.norm(vel)) ** 0.687) * vel
        mass, alpha = trap.mass_ratio[i], trap.alpha[i]
        power += mass * (alpha * phi_t + float(drag @ vel))
        total_tangent = alpha * field + drag
        torque += mass * (pos[0] * total_tangent[1] - pos[1] * total_tangent[0])
    return power, torque
