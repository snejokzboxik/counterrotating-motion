"""Dimensionless full-RF dynamics and finite-time aperture outcomes (V12).

The field is a real Fourier polynomial

    F(rho, tau) = Re[(C(z) cos(tau+phase) + S(z) sin(tau+phase))],
    z = rho_x + i rho_y,

with coefficients normalized to a reference voltage and length.  The
dimensionless equation is rho'' = -Lambda grad(F), tau = Omega*t.  No
pseudopotential averaging is used by the trajectory integrator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class DimensionlessRFModel:
    """A two-quadrature RF field expressed in dimensionless coordinates.

    ``cosine[m]`` and ``sine[m]`` are complex coefficients of z**m in the
    dimensionless potential quadratures, per unit applied voltage.  The
    conversion from dimensional FEM coefficients is made by
    :meth:`from_physical`.
    """

    cosine: np.ndarray
    sine: np.ndarray
    coupling: float
    aperture_rho: float = 1.0
    phase_rad: float = 0.0
    name: str = "unnamed"
    _cosine_derivative: np.ndarray = field(init=False, repr=False, compare=False)
    _sine_derivative: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        c = np.asarray(self.cosine, dtype=np.complex128)
        s = np.asarray(self.sine, dtype=np.complex128)
        if c.ndim != 1 or s.shape != c.shape or len(c) < 2:
            raise ValueError("cosine and sine must be same-length 1-D polynomial arrays")
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(s))):
            raise ValueError("RF coefficients must be finite")
        if not np.isfinite(self.coupling) or not np.isfinite(self.aperture_rho) or self.aperture_rho <= 0:
            raise ValueError("coupling must be finite and aperture_rho positive")
        if not np.isfinite(self.phase_rad):
            raise ValueError("phase_rad must be finite")
        c = c.copy(); s = s.copy()
        c.setflags(write=False); s.setflags(write=False)
        object.__setattr__(self, "cosine", c)
        object.__setattr__(self, "sine", s)
        dc = np.polynomial.polynomial.polyder(c) if len(c) > 1 else np.zeros(1, complex)
        ds = np.polynomial.polynomial.polyder(s) if len(s) > 1 else np.zeros(1, complex)
        dc = np.asarray(dc, complex); ds = np.asarray(ds, complex)
        dc.setflags(write=False); ds.setflags(write=False)
        object.__setattr__(self, "_cosine_derivative", dc)
        object.__setattr__(self, "_sine_derivative", ds)

    @classmethod
    def from_physical(
        cls,
        cosine: np.ndarray,
        sine: np.ndarray,
        *,
        charge_c: float,
        mass_kg: float,
        voltage_v: float,
        omega_rad_s: float,
        length_m: float,
        reference_radius_m: float,
        aperture_m: float,
        phase_rad: float = 0.0,
        name: str = "FEM-derived",
    ) -> "DimensionlessRFModel":
        """Build model from physical FEM polynomial quadratures.

        The supplied polynomial uses ``z=(x+i*y)/reference_radius_m`` and
        coefficients per volt.  It is rescaled to rho=r/length_m, and
        ``coupling=q*U/(M*Omega**2*length_m**2)``.  This supports distinct
        reference and dynamical lengths without changing the field.
        """
        vals = (charge_c, mass_kg, voltage_v, omega_rad_s, length_m,
                reference_radius_m, aperture_m, phase_rad)
        if not np.all(np.isfinite(vals)) or mass_kg <= 0 or omega_rad_s <= 0 or length_m <= 0 or reference_radius_m <= 0:
            raise ValueError("physical parameters must be finite; mass, frequency and lengths positive")
        if aperture_m <= 0:
            raise ValueError("aperture_m must be positive")
        ratio = length_m / reference_radius_m
        orders = np.arange(len(cosine), dtype=float)
        scale = ratio ** orders
        return cls(np.asarray(cosine, complex) * scale,
                   np.asarray(sine, complex) * scale,
                   float(charge_c * voltage_v / (mass_kg * omega_rad_s**2 * length_m**2)),
                   float(aperture_m / length_m), float(phase_rad), name)

    @property
    def max_order(self) -> int:
        return len(self.cosine) - 1

    def potential_and_gradient(self, rho_xy: np.ndarray, tau: float) -> tuple[float, np.ndarray]:
        """Return F and its dimensionless spatial gradient at one point."""
        x, y = np.asarray(rho_xy, dtype=float)
        z = complex(x, y)
        angle = math.remainder(float(tau) + self.phase_rad, 2.0 * math.pi)
        c, s = math.cos(angle), math.sin(angle)
        value_c = self._polyval(self.cosine, z)
        value_s = self._polyval(self.sine, z)
        value = (value_c * c + value_s * s).real
        if self.max_order == 0:
            return float(value), np.zeros(2, dtype=float)
        deriv = c * self._polyval(self._cosine_derivative, z) + s * self._polyval(self._sine_derivative, z)
        # For Re(P(x+i*y)), d/dy = -Im(P'(z)).
        grad = np.array([deriv.real, -deriv.imag], dtype=float)
        return float(value), grad

    @staticmethod
    def _polyval(coefficients: np.ndarray, z: complex) -> complex:
        value = 0j
        for coefficient in coefficients[::-1]:
            value = value * z + coefficient
        return value

    def gradient(self, rho_xy: np.ndarray, tau: float) -> np.ndarray:
        """Evaluate only the gradient, avoiding potential work in the RHS."""
        x, y = np.asarray(rho_xy, dtype=float)
        z = complex(x, y)
        angle = math.remainder(float(tau) + self.phase_rad, 2.0 * math.pi)
        c, s = math.cos(angle), math.sin(angle)
        deriv = c * self._polyval(self._cosine_derivative, z) + s * self._polyval(self._sine_derivative, z)
        return np.array([deriv.real, -deriv.imag], dtype=float)

    def acceleration(self, rho_xy: np.ndarray, tau: float) -> np.ndarray:
        return -self.coupling * self.gradient(rho_xy, tau)

    def rhs(self, tau: float, state: np.ndarray) -> np.ndarray:
        query = np.asarray(state[:2], dtype=float)
        radius = float(np.linalg.norm(query))
        # A solver's trial stages can step just beyond a terminal aperture
        # event.  Keep the field query on the audited FEM disk, as the native
        # SI integrator does; the event still uses the unprojected state.
        if radius > self.aperture_rho:
            query = query * (self.aperture_rho / radius)
        return np.r_[state[2:], self.acceleration(query, tau)]

    def hamiltonian(self, state: np.ndarray, tau: float) -> float:
        f, _ = self.potential_and_gradient(state[:2], tau)
        return float(0.5 * np.dot(state[2:], state[2:]) + self.coupling * f)

    def pseudopotential(self, rho_xy: np.ndarray) -> float:
        """Leading averaged energy in units M*Omega**2*R**2.

        This evaluates the two RF quadratures separately and is only an
        approximation to the full RF dynamics.
        """
        # phase_rad rotates both quadratures in time.  Recover the underlying
        # cosine/sine pair directly so their norm is phase invariant.
        z = complex(*np.asarray(rho_xy, dtype=float))
        if self.max_order == 0:
            return 0.0
        dc = np.polynomial.polynomial.polyval(z, np.polynomial.polynomial.polyder(self.cosine))
        ds = np.polynomial.polynomial.polyval(z, np.polynomial.polynomial.polyder(self.sine))
        norm2 = dc.real**2 + dc.imag**2 + ds.real**2 + ds.imag**2
        return float((self.coupling**2 / 4.0) * norm2)


@dataclass(frozen=True)
class SegmentResult:
    status: Literal["retained", "escaped", "numerical_error"]
    final_state: np.ndarray
    start_period: float
    reached_period: float
    escape_period: float | None
    max_radius_sampled_rho: float
    nfev: int
    message: str = ""


def integrate_segment(
    model: DimensionlessRFModel,
    initial_state: np.ndarray,
    start_period: float,
    end_period: float,
    *,
    rtol: float = 2e-10,
    atol: float | np.ndarray = 2e-12,
    max_step_rf: float = 1.0 / 64.0,
    samples_per_rf: int = 8,
) -> SegmentResult:
    """Integrate one resumable RF-period segment with a first-exit event.

    The state is ``(rho_x,rho_y,u_x,u_y)``.  Restarting at an integer or
    fractional RF period keeps absolute drive phase through ``start_period``.
    A solver failure is returned as ``numerical_error``, never as retention.
    """
    y0 = np.asarray(initial_state, dtype=float)
    if y0.shape != (4,) or not np.all(np.isfinite(y0)):
        raise ValueError("initial_state must be a finite four-vector")
    if not np.isfinite(start_period) or not np.isfinite(end_period) or end_period <= start_period:
        raise ValueError("end_period must be finite and greater than start_period")
    atol_array = np.asarray(atol, dtype=float)
    if (rtol <= 0 or not np.all(np.isfinite(atol_array)) or np.any(atol_array <= 0)
            or (atol_array.ndim not in (0, 1))
            or (atol_array.ndim == 1 and atol_array.shape != (4,))
            or max_step_rf <= 0 or samples_per_rf < 1):
        raise ValueError("integration tolerances and sampling parameters must be positive")
    radius0 = float(np.linalg.norm(y0[:2]))
    if radius0 >= model.aperture_rho:
        return SegmentResult("escaped", y0.copy(), start_period, start_period,
                             start_period, radius0, 0, "initial state at/outside aperture")

    tau0 = 2.0 * math.pi * start_period
    duration = 2.0 * math.pi * (end_period - start_period)

    def fun(t_local, state):
        return model.rhs(tau0 + t_local, state)

    def first_exit(t_local, state):
        return float(np.linalg.norm(state[:2]) - model.aperture_rho)
    first_exit.terminal = True
    first_exit.direction = 1

    n_samples = max(2, int(math.ceil((end_period - start_period) * samples_per_rf)) + 1)
    t_eval = np.linspace(0.0, duration, n_samples)
    try:
        sol = solve_ivp(fun, (0.0, duration), y0, method="DOP853", rtol=rtol,
                        atol=atol, max_step=2.0 * math.pi * max_step_rf,
                        events=first_exit, t_eval=t_eval, dense_output=True)
    except Exception as exc:  # preserve numerical failures as explicit statuses
        return SegmentResult("numerical_error", y0.copy(), start_period,
                             start_period, None, radius0, 0,
                             f"{type(exc).__name__}: {exc}")
    if not sol.success:
        return SegmentResult("numerical_error", sol.y[:, -1].copy(), start_period,
                             start_period + sol.t[-1] / (2.0 * math.pi), None,
                             float(np.linalg.norm(sol.y[:2], axis=0).max()),
                             int(sol.nfev), sol.message)
    exit_tau = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
    if exit_tau is None:
        final_state = sol.y[:, -1].copy()
        reached = end_period
        escape_period = None
        status = "retained"
    else:
        final_state = sol.y_events[0][0].copy()
        reached = start_period + exit_tau / (2.0 * math.pi)
        escape_period = reached
        status = "escaped"
    # Dense samples characterize the continuous path better than accepted RK
    # nodes alone; this remains a sampled maximum, not a certified supremum.
    sample_tau = np.linspace(0.0, exit_tau if exit_tau is not None else duration,
                             max(2, int(math.ceil((reached - start_period) * samples_per_rf)) + 1))
    sample_states = sol.sol(sample_tau)
    max_r = max(radius0, float(np.linalg.norm(sample_states[:2], axis=0).max()))
    return SegmentResult(status, final_state, start_period, reached, escape_period,
                         max_r, int(sol.nfev), "")


def integrate_to_horizon(
    model: DimensionlessRFModel,
    initial_state: np.ndarray,
    horizon_periods: float,
    *,
    segment_periods: float = 10.0,
    rtol: float = 2e-10,
    atol: float | np.ndarray = 2e-12,
    max_step_rf: float = 1.0 / 64.0,
    samples_per_rf: int = 8,
    checkpoint_callback=None,
    resume_state: np.ndarray | None = None,
    resume_period: float = 0.0,
) -> SegmentResult:
    """Continue a trajectory segment by segment; callback sees each commit.

    ``checkpoint_callback(period, state, max_radius, nfev)`` must write to a
    per-trajectory file or otherwise synchronize its own output.
    """
    if horizon_periods <= resume_period or segment_periods <= 0:
        raise ValueError("horizon must exceed resume_period and segment_periods be positive")
    state = np.asarray(initial_state if resume_state is None else resume_state, dtype=float).copy()
    current = float(resume_period)
    maximum = float(np.linalg.norm(state[:2]))
    total_nfev = 0
    while current < horizon_periods - 1e-12:
        stop = min(current + segment_periods, horizon_periods)
        result = integrate_segment(model, state, current, stop, rtol=rtol, atol=atol,
                                   max_step_rf=max_step_rf,
                                   samples_per_rf=samples_per_rf)
        total_nfev += result.nfev
        maximum = max(maximum, result.max_radius_sampled_rho)
        if result.status != "retained":
            return SegmentResult(result.status, result.final_state, current,
                                 result.reached_period, result.escape_period,
                                 maximum, total_nfev, result.message)
        state = result.final_state
        current = stop
        if checkpoint_callback is not None:
            checkpoint_callback(current, state.copy(), maximum, total_nfev)
    return SegmentResult("retained", state, float(resume_period), current, None,
                         maximum, total_nfev, "")
