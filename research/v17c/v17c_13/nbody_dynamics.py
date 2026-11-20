"""FP64 N-body extension of the validated V17C local15 dynamics.

State layout is particle-major (x,y,z,vx,vy,vz).  The model keeps the full
eight-mode MFS spectrum, continuous RF phases, finite-sphere contact tests,
and the V17C.11 common-envelope definitions.  Coulomb coefficients are
scaled by both charges and each particle's inverse mass.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
V17C11 = HERE.parent / "v17c_11"
if str(V17C11) not in sys.path:
    sys.path.insert(0, str(V17C11))
import loading_dynamics as ld  # noqa: E402

TWOPI = 2.0 * math.pi
RMIN_CLASS = 0.02
ESCAPE_R = 1.0
ESCAPE_Z = 2.0
F1_HZ = 100_000.0


def load_local15() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load local15 parameters and its complete eight-harmonic MFS spectrum."""
    cfg = json.loads((V17C11 / "local15_parameters.json").read_text(encoding="utf-8"))
    p = np.asarray(cfg["candidate"]["p"], dtype=np.float64)
    field = cfg["field"]
    family = np.asarray(field["family"], dtype=np.int64)
    modes = np.asarray(field["k"], dtype=np.int64)
    coeff = np.asarray(field["coeff_real_imag"], dtype=np.float64)
    coeff = coeff[:, 0] + 1j * coeff[:, 1]
    qratio = float(cfg["candidate"]["q2_over_q1"])
    return p, family, modes, coeff.astype(np.complex128), qratio


def load_protocols() -> dict[str, np.ndarray]:
    """Return constant local15 and exact published #505/#353/#292 schedules."""
    data = json.loads((V17C11 / "strict_loading_holdout_n256" / "holdout_protocols.json").read_text(encoding="utf-8"))
    result = {"constant": ld.constant_protocol()}
    for desc in data:
        result[desc["label"]] = np.asarray(desc["protocol_vector_19"], dtype=np.float64)
    return result


@njit(cache=True, nogil=True)
def _contact_status(y, groups, radii):
    """Return 0=clear, 1=finite-sphere collision, 2=electrode/end escape."""
    n = groups.size
    for i in range(n - 1):
        bi = 6 * i
        for j in range(i + 1, n):
            bj = 6 * j
            dx = y[bi] - y[bj]
            dy = y[bi + 1] - y[bj + 1]
            dz = y[bi + 2] - y[bj + 2]
            rs = radii[i] + radii[j]
            if dx * dx + dy * dy + dz * dz < rs * rs:
                return 1
    for i in range(n):
        b = 6 * i
        if (math.hypot(y[b], y[b + 1]) + radii[i] > ESCAPE_R or
                abs(y[b + 2]) + radii[i] > ESCAPE_Z):
            return 2
    return 0


@njit(cache=True, nogil=True)
def _rhs_nbody(t, y, p, protocol, phase, groups, family, modes, coeff,
               qratio, radii, out, parts):
    """RHS; parts[:,0:4] are RF, DC, Coulomb and drag accelerations."""
    n = groups.size
    ga, gb, gd = ld._envelopes(t, protocol)
    amp1 = p[0] * ga
    amp2 = p[1] * gb
    beta = p[3] * gd
    nu = p[2]
    coupling = p[5]

    # External RF, DC and drag accelerations for every particle.
    for i in range(n):
        b = 6 * i
        x, yy, z = y[b], y[b + 1], y[b + 2]
        vx, vy, vz = y[b + 3], y[b + 4], y[b + 5]
        u = complex(x, yy)
        ex = 0.0
        ey = 0.0
        for h in range(modes.size):
            mode = modes[h]
            degree = abs(mode)
            base = u if mode > 0 else u.conjugate()
            if family[h] == 0:
                carrier = amp1 * complex(math.cos(t + phase[0]), -math.sin(t + phase[0]))
            else:
                carrier = amp2 * complex(math.cos(nu * t + phase[1]), -math.sin(nu * t + phase[1]))
            grad = carrier * coeff[h] * degree * base ** (degree - 1)
            ex -= grad.real
            if mode > 0:
                ey += grad.imag
            else:
                ey -= grad.imag

        if groups[i] == 0:
            q_i = 1.0
            invmass = 1.0
            lam = p[6]
        else:
            q_i = qratio
            invmass = p[4]
            lam = p[7]
        electric_factor = q_i * invmass
        parts[i, 0, 0] = electric_factor * ex
        parts[i, 0, 1] = electric_factor * ey
        parts[i, 0, 2] = 0.0
        parts[i, 1, 0] = electric_factor * beta * x
        parts[i, 1, 1] = electric_factor * beta * yy
        parts[i, 1, 2] = -2.0 * electric_factor * beta * z
        parts[i, 2, 0] = 0.0
        parts[i, 2, 1] = 0.0
        parts[i, 2, 2] = 0.0
        parts[i, 3, 0] = -lam * vx
        parts[i, 3, 1] = -lam * vy
        parts[i, 3, 2] = -lam * vz

    # One loop over all unique pairs. The mass-weighted pair accelerations
    # cancel exactly up to floating-point rounding.
    for i in range(n - 1):
        bi = 6 * i
        if groups[i] == 0:
            qi = 1.0
            mi_inv = 1.0
        else:
            qi = qratio
            mi_inv = p[4]
        for j in range(i + 1, n):
            bj = 6 * j
            dx = y[bi] - y[bj]
            dy = y[bi + 1] - y[bj + 1]
            dz = y[bi + 2] - y[bj + 2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > 1.0e-28:
                if groups[j] == 0:
                    qj = 1.0
                    mj_inv = 1.0
                else:
                    qj = qratio
                    mj_inv = p[4]
                coeff_pair = coupling * (qi * qj / qratio) / (d2 * math.sqrt(d2))
                cix = mi_inv * coeff_pair * dx
                ciy = mi_inv * coeff_pair * dy
                ciz = mi_inv * coeff_pair * dz
                cjx = -mj_inv * coeff_pair * dx
                cjy = -mj_inv * coeff_pair * dy
                cjz = -mj_inv * coeff_pair * dz
                parts[i, 2, 0] += cix
                parts[i, 2, 1] += ciy
                parts[i, 2, 2] += ciz
                parts[j, 2, 0] += cjx
                parts[j, 2, 1] += cjy
                parts[j, 2, 2] += cjz

    for i in range(n):
        b = 6 * i
        out[b] = y[b + 3]
        out[b + 1] = y[b + 4]
        out[b + 2] = y[b + 5]
        for a in range(3):
            out[b + 3 + a] = parts[i, 0, a] + parts[i, 1, a] + parts[i, 2, a] + parts[i, 3, a]


@njit(cache=True, nogil=True)
def _pair_distance_update(y, n, min_pair_distances):
    idx = 0
    for i in range(n - 1):
        bi = 6 * i
        for j in range(i + 1, n):
            bj = 6 * j
            dx = y[bi] - y[bj]
            dy = y[bi + 1] - y[bj + 1]
            dz = y[bi + 2] - y[bj + 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < min_pair_distances[idx]:
                min_pair_distances[idx] = d
            idx += 1


@njit(cache=True, nogil=True)
def integrate(initial, phase, p, protocol, groups, family, modes, coeff,
              qratio, radii, periods, steps, skip, log_every_period=False,
              energy_balance=False):
    """Adaptive-free RK4, checking all pair and wall contacts at RK stages."""
    initial = np.asarray(initial, dtype=np.float64)
    y = initial.copy()
    n = groups.size
    pairs = n * (n - 1) // 2
    dt = TWOPI / steps
    y1 = np.empty(6 * n); y2 = np.empty(6 * n)
    y3 = np.empty(6 * n); y4 = np.empty(6 * n)
    tmp = np.empty(6 * n); yn = np.empty(6 * n)
    parts1 = np.empty((n, 4, 3)); parts2 = np.empty((n, 4, 3))
    parts3 = np.empty((n, 4, 3)); parts4 = np.empty((n, 4, 3))
    min_radius = np.empty(n); max_radius = np.empty(n); max_abs_z = np.empty(n)
    max_abs_dtheta = np.zeros(n); turns = np.zeros(n); windows = np.zeros((3, n))
    theta_unwrapped = np.zeros(n)
    for i in range(n):
        b = 6 * i
        r = math.hypot(y[b], y[b + 1])
        min_radius[i] = r; max_radius[i] = r; max_abs_z[i] = abs(y[b + 2])
        theta_unwrapped[i] = math.atan2(y[b + 1], y[b])
    pair_min = np.full(pairs, 1.0e300)
    _pair_distance_update(y, n, pair_min)
    window1 = skip + (periods - skip) // 3
    window2 = skip + 2 * (periods - skip) // 3
    steps_done = 0
    log_count = 0
    logs = np.empty((0, n, 8), dtype=np.float64)
    work_log = np.empty((0, 5), dtype=np.float64)
    work_count = 0
    work_rf = 0.0
    work_drag = 0.0
    energy_start = 0.0
    if energy_balance:
        work_log = np.empty((max(1, periods - skip + 1), 5), dtype=np.float64)
    if log_every_period:
        logs = np.empty((periods + 1, n, 8), dtype=np.float64)
        log_count = 1
        for i in range(n):
            b = 6 * i
            logs[0, i, 0] = 0.0
            for q in range(6):
                logs[0, i, 1 + q] = y[b + q]
            logs[0, i, 7] = theta_unwrapped[i]
    status = _contact_status(y, groups, radii)
    for step in range(periods * steps):
        if status != 0:
            steps_done = step
            break
        t = step * dt
        cycle = step // steps
        if energy_balance and cycle == skip and step % steps == 0:
            energy_start = _system_energy(t, y, p, protocol, phase, groups,
                                          family, modes, coeff, qratio)
            work_rf = 0.0
            work_drag = 0.0
            work_log[0, 0] = t
            work_log[0, 1] = energy_start
            work_log[0, 2] = 0.0
            work_log[0, 3] = 0.0
            work_log[0, 4] = 0.0
            work_count = 1
        _rhs_nbody(t, y, p, protocol, phase, groups, family, modes, coeff,
                   qratio, radii, y1, parts1)
        for q in range(6 * n):
            tmp[q] = y[q] + 0.5 * dt * y1[q]
        _pair_distance_update(tmp, n, pair_min)
        status = _contact_status(tmp, groups, radii)
        if status != 0:
            steps_done = step + 1
            break
        _rhs_nbody(t + 0.5 * dt, tmp, p, protocol, phase, groups, family, modes, coeff,
                   qratio, radii, y2, parts2)
        for q in range(6 * n):
            tmp[q] = y[q] + 0.5 * dt * y2[q]
        _pair_distance_update(tmp, n, pair_min)
        status = _contact_status(tmp, groups, radii)
        if status != 0:
            steps_done = step + 1
            break
        _rhs_nbody(t + 0.5 * dt, tmp, p, protocol, phase, groups, family, modes, coeff,
                   qratio, radii, y3, parts3)
        for q in range(6 * n):
            tmp[q] = y[q] + dt * y3[q]
        _pair_distance_update(tmp, n, pair_min)
        status = _contact_status(tmp, groups, radii)
        if status != 0:
            steps_done = step + 1
            break
        _rhs_nbody(t + dt, tmp, p, protocol, phase, groups, family, modes, coeff,
                   qratio, radii, y4, parts4)

        if energy_balance and cycle >= skip:
            prf1, pdrag1 = _explicit_rf_and_drag_power(
                t, y, p, protocol, phase, groups, family, modes, coeff, qratio)
            for q in range(6 * n):
                tmp[q] = y[q] + 0.5 * dt * y1[q]
            prf2, pdrag2 = _explicit_rf_and_drag_power(
                t + 0.5 * dt, tmp, p, protocol, phase, groups, family, modes, coeff, qratio)
            for q in range(6 * n):
                tmp[q] = y[q] + 0.5 * dt * y2[q]
            prf3, pdrag3 = _explicit_rf_and_drag_power(
                t + 0.5 * dt, tmp, p, protocol, phase, groups, family, modes, coeff, qratio)
            for q in range(6 * n):
                tmp[q] = y[q] + dt * y3[q]
            prf4, pdrag4 = _explicit_rf_and_drag_power(
                t + dt, tmp, p, protocol, phase, groups, family, modes, coeff, qratio)
            work_rf += dt * (prf1 + 2.0 * prf2 + 2.0 * prf3 + prf4) / 6.0
            work_drag += dt * (pdrag1 + 2.0 * pdrag2 + 2.0 * pdrag3 + pdrag4) / 6.0
        for q in range(6 * n):
            yn[q] = y[q] + dt * (y1[q] + 2.0 * y2[q] + 2.0 * y3[q] + y4[q]) / 6.0
        _pair_distance_update(yn, n, pair_min)

        # Match the validated V17C classifier's center-coordinate extrema.
        for i in range(n):
            b = 6 * i
            r0 = math.hypot(y[b], y[b + 1]); r1 = math.hypot(yn[b], yn[b + 1])
            if r0 < min_radius[i]: min_radius[i] = r0
            if r1 < min_radius[i]: min_radius[i] = r1
            if r0 > max_radius[i]: max_radius[i] = r0
            if r1 > max_radius[i]: max_radius[i] = r1
            z0 = abs(y[b + 2]); z1 = abs(yn[b + 2])
            if z0 > max_abs_z[i]: max_abs_z[i] = z0
            if z1 > max_abs_z[i]: max_abs_z[i] = z1

        status = _contact_status(yn, groups, radii)
        steps_done = step + 1
        if status != 0:
            y[:] = yn
            break
        if not np.all(np.isfinite(yn)):
            status = 3
            break
        cycle = step // steps
        for i in range(n):
            b = 6 * i
            r0 = math.hypot(y[b], y[b + 1]); r1 = math.hypot(yn[b], yn[b + 1])
            if r0 > RMIN_CLASS and r1 > RMIN_CLASS:
                cross = y[b] * yn[b + 1] - y[b + 1] * yn[b]
                dot = y[b] * yn[b] + y[b + 1] * yn[b + 1]
                dtheta = math.atan2(cross, dot)
                theta_unwrapped[i] += dtheta
                if cycle >= skip:
                    if cycle < window1: wi = 0
                    elif cycle < window2: wi = 1
                    else: wi = 2
                    turns[i] += dtheta / TWOPI
                    windows[wi, i] += dtheta / TWOPI
                    dtheta_rad = abs(dtheta)
                    if dtheta_rad > max_abs_dtheta[i]: max_abs_dtheta[i] = dtheta_rad
        y, yn = yn, y
        if log_every_period and (step + 1) % steps == 0:
            cycle_index = (step + 1) // steps
            for i in range(n):
                b = 6 * i
                logs[log_count, i, 0] = (step + 1) * dt
                for q in range(6):
                    logs[log_count, i, 1 + q] = y[b + q]
                logs[log_count, i, 7] = theta_unwrapped[i]
            log_count += 1
        if energy_balance and (step + 1) % steps == 0 and cycle >= skip:
            energy_now = _system_energy(t + dt, y, p, protocol, phase, groups,
                                        family, modes, coeff, qratio)
            work_log[work_count, 0] = t + dt
            work_log[work_count, 1] = energy_now
            work_log[work_count, 2] = work_rf
            work_log[work_count, 3] = work_drag
            work_log[work_count, 4] = energy_now - energy_start - work_rf - work_drag
            work_count += 1

    completed = steps_done // steps
    retained = status == 0 and completed >= periods
    axis_ambiguous = bool(np.any(min_radius < RMIN_CLASS))
    enough = bool(np.all(np.abs(turns) >= 10.0))
    stable = True
    for w in range(3):
        for i in range(n):
            if abs(windows[w, i]) < 0.05:
                stable = False
    signs = np.sign(windows)
    if status == 1:
        category = "finite_particle_contact"
    elif status == 2:
        category = "finite_particle_wall_contact"
    elif status == 3:
        category = "point_singularity_unresolved"
    elif axis_ambiguous:
        category = "axis_ambiguous"
    elif retained and enough and stable:
        group_sign = np.zeros((2, 3))
        for w in range(3):
            for i in range(n):
                group_sign[w, groups[i]] += math.copysign(1.0, windows[w, i]) if abs(windows[w, i]) >= 0.05 else 0.0
        if n == 2 and np.all(signs[:, 0] * signs[:, 1] < 0.0):
            category = "CR_screen"
        elif np.all(group_sign[:, 0] == 3.0) and np.all(group_sign[:, 1] == -3.0) or \
                np.all(group_sign[:, 0] == -3.0) and np.all(group_sign[:, 1] == 3.0):
            category = "COUNTERROTATING_UNORDERED"
        elif np.all(group_sign[:, 0] == 3.0) and np.all(group_sign[:, 1] == 3.0) or \
                np.all(group_sign[:, 0] == -3.0) and np.all(group_sign[:, 1] == -3.0):
            category = "COROTATING_UNORDERED"
        else:
            category = "PARTIAL_CRYSTALLIZATION"
    elif retained:
        if np.max(np.abs(turns)) < 0.25:
            category = "NONROTATING"
        else:
            category = "UNRESOLVED"
    else:
        category = "UNRESOLVED"
    frequencies = np.zeros(n)
    measured = max(1, completed - skip)
    for i in range(n):
        frequencies[i] = turns[i] * F1_HZ / measured

    if log_every_period:
        logs = logs[:log_count].copy()
    if energy_balance:
        work_log = work_log[:work_count].copy()
    return (y, turns, windows, frequencies, min_radius, max_radius, max_abs_z,
            max_abs_dtheta, pair_min, completed, retained, category, status, logs,
            work_log)


@njit(cache=True, nogil=True)
def external_potential(t, y, p, protocol, phase, groups, family, modes, coeff,
                       qratio):
    """Dimensionless RF+DC potential energy divided by m1 R^2 Omega1^2."""
    ga, gb, gd = ld._envelopes(t, protocol)
    amp1 = p[0] * ga
    amp2 = p[1] * gb
    beta = p[3] * gd
    n = groups.size
    energy = 0.0
    for i in range(n):
        b = 6 * i
        x, yy, z = y[b], y[b + 1], y[b + 2]
        u = complex(x, yy)
        phi_rf = 0.0
        for h in range(modes.size):
            mode = modes[h]
            degree = abs(mode)
            base = u if mode > 0 else u.conjugate()
            if family[h] == 0:
                carrier = amp1 * complex(math.cos(t + phase[0]), -math.sin(t + phase[0]))
            else:
                carrier = amp2 * complex(math.cos(p[2] * t + phase[1]), -math.sin(p[2] * t + phase[1]))
            phi_rf += (carrier * coeff[h] * base ** degree).real
        qi = 1.0 if groups[i] == 0 else qratio
        energy += qi * (phi_rf + beta * (z * z - 0.5 * (x * x + yy * yy)))
    for i in range(n - 1):
        bi = 6 * i
        qi = 1.0 if groups[i] == 0 else qratio
        for j in range(i + 1, n):
            bj = 6 * j
            qj = 1.0 if groups[j] == 0 else qratio
            dx = y[bi] - y[bj]; dy = y[bi + 1] - y[bj + 1]; dz = y[bi + 2] - y[bj + 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d > 1.0e-14:
                energy += p[5] * (qi * qj / qratio) / d
    return energy


@njit(cache=True, nogil=True)
def _system_energy(t, y, p, protocol, phase, groups, family, modes, coeff,
                   qratio):
    """Dimensionless kinetic plus RF, DC and all-pair Coulomb potential."""
    kinetic = 0.0
    for i in range(groups.size):
        b = 6 * i
        mass = 1.0 if groups[i] == 0 else 1.0 / p[4]
        kinetic += 0.5 * mass * (y[b + 3] ** 2 + y[b + 4] ** 2 + y[b + 5] ** 2)
    return kinetic + external_potential(t, y, p, protocol, phase, groups,
                                        family, modes, coeff, qratio)


@njit(cache=True, nogil=True)
def _explicit_rf_and_drag_power(t, y, p, protocol, phase, groups, family,
                                modes, coeff, qratio):
    """Return dU/dtau from RF carriers and drag power after envelopes settle.

    The diagnostic is only accumulated after ``skip`` in ``integrate``, and
    callers set skip beyond the loading schedule. DC and Coulomb potentials
    are then static, so their explicit time derivative is exactly zero.
    """
    ga, gb, _ = ld._envelopes(t, protocol)
    amp1 = p[0] * ga
    amp2 = p[1] * gb
    nu = p[2]
    rf_partial = 0.0
    drag = 0.0
    for i in range(groups.size):
        b = 6 * i
        x, yy = y[b], y[b + 1]
        vx, vy, vz = y[b + 3], y[b + 4], y[b + 5]
        base_group = 1.0 if groups[i] == 0 else qratio
        lam = p[6] if groups[i] == 0 else p[7]
        mass = 1.0 if groups[i] == 0 else 1.0 / p[4]
        drag -= mass * lam * (vx * vx + vy * vy + vz * vz)
        u = complex(x, yy)
        for h in range(modes.size):
            mode = modes[h]
            degree = abs(mode)
            base = u if mode > 0 else u.conjugate()
            if family[h] == 0:
                omega = 1.0
                carrier = amp1 * complex(math.cos(t + phase[0]), -math.sin(t + phase[0]))
            else:
                omega = nu
                carrier = amp2 * complex(math.cos(nu * t + phase[1]), -math.sin(nu * t + phase[1]))
            rf_partial += base_group * omega * (carrier * coeff[h] * base ** degree).imag
    return rf_partial, drag


def group_masses(groups: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(groups, dtype=np.int64) == 0, 1.0, 1.0 / p[4])


def initialize_rings(radii: tuple[float, float] = (0.45, 0.63),
                     z_centers: tuple[float, float] = (0.0, 0.1),
                     relative_angle: float = 0.0,
                     phase_order: tuple[int, int] = (1, 1),
                     groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two equilateral triangles at rest; z may be shared or offset by group."""
    if groups is None:
        groups = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    state = np.zeros(6 * len(groups), dtype=np.float64)
    for i, g in enumerate(groups):
        offset = 0.0 if g == 0 else relative_angle
        order = phase_order[g]
        theta = offset + 2.0 * math.pi * ((i % 3) * order % 3) / 3.0
        rr = radii[g]
        zz = z_centers[g]
        state[6 * i:6 * i + 6] = (rr * math.cos(theta), rr * math.sin(theta), zz, 0.0, 0.0, 0.0)
    phases = np.asarray([0.0, 0.0], dtype=np.float64)
    return state, phases

