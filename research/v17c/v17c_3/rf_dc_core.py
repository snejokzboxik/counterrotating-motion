"""Dimensionless RF--DC dynamics with Coulomb coupling for V17C.3.

The RF field is translationally invariant in z and uses the complete signed
harmonic spectrum from V17C.2.  The DC quadrupole is three-dimensional.
Coordinates and time are scaled by R and 1/Omega, respectively.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input"
RESULTS = HERE / "results"
NORM = 13  # x,y,z,vx,vy,vz + four work channels + three torque channels


def load_geometry(name: str):
    """Return metadata and harmonics normalized to the leading coefficient."""
    data = json.loads((INPUT / "harmonics.json").read_text(encoding="utf-8"))
    key = {
        "G1": "6_ABC_moderate", "G2": "6_ABC_thin",
        "G3": "9_ABC_moderate", "G3thin": "9_ABC_thin",
        "G5": "12_ABC_moderate",
    }.get(name)
    if key:
        g = next(item for item in data if item["name"] == key)
        rows = np.asarray(g["coefficients"], dtype=float)
        modes = rows[:, 0].astype(np.int64)
        coeff = rows[:, 1] + 1j * rows[:, 2]
        lead = coeff[np.flatnonzero(modes == int(g["m"]))[0]]
        coeff = coeff / abs(lead)
        # Drop only round-off/remote orders below 1e-13 in potential at .95R.
        keep = np.abs(coeff) * .95 ** np.abs(modes) >= 1e-13
        return {**g, "label": name, "A_lead": complex(lead)}, modes[keep], coeff[keep]

    if name in ("G0_h50", "G0_h35"):
        fem = json.loads((INPUT / "native_p4_h50_h35_coefficients.json").read_text(encoding="utf-8"))
        mesh = "h50" if name.endswith("h50") else "h35"
        record = fem[mesh]
        # FEM phase is expressed in its leading-wave frame; retaining its full
        # complex relative coefficient preserves the measured h50/h35 phase.
        rel = complex(record["relative_re"], record["relative_im"])
        meta = {"label": name, "N": 12, "m": 4, "R_D": None,
                "native_mesh": mesh, "rho": abs(rel),
                "relative_phase_rad": float(np.angle(rel))}
        return meta, np.asarray([4, -8], dtype=np.int64), np.asarray([1.0 + 0j, rel])
    raise KeyError(name)


def radial_shape(modes, coeff, r):
    """S(r) where beta_eq = alpha^2*S(r), full-spectrum azimuthal mean."""
    n = np.abs(modes).astype(float)
    return float(np.sum(n*n*(n-1)*np.abs(coeff)**2*np.asarray(r)**(2*n-4)))


def beta_equilibrium(modes, coeff, alpha, r, lam=0.0, omega=0.0):
    """Weak-micromotion ring estimate, with damping and centrifugal terms."""
    # The driven micromotion amplitude is smaller by sqrt(1+lambda^2).
    return alpha*alpha*radial_shape(modes, coeff, r)/(1.0+lam*lam) - omega*omega


def torque_omega_estimate(modes, coeff, alpha, r, lam=5e-4):
    """Leading averaged angular speed from RF torque = linear drag torque."""
    n = np.abs(modes).astype(float)
    sgn = np.sign(modes.astype(float))
    term = sgn*n*n*(n-1)*np.abs(coeff)**2*np.asarray(r)**(2*n-4)
    return float(alpha*alpha*np.sum(term)/(1.0+lam*lam))


def torque_sign_roots(modes, coeff, rlo=.02, rhi=.999, samples=10000):
    """Roots of the azimuthally averaged guiding-centre torque polynomial."""
    grid = np.linspace(rlo, rhi, samples)
    n = np.abs(modes).astype(float)
    sgn = np.sign(modes.astype(float))
    values = np.zeros_like(grid)
    for k, (nn, ss, cc) in enumerate(zip(n, sgn, coeff)):
        values += ss*nn*nn*(nn-1)*abs(cc)**2*grid**(2*nn-3)
    roots = []
    for i in np.flatnonzero(values[:-1]*values[1:] < 0):
        lo, hi = grid[i], grid[i+1]
        flo = values[i]
        for _ in range(60):
            mid = .5*(lo+hi)
            fm = sum(ss*nn*nn*(nn-1)*abs(cc)**2*mid**(2*nn-3)
                     for nn, ss, cc in zip(n, sgn, coeff))
            if flo*fm <= 0: hi = mid
            else: lo, flo = mid, fm
        roots.append(.5*(lo+hi))
    return roots


@njit(cache=True)
def _rf_accel(x, y, tau, modes, coeff, alpha):
    u = complex(x, y)
    phase = complex(np.cos(tau), -np.sin(tau))
    ax = 0.0
    ay = 0.0
    for j in range(len(modes)):
        k = modes[j]
        n = abs(k)
        p = 1.0 + 0.0j
        base = u if k > 0 else np.conj(u)
        for _ in range(n-1):
            p *= base
        wp = n * coeff[j] * p * phase
        ax -= wp.real
        if k > 0:
            ay -= (1j*wp).real
        else:
            ay += (1j*wp).real
    return alpha*ax, alpha*ay


@njit(cache=True)
def _rhs_aug(state, tau, modes, coeff, alpha, beta, lam, mass, charge, chi):
    count = len(mass)
    out = np.zeros_like(state)
    arx = np.zeros(count); ary = np.zeros(count); arz = np.zeros(count)
    acx = np.zeros(count); acy = np.zeros(count); acz = np.zeros(count)
    for i in range(count):
        b = NORM*i
        x, y, z = state[b], state[b+1], state[b+2]
        vx, vy, vz = state[b+3], state[b+4], state[b+5]
        arx[i], ary[i] = _rf_accel(x, y, tau, modes, coeff, alpha[i])
        arz[i] = 0.0
        acx[i] = beta[i]*x
        acy[i] = beta[i]*y
        acz[i] = -2.0*beta[i]*z
    for i in range(count):
        for j in range(i+1, count):
            bi, bj = NORM*i, NORM*j
            dx = state[bi]-state[bj]
            dy = state[bi+1]-state[bj+1]
            dz = state[bi+2]-state[bj+2]
            rr2 = dx*dx+dy*dy+dz*dz
            # An encounter threshold is checked by the caller; this floor only
            # prevents a floating-point singularity from corrupting a run.
            if rr2 < 1e-14:
                rr2 = 1e-14
            factor = chi*charge[i]*charge[j]/(rr2*np.sqrt(rr2))
            acx[i] += factor*dx/mass[i]
            acy[i] += factor*dy/mass[i]
            acz[i] += factor*dz/mass[i]
            acx[j] -= factor*dx/mass[j]
            acy[j] -= factor*dy/mass[j]
            acz[j] -= factor*dz/mass[j]
    for i in range(count):
        b = NORM*i
        x, y, z = state[b], state[b+1], state[b+2]
        vx, vy, vz = state[b+3], state[b+4], state[b+5]
        dragx, dragy, dragz = -lam[i]*vx, -lam[i]*vy, -lam[i]*vz
        out[b] = vx; out[b+1] = vy; out[b+2] = vz
        out[b+3] = arx[i]+acx[i]+dragx
        out[b+4] = ary[i]+acy[i]+dragy
        out[b+5] = arz[i]+acz[i]+dragz
        # Work channels are specific (per unit particle mass).
        out[b+6] = vx*arx[i]+vy*ary[i]+vz*arz[i]
        # Separate conservative DC acceleration from Coulomb acceleration.
        ccx, ccy, ccz = acx[i]-beta[i]*x, acy[i]-beta[i]*y, acz[i]+2.0*beta[i]*z
        out[b+7] = vx*beta[i]*x+vy*beta[i]*y-2.0*vz*beta[i]*z
        out[b+8] = vx*ccx+vy*ccy+vz*ccz
        out[b+9] = vx*dragx+vy*dragy+vz*dragz
        out[b+10] = x*ary[i]-y*arx[i]
        out[b+11] = x*(acy[i]-beta[i]*y)-y*(acx[i]-beta[i]*x)
        out[b+12] = x*dragy-y*dragx
    return out


@njit(cache=True)
def integrate_rk4(y0, periods, steps_per_period, save_stride,
                  modes, coeff, alpha, beta, lam, mass, charge, chi,
                  phase0=0.0, escape_radius=.995, z_limit=5.0):
    dt = 2.0*np.pi/steps_per_period
    nsteps = periods*steps_per_period
    nsaved = nsteps//save_stride+2
    out = np.empty((nsaved, len(y0)), dtype=np.float64)
    times = np.empty(nsaved, dtype=np.float64)
    y = y0.copy()
    out[0] = y; times[0] = 0.0
    used = 1
    escaped = False
    collided = False
    for it in range(nsteps):
        t = phase0+it*dt
        k1 = _rhs_aug(y, t, modes, coeff, alpha, beta, lam, mass, charge, chi)
        k2 = _rhs_aug(y+.5*dt*k1, t+.5*dt, modes, coeff, alpha, beta, lam, mass, charge, chi)
        k3 = _rhs_aug(y+.5*dt*k2, t+.5*dt, modes, coeff, alpha, beta, lam, mass, charge, chi)
        k4 = _rhs_aug(y+dt*k3, t+dt, modes, coeff, alpha, beta, lam, mass, charge, chi)
        y += (dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
        if not np.all(np.isfinite(y)):
            escaped = True
            break
        for i in range(len(mass)):
            b = NORM*i
            r = np.sqrt(y[b]*y[b]+y[b+1]*y[b+1])
            if r >= escape_radius or abs(y[b+2]) >= z_limit:
                escaped = True
                break
        if escaped:
            if (it+1) % save_stride != 0:
                out[used] = y; times[used] = (it+1)*dt; used += 1
            break
        if len(mass) > 1:
            dx=y[0]-y[NORM]; dy=y[1]-y[NORM+1]; dz=y[2]-y[NORM+2]
            if dx*dx+dy*dy+dz*dz < 1e-8:
                collided = True
                if (it+1) % save_stride != 0:
                    out[used] = y; times[used] = (it+1)*dt; used += 1
                break
        if (it+1) % save_stride == 0:
            out[used] = y; times[used] = (it+1)*dt; used += 1
    return times[:used], out[:used], escaped, collided


def initial_augmented(radii, thetas, omegas, z0=None, vr=None, vz=None):
    count = len(radii)
    z0 = np.zeros(count) if z0 is None else np.asarray(z0, float)
    vr = np.zeros(count) if vr is None else np.asarray(vr, float)
    vz = np.zeros(count) if vz is None else np.asarray(vz, float)
    y = np.zeros(NORM*count, dtype=np.float64)
    for i, (r, th, om) in enumerate(zip(radii, thetas, omegas)):
        b = NORM*i
        c, s = np.cos(th), np.sin(th)
        y[b:b+6] = [r*c, r*s, z0[i], vr[i]*c-om*r*s,
                    vr[i]*s+om*r*c, vz[i]]
    return y


def run_rk4(geom, radii, thetas, omegas, alpha, beta, lam, mass=None,
            charge=None, chi=0.0, z0=None, vr=None, vz=None,
            periods=1000, steps_per_period=48, save_per_period=1,
            phase0=0.0, add_micromotion=True,
            escape_radius=.995, z_limit=5.0):
    if mass is None: mass = np.ones(len(radii))
    if charge is None: charge = np.ones(len(radii))
    y0 = initial_augmented(radii, thetas, omegas, z0, vr, vz)
    modes, coeff = geom["modes"], geom["coeff"]
    if add_micromotion:
        for i in range(len(radii)):
            b = NORM*i
            x, y = y0[b], y0[b+1]
            fx = 0j; fy = 0j; u = complex(x, y)
            phase = np.exp(-1j*phase0)
            for k, c in zip(modes, coeff):
                n = abs(int(k)); base = u if k > 0 else u.conjugate()
                force_x = -n*c*base**(n-1)*phase
                force_y = (-1j*n*c*base**(n-1)*phase) if k > 0 else (1j*n*c*base**(n-1)*phase)
                fx += alpha[i]*force_x; fy += alpha[i]*force_y
            delta_x = fx/(-1.0-1j*lam[i])
            delta_y = fy/(-1.0-1j*lam[i])
            y0[b] += delta_x.real; y0[b+1] += delta_y.real
            y0[b+3] += (-1j*delta_x).real; y0[b+4] += (-1j*delta_y).real
    save_stride = max(1, steps_per_period//save_per_period)
    t, data, escaped, collided = integrate_rk4(
        y0, int(periods), int(steps_per_period), int(save_stride),
        modes, coeff, np.asarray(alpha, float), np.asarray(beta, float),
        np.asarray(lam, float), np.asarray(mass, float), np.asarray(charge, float),
        float(chi), float(phase0), float(escape_radius), float(z_limit))
    return summarize_run(t, data, np.asarray(mass), np.asarray(charge), float(chi), escaped, collided)


def summarize_run(t, data, mass, charge, chi, escaped=False, collided=False):
    count = len(mass)
    particles = []
    for i in range(count):
        b = NORM*i
        p = data[:, b:b+6]
        theta = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
        radius = np.hypot(p[:, 0], p[:, 1])
        duration = max(t[-1]-t[0], 1e-30)
        win = []
        for a, c in ((.5,.7),(.7,.85),(.85,1.0)):
            ia, ic = int(a*(len(t)-1)), max(int(c*(len(t)-1)), int(a*(len(t)-1))+1)
            win.append(float((theta[ic]-theta[ia])/(2*np.pi)))
        specific = data[:, b+6:b+13]
        particles.append({
            "turns": float((theta[-1]-theta[0])/(2*np.pi)),
            "mean_omega": float((theta[-1]-theta[0])/duration),
            "window_turns": win,
            "radius_mean": float(np.mean(radius)),
            "radius_std": float(np.std(radius)),
            "radius_min": float(np.min(radius)),
            "radius_max": float(np.max(radius)),
            "z_mean": float(np.mean(p[:,2])),
            "z_rms": float(np.sqrt(np.mean(p[:,2]**2))),
            "z_max": float(np.max(np.abs(p[:,2]))),
            "K_start": float(.5*mass[i]*np.sum(p[0,3:6]**2)),
            "K_end": float(.5*mass[i]*np.sum(p[-1,3:6]**2)),
            "work_rf": float(mass[i]*specific[-1,0]),
            "work_dc": float(mass[i]*specific[-1,1]),
            "work_coulomb": float(mass[i]*specific[-1,2]),
            "work_drag": float(mass[i]*specific[-1,3]),
            "torque_rf_impulse": float(mass[i]*specific[-1,4]),
            "torque_coulomb_impulse": float(mass[i]*specific[-1,5]),
            "torque_drag_impulse": float(mass[i]*specific[-1,6]),
        })
    min_sep = None
    mean_sep = None
    coulomb_energy_end = 0.0
    if count > 1:
        d = np.linalg.norm(data[:, :3]-data[:, NORM:NORM+3], axis=1)
        min_sep = float(np.min(d))
        mean_sep = float(np.mean(d))
        coulomb_energy_end = float(chi*charge[0]*charge[1]/d[-1])
    return {"t": t, "state": data, "particles": particles,
            "escaped": bool(escaped), "collided": bool(collided),
            "duration_rf_periods": float((t[-1]-t[0])/(2*np.pi)),
            "minimum_pair_distance": min_sep,
            "mean_pair_distance": mean_sep,
            "coulomb_energy_end_unit_chi": coulomb_energy_end}


def pseudopotential(alpha, beta, x, y, modes, coeff, lam=0.0):
    """Full-spectrum ponderomotive plus DC potential per unit mass."""
    fx = 0j; fy = 0j
    u = complex(x, y)
    for k, c in zip(modes, coeff):
        n = abs(int(k)); base = u if k > 0 else u.conjugate()
        f = base**(n-1)
        fx += -n*c*f
        fy += (-1j*n*c*f) if k > 0 else (1j*n*c*f)
    rf = alpha**2*(abs(fx)**2+abs(fy)**2)/(4*(1+lam**2))
    return float(rf + beta*(-.5*(x*x+y*y)))


def complex_dc_check():
    """Exact Laplace/force check and centered finite-difference residual."""
    beta = .013
    points = np.array([[.2,-.1,.4],[-.3,.05,-.2],[.7,.2,.1]])
    h = 1e-5
    rows=[]
    for x,y,z in points:
        phi=lambda a,b,c: beta*(c*c-(a*a+b*b)/2)
        grad=np.array([(phi(x+h,y,z)-phi(x-h,y,z))/(2*h),
                       (phi(x,y+h,z)-phi(x,y-h,z))/(2*h),
                       (phi(x,y,z+h)-phi(x,y,z-h))/(2*h)])
        rows.append(float(np.max(np.abs(grad-beta*np.array([-x,-y,2*z])))))
    return {"analytic_laplacian": 0.0,
            "force_over_mass": "beta*(x,y,-2z)",
            "max_centered_gradient_error": max(rows),
            "max_analytic_trace_error": 0.0,
            "azimuthal_torque_identity": "x*a_y-y*a_x = 0"}
