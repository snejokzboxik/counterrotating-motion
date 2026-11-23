from __future__ import annotations

"""V17C.21 reduced moving-orbit model for the published refine12 case.

M0 is the stationary pseudopotential limit. M1/M2 are the moving-orbit,
linear-Stokes local-response balances (leading modes / all MFS modes). M3 is a
first self-consistent Fourier-Galerkin correction with the exact V20B
Schiller-Naumann drag law and the full MFS basis. Coulomb is phase-averaged in
the slow balances; its fast harmonic response is bounded and omitted.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research" / "v17c" / "v17c_21"
SOURCE = ROOT / "research" / "v17c" / "v17c_20a"
TARGET = ROOT / "research" / "v17c" / "v17c_20b" / "results" / "refine12_240k_corrected_drag_summary.json"
J = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


@dataclass(frozen=True)
class Mode:
    family: int
    k: int
    coefficient: complex
    amplitude: float
    frequency: float

    @property
    def n(self) -> int:
        return abs(self.k)

    @property
    def chirality(self) -> float:
        return 1.0 if self.k > 0 else -1.0


def load_inputs():
    registry = json.loads((SOURCE / "ATTRACTOR_CANDIDATES.json").read_text(encoding="utf-8"))
    candidate = next(c for c in registry["candidates"] if c["id"] == "refine12_sobol17")
    p = np.asarray(candidate["parameters"]["p_corrected"], dtype=float)
    A, B, nu, beta, m1_m2, C, lam1, lam2 = p
    modes = [Mode(int(fam), int(k), complex(*c), (A, B)[int(fam)], (1.0, nu)[int(fam)])
             for fam, k, c in zip(registry["shared_field_basis"]["MFS_family_index"],
                                  registry["shared_field_basis"]["azimuthal_modes_k"],
                                  registry["shared_field_basis"]["complex_coefficients_real_imag"])]
    cases = candidate["parameters"]
    phys = candidate["physical_parameter_mapping_from_V20B"]
    particles = [phys["particle1"], phys["particle2"]]
    data = {
        "candidate": candidate,
        "modes": modes,
        "alpha": [1.0, float(cases["alpha2"])],
        "lambda": [float(lam1), float(lam2)],
        "mass_ratio_m1_m2": float(m1_m2),
        "charge_ratio_q2_q1": float(cases["q2_over_q1"]),
        "C": float(C),
        "beta": float(beta),
        "gravity_z": float(cases["gravity_z"]),
        "Re_per_speed": [float(particles[i]["Re_per_dimensionless_speed"]) for i in range(2)],
        "f1_Hz": float(cases["f1_Hz"]),
        "R_m": float(phys["trap_radius_m"]),
        "exact_p": p.tolist(),
        "field_sha256": registry["code_sha256"]["field_basis"],
        "pair_rhs_sha256": registry["code_sha256"]["pair_integrator"],
    }
    data["target"] = json.loads(TARGET.read_text(encoding="utf-8"))
    return data


def leading_indices(modes: list[Mode]) -> list[int]:
    result = []
    for family, k in ((0, 2), (1, -4)):
        result.append(next(i for i, m in enumerate(modes) if m.family == family and m.k == k))
    return result


def linear_response(r: float, omega: float, alpha: float, lam: float,
                    beta: float, mode: Mode) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """V15/V16 co-rotating susceptibility and first averaged RF force.

    Phasors use Re[xhat exp(-i sigma tau)]; sigma=Omega_j-k*omega.
    """
    n, s = mode.n, mode.chirality
    sigma = mode.frequency - mode.k * omega
    amp = mode.amplitude * mode.coefficient
    fhat = -alpha * n * amp * r ** (n - 1) * np.array([1.0, 1j * s])
    hess = -alpha * n * (n - 1) * amp * r ** (n - 2) * np.array(
        [[1.0, 1j * s], [1j * s, -1.0]], dtype=complex)
    D = (-sigma**2 * I2 - 1j * sigma * (lam * I2 + 2.0 * omega * J)
         - omega**2 * I2 + lam * omega * J - alpha * beta * I2)
    xi = np.linalg.solve(D, fhat)
    mean_force = 0.5 * np.real(hess @ np.conjugate(xi))
    return xi, mean_force, float(sigma), D


def coulomb_ring_average(r1: float, r2: float, z1: float, z2: float,
                         C: float, s: float, samples: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Mean two-body Coulomb accelerations on independently sampling circular rings."""
    phi = 2.0 * np.pi * np.arange(samples) / samples
    cp = np.cos(phi)
    dz = z1 - z2
    d2 = np.maximum(r1*r1 + r2*r2 - 2*r1*r2*cp + dz*dz, 1e-30)
    invd3 = d2 ** -1.5
    f1 = np.array([C*np.mean((r1-r2*cp)*invd3), 0.0, C*dz*np.mean(invd3)])
    f2 = np.array([s*C*np.mean((r2-r1*cp)*invd3), 0.0, -s*C*dz*np.mean(invd3)])
    return f1, f2


def mfs_rf_means(r: float, omega: float, alpha: float, lam: float,
                 beta: float, modes: list[Mode], indices: list[int]) -> tuple[np.ndarray, list[dict]]:
    total = np.zeros(2)
    components = []
    for idx in indices:
        mode = modes[idx]
        xi, mean, sigma, D = linear_response(r, omega, alpha, lam, beta, mode)
        amp = mode.amplitude * mode.coefficient
        f0_hat = -alpha * mode.n * amp * r ** (mode.n - 1) * np.array(
            [1.0, 1j * mode.chirality], dtype=complex)
        micro_rf_torque = 0.5 * np.real(xi[0]*np.conjugate(f0_hat[1])
                                        - xi[1]*np.conjugate(f0_hat[0]))
        xdot_hat = -1j * sigma * xi
        micro_L = 0.5 * float(np.vdot(xi, xi).real) * omega + 0.5 * np.real(
            xi[0]*np.conjugate(xdot_hat[1]) - xi[1]*np.conjugate(xdot_hat[0]))
        total_rf_torque = r * mean[1] + micro_rf_torque
        total += mean
        components.append({
            "family": mode.family + 1, "k": mode.k, "n": mode.n,
            "sigma_rf_over_Omega1": sigma,
            "micromotion_amplitude_R": float(np.linalg.norm(xi)),
            "mean_rf_force_rotating_xy": mean.tolist(),
            "mean_rf_torque_about_mean_orbit": float(r * mean[1]),
            "micromotion_RF_torque": float(micro_rf_torque),
            "mean_total_RF_torque": float(total_rf_torque),
            "micromotion_angular_momentum": float(micro_L),
            "response_condition_number": float(np.linalg.cond(D)),
        })
    return total, components


def m12_balance(x: np.ndarray, data: dict, indices: list[int]) -> np.ndarray:
    r1, w1, r2, w2, z1, z2 = map(float, x)
    alpha, lam = data["alpha"], data["lambda"]
    beta, C, s, gz = data["beta"], data["C"], data["mass_ratio_m1_m2"], data["gravity_z"]
    fc1, fc2 = coulomb_ring_average(r1, r2, z1, z2, C, s)
    residual = []
    for i, (r, w, z, fc) in enumerate(((r1, w1, z1, fc1), (r2, w2, z2, fc2))):
        rf, components = mfs_rf_means(r, w, alpha[i], lam[i], beta, data["modes"], indices)
        drag = np.array([-lam[i] * 0.0, -lam[i] * w * r])
        rf_torque = sum(c["mean_total_RF_torque"] for c in components)
        micro_L = sum(c["micromotion_angular_momentum"] for c in components)
        drag_torque = -lam[i] * (r*r*w + micro_L)
        residual.extend([
            (r*w*w + alpha[i]*beta*r + fc[0] + rf[0] + drag[0]) * 100.0,
            (rf_torque + drag_torque + r*fc[1]) * 1.0e4,
            (-2.0*alpha[i]*beta*z + gz + fc[2]) * 100.0,
        ])
    return np.asarray(residual)


def solve_m12(data: dict, indices: list[int]) -> dict:
    # Seeds are broad analytical scales, not the archived radii or frequencies.
    zseed = [data["gravity_z"]/(2*a*data["beta"]) for a in data["alpha"]]
    x0 = np.array([0.46, 0.001, 0.82, -0.001, zseed[0], zseed[1]])
    sol = least_squares(lambda x: m12_balance(x, data, indices), x0,
                        bounds=([0.1, -0.05, 0.1, -0.05, -2.0, -2.0],
                                [0.99, 0.05, 0.99, 0.05, 2.0, 2.0]),
                        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=4000)
    r1, w1, r2, w2, z1, z2 = map(float, sol.x)
    alpha, lam = data["alpha"], data["lambda"]
    rf1, c1 = mfs_rf_means(r1, w1, alpha[0], lam[0], data["beta"], data["modes"], indices)
    rf2, c2 = mfs_rf_means(r2, w2, alpha[1], lam[1], data["beta"], data["modes"], indices)
    return {
        "success": bool(sol.success), "cost": float(sol.cost), "solver_residual_scaled": m12_balance(sol.x, data, indices).tolist(),
        "radii_R": [r1, r2], "omega_over_Omega1": [w1, w2],
        "orbital_frequency_Hz": [w1*data["f1_Hz"], w2*data["f1_Hz"]], "mean_z_R": [z1, z2],
        "mean_RF_force_rotating_xy": [rf1.tolist(), rf2.tolist()],
        "mode_torque_contributions": [c1, c2],
        "ring_averaged_Coulomb_acceleration": [x.tolist() for x in coulomb_ring_average(r1, r2, z1, z2, data["C"], data["mass_ratio_m1_m2"])],
        "used_MFS_indices": indices,
    }


def _phase_grid(ntheta: int, nphase: int):
    theta = 2*np.pi*np.arange(ntheta)/ntheta
    phase = 2*np.pi*np.arange(nphase)/nphase
    th, p1, p2 = np.meshgrid(theta, phase, phase, indexing="ij")
    return th, (p1, p2)


def _rf_field_rotating(w: np.ndarray, theta: np.ndarray, phases: tuple[np.ndarray, np.ndarray],
                       alpha: float, modes: list[Mode]) -> np.ndarray:
    out = np.zeros(w.shape, dtype=np.complex128)
    for m in modes:
        n = m.n
        psi = phases[m.family]
        P = m.amplitude * m.coefficient
        if m.k > 0:
            z = -alpha*n*np.conjugate(P)*np.exp(1j*psi-1j*n*theta)*np.conjugate(w)**(n-1)
        else:
            z = -alpha*n*P*np.exp(-1j*psi-1j*n*theta)*np.conjugate(w)**(n-1)
        out += z
    return out


def nonlinear_micromotion(r: float, omega: float, alpha: float, lam: float,
                          re_per_speed: float, beta: float, modes: list[Mode],
                          ntheta: int = 96, nphase: int = 12,
                          tolerance: float = 2e-9, max_iter: int = 160) -> dict:
    """Self-consistent retained-mode HB with exact Schiller-Naumann drag.

    It projects the exact polynomial MFS field and nonlinear drag onto each
    retained RF mode. Higher combination modes are measured as a residual.
    """
    theta, phases = _phase_grid(ntheta, nphase)
    shape = theta.shape
    eta = []
    sigma = []
    D = []
    X = []
    for m in modes:
        phase = np.angle(m.coefficient) + m.k*theta - phases[m.family]
        eta.append(np.exp(1j*phase))
        sig = m.frequency - m.k*omega
        sigma.append(sig)
        Dm = (-sig**2*I2 - 1j*sig*(lam*I2+2.0*omega*J)
              -omega**2*I2 + lam*omega*J - alpha*beta*I2)
        D.append(Dm)
        n = m.n
        fhat = -alpha*n*abs(m.amplitude*m.coefficient)*r**(n-1)*np.array([1.0, 1j*m.chirality])
        X.append(np.linalg.solve(Dm, fhat))
    X = np.asarray(X, dtype=np.complex128)
    eta = np.asarray(eta)

    for iteration in range(1, max_iter+1):
        xi = np.zeros(shape+(2,), dtype=float)
        dxi = np.zeros_like(xi)
        for h in range(len(modes)):
            xi += np.real(eta[h][..., None]*X[h])
            dxi += np.real((-1j*sigma[h])*eta[h][..., None]*X[h])
        w = (r+xi[..., 0]) + 1j*xi[..., 1]
        v = np.zeros_like(xi)
        v[..., 0] = 0.0
        v[..., 1] = omega*r
        v += dxi + omega*np.einsum("ab,...b->...a", J.real, xi)
        speed = np.linalg.norm(v, axis=-1)
        rf_complex = _rf_field_rotating(w, theta, phases, alpha, modes)
        rf_vec = np.stack([rf_complex.real, rf_complex.imag], axis=-1)
        sn = (-0.15*lam*(re_per_speed*speed)**0.687)[..., None]*v
        targets = []
        for h in range(len(modes)):
            proj_rf = 2.0*np.mean(rf_vec*eta[h].conj()[..., None], axis=(0, 1, 2))
            proj_sn = 2.0*np.mean(sn*eta[h].conj()[..., None], axis=(0, 1, 2))
            targets.append(np.linalg.solve(D[h], proj_rf+proj_sn))
        targets = np.asarray(targets)
        err = np.max(np.abs(targets-X))/max(1e-15, np.max(np.abs(X)))
        X = 0.55*targets + 0.45*X
        if err < tolerance:
            break

    # Re-evaluate at the converged coefficient vector.
    xi = np.zeros(shape+(2,), dtype=float); dxi = np.zeros_like(xi)
    for h in range(len(modes)):
        xi += np.real(eta[h][..., None]*X[h])
        dxi += np.real((-1j*sigma[h])*eta[h][..., None]*X[h])
    w = (r+xi[..., 0]) + 1j*xi[..., 1]
    v = np.zeros_like(xi); v[..., 1] = omega*r
    v += dxi + omega*np.einsum("ab,...b->...a", J.real, xi)
    speed = np.linalg.norm(v, axis=-1)
    rf_complex = _rf_field_rotating(w, theta, phases, alpha, modes)
    rf_vec = np.stack([rf_complex.real, rf_complex.imag], axis=-1)
    drag = -lam*(1.0+0.15*(re_per_speed*speed)**0.687)[..., None]*v
    sn = drag + lam*v
    pos = np.stack((r+xi[..., 0], xi[..., 1]), axis=-1)
    rf_torque = float(np.mean(pos[..., 0]*rf_vec[..., 1]-pos[..., 1]*rf_vec[..., 0]))
    drag_torque = float(np.mean(pos[..., 0]*drag[..., 1]-pos[..., 1]*drag[..., 0]))
    mean_L = float(np.mean(pos[..., 0]*v[..., 1]-pos[..., 1]*v[..., 0]))
    # Exact average force and drag on the retained-mode orbit.
    mean_rf = np.mean(rf_vec, axis=(0, 1, 2))
    mean_drag = np.mean(drag, axis=(0, 1, 2))
    # RMS residual in unretained combination harmonics of the fast equation.
    op = np.zeros(shape+(2,), dtype=float)
    for h in range(len(modes)):
        op += np.real(eta[h][..., None]*(D[h]@X[h]))
    res = op-rf_vec-sn
    res -= np.mean(res, axis=(0, 1, 2), keepdims=True)
    for h in range(len(modes)):
        coef = 2.0*np.mean(res*eta[h].conj()[..., None], axis=(0, 1, 2))
        res -= np.real(eta[h][..., None]*coef)
    return {
        "xi_hat_real_imag": [[[float(z.real), float(z.imag)] for z in pair] for pair in X],
        "sigma": np.asarray(sigma).tolist(),
        "converged": bool(err < tolerance), "iterations": iteration,
        "relative_fixed_point_residual": float(err),
        "mean_rf_force_rotating_xy": mean_rf.tolist(),
        "mean_drag_force_rotating_xy": mean_drag.tolist(),
        "mean_total_RF_torque": rf_torque,
        "mean_total_drag_torque": drag_torque,
        "mean_angular_momentum": mean_L,
        "mean_speed": float(np.mean(speed)),
        "rms_micromotion": float(np.sqrt(np.mean(np.sum(xi*xi, axis=-1)))),
        "max_micromotion": float(np.max(np.linalg.norm(xi, axis=-1))),
        "rms_omitted_harmonic_force_residual": float(np.sqrt(np.mean(np.sum(res*res, axis=-1)))),
        "phase_grid": [ntheta, nphase, nphase],
    }


def nonlinear_micromotion_full(r: float, omega: float, alpha: float, lam: float,
                               re_per_speed: float, beta: float, modes: list[Mode],
                               ntheta: int = 192, nphase: int = 16,
                               tolerance: float = 2e-9, max_iter: int = 120,
                               slow_mode_cutoff: float = 0.05) -> dict:
    """Pseudospectral solve on the full (orbital, RF1, RF2) phase torus.

    Unlike nonlinear_micromotion(), this retains all Fourier combination modes
    representable by the collocation grid. The zero mode is assigned to the slow
    orbit equations and removed from the fast displacement problem.
    """
    theta, phases = _phase_grid(ntheta, nphase)
    shape = theta.shape
    ktheta = np.fft.fftfreq(ntheta)*ntheta
    kphase = np.fft.fftfreq(nphase)*nphase
    kt, kp1, kp2 = np.meshgrid(ktheta, kphase, kphase, indexing="ij")
    carrier_ratio = modes[0].frequency
    nu = next(m.frequency for m in modes if m.family == 1)
    deriv_freq = kt*omega + kp1*carrier_ratio + kp2*nu
    # Orbital-only and near-resonant combination modes belong to slow dynamics,
    # not the RF micromotion inverse. They are retained in the residual ledger.
    fast_mask = ((kp1 != 0) | (kp2 != 0)) & (np.abs(deriv_freq) >= slow_mode_cutoff)
    d0 = (-deriv_freq**2 + 1j*deriv_freq*lam - omega**2 - alpha*beta)
    dJ = (2j*omega*deriv_freq + lam*omega)
    den = d0*d0 + dJ*dJ
    if np.any(np.abs(den[1:, :, :]) < 1e-15):
        raise ArithmeticError("Near-resonant/singular nonzero Fourier mode in micromotion operator")
    xi = np.zeros(shape+(2,),dtype=float)
    err = math.inf
    for iteration in range(1,max_iter+1):
        w=(r+xi[...,0])+1j*xi[...,1]
        xihat=np.fft.fftn(xi,axes=(0,1,2))/np.prod(shape)
        xidot_hat=1j*deriv_freq[...,None]*xihat
        xidot=np.fft.ifftn(xidot_hat*np.prod(shape),axes=(0,1,2)).real
        v=np.zeros_like(xi); v[...,1]=omega*r
        v+=xidot+omega*np.einsum("ab,...b->...a",J.real,xi)
        speed=np.linalg.norm(v,axis=-1)
        rf_complex=_rf_field_rotating(w,theta,phases,alpha,modes)
        rf=np.stack((rf_complex.real,rf_complex.imag),axis=-1)
        extra=(-0.15*lam*(re_per_speed*speed)**0.687)[...,None]*v
        force_hat=np.fft.fftn(rf+extra,axes=(0,1,2))/np.prod(shape)
        fx,fy=force_hat[...,0],force_hat[...,1]
        target_hat=np.empty_like(force_hat)
        target_hat[...,0]=(d0*fx+dJ*fy)/den
        target_hat[...,1]=(-dJ*fx+d0*fy)/den
        target_hat[~fast_mask,:]=0.0
        target=np.fft.ifftn(target_hat*np.prod(shape),axes=(0,1,2)).real
        err=float(np.max(np.abs(target-xi))/max(1e-14,np.max(np.abs(target))))
        xi=0.5*target+0.5*xi
        if err<tolerance:
            break
    # Recompute all means/residuals at the returned fixed-point iterate.
    w=(r+xi[...,0])+1j*xi[...,1]
    xihat=np.fft.fftn(xi,axes=(0,1,2))/np.prod(shape)
    xidot_hat=1j*deriv_freq[...,None]*xihat
    xidot=np.fft.ifftn(xidot_hat*np.prod(shape),axes=(0,1,2)).real
    xiacc_hat=-(deriv_freq**2)[...,None]*xihat
    xiacc=np.fft.ifftn(xiacc_hat*np.prod(shape),axes=(0,1,2)).real
    v=np.zeros_like(xi); v[...,1]=omega*r
    v+=xidot+omega*np.einsum("ab,...b->...a",J.real,xi)
    speed=np.linalg.norm(v,axis=-1)
    rf_complex=_rf_field_rotating(w,theta,phases,alpha,modes)
    rf=np.stack((rf_complex.real,rf_complex.imag),axis=-1)
    drag=-lam*(1+0.15*(re_per_speed*speed)**0.687)[...,None]*v
    extra=drag+lam*v
    linear_op=xiacc+lam*xidot+2*omega*np.einsum("ab,...b->...a",J.real,xidot)-omega**2*xi+lam*omega*np.einsum("ab,...b->...a",J.real,xi)-alpha*beta*xi
    residual=linear_op-rf-extra
    residual-=np.mean(residual,axis=(0,1,2),keepdims=True)
    residual_hat=np.fft.fftn(residual,axes=(0,1,2))/np.prod(shape)
    rms_slow=float(np.sqrt(np.sum(np.abs(residual_hat[~fast_mask])**2)))
    rms_fast=float(np.sqrt(np.sum(np.abs(residual_hat[fast_mask])**2)))
    pos=np.stack((r+xi[...,0],xi[...,1]),axis=-1)
    return {"converged":bool(err<tolerance),"iterations":iteration,
      "relative_fixed_point_residual":err,"mean_rf_force_rotating_xy":np.mean(rf,axis=(0,1,2)).tolist(),
      "mean_drag_force_rotating_xy":np.mean(drag,axis=(0,1,2)).tolist(),
      "mean_total_RF_torque":float(np.mean(pos[...,0]*rf[...,1]-pos[...,1]*rf[...,0])),
      "mean_total_drag_torque":float(np.mean(pos[...,0]*drag[...,1]-pos[...,1]*drag[...,0])),
      "mean_angular_momentum":float(np.mean(pos[...,0]*v[...,1]-pos[...,1]*v[...,0])),
      "mean_speed":float(np.mean(speed)),"rms_micromotion":float(np.sqrt(np.mean(np.sum(xi*xi,axis=-1)))),
      "max_micromotion":float(np.max(np.linalg.norm(xi,axis=-1))),
      "max_speed":float(np.max(speed)),"mean_Re_estimate":float(re_per_speed*np.mean(speed)),
      "max_Re_estimate":float(re_per_speed*np.max(speed)),
      "rms_rf_force":float(np.sqrt(np.mean(np.sum(rf*rf,axis=-1)))),
      "rms_drag_force":float(np.sqrt(np.mean(np.sum(drag*drag,axis=-1)))),
      "rms_unresolved_force":float(np.sqrt(np.mean(np.sum(residual*residual,axis=-1)))),
      "rms_slow_mode_residual":rms_slow,"rms_fast_mode_equation_residual":rms_fast,
      "slow_mode_cutoff":slow_mode_cutoff,"number_of_fast_modes":int(np.count_nonzero(fast_mask)),
      "phase_grid":[ntheta,nphase,nphase],"minimum_fast_mode_denominator":float(np.min(np.abs(den[fast_mask])))}


def m3_balance(x: np.ndarray, data: dict, quadrature: tuple[int, int]=(96,12)) -> tuple[np.ndarray, dict]:
    r1, w1, r2, w2, z1, z2 = map(float, x)
    alpha, lam = data["alpha"], data["lambda"]
    beta, C, s, gz = data["beta"], data["C"], data["mass_ratio_m1_m2"], data["gravity_z"]
    fc1, fc2 = coulomb_ring_average(r1, r2, z1, z2, C, s)
    rows=[]; summaries=[]
    for i,(r,w,z,fc) in enumerate(((r1,w1,z1,fc1),(r2,w2,z2,fc2))):
        hb=nonlinear_micromotion_full(r,w,alpha[i],lam[i],data["Re_per_speed"][i],beta,data["modes"],
                                      *quadrature,tolerance=2e-6,max_iter=120)
        rf=hb["mean_rf_force_rotating_xy"]; drag=hb["mean_drag_force_rotating_xy"]
        rows.extend([(r*w*w+alpha[i]*beta*r+fc[0]+rf[0]+drag[0])*100.0,
                     (hb["mean_total_RF_torque"]+hb["mean_total_drag_torque"]+r*fc[1])*1e4,
                     (-2.0*alpha[i]*beta*z+gz+fc[2]-lam[i]*0.0)*100.0])
        summaries.append(hb)
    return np.asarray(rows), {"particles":summaries}


def solve_m3(data: dict, seed: dict, quadrature: tuple[int, int] = (96, 12)) -> dict:
    x0=np.array([seed["radii_R"][0],seed["omega_over_Omega1"][0],seed["radii_R"][1],
                 seed["omega_over_Omega1"][1],seed["mean_z_R"][0],seed["mean_z_R"][1]])
    cache={"x":None,"balance":None,"extra":None}
    def fun(x):
        bal,extra=m3_balance(x,data,quadrature)
        cache.update(x=x.copy(),balance=bal,extra=extra)
        return bal
    sol=least_squares(fun,x0,bounds=([0.1,-0.05,0.1,-0.05,-2.,-2.],[0.99,0.05,0.99,0.05,2.,2.]),
                      xtol=3e-10,ftol=3e-10,gtol=3e-10,max_nfev=160)
    residual,extra=m3_balance(sol.x,data,quadrature)
    r1,w1,r2,w2,z1,z2=map(float,sol.x)
    return {"success":bool(sol.success),"message":str(sol.message),"cost":float(sol.cost),
            "solver_residual_scaled":residual.tolist(),"radii_R":[r1,r2],
            "omega_over_Omega1":[w1,w2],"orbital_frequency_Hz":[w1*data["f1_Hz"],w2*data["f1_Hz"]],
            "mean_z_R":[z1,z2],"inner_harmonic_balance":extra,"phase_grid":list(quadrature)}


def write_results(data: dict) -> dict:
    modes=data["modes"]
    lead=solve_m12(data,leading_indices(modes))
    full=solve_m12(data,list(range(len(modes))))
    m3=solve_m3(data,full)
    c0_data=dict(data); c0_data["C"]=0.0
    c0=solve_m12(c0_data,list(range(len(modes))))
    target=data["target"]
    target_r=target["late_mean_radius_R"]
    target_f=target["late_frequency_Hz"]
    models={"M1_leading_modes_linear_Stokes":lead,"M2_full_MFS_linear_Stokes":full,"M3_full_MFS_SN_torus_Galerkin":m3}
    for name,row in models.items():
        row["radius_relative_error_percent"]=[100*(a-b)/b for a,b in zip(row["radii_R"],target_r)]
        row["frequency_absolute_error_Hz"]=[a-b for a,b in zip(row["orbital_frequency_Hz"],target_f)]
        row["frequency_relative_error_percent"]=[100*(a-b)/abs(b) for a,b in zip(row["orbital_frequency_Hz"],target_f)]
    result={"study":"V17C.21","base_commit":"43c84bf72ce0807d244475518588770fbaf0fb69",
            "source_input":data["exact_p"],"alpha":data["alpha"],"lambda":data["lambda"],
            "full_modes":[{"family":m.family+1,"k":m.k,"coefficient":[m.coefficient.real,m.coefficient.imag],
                           "amplitude":m.amplitude,"carrier_frequency_over_Omega1":m.frequency} for m in modes],
            "numerical_reference":{"radii_R":target_r,"frequency_Hz":target_f,"horizon_periods":target["completed_periods"],
                                   "turns":target["late_turns"],"source":"V20B refine12_240k_corrected_drag_summary.json"},
            "models":models,
            "coulomb_off_M2_control":{"radii_R":c0["radii_R"],"omega_over_Omega1":c0["omega_over_Omega1"],
                "frequency_Hz":c0["orbital_frequency_Hz"],
                "frequency_shift_from_C_on_Hz":[float(a-b) for a,b in zip(c0["orbital_frequency_Hz"],full["orbital_frequency_Hz"])],
                "scope":"Analytical torus-averaged M2 control with the same particle and field parameters; not an additional full trajectory."},
            "interpretation":{"M0":"Stationary leading pseudopotential has zero secular azimuthal torque and cannot predict nonzero counterrotation.",
                "M1":"Moving-orbit susceptibility for k=+2/-4, linear Stokes drag, phase-averaged Coulomb.",
                "M2":"Same closure with all eight complex MFS modes; spatial truncation effect is measured by M1-to-M2 difference.",
            "M3":"Full MFS polynomial and self-consistent Fourier-collocation solution on the orbital/RF phase torus with Schiller-Naumann drag; orbital-only and near-resonant modes below the stated cutoff are assigned to the slow residual. Coulomb enters the slow torus-averaged balance, not the fast micromotion RHS.",
                "torus_assumption":"RF phases and orbital phase are averaged on a nonresonant torus; V20A found phase drift, no locking. This is a conditional averaging hypothesis, not a proof of rational independence.",
                "stability":"The averaged slow vector-field Jacobian and its numerical spectrum are stored separately by stability_analysis.py."}}
    out=BASE/"results"/"analytical_predictions.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    return result


if __name__ == "__main__":
    data=load_inputs()
    result=write_results(data)
    print(json.dumps({k:{"r":v["radii_R"],"f":v["orbital_frequency_Hz"],"dr":v["radius_relative_error_percent"],"df":v["frequency_relative_error_percent"]} for k,v in result["models"].items()},indent=2))
