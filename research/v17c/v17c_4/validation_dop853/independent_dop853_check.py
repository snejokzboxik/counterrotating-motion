"""Independent DOP853 audit of the V17C.4 ADCFEB RF--DC pair equations.

This file deliberately does not call the V17C.3 RHS or its RF-force helper.
It uses the saved full MFS coefficient list, integrates the Cartesian equations
and mechanical-work channels independently, and compares against the existing
RK4 implementation at two fixed resolutions.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
sys.path.insert(0, str(V4.parent / "v17c_3"))
from rf_dc_core import run_rk4, _rf_accel  # only comparison implementations

SPEC = V4 / "results" / "fem_sixphase" / "G1_G2_12electrode_6phase_P1_FEM.json"
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
N = 2
STRIDE = 13


def read_spectrum():
    doc = json.loads(SPEC.read_text(encoding="utf-8"))
    rows = doc["geometries"]["ADCFEBADCFEB"]["mesh_levels"][-1]["mfs_coefficients"]
    coeff = {int(k): complex(re, im) for k, re, im in rows}
    # The MFS solve returns tiny round-off values in orders forbidden by the
    # exact pi rotational symmetry of the boundary data.
    coeff = {k: (v if k % 2 == 0 else 0j) for k, v in coeff.items()}
    modes = np.asarray(sorted(coeff), dtype=np.int64)
    values = np.asarray([coeff[int(k)] for k in modes], dtype=np.complex128)
    return modes, values


def rf_accel(x: float, y: float, tau: float, alpha: float,
             modes: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """Direct Cartesian gradient of Re(sum Ak z^|k| exp(-i tau))."""
    z = complex(x, y)
    carrier = np.exp(-1j * tau)
    ax = 0.0
    ay = 0.0
    for k, ak in zip(modes, coeff):
        k = int(k)
        n = abs(k)
        base = z if k > 0 else z.conjugate()
        derivative = n * ak * base ** (n - 1) * carrier
        ax -= alpha * derivative.real
        # d/dy z^n = +i n z^(n-1); d/dy zbar^n = -i n zbar^(n-1).
        if k > 0:
            ay -= alpha * (1j * derivative).real
        else:
            ay += alpha * (1j * derivative).real
    return np.array([ax, ay, 0.0])


def rhs(t: float, y: np.ndarray, modes, coeff, alpha, beta, lam,
        mass, charge, chi, phase0: float) -> np.ndarray:
    state = y.reshape(N, STRIDE)
    out = np.zeros_like(state)
    a_rf = np.zeros((N, 3))
    a_dc = np.zeros((N, 3))
    a_c = np.zeros((N, 3))
    a_drag = np.zeros((N, 3))
    for i in range(N):
        x, yy, z = state[i, :3]
        a_rf[i] = rf_accel(x, yy, t + phase0, alpha[i], modes, coeff)
        a_dc[i] = beta[i] * np.array([x, yy, -2.0 * z])
        a_drag[i] = -lam[i] * state[i, 3:6]
    dr = state[0, :3] - state[1, :3]
    r2 = float(np.dot(dr, dr))
    if r2 < 1e-14:
        raise FloatingPointError("pair separation reached the model singularity floor")
    a_c[0] = chi * charge[0] * charge[1] * dr / (mass[0] * r2 ** 1.5)
    a_c[1] = -chi * charge[0] * charge[1] * dr / (mass[1] * r2 ** 1.5)
    for i in range(N):
        pos = state[i, :3]
        vel = state[i, 3:6]
        out[i, :3] = vel
        out[i, 3:6] = a_rf[i] + a_dc[i] + a_c[i] + a_drag[i]
        out[i, 6] = np.dot(vel, a_rf[i])
        out[i, 7] = np.dot(vel, a_dc[i])
        out[i, 8] = np.dot(vel, a_c[i])
        out[i, 9] = np.dot(vel, a_drag[i])
        out[i, 10] = pos[0] * a_rf[i, 1] - pos[1] * a_rf[i, 0]
        out[i, 11] = pos[0] * a_c[i, 1] - pos[1] * a_c[i, 0]
        out[i, 12] = pos[0] * a_drag[i, 1] - pos[1] * a_drag[i, 0]
    return out.reshape(-1)


def energy_and_torque(state, beta, mass, charge, chi):
    st = state.reshape(N, STRIDE)
    energy = 0.0
    lz = np.zeros(N)
    for i in range(N):
        x, y, z, vx, vy, vz = st[i, :6]
        energy += mass[i] * (0.5 * (vx*vx + vy*vy + vz*vz)
                            + beta[i] * (z*z - 0.5*(x*x + y*y)))
        lz[i] = mass[i] * (x*vy - y*vx)
    sep = np.linalg.norm(st[0, :3] - st[1, :3])
    energy += chi * charge[0] * charge[1] / sep
    return energy, lz, sep


def dc_checks():
    # Phi_DC = beta (z^2 - (x^2+y^2)/2) has Hessian diag(-beta,-beta,2beta).
    hess = np.diag([-1.0, -1.0, 2.0])
    points = np.array([[0.2, -0.1, 0.4], [-0.3, 0.05, -0.2], [0.7, 0.2, 0.1]])
    beta0 = 0.013
    force = beta0 * points * np.array([-1.0, -1.0, 2.0])
    tau_z = points[:, 0] * force[:, 1] - points[:, 1] * force[:, 0]
    h = 1e-5
    def phi(a, b, c):
        return beta0 * (c*c - 0.5*(a*a+b*b))
    fd_errors=[]
    for x,y,z in points:
        grad_fd=np.array([(phi(x+h,y,z)-phi(x-h,y,z))/(2*h),
                          (phi(x,y+h,z)-phi(x,y-h,z))/(2*h),
                          (phi(x,y,z+h)-phi(x,y,z-h))/(2*h)])
        grad_exact=beta0*np.array([-x,-y,2*z])
        fd_errors.append(float(np.max(np.abs(grad_fd-grad_exact))))
    return {"laplacian_hessian_trace_over_beta": float(np.trace(hess)),
            "force_per_mass": "beta*(x,y,-2z)",
            "max_absolute_dc_azimuthal_torque": float(np.max(np.abs(tau_z))),
            "analytic_dc_gradient": "beta*(-x,-y,2z)",
            "max_centered_finite_difference_gradient_error": max(fd_errors)}


def rf_checks(modes, coeff):
    alpha=0.137
    points=[(0.24,-0.13,0.18),(-0.41,0.22,2.03),(0.58,0.07,4.7)]
    core_errors=[]
    fd_errors=[]
    h=1e-6
    def potential(x,y,tau):
        z=complex(x,y); carrier=np.exp(-1j*tau); total=0j
        for k,ak in zip(modes,coeff):
            k=int(k); n=abs(k); base=z if k>0 else z.conjugate()
            total += ak*base**n*carrier
        return float(total.real)
    for x,y,tau in points:
        ours=rf_accel(x,y,tau,alpha,modes,coeff)
        core=np.asarray(_rf_accel(x,y,tau,modes,coeff,alpha),float)
        core_errors.append(float(np.max(np.abs(ours[:2]-core))))
        fd=np.array([-(potential(x+h,y,tau)-potential(x-h,y,tau))/(2*h),
                     -(potential(x,y+h,tau)-potential(x,y-h,tau))/(2*h)])*alpha
        fd_errors.append(float(np.max(np.abs(ours[:2]-fd))))
    return {"potential_convention": "Re(sum Ak*z^k*exp(-i*tau)); negative k uses conjugate(z)^|k|",
            "max_component_error_vs_existing_core_rf_accel": max(core_errors),
            "max_component_error_vs_negative_finite_difference_potential_gradient": max(fd_errors)}


def run():
    modes, coeff = read_spectrum()
    geom = {"label": "ADCFEBADCFEB", "N": 12, "modes": modes, "coeff": coeff}
    # A deterministic modest-amplitude case. beta_i/alpha_i is shared, so
    # both test particles see the same RF and DC voltage ratio.
    alpha = np.array([0.11, 0.072])
    common_dc_ratio = 0.031
    beta = alpha * common_dc_ratio
    lam = np.array([0.007, 0.011])
    mass = np.array([1.0, 1.75])
    charge = np.array([1.0, 1.05])
    chi = 8.0e-5
    radii = [0.29, 0.54]
    angles = [0.31, 2.20]
    omegas = [0.028, -0.022]
    z0 = [-0.012, 0.019]
    vr = [0.002, -0.001]
    vz = [0.003, -0.002]
    phase0 = 0.37
    periods = 10

    rk = {}
    for spp in (64, 128):
        result = run_rk4(geom, radii, angles, omegas, alpha, beta, lam,
                         mass=mass, charge=charge, chi=chi,
                         z0=z0, vr=vr, vz=vz, periods=periods,
                         steps_per_period=spp, save_per_period=spp,
                         phase0=phase0, escape_radius=0.995, z_limit=1.0)
        rk[spp] = result

    # Use the fine RK4 initial state, including its analytic micromotion kick.
    y0 = rk[128]["state"][0].reshape(-1).copy()
    ts = rk[128]["t"]
    dop = solve_ivp(rhs, (0.0, float(ts[-1])), y0, method="DOP853", t_eval=ts,
                    rtol=2e-12, atol=2e-14, max_step=2*np.pi/128,
                    args=(modes, coeff, alpha, beta, lam, mass, charge, chi, phase0))
    if not dop.success:
        raise RuntimeError(dop.message)

    d = dop.y.T
    y64, y128 = rk[64]["state"], rk[128]["state"]
    # Compare common temporal nodes (one saved state per integration step).
    d_at_coarse = d[::2]
    state_error_64 = np.max(np.abs(y64[:, :N*6] - d_at_coarse[:, :N*6]))
    state_error_128 = np.max(np.abs(y128[:, :N*6] - d[:, :N*6]))
    scale = max(1.0, float(np.max(np.abs(d[:, :N*6]))))
    energy_start = energy_and_torque(d[0], beta, mass, charge, chi)[0]
    energy_end, lz_end, sep_end = energy_and_torque(d[-1], beta, mass, charge, chi)
    work_rf = float(sum(mass[i] * d[-1].reshape(N, STRIDE)[i, 6] for i in range(N)))
    work_drag = float(sum(mass[i] * d[-1].reshape(N, STRIDE)[i, 9] for i in range(N)))
    energy_residual = energy_end - energy_start - work_rf - work_drag

    # Integrated internal Coulomb z-torque is equal and opposite in total.
    final_aug = d[-1].reshape(N, STRIDE)
    coul_torque_integral = np.asarray([mass[i] * final_aug[i, 11] for i in range(N)])
    rk_energy_audits = {}
    for spp in (64, 128):
        sr = rk[spp]["state"]
        ein = energy_and_torque(sr[0], beta, mass, charge, chi)[0]
        eout = energy_and_torque(sr[-1], beta, mass, charge, chi)[0]
        shaped = sr[-1].reshape(N, STRIDE)
        wrf = float(sum(mass[i] * shaped[i, 6] for i in range(N)))
        wdrag = float(sum(mass[i] * shaped[i, 9] for i in range(N)))
        rk_energy_audits[str(spp)] = {
            "residual_after_RF_and_drag": float(eout-ein-wrf-wdrag),
            "relative_residual": float(abs(eout-ein-wrf-wdrag)/max(abs(ein),abs(eout),1e-30))}
    result = {
        "scope": "10 RF periods, two 3D particles, ADCFEB full saved MFS spectrum",
        "spectrum_modes_count": int(len(modes)),
        "spectrum_nonzero_modes_count": int(np.count_nonzero(np.abs(coeff) > 1e-15)),
        "parameters": {"alpha": alpha.tolist(), "beta": beta.tolist(),
                       "common_Vdc_over_U0_dimensionless": common_dc_ratio,
                       "lambda": lam.tolist(), "mass": mass.tolist(),
                       "charge": charge.tolist(), "chi": chi, "phase0": phase0,
                       "periods": periods, "initial_radii": radii,
                       "initial_azimuths": angles, "initial_omega": omegas,
                       "initial_z": z0, "initial_radial_velocity": vr,
                       "initial_axial_velocity": vz},
        "dc_analytic_checks": dc_checks(),
        "rf_analytic_checks": rf_checks(modes, coeff),
        "integrators": {"DOP853": {"success": bool(dop.success),
                                     "rtol": 2e-12, "atol": 2e-14,
                                     "max_step_over_RF_period": 1/128,
                                     "nfev": int(dop.nfev)},
                        "RK4_64": {"steps_per_period": 64},
                        "RK4_128": {"steps_per_period": 128}},
        "max_cartesian_state_error_vs_DOP853": {
            "RK4_64": float(state_error_64),
            "RK4_128": float(state_error_128),
            "RK4_64_over_128_error_ratio": float(state_error_64/max(state_error_128, 1e-300)),
            "normalized_by_max_1_or_state": {"RK4_64": float(state_error_64/scale),
                                             "RK4_128": float(state_error_128/scale)}},
        "energy_balance": {"initial_total_K_plus_DC_plus_Coulomb": energy_start,
                           "final_total_K_plus_DC_plus_Coulomb": energy_end,
                           "integrated_RF_work": work_rf,
                           "integrated_drag_work": work_drag,
                           "residual_after_RF_and_drag": float(energy_residual),
                           "relative_residual": float(abs(energy_residual)/max(abs(energy_start), abs(energy_end), 1e-30)),
                           "RK4_energy_residual_by_steps_per_period": rk_energy_audits},
        "angular_momentum": {"final_Lz": lz_end.tolist(),
                              "integrated_coulomb_torque_per_particle": coul_torque_integral.tolist(),
                              "sum_internal_coulomb_torque_integrals": float(np.sum(coul_torque_integral)),
                              "final_separation": float(sep_end)},
        "notes": ["Independent RHS re-derives the field gradient and does not import _rf_accel/_rhs_aug.",
                  "Potential and fields are z-translation invariant RF plus the separate Laplace DC quadrupole.",
                  "This is an integration/model consistency check, not a physical realization or candidate search."]}
    (OUT / "dop853_rk4_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "dop853_rk4_states.npz", t=ts, dop853=d,
                        rk4_64=y64, rk4_128=y128,
                        initial_state=d[0], final_state=d[-1],
                        modes=modes, coefficients=coeff)
    print(json.dumps({"state_error_64": state_error_64,
                      "state_error_128": state_error_128,
                      "error_ratio": state_error_64/max(state_error_128,1e-300),
                      "energy_residual": energy_residual,
                      "coulomb_torque_sum": float(np.sum(coul_torque_integral)),
                      "dop853_nfev": int(dop.nfev)}, indent=2))


if __name__ == "__main__":
    run()
