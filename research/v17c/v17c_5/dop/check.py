"""Independent polar gradients of the electric potentials with SciPy's DOP853.

The 22 states are two Cartesian particles, four work channels per particle
(RF, DC, Coulomb and drag) and independent time integrals of angular speed.
All units and coefficients are specified in ../input/config.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy
from numba import njit
from scipy.integrate import ode

HERE = Path(__file__).resolve().parent
T = 2 * np.pi
CONFIG = json.loads((HERE.parent / "input" / "config.json").read_text(encoding="utf-8"))
P = np.array(CONFIG["parameters"])
HARMONICS = np.array(CONFIG["k"])
FAMILY = np.array(CONFIG["family"])
COEFF = np.array(CONFIG["coeff"])
RTOL, ATOL, MAX_STEP = 2e-11, 2e-13, T / 24


@njit(cache=True)
def derivative(t, y):
    out = np.zeros(22)
    delta = y[:3] - y[6:9]
    electric_repulsion = P[5] * delta / np.linalg.norm(delta)**3
    for particle in range(2):
        base = 6 * particle
        x, yy, z = y[base:base+3]
        v = y[base+3:base+6]
        r = np.hypot(x, yy)
        theta = np.arctan2(yy, x)
        radial, azimuthal = 0., 0.
        for h in range(len(HARMONICS)):
            mode = HARMONICS[h]
            order = abs(mode)
            family = FAMILY[h]
            phase = mode * theta - t * (1. if family == 0 else P[2])
            re, im = COEFF[h]
            cosine, sine = np.cos(phase), np.sin(phase)
            strength = P[family] * r**(order-1)
            radial -= strength * order * (re * cosine - im * sine)
            azimuthal += strength * mode * (re * sine + im * cosine)
        # The supplied modes all have order >=2 and vanish at the origin.
        ct, st = np.cos(theta), np.sin(theta)
        rf_force = np.array([radial*ct-azimuthal*st, radial*st+azimuthal*ct, 0.])
        dc_force = P[3] * np.array([x, yy, -2*z])
        pair_force = electric_repulsion if particle == 0 else -electric_repulsion
        inverse_mass = 1. if particle == 0 else P[4]
        drag_force = -(P[6+particle]/inverse_mass)*v
        out[base:base+3] = v
        out[base+3:base+6] = inverse_mass * (rf_force + dc_force + pair_force + drag_force)
        channel = 12 + 4*particle
        out[channel] = np.dot(rf_force, v)
        out[channel+1] = np.dot(dc_force, v)
        out[channel+2] = np.dot(pair_force, v)
        out[channel+3] = np.dot(drag_force, v)
        out[20+particle] = (x*v[1] - yy*v[0])/r**2
    return out


def mechanical(state):
    k = .5*(np.sum(state[:,3:6]**2, axis=1)+np.sum(state[:,9:12]**2,axis=1)/P[4])
    dc = P[3]*(state[:,2]**2+state[:,8]**2-.5*np.sum(state[:,[0,1,6,7]]**2,axis=1))
    return k + dc + P[5]/np.linalg.norm(state[:,:3]-state[:,6:9],axis=1)


def evaluate(times, state, transient):
    e = mechanical(state)
    work = state[:,12:20].reshape(-1,2,4)
    ext_work = work[:,:,0].sum(axis=1)+work[:,:,3].sum(axis=1)
    energy_res = e-e[0]-ext_work
    first = transient
    pos = np.stack((state[:,:3],state[:,6:9]),axis=1)
    radius = np.linalg.norm(pos[:,:,:2],axis=2)
    angle = np.unwrap(np.arctan2(pos[:,:,1],pos[:,:,0]),axis=0)
    total = (angle[-1]-angle[first])/T
    cuts = np.linspace(first,len(angle)-1,4,dtype=int)
    turns_by_integral = (state[-1,20:22]-state[first,20:22])/T
    return dict(absolute_RF_interval=[float(times[0]/T),float(times[-1]/T)],
        discard_initial_periods=transient,frequency_Hz=(total/(len(angle)-1-first)*CONFIG["f1_Hz"]).tolist(),
        turns_after_transient=total.tolist(),turns_by_omega_integral=turns_by_integral.tolist(),
        max_abs_turn_difference_integral_vs_unwrap=float(np.max(abs((angle-angle[0])/T-state[:,20:22]/T))),
        turns_three_windows=(np.diff(angle[cuts],axis=0)/T).tolist(),
        radius_mean_stroboscopic=radius[first:].mean(axis=0).tolist(), radius_min_stroboscopic=radius.min(axis=0).tolist(),
        radius_max_stroboscopic=radius.max(axis=0).tolist(), max_abs_z_at_samples=float(abs(pos[:,:,2]).max()),
        work_after_transient_rf_dc_coulomb_drag=(work[-1]-work[first]).tolist(),
        energy_change_late=float(e[-1]-e[first]),max_abs_total_energy_residual=float(np.max(abs(energy_res))),
        residual_normalized_integrated_rf=float(np.max(abs(energy_res))/np.abs(work[-1,:,0]).sum()),
        last_state=state[-1,:12].tolist())


def run(start, end, initial, name, discard):
    tic=time.perf_counter()
    times = np.arange(start,end+1,dtype=float)*T
    state=np.zeros((len(times),22))
    state[0,:12]=initial
    # The Fortran-compatible SciPy driver calls a JIT-compiled, polar-coordinate
    # field gradient independent of the archive's complex Cartesian RHS.
    ode_driver = ode(derivative).set_integrator('dop853',rtol=RTOL,atol=ATOL,max_step=MAX_STEP,nsteps=512)
    ode_driver.set_initial_value(state[0].copy(),times[0])
    for i,t in enumerate(times[1:],1):
        state[i]=ode_driver.integrate(t)
        if not ode_driver.successful() or not np.isfinite(state[i]).all():
            raise RuntimeError(f"DOP853 failed on primary period {start+i}")
    diag=evaluate(times,state,discard)
    diag.update(wall_s=time.perf_counter()-tic,scipy=scipy.__version__,rtol=RTOL,atol=ATOL,max_step_tau=MAX_STEP,
                solver="scipy.integrate.ode dop853; continuous initial-value solution sampled once per primary period")
    np.savez_compressed(HERE/(name+".npz"),tau=times,state=state)
    (HERE/(name+".json")).write_text(json.dumps(diag,indent=2)+"\n",encoding="utf-8")
    print(name,diag["frequency_Hz"],diag["turns_after_transient"],diag["wall_s"],flush=True)
    return diag


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["windows","full"])
    args=parser.parse_args()
    if args.mode=="full":
        run(0,15000,np.asarray(CONFIG["initial"]),"continuous",2000)
    else:
        refs=np.load(HERE/"archived_window_states.npz")
        for i,(start,end) in enumerate([(0,150),(14000,14200)]):
            d=run(start,end,refs[f"reference{i}"][0],f"window{i}",0)
            current=np.load(HERE/f"window{i}.npz")["state"][:,:12]
            d["restart_state_source"]="archived base128 RK4" if start else "nominal initial condition"
            d["max_abs_cartesian_difference_vs_archive_RK128"]=float(abs(current-refs[f"reference{i}"]).max())
            (HERE/f"window{i}.json").write_text(json.dumps(d,indent=2)+"\n",encoding="utf-8")

