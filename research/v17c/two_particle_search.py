"""Independent 3D two-charge search in the unsegmented ABC trap.

Dimensionless tau=Omega*t, xyz/R, masses/M0, charges/Q0.  This script is a
screening calculation, not a calibrated physical experiment.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEM = ROOT / "outputs/abc_validation/direct_fem_validation/v1/h35_P4/diagnostic_N48.json"
OUT = HERE / "results"


def coefficients():
    d = json.loads(FEM.read_text(encoding="utf-8"))
    c = np.array(d["cosine_real"]) + 1j * np.array(d["cosine_imag"])
    s = np.array(d["sine_real"]) + 1j * np.array(d["sine_imag"])
    # Scale the FEM unit-voltage reconstruction so |m4| equals historical k.
    scale = 0.1044 / abs(c[4])
    return np.array([c[4], s[4], c[8], s[8]], complex) * scale


@njit(cache=True)
def rhs(y, tau, cf, C, wz, gamma, masses, charges, coulomb):
    out = np.empty_like(y)
    ct, st = np.cos(tau), np.sin(tau)
    for i in range(2):
        x, yy, z, vx, vy, vz = y[i]
        u = x + 1j*yy
        D = 4*u**3*(cf[0]*ct + cf[1]*st) + 8*u**7*(cf[2]*ct + cf[3]*st)
        # Phi=Re(P(u)); E=(-Re(P'), +Im(P')).
        ex = -D.real + wz*wz*x/2
        ey = D.imag + wz*wz*yy/2
        ez = -wz*wz*z
        out[i, :3] = y[i, 3:]
        out[i, 3] = charges[i]/masses[i]*ex - gamma[i]*vx
        out[i, 4] = charges[i]/masses[i]*ey - gamma[i]*vy
        out[i, 5] = charges[i]/masses[i]*ez - gamma[i]*vz
    if coulomb:
        dx = y[0,0]-y[1,0]; dy = y[0,1]-y[1,1]; dz = y[0,2]-y[1,2]
        r2 = dx*dx + dy*dy + dz*dz
        if r2 < 1e-20: r2 = 1e-20
        f = C*charges[0]*charges[1]/(r2*np.sqrt(r2))
        out[0,3] += f*dx/masses[0]; out[1,3] -= f*dx/masses[1]
        out[0,4] += f*dy/masses[0]; out[1,4] -= f*dy/masses[1]
        out[0,5] += f*dz/masses[0]; out[1,5] -= f*dz/masses[1]
    return out


@njit(cache=True)
def integrate(y0, cf, C, wz, gamma, masses, charges, phase, periods, steps,
              coulomb=True, stride=48):
    y = y0.copy()
    dt = 2*np.pi/steps
    nsteps = periods*steps
    samples = np.empty((nsteps//stride+1,2,6))
    times = np.empty(nsteps//stride+1)
    samples[0] = y; times[0] = 0.
    count = 1; minsep = 1e10; maxr = 0.; escaped = False
    for j in range(nsteps):
        t = phase + j*dt
        a = rhs(y,t,cf,C,wz,gamma,masses,charges,coulomb)
        b = rhs(y+dt*a/2,t+dt/2,cf,C,wz,gamma,masses,charges,coulomb)
        c = rhs(y+dt*b/2,t+dt/2,cf,C,wz,gamma,masses,charges,coulomb)
        d = rhs(y+dt*c,t+dt,cf,C,wz,gamma,masses,charges,coulomb)
        y += dt*(a+2*b+2*c+d)/6
        rr0 = np.hypot(y[0,0],y[0,1]); rr1 = np.hypot(y[1,0],y[1,1])
        sep = np.sqrt((y[0,0]-y[1,0])**2+(y[0,1]-y[1,1])**2+(y[0,2]-y[1,2])**2)
        minsep = min(minsep,sep); maxr = max(maxr,rr0,rr1)
        if not np.isfinite(rr0+rr1+sep) or maxr >= .8 or abs(y[0,2]) >= 1.2 or abs(y[1,2]) >= 1.2 or sep < .005:
            escaped = True
            break
        if (j+1)%stride == 0:
            samples[count] = y; times[count] = (j+1)*dt; count += 1
    return times[:count], samples[:count], minsep, maxr, escaped


def diagnostics(t, tr, periods, escaped):
    angle = np.unwrap(np.arctan2(tr[:,:,1],tr[:,:,0]),axis=0)
    windows = []
    for lo,hi in [(.4,.6),(.6,.8),(.8,1.)]:
        a=np.searchsorted(t,2*np.pi*periods*lo)
        b=np.searchsorted(t,2*np.pi*periods*hi)
        if b>=len(t): b=len(t)-1
        windows.append(((angle[b]-angle[a])/(2*np.pi)).tolist() if a<b else [0.,0.])
    windows=np.asarray(windows)
    radius=np.hypot(tr[:,:,0],tr[:,:,1])
    valid=not escaped and np.min(radius[int(len(radius)*.4):])>.015
    mixed=bool(valid and ((np.all(windows[:,0]>1) and np.all(windows[:,1]<-1)) or
                          (np.all(windows[:,1]>1) and np.all(windows[:,0]<-1))))
    return dict(windows_turns=windows.tolist(),total_turns=((angle[-1]-angle[0])/(2*np.pi)).tolist(),
                min_radius=float(radius.min()),mixed=mixed,escaped=bool(escaped))


def initial(r1,r2,dtheta,dz,spin1,spin2,vr1,vr2,vz1,vz2):
    y=np.zeros((2,6))
    y[0,:3]=[r1,0,-dz/2]
    y[1,:3]=[r2*np.cos(dtheta),r2*np.sin(dtheta),dz/2]
    for i,(spin,vr,vz) in enumerate(((spin1,vr1,vz1),(spin2,vr2,vz2))):
        u=y[i,:2]; r=np.linalg.norm(u)
        y[i,3:5]=vr*u/r+spin*np.array([-u[1],u[0]])
        y[i,5]=vz
    return y


def main():
    OUT.mkdir(exist_ok=True)
    cf=coefficients(); rng=np.random.default_rng(1701)
    records=[]; starts=[]
    # Deliberate coverage of angular/radial/axial offsets and initial spin.
    for j in range(64):
        r1=rng.uniform(.16,.42); r2=r1 if j%4==0 else rng.uniform(.16,.42)
        dz=0 if j%4==1 else rng.uniform(0,.75)
        angle=rng.uniform(.3,2.9)
        spin=rng.uniform(.015,.09)
        spins=(spin,-spin) if j%4 in (0,1) else ((0.,0.) if j%4==2 else (spin,spin))
        y=initial(r1,r2,angle,dz,*spins,*rng.uniform(-.01,.01,2),*rng.uniform(-.01,.01,2))
        starts.append(y)
        C=(2e-4 if j%2==0 else 2e-5); wz=(.045 if j%3 else .08)
        masses=np.array([1.,1. if j%5 else 1.5]); charges=np.array([1.,1.])
        gamma=np.array([.003,.003 if j%6 else .006])
        phase=rng.uniform(0,2*np.pi)
        t0=time.perf_counter()
        t,tr,sep,maxr,escaped=integrate(y,cf,C,wz,gamma,masses,charges,phase,400,48)
        diag=diagnostics(t,tr,400,escaped)
        row=dict(index=j,C=C,wz=wz,masses=masses.tolist(),charges=charges.tolist(),
                 gamma=gamma.tolist(),phase=phase,min_separation=sep,max_radius=maxr,
                 elapsed_s=time.perf_counter()-t0,**diag)
        records.append(row)
        if j%8==7:
            (OUT/'search_checkpoint.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            np.savez_compressed(OUT/'initial_states.npz',starts=np.asarray(starts))
            print(j+1,'mixed',sum(x['mixed'] for x in records),flush=True)
    best=sorted(records,key=lambda x: min(abs(np.array(x['windows_turns']).sum(axis=0))))[-1]
    (OUT/'summary.json').write_text(json.dumps(dict(records=records,best_screening_index=best['index'],
          coefficients=[dict(real=x.real,imag=x.imag) for x in cf]),indent=2),encoding='utf-8')
    print('done; mixed',sum(x['mixed'] for x in records),'best',best['index'])


if __name__=='__main__': main()
