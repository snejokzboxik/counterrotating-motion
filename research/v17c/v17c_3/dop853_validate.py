"""Independent SciPy DOP853 replay for the bounded G0 RF--DC orbit."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from rf_dc_core import (NORM, RESULTS, beta_equilibrium, load_geometry,
                        run_rk4, torque_omega_estimate)


def rhs_numpy(y, tau, modes, coeff, alpha, beta, lam, mass, charge, chi, phase0):
    count=len(mass); out=np.zeros_like(y)
    ar=np.zeros((count,3)); dc=np.zeros((count,3)); coul=np.zeros((count,3))
    phase=np.exp(-1j*(tau+phase0))
    for i in range(count):
        b=NORM*i; x,yc,z=y[b:b+3]
        u=complex(x,yc); fx=0j; fy=0j
        for k,c in zip(modes,coeff):
            n=abs(int(k)); base=u if k>0 else u.conjugate()
            f=base**(n-1)
            fx += -alpha[i]*n*c*f
            fy += (-1j*alpha[i]*n*c*f) if k>0 else (1j*alpha[i]*n*c*f)
        ar[i,0]=float(np.real(fx*phase)); ar[i,1]=float(np.real(fy*phase))
        dc[i]=beta[i]*np.array([x,yc,-2*z])
    for i in range(count):
        for j in range(i+1,count):
            bi,bj=NORM*i,NORM*j
            d=y[bi:bi+3]-y[bj:bj+3]
            rr=np.linalg.norm(d)
            f=chi*charge[i]*charge[j]*d/max(rr**3,1e-21)
            coul[i]+=f/mass[i];coul[j]-=f/mass[j]
    for i in range(count):
        b=NORM*i; pos=y[b:b+3]; vel=y[b+3:b+6]
        drag=-lam[i]*vel
        out[b:b+3]=vel; out[b+3:b+6]=ar[i]+dc[i]+coul[i]+drag
        out[b+6]=np.dot(vel,ar[i]);out[b+7]=np.dot(vel,dc[i])
        out[b+8]=np.dot(vel,coul[i]);out[b+9]=np.dot(vel,drag)
        out[b+10]=pos[0]*ar[i,1]-pos[1]*ar[i,0]
        out[b+11]=pos[0]*coul[i,1]-pos[1]*coul[i,0]
        out[b+12]=pos[0]*drag[1]-pos[1]*drag[0]
    return out


def main():
    meta,modes,coeff=load_geometry("G0_h50")
    geom={**meta,"modes":modes,"coeff":coeff}
    alpha0=.025; r0=.55; theta=.371; lam0=.0005; phase0=.23
    beta0=beta_equilibrium(modes,coeff,alpha0,r0)
    omega0=torque_omega_estimate(modes,coeff,alpha0,r0,lam0)
    rk=run_rk4(geom,[r0],[theta],[omega0],[alpha0],[beta0],[lam0],
               periods=2400,steps_per_period=64,save_per_period=1,
               phase0=.23)
    y0=rk["state"][0].copy(); period=2*np.pi
    t_eval=np.arange(2401,dtype=float)*period
    sol=solve_ivp(lambda t,y:rhs_numpy(y,t,modes,coeff,np.array([alpha0]),
        np.array([beta0]),np.array([lam0]),np.array([1.]),np.array([1.]),0.,phase0),
        (0,t_eval[-1]),y0,method="DOP853",rtol=2e-11,atol=2e-13,
        max_step=.12,t_eval=t_eval)
    if not sol.success: raise RuntimeError(sol.message)
    delta=np.abs(sol.y.T-rk["state"])
    s0=sol.y[:,-1]
    k0=.5*float(np.sum(y0[3:6]**2)); k1=.5*float(np.sum(s0[3:6]**2))
    work=s0[6:10]
    l0=y0[0]*y0[4]-y0[1]*y0[3]
    l1=s0[0]*s0[4]-s0[1]*s0[3]
    torque=s0[10:13]
    out={"geometry":"native P4 FEM h50 two-harmonic normalized field",
         "periods":2400,"steps_per_period_RK4":64,
         "DOP853":{"success":bool(sol.success),"nfev":int(sol.nfev),
                    "rtol":2e-11,"atol":2e-13,"max_step":.12},
         "max_state_abs_difference":float(np.max(delta[:,:6])),
         "max_position_abs_difference":float(np.max(delta[:,:3])),
         "turns_RK4":float(np.unwrap(np.arctan2(rk["state"][:,1],rk["state"][:,0]))[-1]/(2*np.pi)
                             -np.unwrap(np.arctan2(rk["state"][:,1],rk["state"][:,0]))[0]/(2*np.pi)),
         "turns_DOP853":float(np.unwrap(np.arctan2(sol.y[1],sol.y[0]))[-1]/(2*np.pi)
                             -np.unwrap(np.arctan2(sol.y[1],sol.y[0]))[0]/(2*np.pi)),
         "kinetic_work_residual_DOP853":float(k1-k0-np.sum(work)),
         "rf_plus_drag_work_DOP853":float(work[0]+work[3]),
         "angular_momentum_balance_residual_DOP853":float(l1-l0-np.sum(torque)),
         "radius_range_DOP853":[float(np.min(np.hypot(sol.y[0],sol.y[1])),),
                                float(np.max(np.hypot(sol.y[0],sol.y[1])))],
         "parameters":{"r0":r0,"theta0":theta,"omega0":omega0,
                       "alpha":alpha0,"beta":beta0,"lambda":lam0}}
    (RESULTS/"dop853_G0_validation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    np.savez_compressed(RESULTS/"dop853_G0_2400RF.npz",t=sol.t,state=sol.y.T,
                        initial=y0,params=np.array([alpha0,beta0,lam0,phase0]))
    print(json.dumps(out,indent=2),flush=True)

if __name__=="__main__":main()
