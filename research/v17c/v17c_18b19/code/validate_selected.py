"""Independent RK4 refinement and DOP853 check for a frozen PS benchmark.

The case is selected before inspecting its convergence output: the V17C.16
PS pair (alpha1=1, alpha2=0.3412603), with the pair-map zero-velocity start.
It is a numerical-method check of the planar leading-harmonic model, not a
3-D continuous-rod/endcap-trap validation.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from model import Params, pair_rhs, rk4_pair_fast, rk4_single_fast

ROOT=Path(__file__).resolve().parents[1]
P=Params()
PERIODS=1000
TAU=2*math.pi*PERIODS
PAIR0=(0.55,0.0,0.0,0.0, 0.766,math.pi/2,0.0,0.0)


def integrate_pair_rk4(nper: int):
    dt=2*math.pi/nper
    s=np.array([PAIR0[0],PAIR0[1],PAIR0[2],PAIR0[3],
                PAIR0[4]*math.cos(PAIR0[5]),PAIR0[4]*math.sin(PAIR0[5]),
                PAIR0[6],PAIR0[7]],float)
    phase=[0.0,0.0]; radii=[[],[]]; turn_series=[[0.0],[0.0]]
    for k in range(PERIODS*nper):
        sn=np.asarray(rk4_pair_fast(tuple(s),k*dt,dt,(1.0,0.3412603,P,0.0,0.0)))
        for i,off in enumerate((0,4)):
            x,y=s[off:off+2]; xn,yn=sn[off:off+2]
            phase[i]+=math.atan2(x*yn-y*xn,x*xn+y*yn)
        s=sn
        if (k+1)%nper==0:
            for i,off in enumerate((0,4)):
                radii[i].append(math.hypot(s[off],s[off+1]))
                turn_series[i].append(phase[i]/(2*math.pi))
    late=np.asarray(radii)[:,int(.8*PERIODS):]
    return {"mean_radius_R_last_200_periods":np.asarray(late).mean(axis=1).tolist(),
            "std_radius_R_last_200_periods":np.asarray(late).std(axis=1).tolist(),
            "turns_per_RF1_period_last_200":[(u[-1]-u[-201])/200 for u in turn_series]}


def main():
    # Single particle, Stokes-only, P0.
    ss=[]
    for nper in (64,128,256):
        dt=2*math.pi/nper
        s=(.55*math.cos(.71),.55*math.sin(.71),0.0,0.0)
        angle=0.0; r=[]; turns_by_period=[0.0]
        for k in range(PERIODS*nper):
            sn=rk4_single_fast(s,k*dt,dt,(1.0,P.lam1,P.A,P.B,P.nu,P.beta,0.0,0.0,0.0))
            angle+=math.atan2(s[0]*sn[1]-s[1]*sn[0],s[0]*sn[0]+s[1]*sn[1])
            s=sn
            if (k+1)%nper==0:
                r.append(math.hypot(s[0],s[1])); turns_by_period.append(angle/(2*math.pi))
        lo=int(.8*PERIODS)
        ss.append({"steps_per_RF1_period":nper,"mean_radius_R_last_200":float(np.mean(r[lo:])),
                   "std_radius_R_last_200":float(np.std(r[lo:])),
                   "turns_per_period_last_200":float((turns_by_period[-1]-turns_by_period[lo])/200)})

    # Pair RK4 refinement, using the map's exact planar optimized implementation.
    pair=[]
    for nper in (64,128,256):
        pair.append({"steps_per_RF1_period":nper,**integrate_pair_rk4(nper)})

    # Independent 3-D vector RHS and DOP853 (z remains identically zero).
    y0=np.array([PAIR0[0],PAIR0[1],0.,PAIR0[2],PAIR0[3],0.,
                 PAIR0[4]*math.cos(PAIR0[5]),PAIR0[4]*math.sin(PAIR0[5]),0.,
                 PAIR0[6],PAIR0[7],0.])
    eval_t=np.arange(0,TAU+1e-8,2*math.pi)
    sol=solve_ivp(lambda t,y:pair_rhs(y,t,1.0,0.3412603,P,coulomb=True,
                                      gravity=False,pol_scale=0.0),
                  (0.,TAU),y0,method="DOP853",rtol=2e-10,atol=2e-12,
                  max_step=2*math.pi/128,t_eval=eval_t)
    if not sol.success:
        raise RuntimeError(sol.message)
    traj=sol.y.T.reshape(-1,2,6)
    phase=np.unwrap(np.arctan2(traj[:,:,1],traj[:,:,0]),axis=0)
    radii=np.hypot(traj[:,:,0],traj[:,:,1]).T
    lo=int(.8*PERIODS)
    dop={"method":"DOP853; 3-D pair_rhs with z=0 invariant; max_step=2pi/128; rtol=2e-10; atol=2e-12",
         "n_output_periods":len(eval_t)-1,
         "mean_radius_R_last_200_periods":radii[:,lo:].mean(axis=1).tolist(),
         "std_radius_R_last_200_periods":radii[:,lo:].std(axis=1).tolist(),
         "turns_per_RF1_period_last_200":((phase[-1]-phase[lo])/(2*math.pi*200)).tolist()}
    output={"case":"V17C.16 PS benchmark; alpha=(1,0.3412603); Coulomb on; P0; common leading-harmonic field",
            "horizon_RF1_periods":PERIODS,"pair_initial_state":"r1/R=.55 theta1=0; r2/R=.766 theta2=pi/2; zero velocities",
            "single_particle_refinement":ss,"pair_RK4_refinement":pair,"pair_DOP853":dop,
            "warning":"method convergence applies only to the planar leading-harmonic model; not finite-length FEM or physical validation"}
    (ROOT/"results"/"selected_numerical_validation.json").write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
