"""Reproduce the archive's full-field, 3-D classical RF–DC pair with fixed-step RK4."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PERIODS = 15000
SKIP = 2000
TWOPI = 2 * np.pi

@njit
def derivative(tau, y, p, speed_coef, family, k, coeff, dy):
    """12 Cartesian derivatives followed by 4 mechanical powers per particle."""
    a, b, nu, beta, s, coupling, lam1, lam2 = p
    ph1 = a * complex(np.cos(tau), -np.sin(tau))
    ph2 = b * complex(np.cos(nu*tau), -np.sin(nu*tau))
    delta0, delta1, delta2 = y[0]-y[6], y[1]-y[7], y[2]-y[8]
    distance2 = delta0**2 + delta1**2 + delta2**2
    cscale = coupling / distance2**1.5
    for i in range(2):
        n = 6*i
        x,yc,z,vx,vy,vz = y[n:n+6]
        u = complex(x,yc)
        ex,ey = 0.,0.
        for j in range(k.size):
            harmonic = k[j]
            base = u if harmonic > 0 else u.conjugate()
            degree = abs(harmonic)
            ph = ph1 if family[j]==0 else ph2
            gradient = ph * coeff[j] * degree * base**(degree-1)
            ex -= gradient.real
            ey += gradient.imag if harmonic>0 else -gradient.imag
        inv_mass = 1. if i==0 else s
        drag = (lam1 if i==0 else lam2)*(1+speed_coef*np.sqrt(vx*vx+vy*vy+vz*vz))
        cs = cscale if i==0 else -cscale
        dy[n],dy[n+1],dy[n+2] = vx,vy,vz
        dy[n+3] = inv_mass*(ex+beta*x+cs*delta0)-drag*vx
        dy[n+4] = inv_mass*(ey+beta*yc+cs*delta1)-drag*vy
        dy[n+5] = inv_mass*(-2*beta*z+cs*delta2)-drag*vz
        w = 12+4*i
        dy[w] = vx*ex+vy*ey
        dy[w+1] = beta*(vx*x+vy*yc-2*vz*z)
        dy[w+2] = cs*(vx*delta0+vy*delta1+vz*delta2)
        dy[w+3] = -drag*(vx*vx+vy*vy+vz*vz)/inv_mass

@njit(nogil=True)
def integrate(initial, p, speed_coef, family, k, coeff, periods, steps, skip):
    dt = TWOPI/steps
    y = np.zeros(20)
    y[:12] = initial
    traces = np.empty((periods+1,20))
    traces[0] = y
    angles = np.zeros((periods+1,2))
    angle = np.zeros(2)
    max_dtheta = np.zeros(2)
    rsum = np.zeros(2)
    rmin = np.full(2,np.inf)
    rmax = np.zeros(2)
    zmax = np.array([abs(initial[2]),abs(initial[8])])
    rmaxall = np.array([np.hypot(y[0],y[1]),np.hypot(y[6],y[7])])
    dmin = np.inf
    k1,k2,k3,k4 = np.empty(20),np.empty(20),np.empty(20),np.empty(20)
    aux,yn = np.empty(20),np.empty(20)
    completed = 0
    for it in range(periods*steps):
        t = it*dt
        derivative(t,y,p,speed_coef,family,k,coeff,k1)
        for j in range(12): aux[j] = y[j]+dt*k1[j]*0.5
        derivative(t+dt/2,aux,p,speed_coef,family,k,coeff,k2)
        for j in range(12): aux[j] = y[j]+dt*k2[j]*0.5
        derivative(t+dt/2,aux,p,speed_coef,family,k,coeff,k3)
        for j in range(12): aux[j] = y[j]+dt*k3[j]
        derivative(t+dt,aux,p,speed_coef,family,k,coeff,k4)
        for j in range(20): yn[j] = y[j]+dt/6*(k1[j]+2*k2[j]+2*k3[j]+k4[j])
        dist = np.sqrt((yn[0]-yn[6])**2+(yn[1]-yn[7])**2+(yn[2]-yn[8])**2)
        dmin = min(dmin,dist)
        for i in range(2):
            q=6*i
            r = np.hypot(yn[q],yn[q+1])
            rmaxall[i] = max(r,rmaxall[i])
            zmax[i] = max(abs(yn[q+2]),zmax[i])
            dtheta = np.arctan2(y[q]*yn[q+1]-y[q+1]*yn[q],y[q]*yn[q]+y[q+1]*yn[q+1])
            angle[i] += dtheta
            max_dtheta[i] = max(max_dtheta[i],abs(dtheta))
            if it >= skip*steps:
                rsum[i] += r
                rmin[i] = min(rmin[i],r)
                rmax[i] = max(rmax[i],r)
        if max(rmaxall)>1 or max(zmax)>2 or not np.all(np.isfinite(yn)):
            break
        y,yn = yn,y
        if (it+1)%steps==0:
            completed += 1
            traces[completed] = y
            angles[completed] = angle/TWOPI
    return (traces[:completed+1], angles[:completed+1], max_dtheta, rsum, rmin,
            rmax, rmaxall, zmax, dmin)

def run(case, cfg):
    name, steps, coupling, rscale, speed_coef = case
    p = np.array(cfg["parameters"])
    p[5] = coupling
    initial = np.asarray(cfg["initial"]).copy()
    initial[[0,1,6,7]] *= rscale
    coeff = np.asarray(cfg["coeff"])
    coeff = coeff[:,0]+1j*coeff[:,1]
    data = integrate(initial,p,speed_coef,np.array(cfg["family"]),np.array(cfg["k"]),coeff,
                     PERIODS,steps,SKIP)
    trace, turns, max_increment, rsum, rmin, rmax, rmaxall, zmax, dmin = data
    state = trace[:,:12]
    work = trace[:,12:].reshape(-1,2,4)
    tau = np.arange(trace.shape[0])*TWOPI
    np.savez_compressed(OUT/(name+".npz"),tau=tau,state=state,work=work,turns=turns,
                        initial=initial,parameters=p,oseen_speed_coefficient=speed_coef)
    n = len(tau)-1
    diag = dict(retained=bool(n==PERIODS),completed=n,steps=steps,coupling=coupling,
                rscale=rscale,speed_coefficient=speed_coef,max_axial_abs=zmax.tolist(),
                max_radius_all=rmaxall.tolist(),min_distance_all=dmin,
                max_abs_inner_step_delta_angle_rad=max_increment.tolist())
    if n>SKIP:
        mass = np.array([1.,1/p[4]])
        positions = state.reshape(-1,2,6)[:,:,:3]
        vel = state.reshape(-1,2,6)[:,:,3:]
        r = np.linalg.norm(positions[:,:,:2],axis=2)
        kinetic = .5*mass*np.sum(vel**2,axis=2)
        dc = p[3]*(positions[:,:,2]**2-.5*r**2).sum(axis=1)
        uc = p[5]/np.linalg.norm(positions[:,0,:]-positions[:,1,:],axis=1)
        total = kinetic.sum(axis=1)+dc+uc
        external = work[:,:,[0,3]].sum(axis=(1,2))
        residual = (total-total[0])-external
        interval = [SKIP,6333,10666,n]
        tlate = turns[-1]-turns[SKIP]
        rf = cfg["f1_Hz"]
        strob = np.unwrap(np.arctan2(positions[:,:,1],positions[:,:,0]),axis=0)/TWOPI
        late_work = work[-1]-work[SKIP]
        denergy = total[-1]-total[SKIP]
        diag.update(late_interval=[SKIP,n],turns=tlate.tolist(),hz=(tlate*rf/(n-SKIP)).tolist(),
                    window_bounds=interval,window_turns=np.diff(turns[interval],axis=0).tolist(),
                    window_hz=(rf*np.diff(turns[interval],axis=0)/np.diff(interval)[:,None]).tolist(),
                    mean_r_strobe=r[SKIP:].mean(axis=0).tolist(),
                    mean_r_inner=(rsum/((n-SKIP)*steps)).tolist(),min_r_inner=rmin.tolist(),
                    max_r_inner=rmax.tolist(),max_strobe_vs_inner_turns=np.max(
                        abs(strob-strob[0]-turns),axis=0).tolist(),
                    late_power_work_rf_dc_coul_drag=late_work.tolist(),
                    kinetic_change_late=(kinetic[-1]-kinetic[SKIP]).tolist(),
                    particle_balance_abs_max=np.max(abs(kinetic-kinetic[0]-work.sum(axis=2)),axis=0).tolist(),
                    system_energy_change_late=float(denergy),late_balance=float(denergy-late_work[:,[0,3]].sum()),
                    max_energy_residual=float(np.max(abs(residual))),
                    max_energy_residual_over_rf_work=float(np.max(abs(residual))/abs(work[-1,:,0].sum())),
                    late_coulomb_energy_mean=float(uc[SKIP:].mean()))
    (OUT/(name+".json")).write_text(json.dumps(diag,indent=2),encoding="utf-8")
    print(name,diag.get("hz"),diag.get("mean_r_inner"),flush=True)
    return name,diag

def main():
    cfg=json.loads((HERE.parent/"input"/"config.json").read_text(encoding="utf-8-sig"))
    OUT.mkdir(exist_ok=True)
    cases=[("nom128",128,1e-5,1.,0.),("nom64",64,1e-5,1.,0.),
           ("c0",64,0.,1.,0.),("c5e6",64,5e-6,1.,0.),("c2e5",64,2e-5,1.,0.),
           ("oseen",64,1e-5,1.,.5),("rplus1",64,1e-5,1.01,0.),
           ("combined",64,1e-5,1.01,.5)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = dict(f.result() for f in as_completed([ex.submit(run,case,cfg) for case in cases]))
    (OUT/"diagnostics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()

