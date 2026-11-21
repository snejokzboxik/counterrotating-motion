"""Vectorized fixed-step exact-RF Local15 validation (no Coulomb, full ideal modes).

This protocol integrates a frozen paired panel for both Local15 species. The
carrier phases are independent per start; the secondary frequency is not
rounded to a rational ratio. Escapes at rho>=1 are retained as losses.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/"results/frozen_local15_single_particle_panel.csv"
OUT=ROOT/"results/local15_exact_rf"
PARAM={
 "A":0.03716194907442204,
 "B":0.1626938199343468,
 "nu":3.1118029452860356,
 "beta":0.017273303124515367,
 "species":{
   "species1":{"alpha":1.0,"lambda":0.20975541409005094},
   "species2":{"alpha":0.34126014359705537,"lambda":0.12617311138613094}
 }
}

def load_panel():
    with PANEL.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    keys=("rho0","theta0","x0","y0","vx0","vy0","phase1","phase2")
    return rows,{k:np.array([float(r[k]) for r in rows]) for k in keys}

def acceleration(t,y,phase1,phase2,alpha,lam):
    x,yc,vx,vy=y.T
    z=x+1j*yc
    g0=2.0*PARAM["A"]*np.exp(1j*phase1)*z*np.exp(-1j*t)
    g1=4.0*PARAM["B"]*np.exp(1j*phase2)*np.conjugate(z)**3*np.exp(-1j*PARAM["nu"]*t)
    gx=np.real(g0+g1)
    gy=np.real(1j*g0-1j*g1)
    ax=-alpha*gx+alpha*PARAM["beta"]*x-lam*vx
    ay=-alpha*gy+alpha*PARAM["beta"]*yc-lam*vy
    return np.column_stack((vx,vy,ax,ay))

def rk4_step(t,y,h,phase1,phase2,alpha,lam,active):
    def f(tt,yy):
        a=acceleration(tt,yy,phase1,phase2,alpha,lam)
        a[~active]=0.0
        return a
    k1=f(t,y)
    k2=f(t+0.5*h,y+0.5*h*k1)
    k3=f(t+0.5*h,y+0.5*h*k2)
    k4=f(t+h,y+h*k3)
    return y+(h/6.0)*(k1+2*k2+2*k3+k4)

def run_species(name, initials, periods, steps_per_period, checkpoint_periods=500):
    meta,arr=load_panel()
    m=len(meta)
    phases1,phases2=arr["phase1"],arr["phase2"]
    alpha=float(PARAM["species"][name]["alpha"])
    lam=float(PARAM["species"][name]["lambda"])
    y=np.column_stack((arr["x0"],arr["y0"],arr["vx0"],arr["vy0"]))
    active=np.ones(m,dtype=bool)
    rho=np.hypot(y[:,0],y[:,1])
    escaped=rho>=1.0
    active &= ~escaped
    escape_period=np.where(escaped,0.0,np.nan)
    maxrho=rho.copy()
    ang=np.arctan2(y[:,1],y[:,0])
    angle_unwrapped=ang.copy()
    sample_rho=[rho.copy()]
    sample_maxrho=[maxrho.copy()]
    sample_angle=[angle_unwrapped.copy()]
    sample_time=[0]
    dt=2*np.pi/steps_per_period
    t0=time.perf_counter()
    for step in range(1,periods*steps_per_period+1):
        tau=(step-1)*dt
        y=rk4_step(tau,y,dt,phases1,phases2,alpha,lam,active)
        current_rho=np.hypot(y[:,0],y[:,1])
        maxrho=np.maximum(maxrho,current_rho)
        newly=active & (current_rho>=1.0)
        if np.any(newly):
            escaped[newly]=True
            escape_period[newly]=step/steps_per_period
            active[newly]=False
        if step%steps_per_period==0:
            raw=np.arctan2(y[:,1],y[:,0])
            delta=np.arctan2(np.sin(raw-ang),np.cos(raw-ang))
            angle_unwrapped += delta
            ang=raw
            sample_rho.append(current_rho.copy())
            sample_maxrho.append(maxrho.copy())
            sample_angle.append(angle_unwrapped.copy())
            sample_time.append(step//steps_per_period)
        if step%(checkpoint_periods*steps_per_period)==0:
            np.savez_compressed(OUT/f"{name}_checkpoint.npz",step=step,state=y,
                active=active,escaped=escaped,escape_period=escape_period,
                maxrho=maxrho,angle_unwrapped=angle_unwrapped,angle_last=ang,
                sample_rho=np.asarray(sample_rho),sample_maxrho=np.asarray(sample_maxrho),
                sample_angle=np.asarray(sample_angle),
                sample_time=np.asarray(sample_time))
    elapsed=time.perf_counter()-t0
    rho_hist=np.asarray(sample_rho)
    maxrho_hist=np.asarray(sample_maxrho)
    angle_hist=np.asarray(sample_angle)
    times=np.asarray(sample_time,dtype=float)
    windows=[]
    for horizon in (5000,10000,periods):
        if horizon>periods: continue
        width=min(500,horizon)
        lo=horizon-width
        i0=int(np.searchsorted(times,lo))
        i1=int(np.searchsorted(times,horizon))
        if i1<=i0: continue
        for j,row in enumerate(meta):
            dtheta=angle_hist[i1,j]-angle_hist[i0,j]
            omega=dtheta/(2*np.pi*(times[i1]-times[i0]))
            rs=rho_hist[i0:i1+1,j]
            escaped_by_horizon=bool(np.isfinite(escape_period[j]) and escape_period[j]<=horizon)
            windows.append({"species":name,"start_id":row["start_id"],
                "horizon_periods":horizon,"window_periods":float(times[i1]-times[i0]),
                "rho_mean":float(np.mean(rs)),"rho_std":float(np.std(rs)),
                "rho_min":float(np.min(rs)),"rho_max":float(np.max(rs)),
                "omega_mean":float(omega),"escaped_by_horizon":escaped_by_horizon,
                "escape_period":None if not np.isfinite(escape_period[j]) else float(escape_period[j]),
                "max_rho":float(maxrho_hist[i1,j]),"active_at_horizon":not escaped_by_horizon})
    return {"species":name,"periods":periods,"steps_per_period":steps_per_period,
            "elapsed_seconds":elapsed,"steps":periods*steps_per_period,
            "windows":windows,"maxrho":maxrho.tolist(),"escaped":escaped.tolist()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--periods",type=int,default=200)
    ap.add_argument("--steps-per-period",type=int,default=96)
    ap.add_argument("--checkpoint-periods",type=int,default=500)
    ap.add_argument("--benchmark-only",action="store_true")
    a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    panel_hash=hashlib.sha256(PANEL.read_bytes()).hexdigest()
    results=[]
    for name in ("species1","species2"):
        results.append(run_species(name,None,a.periods,a.steps_per_period,a.checkpoint_periods))
    doc={"parameters":PARAM,"panel_sha256":panel_hash,"protocol":{
       "periods":a.periods,"steps_per_primary_rf_period":a.steps_per_period,
       "secondary_frequency_ratio":PARAM["nu"],"escape_radius_over_R":1.0,
       "classification_windows":"last 500 primary periods at 5k, 10k, and final requested horizon",
       "initial_panel":"24 frozen starts shared between species; no postselection"},
       "results":results}
    out=OUT/("benchmark.json" if a.benchmark_only else f"rk4_{a.steps_per_period}_{a.periods}.json")
    out.write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"out":str(out),"panel_sha256":panel_hash,
        "elapsed_seconds":[x["elapsed_seconds"] for x in results],
        "total_steps":[x["steps"] for x in results]},indent=2))
if __name__=="__main__": main()
