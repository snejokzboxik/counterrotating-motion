"""Independent high-accuracy DOP853 checks for P1 candidate trajectories."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy
from numba import njit
from scipy.integrate import ode

HERE=Path(__file__).resolve().parent; BASE=HERE.parent
CFG=json.loads((BASE.parent/"v17c_5"/"input"/"config.json").read_text(encoding="utf-8-sig"))
K=np.asarray(CFG["k"],np.int64); FAMILY=np.asarray(CFG["family"],np.int64)
CO=np.asarray(CFG["coeff"],float); CO=CO[:,0]+1j*CO[:,1]
T=2*np.pi; F1=float(CFG["f1_Hz"])

@njit(cache=True)
def derivative(t,y,p,qratio,phase):
    A,B,nu,beta,s,C,lam1,lam2=p
    delta=y[:3]-y[6:9]; d2=delta[0]**2+delta[1]**2+delta[2]**2
    pair=C*delta/(d2*np.sqrt(d2))
    out=np.zeros(24)
    for i,base in enumerate((0,6)):
        x,yy,z=y[base:base+3]; v=y[base+3:base+6]
        r=np.hypot(x,yy)
        theta=np.arctan2(yy,x)
        radial=np.zeros(2); azimuthal=np.zeros(2)
        for h,mode in enumerate(K):
            fam=FAMILY[h]; omega=1. if fam==0 else nu
            ph=mode*theta-omega*t-phase[fam]
            re=CO[h].real; im=CO[h].imag
            c=np.cos(ph); sn=np.sin(ph)
            strength=(A if fam==0 else B)*r**(abs(mode)-1)
            radial[fam]-=strength*abs(mode)*(re*c-im*sn)
            azimuthal[fam]+=strength*mode*(re*sn+im*c)
        ct=np.cos(theta); st=np.sin(theta)
        rf=np.zeros((2,3))
        for f in range(2):
            rf[f,0]=radial[f]*ct-azimuthal[f]*st
            rf[f,1]=radial[f]*st+azimuthal[f]*ct
        dc=beta*np.array([x,yy,-2*z])
        invmass=1. if i==0 else s
        qr=1. if i==0 else qratio
        lam=lam1 if i==0 else lam2
        cf=pair if i==0 else -pair
        drag=-lam*v/invmass
        out[base:base+3]=v
        out[base+3:base+6]=invmass*(qr*(rf[0]+rf[1]+dc)+cf+drag)
        w=12+5*i
        out[w]=qr*np.dot(rf[0],v)
        out[w+1]=qr*np.dot(rf[1],v)
        out[w+2]=qr*np.dot(dc,v)
        out[w+3]=np.dot(cf,v)
        out[w+4]=np.dot(drag,v)
        if r>1e-14: out[22+i]=(x*v[1]-yy*v[0])/(r*r)
    return out

def evaluate(times,state,p,qratio,skip):
    pos=state[:,[0,1,2,6,7,8]].reshape(-1,2,3)
    vel=state[:,[3,4,5,9,10,11]].reshape(-1,2,3)
    radius=np.linalg.norm(pos[:,:,:2],axis=2)
    theta=np.unwrap(np.arctan2(pos[:,:,1],pos[:,:,0]),axis=0)
    turns=(theta[-1]-theta[skip])/T
    integ=(state[-1,22:24]-state[skip,22:24])/T
    cuts=np.linspace(skip,len(times)-1,4,dtype=int)
    windows=np.diff(theta[cuts],axis=0)/T
    freq=turns/(len(times)-1-skip)*F1
    s=p[4]; q=np.array([1.,qratio]); mass=np.array([1.,1/s])
    ke=.5*np.sum(mass[None,:,None]*vel**2,axis=(1,2))
    dc=p[3]*np.sum(q[None,:]*(pos[:,:,2]**2-.5*radius**2),axis=1)
    uc=p[5]/np.linalg.norm(pos[:,0]-pos[:,1],axis=1)
    work=state[:,12:22].reshape(-1,2,5)
    ext=work[:,:,:2].sum(axis=(1,2))+work[:,:,4].sum(axis=1)
    res=(ke+dc+uc)-(ke[0]+dc[0]+uc[0])-ext
    return {"turns_after_transient":turns.tolist(),"turns_by_omega_integral":integ.tolist(),
       "max_angle_integral_difference_turns":float(np.max(np.abs((theta-theta[0])/T-state[:,22:24]/T))),
       "window_turns":windows.tolist(),"frequency_Hz":freq.tolist(),
       "mean_radius_stroboscopic":radius[skip:].mean(axis=0).tolist(),
       "radius_min_stroboscopic":radius.min(axis=0).tolist(),"radius_max_stroboscopic":radius.max(axis=0).tolist(),
       "max_abs_z":float(np.abs(pos[:,:,2]).max()),
       "minimum_separation_stroboscopic":float(np.linalg.norm(pos[:,0]-pos[:,1],axis=1).min()),
       "late_work_rf1_rf2_dc_coulomb_drag":(work[-1]-work[skip]).tolist(),
       "energy_residual_max":float(np.max(np.abs(res))),
       "energy_residual_over_RF_work":float(np.max(np.abs(res))/abs(work[-1,:,:2].sum()))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config-id",type=int,required=True)
    ap.add_argument("--keys",nargs="+",required=True,help="holdout indices, e.g. 3 55 101")
    ap.add_argument("--periods",type=int,default=15000); a=ap.parse_args()
    npz=np.load(HERE/"stage1"/"configurations.npz")
    p=np.asarray(npz["parameters"][a.config_id],float); qr=float(npz["charge_ratio"][a.config_id])
    source=HERE/"stage2"/"runs"/f"cfg{a.config_id}_rest_holdout_128_{a.periods}p_64s.jsonl"
    data={int(r["index"]):r for r in (json.loads(x) for x in source.read_text(encoding="utf-8").splitlines() if x.strip())}
    outdir=HERE/"stage2"/"runs"; results=[]
    for token in a.keys:
        ix=int(token); r=data[ix]; tic=time.perf_counter(); times=np.arange(a.periods+1,dtype=float)*T
        initial=np.asarray(r["initial"],float); phase=np.asarray(r["phase"],float)
        y0=np.r_[initial,np.zeros(12)]
        driver=ode(derivative).set_integrator("dop853",rtol=2e-10,atol=2e-12,max_step=T/24,nsteps=512)
        driver.set_f_params(p,qr,phase); driver.set_initial_value(y0,0.)
        state=np.empty((a.periods+1,24)); state[0]=y0
        for i,t in enumerate(times[1:],1):
            state[i]=driver.integrate(t)
            if not driver.successful() or not np.isfinite(state[i]).all(): raise RuntimeError(f"DOP853 failed at cycle {i}")
        diag=evaluate(times,state,p,qr,200)
        diag.update(key=f"{a.config_id}:rest:{ix}",config_id=a.config_id,screen_category=r["category"],
          solver="SciPy DOP853 (Fortran driver), independent polar-gradient RHS",rtol=2e-10,atol=2e-12,
          max_step_tau=T/24,scipy=scipy.__version__,wall_s=time.perf_counter()-tic,
          charge_ratio=qr,parameters=p.tolist(),initial=initial.tolist(),rf_phase=phase.tolist())
        np.savez_compressed(outdir/f"dop_cfg{a.config_id}_rest_{ix}.npz",tau=times,state=state[:,:12],
          work=state[:,12:22].reshape(-1,2,5),angles_integral=state[:,22:24])
        results.append(diag); print(diag["key"],diag["frequency_Hz"],diag["wall_s"],flush=True)
    dst=outdir/f"dop_cfg{a.config_id}_results.json"
    old=json.loads(dst.read_text()) if dst.exists() else []
    merged={x["key"]:x for x in old}
    merged.update({x["key"]:x for x in results})
    dst.write_text(json.dumps(list(merged.values()),indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
