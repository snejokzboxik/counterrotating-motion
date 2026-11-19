"""Independent polar-coordinate SciPy DOP853 verification for selected P0 starts."""
import argparse,json,time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE.parent.parent/"v17c_5"/"input"/"config.json").read_text(encoding="utf-8-sig"))
P=np.asarray(CFG["parameters"],float); K=np.asarray(CFG["k"],int); F=np.asarray(CFG["family"],int)
C=np.asarray(CFG["coeff"],float); COEF=C[:,0]+1j*C[:,1]
TWO_PI=2*np.pi; F1=float(CFG["f1_Hz"]); SKIP=200

@njit(cache=True)
def rhs(t,y,phase):
    d=y[:3]-y[6:9]; d2=np.dot(d,d); pair=P[5]*d/(d2*np.sqrt(d2))
    out=np.zeros(24)
    for i,base in enumerate((0,6)):
        x,yy,z=y[base:base+3]; v=y[base+3:base+6]
        r=np.hypot(x,yy)
        if r<1e-12: raise FloatingPointError("DOP853 state reached cylindrical axis")
        theta=np.arctan2(yy,x); radial=np.zeros(2); azim=np.zeros(2)
        for h,mode in enumerate(K):
            fam=F[h]; omega=1. if fam==0 else P[2]
            ph=mode*theta-omega*t-phase[fam]
            re,im=COEF[h].real,COEF[h].imag
            c,sn=np.cos(ph),np.sin(ph)
            strength=P[fam]*r**(abs(mode)-1)
            radial[fam]-=strength*abs(mode)*(re*c-im*sn)
            azim[fam]+=strength*mode*(re*sn+im*c)
        ct,st=np.cos(theta),np.sin(theta)
        rf=np.zeros((2,3))
        for f in range(2):
            rf[f]=[radial[f]*ct-azim[f]*st,radial[f]*st+azim[f]*ct,0.]
        dc=P[3]*np.array([x,yy,-2*z]); cf=pair if i==0 else -pair
        inv=1. if i==0 else P[4]; lam=P[6+i]
        drag=-lam*v/inv
        out[base:base+3]=v
        out[base+3:base+6]=inv*(rf.sum(axis=0)+dc+cf+drag)
        w=12+5*i
        out[w:w+5]=[np.dot(rf[0],v),np.dot(rf[1],v),np.dot(dc,v),np.dot(cf,v),np.dot(drag,v)]
        out[22+i]=(x*v[1]-yy*v[0])/(r*r)
    return out

def evaluate(sol,discard):
    t=sol.t; y=sol.y.T; pos=y[:,[0,1,2,6,7,8]].reshape(-1,2,3)
    vel=y[:,[3,4,5,9,10,11]].reshape(-1,2,3)
    radius=np.linalg.norm(pos[:,:,:2],axis=2)
    theta=np.unwrap(np.arctan2(pos[:,:,1],pos[:,:,0]),axis=0)
    turns=(theta[-1]-theta[discard])/TWO_PI
    turns_integral=(y[-1,22:24]-y[discard,22:24])/TWO_PI
    cuts=np.linspace(discard,len(t)-1,4,dtype=int)
    windows=np.diff(theta[cuts],axis=0)/TWO_PI
    freq=turns/(len(t)-1-discard)*F1
    mass=np.array([1.,1/P[4]])
    ke=.5*np.sum(mass[None,:,None]*vel**2,axis=(1,2))
    dc=P[3]*(pos[:,:,2]**2-.5*radius**2).sum(axis=1)
    uc=P[5]/np.linalg.norm(pos[:,0,:]-pos[:,1,:],axis=1)
    work=y[:,12:22].reshape(-1,2,5)
    res=ke+dc+uc-(ke[0]+dc[0]+uc[0])-work[:,:,:2].sum(axis=(1,2))-work[:,:,4].sum(axis=1)
    return {"turns_after_transient":turns.tolist(),"turns_integral":turns_integral.tolist(),
       "max_angle_integral_difference_turns":float(np.max(np.abs((theta-theta[0])/TWO_PI-y[:,22:24]/TWO_PI))),
       "frequency_Hz":freq.tolist(),"window_turns":windows.tolist(),
       "mean_radius_stroboscopic":radius[discard:].mean(axis=0).tolist(),
       "max_radius_stroboscopic":radius.max(axis=0).tolist(),"max_abs_z":float(np.abs(pos[:,:,2]).max()),
       "min_separation_stroboscopic":float(np.linalg.norm(pos[:,0]-pos[:,1],axis=1).min()),
       "work_after_transient_rf1_rf2_dc_coulomb_drag":(work[-1]-work[discard]).tolist(),
       "energy_residual_max":float(np.max(np.abs(res))),
       "energy_residual_over_rf_work":float(np.max(np.abs(res))/abs(work[-1,:,:2].sum()))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--keys",nargs="+",required=True); ap.add_argument("--periods",type=int,default=5000); a=ap.parse_args()
    sources=[]
    for group in ("rest","weak","wide"):
        path=HERE/"runs"/f"{group}_{a.periods}p_64s.jsonl"
        if not path.exists(): continue
        sources += [(group,json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    wanted={k for k in a.keys}; selected=[(g,r) for g,r in sources if f"{g}:{r['index']}" in wanted]
    found={str(g)+":"+str(r["index"]) for g,r in selected}
    if len(selected)!=len(wanted): raise ValueError(f"keys not found: {wanted-found}")
    outdir=HERE/"runs"; results=[]
    for group,r in selected:
        init=np.asarray(r["initial"]); phase=np.asarray(r["phase"]); t0=time.perf_counter()
        sol=solve_ivp(lambda t,y:rhs(t,y,phase),(0,a.periods*TWO_PI),np.r_[init,np.zeros(12)],
          method="DOP853",rtol=2e-10,atol=2e-12,max_step=TWO_PI/24,
          t_eval=np.arange(a.periods+1)*TWO_PI)
        if not sol.success or sol.y.shape[1]!=a.periods+1: raise RuntimeError(sol.message)
        diag=evaluate(sol,SKIP); diag.update(key=f"{group}:{r['index']}",solver="scipy.solve_ivp DOP853 independent polar RF gradients",
           rtol=2e-10,atol=2e-12,max_step_tau=TWO_PI/24,nfev=sol.nfev,wall_s=time.perf_counter()-t0,
           initial=init.tolist(),rf_phase=phase.tolist(),steps64_category=r["category"])
        np.savez_compressed(outdir/f"dop_{group}_{r['index']}.npz",tau=sol.t,state=sol.y[:12].T,
                            work=sol.y[12:22].T.reshape(-1,2,5),angle_integral=sol.y[22:24].T)
        results.append(diag); print(diag["key"],diag["frequency_Hz"],diag["wall_s"],flush=True)
    dest=outdir/"dop_candidate_verification.json"
    dest.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
