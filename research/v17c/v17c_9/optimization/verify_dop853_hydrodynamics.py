"""Independent DOP853 energy/torque validation with Oseen/RPY hydrodynamics."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import ode
from numba import njit

HERE=Path(__file__).resolve().parent;BASE=HERE.parent
CFG=json.loads((BASE.parent/"v17c_5"/"input"/"config.json").read_text(encoding="utf-8-sig"))
K=np.asarray(CFG["k"],np.int64);FAMILY=np.asarray(CFG["family"],np.int64)
CO=np.asarray(CFG["coeff"],float);CO=CO[:,0]+1j*CO[:,1]
T=2*math.pi;F1=float(CFG["f1_Hz"])

@njit(cache=True)
def rhs(t,y,p,qratio,phase,hydro_mode,a1,a2,chi1,chi2):
    A,B,nu,beta,s,C,lam1,lam2=p
    dx,dy,dz=y[0]-y[6],y[1]-y[7],y[2]-y[8]
    d2=dx*dx+dy*dy+dz*dz
    d=math.sqrt(d2);pair=C*np.array([dx,dy,dz])/(d2*d)
    out=np.zeros(38)
    for i,base in enumerate((0,6)):
        x,yy,z=y[base],y[base+1],y[base+2]
        v=np.array([y[base+3],y[base+4],y[base+5]])
        r=math.hypot(x,yy);theta=math.atan2(yy,x)
        radial=np.zeros(2);az=np.zeros(2)
        for h,mode in enumerate(K):
            fam=FAMILY[h];omega=1. if fam==0 else nu
            ph=mode*theta-omega*t-phase[fam]
            re,im=CO[h].real,CO[h].imag;c,ss=math.cos(ph),math.sin(ph)
            strength=(A if fam==0 else B)*r**(abs(mode)-1)
            radial[fam]-=strength*abs(mode)*(re*c-im*ss)
            az[fam]+=strength*mode*(re*ss+im*c)
        ct,st=math.cos(theta),math.sin(theta)
        rf=np.zeros((2,3))
        for f in range(2):
            rf[f,0]=radial[f]*ct-az[f]*st
            rf[f,1]=radial[f]*st+az[f]*ct
        dc=beta*np.array([x,yy,-2*z])
        invmass=1. if i==0 else s;mass=1./invmass
        qr=1. if i==0 else qratio
        lam=lam1 if i==0 else lam2;chi=chi1 if i==0 else chi2
        speed=math.sqrt(v[0]**2+v[1]**2+v[2]**2)
        drag_acc=-lam*(1.+.1875*chi*speed)*v
        hydro_acc=np.zeros(3)
        if hydro_mode>0 and d>0.:
            aj=a2 if i==0 else a1
            other=9 if i==0 else 3
            vo=np.array([y[other],y[other+1],y[other+2]])
            nhat=np.array([dx,dy,dz])/d
            if i==1:nhat=-nhat
            ai=a1 if i==0 else a2
            giso=1.;glong=1.
            if hydro_mode==2:
                r2=ai*ai+aj*aj
                giso=1.+r2/(3.*d*d);glong=1.-r2/(d*d)
            hscale=.75*aj/d
            hydro_acc=lam*hscale*(giso*vo+glong*nhat*np.dot(nhat,vo))
        cf=pair if i==0 else -pair
        acc=invmass*(qr*(rf[0]+rf[1]+dc)+cf)+drag_acc+hydro_acc
        out[base:base+3]=v;out[base+3:base+6]=acc
        wi=12+6*i
        out[wi]=qr*np.dot(rf[0],v);out[wi+1]=qr*np.dot(rf[1],v)
        out[wi+2]=qr*np.dot(dc,v);out[wi+3]=np.dot(cf,v)
        out[wi+4]=mass*np.dot(drag_acc,v);out[wi+5]=mass*np.dot(hydro_acc,v)
        if r>1e-14:out[24+i]=(x*v[1]-yy*v[0])/(r*r)
        ti=26+6*i
        out[ti]=qr*(x*rf[0,1]-yy*rf[0,0])
        out[ti+1]=qr*(x*rf[1,1]-yy*rf[1,0])
        out[ti+2]=qr*(x*dc[1]-yy*dc[0])
        out[ti+3]=x*cf[1]-yy*cf[0]
        out[ti+4]=mass*(x*drag_acc[1]-yy*drag_acc[0])
        out[ti+5]=mass*(x*hydro_acc[1]-yy*hydro_acc[0])
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=("oseen_self","oseen_pair","rpy_pair"),required=True)
    ap.add_argument("--ensemble",choices=("rest","weak","wide"),default="rest")
    ap.add_argument("--index",type=int,default=0);ap.add_argument("--periods",type=int,default=15000)
    a=ap.parse_args()
    z=np.load(HERE/"stage2_speed"/"configurations.npz");p=np.asarray(z["p"][48],float);qratio=float(z["qratio"][48])
    a1,a2=.0002,.00189;chi1,chi2=.2,1.89
    hm=0 if a.model=="oseen_self" else (1 if a.model=="oseen_pair" else 2)
    chis=(chi1,chi2) if a.model=="oseen_self" else (0.,0.)
    src=HERE/"stage3"/"runs"/f"cfg48_{a.ensemble}_holdout_128_{a.periods}p_64s.jsonl"
    rows={int(r["index"]):r for r in (json.loads(s) for s in src.read_text().splitlines() if s.strip())}
    row=rows[a.index];initial=np.asarray(row["initial"],float);phase=np.asarray(row["phase"],float)
    times=np.arange(a.periods+1,dtype=float)*T;y0=np.r_[initial,np.zeros(26)]
    solver=ode(rhs).set_integrator("dop853",rtol=2e-10,atol=2e-12,max_step=T/24,nsteps=512)
    solver.set_f_params(p,qratio,phase,hm,a1,a2,*chis);solver.set_initial_value(y0,0.)
    states=np.empty((a.periods+1,38));states[0]=y0;tic=time.perf_counter()
    for j,t in enumerate(times[1:],1):
        states[j]=solver.integrate(t)
        if not solver.successful() or not np.isfinite(states[j]).all():raise RuntimeError(f"DOP853 failed at cycle {j}")
    skip=200;pos=states[:,[0,1,2,6,7,8]].reshape(-1,2,3);vel=states[:,[3,4,5,9,10,11]].reshape(-1,2,3)
    rad=np.linalg.norm(pos[:,:,:2],axis=2);theta=np.unwrap(np.arctan2(pos[:,:,1],pos[:,:,0]),axis=0)
    turns=(theta[-1]-theta[skip])/T;cuts=np.linspace(skip,a.periods,4,dtype=int)
    windows=np.diff(theta[cuts],axis=0)/T;freq=turns/(a.periods-skip)*F1
    mass=np.array([1.,1/p[4]]);charge=np.array([1.,qratio])
    ke=.5*np.sum(mass[None,:,None]*vel**2,axis=(1,2))
    udc=p[3]*np.sum(charge[None,:]*(pos[:,:,2]**2-.5*rad**2),axis=1)
    uc=p[5]/np.linalg.norm(pos[:,0]-pos[:,1],axis=1)
    work=states[:,12:24].reshape(-1,2,6);torque=states[:,26:38].reshape(-1,2,6)
    ext=work[:,:,:2].sum(axis=(1,2))+work[:,:,4:].sum(axis=(1,2))
    residual=(ke+udc+uc)-(ke[0]+udc[0]+uc[0])-ext
    lz=mass[None,:]*(pos[:,:,0]*vel[:,:,1]-pos[:,:,1]*vel[:,:,0])
    dlz=lz[-1]-lz[skip];dtau=torque[-1]-torque[skip]
    angleint=(states[-1,24:26]-states[skip,24:26])/T
    dmin=np.linalg.norm(pos[:,0]-pos[:,1],axis=1).min()
    out={"key":f"48:{a.model}:{a.ensemble}:{a.index}","model":a.model,
      "solver":"SciPy DOP853; separate polar-gradient and finite-sphere hydrodynamics RHS",
      "rtol":2e-10,"atol":2e-12,"max_step_tau":T/24,"periods":a.periods,
      "turns_after_transient":turns.tolist(),"turns_by_omega_integral":angleint.tolist(),
      "frequency_Hz":freq.tolist(),"window_turns":windows.tolist(),
      "mean_radius":rad[skip:].mean(axis=0).tolist(),"minimum_center_separation":float(dmin),
      "minimum_surface_gap_over_R":float(dmin-a1-a2),"late_work_rf1_rf2_dc_coulomb_selfdrag_hydro":(work[-1]-work[skip]).tolist(),
      "late_torque_rf1_rf2_dc_coulomb_selfdrag_hydro":(torque[-1]-torque[skip]).tolist(),
      "angular_momentum_change":dlz.tolist(),"angular_momentum_balance_residual":(dlz-dtau.sum(axis=1)).tolist(),
      "energy_residual_max":float(np.max(np.abs(residual))),
      "energy_residual_over_RF_work":float(np.max(np.abs(residual))/abs(work[-1,:,:2].sum())),
      "max_angle_integral_difference_turns":float(np.max(np.abs((theta-theta[0])/T-states[:,24:26]/T))),
      "wall_s":time.perf_counter()-tic,"parameters":p.tolist(),"charge_ratio":qratio,
      "initial":initial.tolist(),"phase":phase.tolist(),"chi":[chis[0],chis[1]],"hydro_mode":hm}
    dst=HERE/"stage3"/"runs"/f"dop_hydro_{a.model}_{a.ensemble}_{a.index}"
    np.savez_compressed(str(dst)+".npz",tau=times,state=states[:,:12],work=work,angles_integral=states[:,24:26],torque=torque)
    Path(str(dst)+".json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
