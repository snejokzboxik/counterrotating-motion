"""Independent polar-coordinate DOP853 and energy/angular-momentum ledgers."""
from __future__ import annotations
import json, math, os, time
from pathlib import Path
import sys

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"v17c_9"/"baseline"
sys.path.insert(0,str(BASE))
import basin_scan  # noqa: E402
import loading_dynamics as ld  # noqa: E402

TWOPI=2.0*math.pi
GROUPS=("rest","weak","wide","edge")
PIDS=(0,505,353)
PERIODS=15000
SKIP_RELAX=200
MAX_STEP=TWOPI/48.0
RTOL=2e-10
ATOL=2e-12
OUT=HERE/"strict_long_validation"


@njit(cache=True)
def _shape(sid,s,power):
    if s<=0.0 or s>=1.0:return 0.0 if s<=0.0 else 1.0
    if sid==0:return s
    if sid==1:return 0.5*(1.0-math.cos(math.pi*s))
    if sid==2:return s*s*(3.0-2.0*s)
    if sid==3:return s*s*s*(s*(6.0*s-15.0)+10.0)
    return s**power

@njit(cache=True)
def _shape_d(sid,s,power):
    if s<=0.0 or s>=1.0:return 0.0
    if sid==0:return 1.0
    if sid==1:return 0.5*math.pi*math.sin(math.pi*s)
    if sid==2:return 6.0*s*(1.0-s)
    if sid==3:return 30.0*s*s*(1.0-s)*(1.0-s)
    return power*s**(power-1.0)

@njit(cache=True)
def _ramp(start,end,cycle,delay,duration,sid,power):
    if duration<=0.0:return end
    s=(cycle-delay)/duration
    if s<=0.0:return start
    if s>=1.0:return end
    return start+(end-start)*_shape(sid,s,power)

@njit(cache=True)
def _ramp_d(start,end,cycle,delay,duration,sid,power):
    if duration<=0.0:return 0.0
    s=(cycle-delay)/duration
    if s<=0.0 or s>=1.0:return 0.0
    return (end-start)*_shape_d(sid,s,power)/(duration*TWOPI)

@njit(cache=True)
def _cap(start,cap,cycle,ramp,hold,trans,sid,tsid,power):
    if ramp>0.0 and cycle<ramp:return _ramp(start,cap,cycle,0.0,ramp,sid,power)
    after=cycle-ramp
    if after<=hold:return cap
    if trans<=0.0:return 1.0
    return _ramp(cap,1.0,after,hold,trans,tsid,power)

@njit(cache=True)
def _cap_d(start,cap,cycle,ramp,hold,trans,sid,tsid,power):
    if ramp>0.0 and cycle<ramp:return _ramp_d(start,cap,cycle,0.0,ramp,sid,power)
    after=cycle-ramp
    if after<=hold or trans<=0.0:return 0.0
    return _ramp_d(cap,1.0,after,hold,trans,tsid,power)

@njit(cache=True)
def _env(t,prot):
    kind=int(prot[0]);cycle=t/TWOPI
    if kind==0:return 1.,1.,1.,0.,0.,0.
    sid=int(prot[9]);power=prot[10]
    if kind==1:
        ga=_ramp(prot[6],1.,cycle,0.,prot[1],sid,power)
        gb=_ramp(prot[7],1.,cycle,prot[4],prot[2],sid,power)
        gad=_ramp_d(prot[6],1.,cycle,0.,prot[1],sid,power)
        gbd=_ramp_d(prot[7],1.,cycle,prot[4],prot[2],sid,power)
        if int(prot[11])==1:
            gd=ga*gb;gdd=gad*gb+ga*gbd
        else:
            gd=_ramp(prot[8],1.,cycle,prot[5],prot[3],sid,power)
            gdd=_ramp_d(prot[8],1.,cycle,prot[5],prot[3],sid,power)
        return ga,gb,gd,gad,gbd,gdd
    if kind==3:
        ga=_ramp(prot[6],1.,cycle,prot[5],prot[1],sid,power)
        gb=_ramp(prot[7],1.,cycle,0.,prot[2],sid,power)
        gad=_ramp_d(prot[6],1.,cycle,prot[5],prot[1],sid,power)
        gbd=_ramp_d(prot[7],1.,cycle,0.,prot[2],sid,power)
        if int(prot[11])==1:gd=ga*gb;gdd=gad*gb+ga*gbd
        else:
            gd=_ramp(prot[8],1.,cycle,prot[4],prot[3],sid,power)
            gdd=_ramp_d(prot[8],1.,cycle,prot[4],prot[3],sid,power)
        return ga,gb,gd,gad,gbd,gdd
    ga=_cap(prot[6],prot[12],cycle,prot[15],prot[16],prot[17],sid,int(prot[18]),power)
    gb=_cap(prot[7],prot[13],cycle,prot[15],prot[16],prot[17],sid,int(prot[18]),power)
    gd=_cap(prot[8],prot[14],cycle,prot[15],prot[16],prot[17],sid,int(prot[18]),power)
    gad=_cap_d(prot[6],prot[12],cycle,prot[15],prot[16],prot[17],sid,int(prot[18]),power)
    gbd=_cap_d(prot[7],prot[13],cycle,prot[15],prot[16],prot[17],sid,int(prot[18]),power)
    gdd=_cap_d(prot[8],prot[14],cycle,prot[15],prot[16],prot[17],sid,int(prot[18]),power)
    return ga,gb,gd,gad,gbd,gdd

@njit(cache=True)
def dop_rhs(t,y,p,prot,phase,family,modes,coeff,qratio):
    out=np.zeros(37,dtype=np.float64)
    ga,gb,gd,gad,gbd,gdd=_env(t,prot)
    env=np.empty(2);env[0]=ga;env[1]=gb
    envd=np.empty(2);envd[0]=gad;envd[1]=gbd
    delta=np.array([y[0]-y[6],y[1]-y[7],y[2]-y[8]])
    d2=np.dot(delta,delta);dist=math.sqrt(d2)
    invd3=1.0/(d2*dist)
    for i in range(2):
        b=6*i;x,yy,z=y[b],y[b+1],y[b+2]
        vx,vy,vz=y[b+3],y[b+4],y[b+5]
        rad=math.hypot(x,yy);theta=math.atan2(yy,x)
        mass=1.0 if i==0 else 1.0/p[4]
        invmass=1.0 if i==0 else p[4]
        elec=1.0 if i==0 else p[4]*qratio
        qfac=1.0 if i==0 else qratio
        lam=p[6+i]
        radial=np.zeros(2);azim=np.zeros(2)
        for h in range(modes.size):
            mode=modes[h];fam=family[h];deg=abs(mode)
            omega=1.0 if fam==0 else p[2]
            ang=mode*theta-omega*t-phase[fam]
            c=math.cos(ang);s=math.sin(ang)
            re=coeff[h].real;im=coeff[h].imag
            strength=p[fam]*env[fam]*rad**(deg-1)
            radial[fam]-=strength*deg*(re*c-im*s)
            azim[fam]+=strength*mode*(re*s+im*c)
        rf=np.zeros((2,3))
        ct=math.cos(theta);st=math.sin(theta)
        for f in range(2):
            rf[f,0]=elec*(radial[f]*ct-azim[f]*st)
            rf[f,1]=elec*(radial[f]*st+azim[f]*ct)
        dc0=np.array([p[3]*gd*x,p[3]*gd*yy,-2.0*p[3]*gd*z])
        dc=elec*dc0
        sign=1.0 if i==0 else -1.0
        cf=invmass*p[5]*sign*delta*invd3
        drag=-lam*np.array([vx,vy,vz])
        vel=np.array([vx,vy,vz]);acc=rf[0]+rf[1]+dc+cf+drag
        out[b]=vx;out[b+1]=vy;out[b+2]=vz
        out[b+3:b+6]=acc
        for f in range(2):out[12+5*i+f]=mass*np.dot(rf[f],vel)
        out[12+5*i+2]=mass*np.dot(dc,vel)
        out[12+5*i+3]=mass*np.dot(cf,vel)
        out[12+5*i+4]=mass*np.dot(drag,vel)
        out[22+i]=(x*vy-yy*vx)/(rad*rad)
        # Explicit time derivative of RF/DC potential at fixed coordinates.
        base=complex(x,yy)
        if modes.size>0:
            explicit_rf=np.zeros(2)
            for h in range(modes.size):
                mode=modes[h];fam=family[h];deg=abs(mode)
                omega=1.0 if fam==0 else p[2]
                zbase=base if mode>0 else base.conjugate()
                carr=complex(math.cos(omega*t+phase[fam]),-math.sin(omega*t+phase[fam]))
                amp=p[fam]*env[fam]
                ampdot=p[fam]*envd[fam]
                term=complex(ampdot,-omega*amp)*coeff[h]*(zbase**deg)*carr
                explicit_rf[fam]+=qfac*term.real
            out[24]+=explicit_rf[0];out[25]+=explicit_rf[1]
        quad=z*z-0.5*(x*x+yy*yy)
        out[26]+=qfac*p[3]*gdd*quad
        tq=np.empty((5,3))
        for f in range(2):tq[f]=rf[f]
        tq[2]=dc;tq[3]=cf;tq[4]=drag
        for f in range(5):out[27+5*i+f]=mass*(x*tq[f,1]-yy*tq[f,0])
    return out


def energy_at(t,y,p,protocol,phase,family,modes,coeff,qratio):
    g=ld._envelopes(t,protocol)
    pos=np.asarray([[y[0],y[1],y[2]],[y[6],y[7],y[8]]])
    vel=np.asarray([[y[3],y[4],y[5]],[y[9],y[10],y[11]]])
    mu=np.asarray([1.0,1.0/p[4]])
    charge=np.asarray([1.0,qratio])
    K=.5*np.sum(mu[:,None]*vel*vel)
    radial=pos[:,0]**2+pos[:,1]**2
    quad=pos[:,2]**2-.5*radial
    Udc=float(np.sum(charge*p[3]*g[2]*quad))
    Urf=np.zeros(2)
    for i in range(2):
        z=complex(pos[i,0],pos[i,1])
        for h,mode in enumerate(modes):
            fam=int(family[h]);omega=1.0 if fam==0 else p[2]
            zbase=z if mode>0 else z.conjugate()
            v=p[fam]*g[fam]*coeff[h]*(zbase**abs(int(mode)))*np.exp(-1j*(omega*t+phase[fam]))
            Urf[fam]+=charge[i]*v.real
    UC=p[5]/np.linalg.norm(pos[0]-pos[1])
    return {"K_total":float(K),"U_RF1":float(Urf[0]),"U_RF2":float(Urf[1]),
            "U_DC":Udc,"U_Coulomb":float(UC),
            "H_total":float(K+Urf.sum()+Udc+UC)}


def build_events(p):
    radii=(.0002,.00189)
    def wall1(t,y):return 1.0-radii[0]-math.hypot(y[0],y[1])
    def wall2(t,y):return 1.0-radii[1]-math.hypot(y[6],y[7])
    def collide(t,y):return math.sqrt((y[0]-y[6])**2+(y[1]-y[7])**2+(y[2]-y[8])**2)-sum(radii)
    def z1p(t,y):return 2.0-radii[0]-y[2]
    def z1m(t,y):return 2.0-radii[0]+y[2]
    def z2p(t,y):return 2.0-radii[1]-y[8]
    def z2m(t,y):return 2.0-radii[1]+y[8]
    events=[wall1,wall2,collide,z1p,z1m,z2p,z2m]
    for e in events:e.terminal=True;e.direction=-1
    return events


def strict_classification(tau,states,skip):
    completed=int(math.floor(tau[-1]/TWOPI))
    pos=states[:,[0,1,2,6,7,8]].reshape(-1,2,3)
    rad=np.linalg.norm(pos[:,:,:2],axis=2)
    theta=np.unwrap(np.arctan2(pos[:,:,1],pos[:,:,0]),axis=0)
    cuts=np.linspace(min(skip,len(tau)-1),len(tau)-1,4,dtype=int)
    windows=np.diff(theta[cuts],axis=0)/TWOPI
    turns=(theta[-1]-theta[min(skip,len(tau)-1)])/TWOPI
    stable=np.all(np.abs(windows)>1e-12,axis=1)
    opposite=np.all(windows[:,0]*windows[:,1]<0.0)
    ambiguous=np.any(rad.min(axis=0)<.02)
    retained=completed>=PERIODS
    if completed<PERIODS:category="event_before_horizon"
    elif ambiguous:category="axis_ambiguous"
    elif np.all(np.abs(turns)>=10.0) and np.all(stable) and opposite:category="CR_screen"
    elif np.all(np.abs(turns)>=10.0) and np.all(stable) and np.all(windows[:,0]*windows[:,1]>0):category="same_direction"
    else:category="inconclusive"
    return {"category":category,"completed_periods":completed,"turns_after_relaxation":turns.tolist(),
            "window_turns":windows.tolist(),"min_radius":rad.min(axis=0).tolist(),
            "max_radius":rad.max(axis=0).tolist(),"max_abs_z":np.abs(pos[:,:,2]).max(axis=0).tolist(),
            "minimum_separation":float(np.linalg.norm(pos[:,0]-pos[:,1],axis=1).min()),
            "retained":retained,"axis_ambiguous":bool(ambiguous)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pdoc=json.loads((HERE/"local15_parameters.json").read_text(encoding="utf-8"))
    p=np.asarray(pdoc["candidate"]["p"],float);qratio=float(pdoc["candidate"]["q2_over_q1"])
    _p0,family,modes,coeff=basin_scan.load_model()
    protocols=json.loads((HERE/"loading_screen"/"protocols.json").read_text(encoding="utf-8"))
    rkrows=[json.loads(s) for s in (OUT/"strict_runs.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    reps=[]
    for pid in PIDS:
        a=[r for r in rkrows if r["protocol_id"]==pid and r["ensemble"]=="edge" and r["category"]=="CR_screen"]
        if not a:raise RuntimeError(f"no strict edge representative for protocol {pid}")
        a.sort(key=lambda r:r["start_id"])
        reps.append(a[0])
    outputs=[]
    checkpoint=OUT/"dop853_energy_validation.jsonl"
    done={}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r=json.loads(line);done[(r["protocol_id"],r["ensemble"],r["start_id"])]=r
    for row in reps:
        pid=row["protocol_id"];protocol=np.asarray(protocols[pid]["protocol"],float)
        rowkey=(pid,row["ensemble"],row["start_id"])
        if rowkey in done:
            outputs.append(done[rowkey]);continue
        y0=np.zeros(37);y0[:12]=np.asarray(row["initial"],float)
        transition=int(math.ceil(ld.protocol_duration_cycles(protocol)))
        skip=transition+SKIP_RELAX
        t0=time.perf_counter()
        sol=solve_ivp(lambda t,y:dop_rhs(t,y,p,protocol,np.asarray(row["phase"],float),family,modes,coeff,qratio),
          (0.0,PERIODS*TWOPI),y0,method="DOP853",rtol=RTOL,atol=ATOL,max_step=MAX_STEP,
          t_eval=np.arange(PERIODS+1,dtype=float)*TWOPI,events=build_events(p))
        if not sol.success:raise RuntimeError(sol.message)
        tau=sol.t; states=sol.y[:12].T.copy();diag=strict_classification(tau,states,skip)
        rk_final=np.asarray(row["final_state"],float)
        finaldiff=float(np.max(np.abs(states[-1]-rk_final)))
        work=sol.y[12:22].T.reshape(-1,2,5)
        explicit=sol.y[24:27].T
        torque=sol.y[27:37].T.reshape(-1,2,5)
        iskip=min(skip,len(tau)-1)
        mass=np.asarray([1.0,1.0/p[4]])
        kin=.5*mass[None,:]*np.sum(states[:,[3,4,5,9,10,11]].reshape(-1,2,3)**2,axis=2)
        per_particle={}
        for i in range(2):
            deltaK=float(kin[-1,i]-kin[iskip,i])
            deltaW=(work[-1,i]-work[iskip,i]).tolist()
            per_particle[str(i+1)]={"delta_kinetic_energy":deltaK,
                "work_RF1_RF2_DC_Coulomb_drag":deltaW,
                "kinetic_balance_residual":float(deltaK-sum(deltaW)),
                "delta_angular_momentum":float(mass[i]*(states[-1,6*i]*states[-1,6*i+4]-states[-1,6*i+1]*states[-1,6*i+3])-mass[i]*(states[iskip,6*i]*states[iskip,6*i+4]-states[iskip,6*i+1]*states[iskip,6*i+3])),
                "angular_momentum_change_from_torques":(torque[-1,i]-torque[iskip,i]).tolist()}
        H0=energy_at(float(tau[iskip]),sol.y[:,iskip],p,protocol,np.asarray(row["phase"],float),family,modes,coeff,qratio)
        H1=energy_at(float(tau[-1]),sol.y[:,-1],p,protocol,np.asarray(row["phase"],float),family,modes,coeff,qratio)
        dH=H1["H_total"]-H0["H_total"]
        explicit_work=(explicit[-1]-explicit[iskip]).tolist()
        drag_work=float((work[-1,:,4]-work[iskip,:,4]).sum())
        system_energy_residual=float(dH-sum(explicit_work)-drag_work)
        total_L=np.sum(mass[None,:]*(states[:,[0,6]]*states[:,[4,10]]-states[:,[1,7]]*states[:,[3,9]]),axis=1)
        dL=float(total_L[-1]-total_L[iskip])
        torque_delta=(torque[-1]-torque[iskip]).sum(axis=0)
        angular_residual=float(dL-torque_delta.sum())
        results={"protocol_id":pid,"ensemble":row["ensemble"],"start_id":row["start_id"],
          "rk4_category":row["category"],"rk4_final_max_abs_delta_vs_DOP853":finaldiff,
          "solver":"SciPy solve_ivp DOP853 with independently coded polar gradients and continuous envelopes",
          "rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP,"nfev":int(sol.nfev),
          "elapsed_seconds":time.perf_counter()-t0,"strict_dop853":diag,
          "frequency_Hz":(diag["turns_after_relaxation"] and (np.asarray(diag["turns_after_relaxation"])*100000.0/(diag["completed_periods"]-skip))).tolist(),
          "per_particle_energy_ledger":per_particle,
          "system_energy_after_relaxation":{"H_start":H0,"H_end":H1,"delta_H":dH,
            "explicit_work_RF1_RF2_DC":explicit_work,"drag_work_total":drag_work,
            "balance_residual":system_energy_residual},
          "system_angular_momentum_after_relaxation":{"delta_Lz":dL,
            "integrated_torque_RF1_RF2_DC_Coulomb_drag":torque_delta.tolist(),
            "balance_residual":angular_residual},
          "initial":row["initial"],"rf_phase":row["phase"]}
        file=OUT/f"dop853_protocol{pid}_edge_start{row['start_id']}.npz"
        np.savez_compressed(file,tau=tau, state=states, work=work, explicit_work=explicit,
                            torque=torque, protocol=protocol, initial=np.asarray(row["initial"]),
                            phase=np.asarray(row["phase"]))
        results["trajectory_file"]=file.name
        outputs.append(results)
        with checkpoint.open("a",encoding="utf-8") as f:
            f.write(json.dumps(results,separators=(",",":"))+"\n")
            f.flush();os.fsync(f.fileno())
        print(pid,diag["category"],diag["turns_after_relaxation"],"maxdiff",finaldiff,"sec",results["elapsed_seconds"],flush=True)
    (OUT/"dop853_energy_validation.json").write_text(json.dumps(outputs,indent=2)+"\n",encoding="utf-8")
    print(f"saved {len(outputs)} independent DOP853 reps")

if __name__=="__main__":main()
