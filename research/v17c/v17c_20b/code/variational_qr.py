from __future__ import annotations
from pathlib import Path
import math, sys, json
import numpy as np
from numba import njit
import attractor_search as a; SIM=a.SIM

@njit(cache=True)
def state_jacobian(t,y,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z):
    J=np.zeros((12,12),dtype=np.float64)
    for i in range(3):
        J[i,i+3]=1.0;J[i+6,i+9]=1.0
    s=p[4];C=p[5];nu=p[2];A=p[0];B=p[1];beta=p[3]
    for part in range(2):
        j=6*part;xx=y[j];yy=y[j+1];z=complex(xx,yy)
        efac=1.0 if part==0 else s*qratio
        c1=A*complex(math.cos(t+phase[0]),-math.sin(t+phase[0]))
        c2=B*complex(math.cos(nu*t+phase[1]),-math.sin(nu*t+phase[1]))
        hxx=0.0;hxy=0.0;hyy=0.0
        for h in range(modes.size):
            mode=modes[h];degree=abs(mode)
            base=z if mode>0 else z.conjugate()
            carrier=c1 if family[h]==0 else c2
            hess=carrier*coeff[h]*degree*(degree-1)*base**(degree-2)
            if mode>0:
                hxx-=hess.real;hxy+=hess.imag;hyy+=hess.real
            else:
                hxx-=hess.real;hxy-=hess.imag;hyy+=hess.real
        J[j+3,j]+=efac*(hxx+beta);J[j+3,j+1]+=efac*hxy
        J[j+4,j]+=efac*hxy;J[j+4,j+1]+=efac*(hyy+beta)
        J[j+5,j+2]+=efac*(-2.0*beta)
        vx=y[j+3];vy=y[j+4];vz=y[j+5];speed=math.sqrt(vx*vx+vy*vy+vz*vz)
        lam=p[6] if part==0 else p[7];chi=re1 if part==0 else re2
        re=max(0.0,chi*speed);fac=1.0+0.15*re**0.687
        coeff_v=0.0
        if speed>1e-30:coeff_v=0.15*0.687*(chi**0.687)*(speed**(-1.313))
        vv=(vx,vy,vz)
        for r in range(3):
            for c in range(3):
                J[j+3+r,j+3+c]+=-lam*(fac*(1.0 if r==c else 0.0)+coeff_v*vv[r]*vv[c])
    dx=y[0]-y[6];dy=y[1]-y[7];dz=y[2]-y[8]
    d2=dx*dx+dy*dy+dz*dz
    if C!=0.0 and d2>1e-28:
        d=math.sqrt(d2);d3=d2*d;d5=d3*d2;dv=(dx,dy,dz)
        for r in range(3):
            for c in range(3):
                h=C*((1.0 if r==c else 0.0)/d3-3.0*dv[r]*dv[c]/d5)
                J[3+r,c]+=h;J[3+r,c+6]-=h
                J[9+r,c]-=s*h;J[9+r,c+6]+=s*h
    return J

@njit(cache=True)
def augmented_rhs(t,y,M,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z,a1,a2):
    dy=np.empty(12,dtype=np.float64);SIM._rhs(t,y,p,phase,family,modes,coeff,qratio,dy,re1,0,
      a1,
      a2,re2,gravity_z,2,1.0)
    J=state_jacobian(t,y,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z)
    dM=J@M
    return dy,dM

@njit(cache=True)
def advance_augmented(y,M,t0,dt,nsteps,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z,a1,a2):
    y=y.copy();M=M.copy();t=t0
    for _ in range(nsteps):
        k1y,k1m=augmented_rhs(t,y,M,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z,a1,a2)
        k2y,k2m=augmented_rhs(t+dt/2,y+dt*k1y/2,M+dt*k1m/2,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z,a1,a2)
        k3y,k3m=augmented_rhs(t+dt/2,y+dt*k2y/2,M+dt*k2m/2,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z,a1,a2)
        k4y,k4m=augmented_rhs(t+dt,y+dt*k3y,M+dt*k3m,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z,a1,a2)
        y=y+dt*(k1y+2*k2y+2*k3y+k4y)/6
        M=M+dt*(k1m+2*k2m+2*k3m+k4m)/6
        t+=dt
    return y,M,t

def rhs_numpy(t,y,p,phase,field,qratio,re1,re2,gravity):
    out=np.zeros(12);SIM._rhs(t,y,p,phase,field['family'],field['k'],field['coeff'],qratio,out,
      re1,0,0,0,re2,gravity,2,1.0)
    return out

def validate_jacobian(case,seed=20261123):
    field=a.load_field();p,q=a.arrays(case,field);rng=np.random.default_rng(seed);errors=[]
    for _ in range(8):
        y=rng.normal(0,0.25,12);y[:3]*=.7;y[6:9]*=.7;y[6:9]+=np.array([.7,0.1,-.1]);y[3:6]*=.01;y[9:12]*=.01
        J=state_jacobian(.73,y,p,np.array([1.2,2.1]),field['family'],field['k'],field['coeff'],q,
          case['re1_per_speed'],case['re2_per_speed'],case['gravity_z'])
        Jfd=np.zeros((12,12));h=2e-6
        for c in range(12):
            yp=y.copy();ym=y.copy();yp[c]+=h;ym[c]-=h
            Jfd[:,c]=(rhs_numpy(.73,yp,p,np.array([1.2,2.1]),field,q,case['re1_per_speed'],case['re2_per_speed'],case['gravity_z'])-
              rhs_numpy(.73,ym,p,np.array([1.2,2.1]),field,q,case['re1_per_speed'],case['re2_per_speed'],case['gravity_z']))/(2*h)
        errors.append(float(np.max(np.abs(J-Jfd))/max(1,np.max(np.abs(Jfd)))))
    return {'seed':seed,'samples':len(errors),'max_relative_jacobian_error':max(errors),'errors':errors,
      'method':'analytic MFS Hessian + analytic central Coulomb derivative + Schiller-Naumann drag derivative vs central-difference independent calls to audited RHS'}

def lyapunov_qr(case,state,phase_start,periods=2000,steps=32,qr_periods=20):
    field=a.load_field();p,q=a.arrays(case,field);y=np.asarray(state,float).copy();M=np.eye(12);dt=2*math.pi/steps
    logs=np.zeros(12);rows=[];elapsed=0;time_local=0.0;qrsteps=qr_periods*steps;remaining=periods*steps
    while remaining>0:
        n=min(qrsteps,remaining)
        y,M,time_local=advance_augmented(y,M,time_local,dt,n,p,np.asarray(phase_start,float),field['family'],field['k'],field['coeff'],q,
          case['re1_per_speed'],case['re2_per_speed'],case['gravity_z'],case['a1_over_R'],case['a2_over_R'])
        # QR reorthogonalization suppresses tangent overflow; x time origin resets with continuous phase encoded in phase_start.
        Q,R=np.linalg.qr(M);diag=np.maximum(np.abs(np.diag(R)),1e-300);logs+=np.log(diag);M=Q
        elapsed+=n;remaining-=n
        if elapsed%(100*steps)==0 or remaining==0:
            rows.append({'periods':elapsed/steps,'exponents_per_tau':(logs/(elapsed*dt)).tolist(),
              'top_per_tau':float(np.max(logs/(elapsed*dt))),'top_per_RF_period':float(np.max(logs/(elapsed*dt))*2*math.pi)})
    return {'candidate':case['id'],'periods':periods,'steps_per_period':steps,'qr_interval_periods':qr_periods,
      'independent_RF_phases_fixed':'Conditional mechanical tangent spectrum; the two prescribed carrier phases are held fixed across perturbations, so this is not the full extended phase-torus spectrum.',
      'rows':rows}

@njit(cache=True)
def extended_jacobian(y,ph,p,t,family,modes,coeff,qratio,re1,re2,gravity_z):
    J=np.zeros((14,14),dtype=np.float64)
    J[:12,:12]=state_jacobian(0.0,y,p,ph,family,modes,coeff,qratio,re1,re2,gravity_z)
    h=1e-6;dyp=np.empty(12);dym=np.empty(12)
    for k in range(2):
        pp=ph.copy();pm=ph.copy();pp[k]+=h;pm[k]-=h
        SIM._rhs(0.0,y,p,pp,family,modes,coeff,qratio,dyp,re1,0,0,0,re2,gravity_z,2,1.0)
        SIM._rhs(0.0,y,p,pm,family,modes,coeff,qratio,dym,re1,0,0,0,re2,gravity_z,2,1.0)
        J[:12,12+k]=(dyp-dym)/(2*h)
    return J

@njit(cache=True)
def extended_rhs(y,ph,M,p,nu,family,modes,coeff,qratio,re1,re2,gravity_z):
    dy=np.empty(12);SIM._rhs(0.0,y,p,ph,family,modes,coeff,qratio,dy,re1,0,0,0,re2,gravity_z,2,1.0)
    dM=extended_jacobian(y,ph,p,0.0,family,modes,coeff,qratio,re1,re2,gravity_z)@M
    return dy,np.array([1.0,nu]),dM

@njit(cache=True)
def advance_extended(y,ph,M,dt,nsteps,p,nu,family,modes,coeff,qratio,re1,re2,gravity_z):
    y=y.copy();ph=ph.copy();M=M.copy()
    for _ in range(nsteps):
        k1y,k1p,k1m=extended_rhs(y,ph,M,p,nu,family,modes,coeff,qratio,re1,re2,gravity_z)
        k2y,k2p,k2m=extended_rhs(y+dt*k1y/2,ph+dt*k1p/2,M+dt*k1m/2,p,nu,family,modes,coeff,qratio,re1,re2,gravity_z)
        k3y,k3p,k3m=extended_rhs(y+dt*k2y/2,ph+dt*k2p/2,M+dt*k2m/2,p,nu,family,modes,coeff,qratio,re1,re2,gravity_z)
        k4y,k4p,k4m=extended_rhs(y+dt*k3y,ph+dt*k3p,M+dt*k3m,p,nu,family,modes,coeff,qratio,re1,re2,gravity_z)
        y=y+dt*(k1y+2*k2y+2*k3y+k4y)/6
        ph=ph+dt*(k1p+2*k2p+2*k3p+k4p)/6
        M=M+dt*(k1m+2*k2m+2*k3m+k4m)/6
    return y,ph,M

def full_phase_lyapunov_qr(case,state,phase_start,periods=1000,steps=32,qr_periods=20):
    field=a.load_field();p,q=a.arrays(case,field);y=np.asarray(state,float).copy();ph=np.asarray(phase_start,float).copy();M=np.eye(14)
    dt=2*math.pi/steps;logs=np.zeros(14);rows=[];elapsed=0;remaining=periods*steps;qrsteps=qr_periods*steps
    while remaining>0:
        n=min(qrsteps,remaining)
        y,ph,M=advance_extended(y,ph,M,dt,n,p,float(p[2]),field['family'],field['k'],field['coeff'],q,
          case['re1_per_speed'],case['re2_per_speed'],case['gravity_z'])
        Q,R=np.linalg.qr(M);logs+=np.log(np.maximum(np.abs(np.diag(R)),1e-300));M=Q;elapsed+=n;remaining-=n
        if elapsed%(100*steps)==0 or remaining==0:
            ex=(logs/(elapsed*dt)).tolist()
            rows.append({'periods':elapsed/steps,'exponents_per_tau':ex,'top_per_tau':max(ex),'top_per_RF_period':max(ex)*2*math.pi})
    return {'candidate':case['id'],'periods':periods,'steps_per_period':steps,'qr_interval_periods':qr_periods,
      'state_dimension':14,'phase_variables':['psi1','psi2'],'phase_rates':[1.0,float(p[2])],
      'phase_jacobian':'central finite difference h=1e-6 rad; independent RF phases are explicit dynamical coordinates and are held in the tangent system, yielding two neutral phase directions.',
      'rows':rows}

if __name__=='__main__':
    cfg=a.build_config();case=next(c for c in cfg['candidates'] if c['legacy_candidate']=='refine12_sobol17')
    print(json.dumps(validate_jacobian(case),indent=2))
