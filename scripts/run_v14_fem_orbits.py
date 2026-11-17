"""Local FEM shooting around the ideal V14 off-axis 1-RF periodic orbits."""
from __future__ import annotations

import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_v13_3_fem_controls as fem
from run_v14_ideal_pilot import OUT,ALPHA,BETA,KAPPA,C2,C4,M0,OMEGA,R,E
from three_phase_trap.v14_dynamics import rhs

J=np.block([[np.zeros((2,2)),np.eye(2)],[-np.eye(2),np.zeros((2,2))]])


def make_rhs(model,kappa,delta,mu=1.,tau0=0.):
    scale=E/(M0*mu*OMEGA**2*R)
    voltscale=M0*OMEGA**2*R**2/E
    u2=BETA*voltscale/C2;vdc=ALPHA*voltscale/C2;u4=kappa*voltscale/C4
    def f(t,z):
        x,y=z[:2]
        qx,qy=fem._grad2(x,y)
        if model=='Q':gx=gy=0.
        else:
            cx,cy,sx,sy=fem._grad4(x,y)
            co=math.cos(t+tau0-delta)
            if model=='SO':gx,gy=cx*co,cy*co
            else:
                si=math.sin(t+tau0-delta)
                gx,gy=cx*co+sx*si,cy*co+sy*si
        qdrive=u2*math.cos(t+tau0)+vdc
        return np.array([z[2],z[3],-scale*(qdrive*qx+u4*gx),-scale*(qdrive*qy+u4*gy)])
    return f


def flow(z0,fun,periods=1,step=64,rtol=2e-10,atol=2e-12):
    def ev(t,z):return np.hypot(z[0],z[1])-1
    ev.direction=1;ev.terminal=True
    sol=solve_ivp(fun,(0,2*math.pi*periods),z0,method='DOP853',rtol=rtol,atol=atol,
                  max_step=2*math.pi/step,events=ev)
    if not sol.success or sol.status==1:raise RuntimeError('escaped or unresolved')
    return sol.y[:,-1],float(np.max(np.hypot(sol.y[0],sol.y[1])))


def jacobian(z,fun,eps=1e-6,step=128):
    cols=[]
    for j in range(4):
        e=np.eye(4)[j]*eps
        plus,_=flow(z+e,fun,step=step)
        minus,_=flow(z-e,fun,step=step)
        cols.append((plus-minus)/(2*eps))
    return np.stack(cols,axis=1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--meshes',type=int,nargs='+',default=[50,35]);args=p.parse_args()
    ideals=json.loads((OUT/'ideal_periodic_orbits.json').read_text())
    selection={r['model']:r for r in ideals if r['state'][1]>0 and r['periods']==1}
    results=[]
    for mesh in args.meshes:
      fem.initialize(mesh)
      for name in ('RO','SO_equal_harmonic','SO_equal_rms'):
        k=KAPPA*(math.sqrt(2) if name=='SO_equal_rms' else 1)
        model='SO' if name.startswith('SO') else name
        f=make_rhs(model,k,0.)
        initial=np.array(selection[name]['state'])
        evals=0
        def residual(z):
            nonlocal evals
            evals+=1
            try: final,_=flow(z,f)
            except Exception:return np.full(4,10.)+z-initial
            return final-z
        try:
            sol=root(residual,initial,method='hybr',options={'xtol':1e-10,'maxfev':80})
            final,rmax=flow(sol.x,f,step=128,rtol=1e-11,atol=1e-13)
            resid=float(np.linalg.norm(final-sol.x,np.inf))
            if resid>1e-7:raise RuntimeError(f'closure residual {resid}')
            A=jacobian(sol.x,f,1e-6)
            B=jacobian(sol.x,f,5e-7)
            eig=np.linalg.eigvals(B)
            result=dict(mesh_um=mesh,model=name,initial_ideal=initial.tolist(),state=sol.x.tolist(),
                        closure_residual=resid,max_radius=rmax,root_success=bool(sol.success),
                        evaluations=evals,spectral_radius=float(max(abs(eig))),
                        symplectic_defect=float(np.linalg.norm(B.T@J@B-J,np.inf)),
                        jacobian_refinement_inf=float(np.linalg.norm(A-B,np.inf)),
                        multipliers=[[float(v.real),float(v.imag)] for v in eig])
            print(mesh,name,'resid',resid,'rho',result['spectral_radius'],'defect',result['symplectic_defect'],flush=True)
        except Exception as exc:
            result=dict(mesh_um=mesh,model=name,initial_ideal=initial.tolist(),error=f'{type(exc).__name__}: {exc}',evaluations=evals)
            print(mesh,name,result['error'],flush=True)
        results.append(result)
        (OUT/'fem_periodic_orbits.json').write_text(json.dumps(results,indent=2)+'\n')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [fem.F4_PATH,fem.F2_PATH,fem.MESH_PATH]}
    (OUT/'fem_orbit_manifest.json').write_text(json.dumps(dict(solver='DOP853; shooting 2e-10/2e-12; verification 1e-11/1e-13; step 2pi/128',
                                                                  jacobian='central finite differences eps 1e-6,5e-7',
                                                                  note='hashes below correspond to final mesh; per-mesh sources are V13.3 saved fields',
                                                                  final_mesh_hashes=hashes),indent=2)+'\n')

if __name__=='__main__':main()
