from pathlib import Path
import json,math,argparse
import numpy as np
from scipy.integrate import solve_ivp
import attractor_search as a

OUT=a.OUT

def rhs_independent(t,y,p,phase,field,qratio,re1,re2,gz):
    dy=np.zeros(12);A,B,nu,beta,s,C,lam1,lam2=map(float,p)
    c1=A*np.exp(-1j*(t+phase[0]));c2=B*np.exp(-1j*(nu*t+phase[1]))
    d=y[:3]-y[6:9];d2=float(d@d);inv=0.0 if C==0 or d2<1e-28 else 1/(d2*math.sqrt(d2))
    for i,j in ((0,0),(1,6)):
        x,yy,z,vx,vy,vz=y[j:j+6];u=complex(x,yy);ex=ey=0.0
        for family,mode,coef in zip(field['family'],field['k'],field['coeff']):
            n=abs(int(mode));base=u if mode>0 else u.conjugate();carrier=c1 if family==0 else c2
            g=carrier*coef*n*base**(n-1);ex-=g.real;ey+=g.imag if mode>0 else -g.imag
        ef=1.0 if i==0 else s*qratio;mass_inv=1.0 if i==0 else s;lam=lam1 if i==0 else lam2
        chi=re1 if i==0 else re2;speed=math.sqrt(vx*vx+vy*vy+vz*vz);re=max(0,chi*speed)
        drag=lam*(1+0.15*re**0.687)
        sg=1.0 if i==0 else -1.0
        dy[j:j+3]=[vx,vy,vz]
        dy[j+3]=ef*(ex+beta*x)+mass_inv*sg*C*inv*d[0]-drag*vx
        dy[j+4]=ef*(ey+beta*yy)+mass_inv*sg*C*inv*d[1]-drag*vy
        dy[j+5]=ef*(-2*beta*z)+mass_inv*sg*C*inv*d[2]-drag*vz+gz
    return dy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',default='all',choices=['all','candidate52','Local15','refine12_sobol17']);args=ap.parse_args()
    cfg=a.read_json(OUT/'candidate_parameters.json');field=a.load_field();start=cfg['matched_common_start']
    periods=3000;skip=200;out=[]
    selected=[c for c in cfg['candidates'] if args.candidate=='all' or c['legacy_candidate']==args.candidate]
    for c in selected:
        p,q=a.arrays(c,field);y0=np.asarray(start['initial'],float);phase=np.asarray(start['phase'],float)
        ev=np.arange(0,periods+1,10,dtype=float);te=ev*2*math.pi
        sol=solve_ivp(lambda t,y:rhs_independent(t,y,p,phase,field,q,c['re1_per_speed'],c['re2_per_speed'],c['gravity_z']),
          (0,periods*2*math.pi),y0,method='DOP853',t_eval=te,rtol=1e-9,atol=1e-11,max_step=2*math.pi/96)
        if not sol.success:raise RuntimeError(sol.message)
        theta=np.unwrap(np.arctan2(sol.y[[1,7]],sol.y[[0,6]]))
        # Sampled output every 10 RF cycles; unwrap is safe at these small increments.
        late=(ev>=skip);rates=[];turns=[]
        edges=np.linspace(skip,periods,4)
        for k in range(2):
            tr=(theta[k,-1]-theta[k,np.flatnonzero(ev>=skip)[0]])/(2*math.pi);turns.append(float(tr));rates.append(float(tr*c['f1_Hz']/(periods-skip)))
        windows=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            ii=np.flatnonzero((ev>=lo)&(ev<=hi));windows.append([(theta[k,ii[-1]]-theta[k,ii[0]])/(2*math.pi) for k in range(2)])
        rf=OUT/f'dop853_{c["legacy_candidate"]}_3000p.npz'
        np.savez_compressed(rf,t_eval=sol.t,states=sol.y.T,initial=y0,phase=phase,p=p,qratio=q,rtol=1e-9,atol=1e-11,max_step=2*math.pi/96)
        # Compare to saved high-resolution RK4 baseline at the same exact 3000-period sample.
        rk=np.load(OUT/f'baseline_{c["id"]}_15000p_rk464.npz');period_index=rk['step_index']/64
        ki=int(np.argmin(abs(period_index-periods)))
        state_diff=float(np.max(abs(sol.y[:,-1]-rk['states'][ki])))
        row={'candidate':c['id'],'method':'independent Python RHS + SciPy DOP853','periods':periods,'rtol':1e-9,'atol':1e-11,
          'max_step_RF_periods':1/96,'nfev':int(sol.nfev),'turns_after_200p':turns,'frequency_Hz':rates,'window_turns':windows,
          'rk4_64_final_state_max_abs_difference_at_3000p':state_diff,'success':bool(sol.success)}
        out.append(row);print(f"{c['id']}: nfev={sol.nfev}, f={rates}, RK64 state maxdiff={state_diff:.3g}",flush=True)
    a.write_json(OUT/'independent_dop853_validation.json',{'rows':out,'field':'full 8-mode complex MFS','drag':'lambda_Stokes/Cc with Schiller-Naumann factor'})

if __name__=='__main__':main()
