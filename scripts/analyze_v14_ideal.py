"""V14 analytical minima, paired pilot summary and bounded periodic-orbit shooting."""
from __future__ import annotations

import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import t as t_dist

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from three_phase_trap.v14_dynamics import acceleration_and_jacobian,effective_potential,rhs
from run_v14_ideal_pilot import OUT,ALPHA,BETA,KAPPA

J=np.block([[np.zeros((2,2)),np.eye(2)],[-np.eye(2),np.zeros((2,2))]])


def write_csv(path,rows):
    if not rows:return
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)


def summary(rows):
    data={(r['replica'],r['phase_index'],r['model'],r['key'].split('|')[-1]):r for r in rows}
    out=[]
    for phase in range(4):
      for model in ('RO','SO_equal_harmonic','SO_equal_rms'):
       differences=[];means=[]
       for rep in range(4):
        x=[];y=[]
        for i in range(16):
            x.append(data[(rep,phase,'RO',str(i))]['status']=='retained')
            y.append(data[(rep,phase,model,str(i))]['status']=='retained')
        means.append(sum(y)/16)
        differences.append((sum(x)-sum(y))/16)
       center=float(np.mean(differences)); half=float(t_dist.ppf(.975,3)*np.std(differences,ddof=1)/2)
       out.append(dict(phase_deg=90*phase,comparison='RO-minus-'+model,
                       P_RO=sum(data[(rep,phase,'RO',str(i))]['status']=='retained' for rep in range(4) for i in range(16))/64,
                       P_control=float(np.mean(means)),delta=center,ci95_low=center-half,ci95_high=center+half,
                       replicate_differences=json.dumps(differences)))
    return out


def minima(model,kappa,delta):
    found=[]
    def f(v):return effective_potential(v[0],v[1],ALPHA,BETA,kappa,model,delta)
    for x in np.linspace(-.6,.6,5):
      for y in np.linspace(-.6,.6,5):
        res=minimize(f,[x,y],method='BFGS',options={'gtol':1e-11,'maxiter':400})
        v=res.x
        if np.linalg.norm(v)>=.95 or any(np.linalg.norm(v-u)<1e-4 for u in found):continue
        h=1e-4; eye=np.eye(2); H=np.zeros((2,2))
        for i in range(2):
          for j in range(2):
            H[i,j]=(f(v+h*eye[i]+h*eye[j])-f(v+h*eye[i]-h*eye[j])-f(v-h*eye[i]+h*eye[j])+f(v-h*eye[i]-h*eye[j]))/(4*h*h)
        ev=np.linalg.eigvalsh(H)
        if min(ev)>1e-7:found.append(v)
    return sorted(found,key=lambda v:(v[0],v[1]))


def flow(initial,model,kappa,delta,periods,step_divisor=64,rtol=1e-11,atol=1e-13):
    z0=np.asarray(initial,float); init=np.r_[z0,np.eye(4).ravel()]
    def vari(t,z):
        force,d=acceleration_and_jacobian(z,t,ALPHA,BETA,kappa,model,delta)
        A=np.block([[np.zeros((2,2)),np.eye(2)],[d,np.zeros((2,2))]])
        return np.r_[z[2:4],force,(A@z[4:].reshape(4,4)).ravel()]
    sol=solve_ivp(vari,(0,2*math.pi*periods),init,method='DOP853',rtol=rtol,atol=atol,
                  max_step=2*math.pi/step_divisor)
    if not sol.success:raise RuntimeError(sol.message)
    radius=float(np.max(np.hypot(sol.y[0],sol.y[1])))
    return sol.y[:4,-1],sol.y[4:,-1].reshape(4,4),radius


def shoot(seed,model,kappa,delta,periods):
    state=np.array(seed,float)
    for _ in range(12):
      final,M,rmax=flow(state,model,kappa,delta,periods)
      residual=final-state
      if np.linalg.norm(residual,np.inf)<2e-9:
        if rmax>=1 or np.linalg.norm(state[:2])<.03:return None
        defect=float(np.linalg.norm(M.T@J@M-J,np.inf))
        eig=np.linalg.eigvals(M)
        return dict(state=state.tolist(),residual=float(np.linalg.norm(residual,np.inf)),
                    radius_max=rmax,symplectic_defect=defect,
                    spectral_radius=float(max(abs(eig))),
                    multipliers=[[float(e.real),float(e.imag)] for e in eig])
      try:step=np.linalg.solve(M-np.eye(4),-residual)
      except np.linalg.LinAlgError:return None
      if not np.all(np.isfinite(step)):return None
      accepted=False
      for fraction in (1.,.5,.25,.125,.0625):
        trial=state+fraction*step
        if np.linalg.norm(trial[:2])>.9 or np.linalg.norm(trial[2:])>1.5:continue
        try: f,_,rm=flow(trial,model,kappa,delta,periods)
        except Exception:continue
        if rm<1.2 and np.linalg.norm(f-trial,np.inf)<np.linalg.norm(residual,np.inf):
          state=trial;accepted=True;break
      if not accepted:return None
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--periods',type=int,nargs='+',default=[1]);args=p.parse_args()
    rows=[json.loads(line) for line in (OUT/'ideal_pilot_checkpoint.jsonl').read_text().splitlines() if line.strip()]
    write_csv(OUT/'ideal_pilot_paired_summary.csv',summary(rows))
    models=[('Q',0.),('RO',KAPPA),('SO_equal_harmonic',KAPPA),('SO_equal_rms',math.sqrt(2)*KAPPA)]
    minima_rows=[];minima_by={}
    for name,k in models:
      model='SO' if name.startswith('SO') else name
      vv=minima(model,k,0.);minima_by[name]=vv
      for v in vv:minima_rows.append(dict(model=name,x=float(v[0]),y=float(v[1]),
                                         value=float(effective_potential(*v,ALPHA,BETA,k,model))))
    write_csv(OUT/'effective_minima.csv',minima_rows)
    orbitrows=[]
    for periods in args.periods:
      for name,k in models:
        model='SO' if name.startswith('SO') else name
        seeds=[]
        for v in minima_by[name]:
          seed=np.r_[v,[0.,0.]]
          # Leading micromotion shift and velocity at RF phase zero.
          f0,_=acceleration_and_jacobian(seed,0,0.,BETA,k,model)
          f90,_=acceleration_and_jacobian(seed,math.pi/2,0.,BETA,k,model)
          seeds.append(np.r_[v-f0,-f90])
          seeds.append(seed)
        candidates=[r for r in rows if r['model']==name and r['phase_index']==0 and r['status']=='retained']
        seeds.extend(np.array(r['final']) for r in candidates[:6])
        if name=='Q':seeds.append(np.array([.1,.1,0.,0.]))
        found=[]
        for ix,seed in enumerate(seeds):
          try: result=shoot(seed,model,k,0.,periods)
          except Exception:result=None
          if result is None or any(np.linalg.norm(np.array(result['state'])-np.array(old['state']))<1e-4 for old in found):continue
          result.update(model=name,periods=periods,seed_index=ix)
          found.append(result)
          print('orbit',name,periods,result['state'],result['spectral_radius'],flush=True)
        orbitrows.extend(found)
        print('searched',name,periods,len(seeds),'found',len(found),flush=True)
    (OUT/'ideal_periodic_orbits.json').write_text(json.dumps(orbitrows,indent=2)+'\n',encoding='utf8')
    print('summary_rows',len(rows),'minima',len(minima_rows),'orbits',len(orbitrows))

if __name__=='__main__':main()
