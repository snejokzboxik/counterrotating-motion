"""Matched full 4D one-RF stroboscopic tracks near the A9 off-axis orbit."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_v13_3_fem_controls as fem
from run_v14_fem_orbits import make_rhs
from run_v14_ideal_pilot import OUT,ALPHA,BETA,KAPPA
from three_phase_trap.v14_dynamics import rhs

def main():
    orbit=json.loads((OUT/'ideal_periodic_orbits.json').read_text())
    center=np.array(next(r['state'] for r in orbit if r['model']=='RO' and r['state'][1]>0))
    starts={'orbit_RO_ideal':center,'near_RO_ideal':center+np.array([.02,0,0,0])}
    rows=[];meta=[]
    for field in ('ideal','fem_h50'):
      if field=='fem_h50':fem.initialize(50)
      for name,k in (('Q',0.),('SO',KAPPA),('RO',KAPPA)):
       for start_id,z0 in starts.items():
        if field=='ideal':
            fun=lambda t,z:rhs(t,z,ALPHA,BETA,k,name)
        else:fun=make_rhs(name,k,0.)
        def ev(t,z):return math.hypot(z[0],z[1])-1
        ev.direction=1;ev.terminal=True
        grid=2*math.pi*np.arange(301)
        sol=solve_ivp(fun,(0,300*2*math.pi),z0,t_eval=grid,method='DOP853',
                      rtol=2e-10,atol=2e-12,max_step=2*math.pi/64,events=ev)
        for n,z in zip(np.rint(sol.t/(2*math.pi)).astype(int),sol.y.T):
            rows.append(dict(field=field,model=name,start_id=start_id,n_rf=int(n),
                             x=z[0],y=z[1],vx=z[2],vy=z[3]))
        status='unresolved' if not sol.success else ('escaped' if sol.status==1 else 'retained')
        exitp=float(sol.t_events[0][0]/(2*math.pi)) if sol.status==1 else None
        meta.append(dict(field=field,model=name,start_id=start_id,status=status,
                         exit_periods=exitp,n_strobes=len(sol.t),initial=z0.tolist()))
        print(field,name,start_id,status,exitp,flush=True)
    with (OUT/'stroboscopic_states.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'stroboscopic_manifest.json').write_text(json.dumps(dict(horizon_rf=300,strobes_every_rf=1,
                     integrator='DOP853',rtol=2e-10,atol=2e-12,max_step_tau=2*math.pi/64,
                     runs=meta),indent=2)+'\n')

if __name__=='__main__':main()
