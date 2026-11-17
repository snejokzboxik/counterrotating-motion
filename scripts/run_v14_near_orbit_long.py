"""Resumable 10,000-RF h50 test of matched small perturbations near Q/SO/RO FEM orbits."""
from __future__ import annotations
import json,math,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_v13_3_fem_controls as fem
from run_v14_fem_orbits import OUT,make_rhs
from run_v14_ideal_pilot import KAPPA

MODELS=('Q','SO_equal_harmonic','RO')

def worker(task):
    model,start,stop,state=task
    bare='SO' if model.startswith('SO') else model
    k=0. if model=='Q' else KAPPA
    fun=make_rhs(bare,k,0.)
    def ev(t,z):return math.hypot(z[0],z[1])-1
    ev.direction=1;ev.terminal=True
    try:
        sol=solve_ivp(fun,(start*2*math.pi,stop*2*math.pi),state,method='DOP853',
                      rtol=2e-10,atol=2e-12,max_step=2*math.pi/64,events=ev)
        status='unresolved' if not sol.success else ('escaped' if sol.status==1 else 'retained')
        exitp=float(sol.t_events[0][0]/(2*math.pi)) if sol.status==1 else None
        final=sol.y[:,-1].tolist();rmax=float(np.max(np.hypot(sol.y[0],sol.y[1])));error=None
    except Exception as exc:
        status='unresolved';exitp=None;final=None;rmax=None;error=f'{type(exc).__name__}:{exc}'
    return dict(key=f'{model}|{stop}',model=model,start_horizon=start,horizon=stop,status=status,
                exit_periods=exitp,final=final,rho_max_segment=rmax,error=error)

def main():
    q=json.loads((OUT/'fem_q_orbit_search.json').read_text())
    oth=json.loads((OUT/'fem_periodic_orbits.json').read_text())
    orbit={'Q':np.array(next(r['state'] for r in q if r['mesh_um']==50))}
    for m in ('SO_equal_harmonic','RO'):
        orbit[m]=np.array(next(r['state'] for r in oth if r['mesh_um']==50 and r['model']==m))
    starts={m:(z+np.array([.01,0,0,0])).tolist() for m,z in orbit.items()}
    (OUT/'near_orbit_long_manifest.json').write_text(json.dumps(dict(field='h50 P4',mu=1,delta_rad=0,
        starts=starts,perturbation='x/R + 0.01 from separately found h50 FEM 1-RF orbit',
        solver=dict(method='DOP853',rtol=2e-10,atol=2e-12,max_step_tau=2*math.pi/64),
        horizons_rf=list(range(1000,10001,1000))),indent=2)+'\n')
    ck=OUT/'near_orbit_long_checkpoint.jsonl';done={}
    if ck.exists():
        for line in ck.read_text(encoding='utf8').splitlines():
            if line.strip():
                r=json.loads(line)
                if r['key'] in done:raise RuntimeError('duplicate near-orbit key')
                done[r['key']]=r
    for stop in range(1000,10001,1000):
        start=stop-1000;tasks=[]
        for model in MODELS:
            key=f'{model}|{stop}'
            if key in done:continue
            prev=done.get(f'{model}|{start}') if start else None
            if prev and prev['status']!='retained':
                r=dict(key=key,model=model,start_horizon=start,horizon=stop,status=prev['status'],
                       exit_periods=prev['exit_periods'],final=prev['final'],
                       rho_max_segment=None,carried=True,error=None)
                done[key]=r
                with ck.open('a',encoding='utf8') as f:f.write(json.dumps(r)+'\n')
            else:tasks.append((model,start,stop,starts[model] if prev is None else prev['final']))
        if tasks:
          with ProcessPoolExecutor(max_workers=3,initializer=fem.initialize,initargs=(50,)) as pool,ck.open('a',encoding='utf8') as f:
            futures=[pool.submit(worker,task) for task in tasks]
            for fu in as_completed(futures):
                r=fu.result();done[r['key']]=r
                f.write(json.dumps(r,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
        print('near orbit',stop,[(m,done[f'{m}|{stop}']['status'],done[f'{m}|{stop}']['rho_max_segment']) for m in MODELS],flush=True)

if __name__=='__main__':main()
