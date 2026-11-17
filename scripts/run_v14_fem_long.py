"""Continue matched h50 A9 survivors without resetting the RF clock."""
from __future__ import annotations
import csv,json,math,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_v13_3_fem_controls as fem
from run_v14_fem_orbits import make_rhs
from run_v14_fem_so import OUT,OLD
from run_v14_ideal_pilot import KAPPA

HORIZONS=(100,300,1000)
MODELS=('Q','RO','SO_equal_harmonic','SO_equal_rms')

def worker(task):
    model,rep,phase_i,i,start,stop,state=task
    k=0 if model=='Q' else KAPPA*(math.sqrt(2) if model=='SO_equal_rms' else 1)
    bare='SO' if model.startswith('SO') else model
    fun=make_rhs(bare,k,0.,tau0=phase_i*math.pi/2)
    def ev(t,z):return math.hypot(z[0],z[1])-1
    ev.terminal=True;ev.direction=1
    try:
        sol=solve_ivp(fun,(2*math.pi*start,2*math.pi*stop),state,
                      method='DOP853',rtol=2e-10,atol=2e-12,max_step=2*math.pi/64,events=ev)
        status='unresolved' if not sol.success else ('escaped' if sol.status==1 else 'retained')
        exitp=float(sol.t_events[0][0]/(2*math.pi)) if sol.status==1 else None
        final=sol.y[:,-1].tolist();rmax=float(np.max(np.hypot(sol.y[0],sol.y[1])))
        error=None
    except Exception as exc:
        status='unresolved';exitp=None;final=None;rmax=None;error=f'{type(exc).__name__}:{exc}'
    return dict(key=f'{model}|{rep}|{phase_i}|{i}|{stop}',model=model,replica=rep,
                phase_index=phase_i,state_i=i,start_horizon=start,horizon=stop,
                status=status,exit_periods=exitp,final=final,rho_max_segment=rmax,error=error)

def main():
    baseline={}
    with OLD.open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['point_id']=='a9_b2_e0' and r['mu']=='1.0' and r['horizon']=='100' and int(r['state_i'])<16:
                key=(r['model'],int(r['replica']),int(r['phase_i']),int(r['state_i']))
                baseline[key]=dict(status=r['status'],exit_periods=None if not r['exit_periods'] else float(r['exit_periods']),
                                   final=json.loads(r['final_state']))
    for line in (OUT/'fem_so_checkpoint.jsonl').read_text().splitlines():
        if line.strip():
            r=json.loads(line);baseline[(r['model'],r['replica'],r['phase_index'],r['state_i'])]=r
    assert len(baseline)==1024
    ck=OUT/'fem_long_checkpoint.jsonl';done={}
    if ck.exists():
        for line in ck.read_text(encoding='utf8').splitlines():
            if line.strip():
                r=json.loads(line)
                if r['key'] in done:raise RuntimeError('duplicate long key')
                done[r['key']]=r
    for start,stop in zip(HORIZONS[:-1],HORIZONS[1:]):
      tasks=[]
      for model in MODELS:
       for rep in range(4):
        for phase in range(4):
         for i in range(16):
            key=f'{model}|{rep}|{phase}|{i}|{stop}'
            if key in done:continue
            previous=baseline[(model,rep,phase,i)] if start==100 else done[f'{model}|{rep}|{phase}|{i}|{start}']
            if previous['status']!='retained':
                row=dict(key=key,model=model,replica=rep,phase_index=phase,state_i=i,
                         start_horizon=start,horizon=stop,status=previous['status'],
                         exit_periods=previous['exit_periods'],final=previous['final'],
                         rho_max_segment=None,carried=True,error=None)
                done[key]=row
            else:tasks.append((model,rep,phase,i,start,stop,previous['final']))
      persisted=existing_keys(ck)
      with ck.open('a',encoding='utf8') as f:
        for row in [r for r in done.values() if r.get('horizon')==stop and r.get('carried') and r['key'] not in persisted]:
            f.write(json.dumps(row,separators=(',',':'))+'\n');persisted.add(row['key'])
        f.flush()
        if tasks:
          with ProcessPoolExecutor(max_workers=3,initializer=fem.initialize,initargs=(50,)) as pool:
            futures=[pool.submit(worker,t) for t in tasks]
            for n,fu in enumerate(as_completed(futures),1):
                row=fu.result();done[row['key']]=row
                f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
                if n%8==0:print(start,stop,'completed',n,'/',len(tasks),flush=True)
      print('horizon',stop,'trajectories integrated',len(tasks),flush=True)
    for horizon in HORIZONS:
        print('H',horizon)
        for model in MODELS:
            if horizon==100:rows=[v for k,v in baseline.items() if k[0]==model]
            else:rows=[v for v in done.values() if v['horizon']==horizon and v['model']==model]
            print(model,sum(r['status']=='retained' for r in rows),'/',len(rows),flush=True)

def existing_keys(path):
    if not path.exists():return set()
    return {json.loads(s)['key'] for s in path.read_text().splitlines() if s.strip()}

if __name__=='__main__':main()
