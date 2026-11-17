"""Additional full h50 P4 RO phase slices, paired to saved V13.3 0°/90°."""
from __future__ import annotations
import csv,json,math,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import t as tdist
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_v13_3_fem_controls as fem
from run_v14_fem_orbits import make_rhs
from run_v14_ideal_pilot import OUT,starts,SEEDS,KAPPA
from analyze_v14_ideal import write_csv

OLD0=ROOT/'outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas/acceptance_4d_fem_h50_selected_N100/trajectory_results.csv'
OLD90=ROOT/'outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas/acceptance_4d_fem_h50_delta90_N100/trajectory_results.csv'
ANGLES=(45,135,180)

def worker(task):
    deg,rep,pi,i,initial=task
    fun=make_rhs('RO',KAPPA,math.radians(deg),tau0=pi*math.pi/2)
    def ev(t,z):return math.hypot(z[0],z[1])-1
    ev.terminal=True;ev.direction=1
    try:
        sol=solve_ivp(fun,(0,200*math.pi),initial,method='DOP853',rtol=2e-10,atol=2e-12,
                      max_step=2*math.pi/64,events=ev)
        status='unresolved' if not sol.success else ('escaped' if sol.status==1 else 'retained')
        exitp=float(sol.t_events[0][0]/(2*math.pi)) if sol.status==1 else None
        final=sol.y[:,-1].tolist();error=None
    except Exception as exc:
        status='unresolved';exitp=None;final=None;error=f'{type(exc).__name__}:{exc}'
    return dict(key=f'{deg}|{rep}|{pi}|{i}',delta_deg=deg,replica=rep,phase_index=pi,state_i=i,
                status=status,exit_periods=exitp,initial=initial,final=final,error=error)

def old_rows(path,deg):
    data={}
    with path.open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['point_id']=='a9_b2_e0' and r['model']=='RO' and r['mu']=='1.0' and r['horizon']=='100' and int(r['state_i'])<16:
                key=f'{deg}|{r["replica"]}|{r["phase_i"]}|{r["state_i"]}'
                if key in data:raise RuntimeError('duplicate old phase key')
                data[key]=dict(key=key,delta_deg=deg,replica=int(r['replica']),phase_index=int(r['phase_i']),
                               state_i=int(r['state_i']),status=r['status'],exit_periods=r['exit_periods'],
                               initial=[float(r[k]) for k in ('x0_rho','y0_rho','vx0_rho_tau','vy0_rho_tau')])
    assert len(data)==256
    return data

def main():
    source={**old_rows(OLD0,0),**old_rows(OLD90,90)}
    for r in source.values():
        z=starts(SEEDS[r['replica']])[r['state_i']]
        if np.max(abs(z-r['initial']))>1e-13:raise RuntimeError('source mismatch')
    ck=OUT/'phase_fem_checkpoint.jsonl';done={}
    if ck.exists():
        for line in ck.read_text(encoding='utf8').splitlines():
            if line.strip():
                r=json.loads(line)
                if r['key'] in done:raise RuntimeError('duplicate FEM phase key')
                done[r['key']]=r
    tasks=[]
    for deg in ANGLES:
     for rep,seed in enumerate(SEEDS):
      init=starts(seed)
      for pi in range(4):
       for i in range(16):
        if f'{deg}|{rep}|{pi}|{i}' not in done:tasks.append((deg,rep,pi,i,init[i].tolist()))
    if tasks:
      with ProcessPoolExecutor(max_workers=3,initializer=fem.initialize,initargs=(50,)) as pool,ck.open('a',encoding='utf8') as f:
        futures=[pool.submit(worker,t) for t in tasks]
        for n,fu in enumerate(as_completed(futures),1):
            r=fu.result();done[r['key']]=r
            f.write(json.dumps(r,separators=(',',':'))+'\n');f.flush()
            if n%32==0:print('phase FEM',n,'/',len(tasks),flush=True)
    allrows={**source,**done};summary=[]
    for deg in (0,45,90,135,180):
        sample=[r for r in allrows.values() if r['delta_deg']==deg]
        assert len(sample)==256
        summary.append(dict(delta_deg=deg,retained=sum(r['status']=='retained' for r in sample),
                            escaped=sum(r['status']=='escaped' for r in sample),
                            unresolved=sum(r['status']=='unresolved' for r in sample),n=256,
                            P_ret=sum(r['status']=='retained' for r in sample)/256))
    write_csv(OUT/'phase_fem_summary.csv',summary)
    paired=[]
    for deg in (0,45,90,135,180):
        replicate=[]
        for rep in range(4):
            replicate.append(sum(int(allrows[f'{deg}|{rep}|{pi}|{i}']['status']=='retained')-
                                 int(allrows[f'0|{rep}|{pi}|{i}']['status']=='retained')
                                 for pi in range(4) for i in range(16))/64)
        mean=float(np.mean(replicate));half=float(tdist.ppf(.975,3)*np.std(replicate,ddof=1)/2)
        paired.append(dict(delta_deg=deg,delta_P_vs_0=mean,ci95_low=mean-half,ci95_high=mean+half,
                           replicate_differences=json.dumps(replicate)))
    write_csv(OUT/'phase_fem_paired_summary.csv',paired)
    print(summary)

if __name__=='__main__':main()
