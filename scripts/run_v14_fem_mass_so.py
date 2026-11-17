"""Paired h50 standing-octupole controls at mu=.70/.85, reusing V13.3 Q/RO."""
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
from run_v14_ideal_pilot import OUT,ALPHA,BETA,KAPPA,M0,OMEGA,R,E,C2,C4
from analyze_v14_ideal import write_csv

BASE=ROOT/'outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas'
SOURCES={.70:BASE/'acceptance_4d_fem_h50_a9_adaptive_N100/trajectory_results.csv',
         .85:BASE/'acceptance_4d_fem_h50_a9_mass_edges_N100/trajectory_results.csv'}

def worker(task):
    mu,rep,pi,i,initial=task
    scale=E/(M0*mu*OMEGA**2*R);voltscale=M0*OMEGA**2*R**2/E
    u2=BETA*voltscale/C2;dc=ALPHA*voltscale/C2;u4=KAPPA*voltscale/C4
    phase=pi*math.pi/2
    def fun(t,z):
        qx,qy=fem._grad2(z[0],z[1]);cx,cy,_,_=fem._grad4(z[0],z[1])
        c=math.cos(t+phase)
        return np.array([z[2],z[3],-scale*((u2*c+dc)*qx+u4*c*cx),
                         -scale*((u2*c+dc)*qy+u4*c*cy)])
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
    return dict(key=f'{mu}|{rep}|{pi}|{i}',mu=mu,model='SO_equal_harmonic',replica=rep,
                phase_index=pi,state_i=i,status=status,exit_periods=exitp,initial=initial,final=final,error=error)

def main():
    source={}
    for mu,path in SOURCES.items():
      with path.open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['point_id']=='a9_b2_e0' and float(r['mu'])==mu and r['horizon']=='100' and int(r['state_i'])<16:
                key=(mu,r['model'],int(r['replica']),int(r['phase_i']),int(r['state_i']))
                if key in source:raise RuntimeError('duplicate mass source')
                source[key]=r
    assert len(source)==1024,len(source)
    ck=OUT/'fem_mass_so_checkpoint.jsonl';done={}
    if ck.exists():
      for line in ck.read_text(encoding='utf8').splitlines():
        if line.strip():
            r=json.loads(line)
            if r['key'] in done:raise RuntimeError('duplicate mass SO key')
            done[r['key']]=r
    tasks=[]
    for mu in SOURCES:
      for rep in range(4):
       for pi in range(4):
        for i in range(16):
            q=source[(mu,'Q',rep,pi,i)];ro=source[(mu,'RO',rep,pi,i)]
            fields=('x0_rho','y0_rho','vx0_rho_tau','vy0_rho_tau')
            initial=[float(q[k]) for k in fields]
            if max(abs(float(ro[k])-initial[j]) for j,k in enumerate(fields))>1e-13:raise RuntimeError('source Q/RO mismatch')
            if f'{mu}|{rep}|{pi}|{i}' not in done:tasks.append((mu,rep,pi,i,initial))
    if tasks:
      with ProcessPoolExecutor(max_workers=3,initializer=fem.initialize,initargs=(50,)) as pool,ck.open('a',encoding='utf8') as f:
        futures=[pool.submit(worker,t) for t in tasks]
        for n,fu in enumerate(as_completed(futures),1):
            r=fu.result();done[r['key']]=r;f.write(json.dumps(r,separators=(',',':'))+'\n');f.flush()
            if n%32==0:print('mass SO',n,'/',len(tasks),flush=True)
    summary=[]
    for mu in SOURCES:
      volume=167.38913929608813/mu
      norm=R**4*OMEGA**2
      for model in ('Q','SO_equal_harmonic','RO'):
        def rget(rep,pi,i):return done[f'{mu}|{rep}|{pi}|{i}'] if model.startswith('SO') else source[(mu,model,rep,pi,i)]
        means=[sum(rget(rep,pi,i)['status']=='retained' for pi in range(4) for i in range(16))/64 for rep in range(4)]
        m=float(np.mean(means));half=float(tdist.ppf(.975,3)*np.std(means,ddof=1)/2)
        p_low=max(0.,m-half);p_high=min(1.,m+half)
        summary.append(dict(mu=mu,model=model,P_ret=m,ci95_low=p_low,ci95_high=p_high,
                            D_volume_m4_s2=volume,A4_m4_s2=m*volume,
                            A4_ci95_low=p_low*volume,A4_ci95_high=p_high*volume,
                            a4=(m*volume)/norm,
                            retained=int(round(m*256)),n=256,replicate_means=json.dumps(means)))
    write_csv(OUT/'fem_mass_so_summary.csv',summary)
    paired=[]
    for mu in SOURCES:
      for a,b in (('RO','Q'),('RO','SO_equal_harmonic'),('SO_equal_harmonic','Q')):
        def status(model,rep,pi,i):
            return done[f'{mu}|{rep}|{pi}|{i}']['status'] if model.startswith('SO') else source[(mu,model,rep,pi,i)]['status']
        diffs=[sum(int(status(a,rep,pi,i)=='retained')-int(status(b,rep,pi,i)=='retained')
                   for pi in range(4) for i in range(16))/64 for rep in range(4)]
        m=float(np.mean(diffs));half=float(tdist.ppf(.975,3)*np.std(diffs,ddof=1)/2)
        volume=167.38913929608813/mu;norm=R**4*OMEGA**2
        paired.append(dict(mu=mu,comparison=a+'-minus-'+b,delta_P=m,ci95_low=m-half,
                           ci95_high=m+half,delta_A4_m4_s2=m*volume,
                           delta_A4_ci95_low=(m-half)*volume,delta_A4_ci95_high=(m+half)*volume,
                           delta_a4=(m*volume)/norm,replicate_differences=json.dumps(diffs)))
    write_csv(OUT/'fem_mass_so_paired.csv',paired)
    print(summary)

if __name__=='__main__':main()
