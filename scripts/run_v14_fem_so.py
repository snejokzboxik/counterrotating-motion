"""Checkpointed h50 P4 standing-octupole paired control; reuses V13.3 Q/RO."""
from __future__ import annotations
import csv,hashlib,json,math,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_v13_3_fem_controls as fem
from run_v14_ideal_pilot import OUT,ALPHA,BETA,KAPPA,starts,SEEDS,PHASES,M0,OMEGA,R,E,C2,C4

OLD=ROOT/'outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas/acceptance_4d_fem_h50_selected_N100/trajectory_results.csv'

def worker(task):
    name,rep,pi,i,initial=task
    k=KAPPA*(math.sqrt(2) if name=='SO_equal_rms' else 1.)
    scale=E/(M0*OMEGA**2*R)
    voltscale=M0*OMEGA**2*R**2/E
    u2=BETA*voltscale/C2;dc=ALPHA*voltscale/C2;u4=k*voltscale/C4
    phase=PHASES[pi]
    def rhs(t,z):
        qx,qy=fem._grad2(z[0],z[1]);cx,cy,_,_=fem._grad4(z[0],z[1])
        qdrive=u2*math.cos(t+phase)+dc; octdrive=u4*math.cos(t+phase)
        return np.array([z[2],z[3],-scale*(qdrive*qx+octdrive*cx),
                         -scale*(qdrive*qy+octdrive*cy)])
    def ev(t,z):return math.hypot(z[0],z[1])-1
    ev.direction=1;ev.terminal=True
    try:
        sol=solve_ivp(rhs,(0,200*math.pi),initial,method='DOP853',rtol=2e-10,atol=2e-12,
                      max_step=2*math.pi/64,events=ev)
        status='unresolved' if not sol.success else ('escaped' if sol.status==1 else 'retained')
        exitp=float(sol.t_events[0][0]/(2*math.pi)) if sol.status==1 else None
        final=sol.y[:,-1].tolist();rmax=float(np.max(np.hypot(sol.y[0],sol.y[1])))
        error=None
    except Exception as exc:
        status='unresolved';exitp=None;final=None;rmax=None;error=f'{type(exc).__name__}:{exc}'
    return dict(key=f'{name}|{rep}|{pi}|{i}',model=name,replica=rep,seed=SEEDS[rep],phase_index=pi,
                tau0_rad=phase,state_i=i,status=status,exit_periods=exitp,rho_max=rmax,
                initial=initial,final=final,error=error,kappa=k,U2_V=u2,VDC_V=dc,U4_V=u4)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source={}
    with OLD.open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['point_id']=='a9_b2_e0' and r['mu']=='1.0' and r['horizon']=='100' and int(r['state_i'])<16:
                key=(r['model'],int(r['replica']),int(r['phase_i']),int(r['state_i']))
                if key in source:raise RuntimeError('duplicate V13 source key')
                source[key]=r
    assert len(source)==512,len(source)
    for rep,seed in enumerate(SEEDS):
        initial=starts(seed)
        for pi in range(4):
            for i in range(16):
                for model in ('Q','RO'):
                    old=source[(model,rep,pi,i)]
                    z=np.array([float(old[k]) for k in ('x0_rho','y0_rho','vx0_rho_tau','vy0_rho_tau')])
                    if np.max(abs(z-initial[i]))>1e-13:raise RuntimeError('V13 source initial mismatch')
    mesh=50
    fem.configure_mesh(mesh)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [OLD,fem.F4_PATH,fem.F2_PATH,fem.MESH_PATH,Path(__file__)]}
    protocol=dict(source_hashes=hashes,source_rows=512,models=['SO_equal_harmonic','SO_equal_rms'],
                  phases=PHASES,seeds=SEEDS,states_per_replica=16,horizon_rf=100,
                  field='full h50 P4',solver=dict(method='DOP853',rtol=2e-10,atol=2e-12,max_step_tau=2*math.pi/64))
    run_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    (OUT/'fem_so_manifest.json').write_text(json.dumps({**protocol,'run_hash':run_hash},indent=2)+'\n')
    ck=OUT/'fem_so_checkpoint.jsonl';done={}
    if ck.exists():
        for line in ck.read_text(encoding='utf8').splitlines():
            if line.strip():
                r=json.loads(line)
                if r['run_hash']!=run_hash or r['key'] in done:raise RuntimeError('FEM SO checkpoint mismatch')
                done[r['key']]=r
    tasks=[]
    for name in protocol['models']:
      for rep,seed in enumerate(SEEDS):
       init=starts(seed)
       for pi in range(4):
        for i in range(16):
         if f'{name}|{rep}|{pi}|{i}' not in done:tasks.append((name,rep,pi,i,init[i].tolist()))
    if tasks:
      with ProcessPoolExecutor(max_workers=3,initializer=fem.initialize,initargs=(50,)) as pool,ck.open('a',encoding='utf8') as f:
        futures=[pool.submit(worker,t) for t in tasks]
        for j,future in enumerate(as_completed(futures),1):
            row=future.result();row['run_hash']=run_hash
            if row['key'] in done:raise RuntimeError('duplicate result')
            f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush();done[row['key']]=row
            if j%16==0:print('completed',j,'/',len(tasks),flush=True)
    print('FEM SO rows',len(done),'unresolved',sum(r['status']=='unresolved' for r in done.values()))

if __name__=='__main__':main()
