"""RO relative-phase pilot and ideal periodic-orbit continuation at A9."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_v14_ideal_pilot import OUT,starts,integrate_batch,SEEDS,PHASES,ALPHA,BETA,KAPPA,M0,OMEGA,R,E,C2,C4
from analyze_v14_ideal import minima,shoot,write_csv

DEGREES=(0,45,90,135,180)

def peak(delta):
    scale=M0*OMEGA**2*R**2/E
    u2=BETA*scale/C2;dc=ALPHA*scale/C2;u4=KAPPA*scale/C4
    values=[]
    for j in range(12):
        th=2*math.pi*j/12;c=math.cos(4*th);s=math.sin(4*th);q=math.cos(2*th)
        values.append(abs(dc*q)+math.hypot(u2*q+u4*(c*math.cos(delta)-s*math.sin(delta)),
                                            u4*(c*math.sin(delta)+s*math.cos(delta))))
    return max(values)

def main():
    ck=OUT/'phase_ideal_checkpoint.jsonl';done={}
    if ck.exists():
        for line in ck.read_text(encoding='utf8').splitlines():
            if line.strip():
                row=json.loads(line); key=row['key']
                if key in done:raise RuntimeError('duplicate phase key')
                done[key]=row
    with ck.open('a',encoding='utf8') as f:
      for deg in DEGREES:
       delta=math.radians(deg)
       for rep,seed in enumerate(SEEDS):
        init=starts(seed)
        for pi,phase in enumerate(PHASES):
         if all(f'{deg}|{rep}|{pi}|{i}' in done for i in range(16)):continue
         final,status,exitp,rmax=integrate_batch(init,'RO',KAPPA,phase,delta)
         for i in range(16):
             key=f'{deg}|{rep}|{pi}|{i}'
             if key in done:continue
             row=dict(key=key,delta_deg=deg,replica=rep,seed=seed,phase_deg=90*pi,state_i=i,
                      status=str(status[i]),exit_periods=None if math.isnan(exitp[i]) else float(exitp[i]),
                      rho_max=float(rmax[i]),initial=init[i].tolist(),final=final[i].tolist())
             f.write(json.dumps(row,separators=(',',':'))+'\n');done[key]=row
         f.flush()
        print('phase pilot',deg,'rep',rep,flush=True)
    phase_rows=[]
    for deg in DEGREES:
        means=[]
        for rep in range(4):
            means.append(sum(done[f'{deg}|{rep}|{pi}|{i}']['status']=='retained' for pi in range(4) for i in range(16))/64)
        m=float(np.mean(means)); half=float(tdist.ppf(.975,3)*np.std(means,ddof=1)/2)
        phase_rows.append(dict(delta_deg=deg,P_ret=m,ci95_low=m-half,ci95_high=m+half,
                               n_retained=int(m*256),n_total=256,voltage_peak_V=peak(math.radians(deg)),
                               replicate_means=json.dumps(means)))
    write_csv(OUT/'phase_ideal_summary.csv',phase_rows)
    # Local search only: existing elliptic orbit continued from delta=0 and
    # independent seeds from each pseudopotential minimum.
    reference=json.loads((OUT/'ideal_periodic_orbits.json').read_text())
    previous=np.array(next(r['state'] for r in reference if r['model']=='RO' and r['state'][1]>0))
    orbits=[]
    for deg in DEGREES:
        delta=math.radians(deg)
        seeds=[previous]
        for v in minima('RO',KAPPA,delta):seeds.append(np.r_[v,0.,0.])
        found=[]
        for ix,seed in enumerate(seeds):
            try:r=shoot(seed,'RO',KAPPA,delta,1)
            except Exception:r=None
            if r is not None and not any(np.linalg.norm(np.array(r['state'])-np.array(s['state']))<1e-4 for s in found):
                r.update(delta_deg=deg,seed_index=ix);found.append(r)
        if found:previous=np.array(found[0]['state'])
        orbits.extend(found)
        print('phase orbit',deg,'found',len(found),flush=True)
    (OUT/'phase_ideal_orbits.json').write_text(json.dumps(orbits,indent=2)+'\n',encoding='utf8')
    print('phase rows',len(done),'orbits',len(orbits))

if __name__=='__main__':main()
