"""Checkpointed paired V14 Q/SO/RO 4D pilot at the A9 fixed-voltage point."""
from __future__ import annotations

import csv, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from three_phase_trap.v14_dynamics import rhs,octupole_rms_field_factor

OUT=ROOT/'outputs/octupole_comparison_home/mass_filter_v14'
E=1.602176634e-19; AMU=1.66053906660e-27
M0=3200.04652261404*AMU; OMEGA=2*math.pi*100_000.; R=.008
C2=.7718255558274212; C4=.5413395699439091
ALPHA=.00625; BETA=.075; KAPPA=-.015
SEEDS=(513217,883391,20261114,941029)
PHASES=(0.,math.pi/2,math.pi,3*math.pi/2)
NSTEPS=64*100; H=2*math.pi/64


def starts(seed,n=16):
    u=qmc.Sobol(4,scramble=True,seed=seed).random_base2(int(math.log2(n)))
    vmax=math.sqrt(10*E/M0)
    radius=.0075*np.sqrt(u[:,0]); theta=2*math.pi*u[:,1]
    speed=vmax*np.sqrt(u[:,2]); veltheta=2*math.pi*u[:,3]
    return np.stack((radius*np.cos(theta)/R,radius*np.sin(theta)/R,
                     speed*np.cos(veltheta)/(R*OMEGA),speed*np.sin(veltheta)/(R*OMEGA)),axis=1)


def integrate_batch(initial,model,kappa,tau0,delta=0.):
    state=initial.copy(); status=np.array(['retained']*len(state),dtype=object)
    exits=np.full(len(state),np.nan); rmax=np.linalg.norm(state[:,:2],axis=1)
    alive=rmax<1
    for n in range(NSTEPS):
        if not np.any(alive):break
        t=n*H; i=np.flatnonzero(alive); z=state[i]
        k1=rhs(t,z,ALPHA,BETA,kappa,model,delta=delta,tau0=tau0)
        k2=rhs(t+H/2,z+H*k1/2,ALPHA,BETA,kappa,model,delta=delta,tau0=tau0)
        k3=rhs(t+H/2,z+H*k2/2,ALPHA,BETA,kappa,model,delta=delta,tau0=tau0)
        k4=rhs(t+H,z+H*k3,ALPHA,BETA,kappa,model,delta=delta,tau0=tau0)
        znew=z+H*(k1+2*k2+2*k3+k4)/6
        oldr=np.linalg.norm(z[:,:2],axis=1); newr=np.linalg.norm(znew[:,:2],axis=1)
        rmax[i]=np.maximum(rmax[i],newr)
        escaped=(newr>=1)|(~np.isfinite(newr))
        if np.any(escaped):
            j=i[escaped]; frac=np.clip((1-oldr[escaped])/(newr[escaped]-oldr[escaped]),0,1)
            exits[j]=(n+frac)/64
            status[j]=np.where(np.isfinite(newr[escaped]),'escaped','unresolved')
            alive[j]=False
        state[i]=znew
    return state,status,exits,rmax


def voltage_table():
    scale=M0*OMEGA**2*R**2/E
    u2=BETA*scale/C2; dc=ALPHA*scale/C2; u4=KAPPA*scale/C4
    rows=[]
    for j in range(12):
        th=2*math.pi*j/12;c4=math.cos(4*th);s4=math.sin(4*th);q2=math.cos(2*th)
        for model,k in [('Q',0.),('RO',u4),('SO_equal_harmonic',u4),('SO_equal_rms',math.sqrt(2)*u4)]:
            if model=='RO':peak=abs(dc*q2)+math.hypot(u2*q2+k*c4,k*s4)
            else:peak=abs(dc*q2)+math.hypot(u2*q2+k*c4,0.)
            rows.append(dict(model=model,electrode=j,theta_deg=30*j,U2_V=u2,VDC_V=dc,U4_V=k,
                             peak_abs_V=peak,within_250V=peak<=250))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(alpha=ALPHA,beta=BETA,kappa=KAPPA,mu=1,frequency_hz=100_000,R_m=R,
                  physical_domain='uniform area r<=7.5 mm and transverse E<=5 eV',
                  seeds=SEEDS,phases=PHASES,states_per_replica=16,horizon_rf=100,
                  integrator='vectorized fixed-step RK4',step_tau=H,
                  models={'Q':0,'RO':KAPPA,'SO_equal_harmonic':KAPPA,'SO_equal_rms':math.sqrt(2)*KAPPA},
                  rms_factors={name:octupole_rms_field_factor('SO' if name.startswith('SO') else name,k)
                               for name,k in [('Q',0),('RO',KAPPA),('SO_equal_harmonic',KAPPA),('SO_equal_rms',math.sqrt(2)*KAPPA)]},
                  source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__),ROOT/'src/three_phase_trap/v14_dynamics.py']})
    h=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    manifest=OUT/'ideal_pilot_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf8'))['run_hash']!=h:
        raise RuntimeError('V14 pilot protocol mismatch')
    manifest.write_text(json.dumps({**protocol,'run_hash':h},indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    ck=OUT/'ideal_pilot_checkpoint.jsonl';done={}
    if ck.exists():
        for line in ck.read_text(encoding='utf8').splitlines():
            if line.strip():
                row=json.loads(line)
                if row['run_hash']!=h or row['key'] in done:raise RuntimeError('checkpoint mismatch or duplicate')
                done[row['key']]=row
    volt=voltage_table()
    with (OUT/'electrode_voltage.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=volt[0]);w.writeheader();w.writerows(volt)
    with ck.open('a',encoding='utf8') as f:
      for rep,seed in enumerate(SEEDS):
       init=starts(seed)
       for pi,phase in enumerate(PHASES):
        for name,k in protocol['models'].items():
         model='SO' if name.startswith('SO') else name
         missing=[i for i in range(len(init)) if f'{rep}|{pi}|{name}|{i}' not in done]
         if not missing:continue
         final,status,exitp,rmax=integrate_batch(init,model,k,phase)
         for i in missing:
          key=f'{rep}|{pi}|{name}|{i}'
          row=dict(key=key,run_hash=h,replica=rep,seed=seed,phase_index=pi,tau0_rad=phase,
                   model=name,alpha=ALPHA,beta=BETA,kappa=k,status=str(status[i]),
                   exit_periods=None if math.isnan(exitp[i]) else float(exitp[i]),
                   rho_max=float(rmax[i]),initial=init[i].tolist(),final=final[i].tolist())
          f.write(json.dumps(row,separators=(',',':'))+'\n');done[key]=row
         f.flush();os.fsync(f.fileno())
         print(rep,pi,name,'retained',sum(status=='retained'),'/',len(status),flush=True)
    rows=list(done.values())
    with (OUT/'ideal_pilot_trajectories.csv').open('w',newline='',encoding='utf8') as f:
        cols=['key','replica','seed','phase_index','tau0_rad','model','alpha','beta','kappa','status','exit_periods','rho_max','x0','y0','vx0','vy0','x100','y100','vx100','vy100']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in rows:
            w.writerow({**{c:r.get(c) for c in cols},**dict(zip(cols[12:16],r['initial'])),
                        **dict(zip(cols[16:20],r['final']))})
    print('rows',len(rows),'unresolved',sum(r['status']=='unresolved' for r in rows))

if __name__=='__main__':main()
