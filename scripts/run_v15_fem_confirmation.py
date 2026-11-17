"""Matched DOP853 h50/h35 FEM confirmation of A9 chirality and torque pilot."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
import run_v13_3_fem_controls as fem
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v15/fem_confirmation_a9"
CONFIRM=ROOT/"outputs/octupole_comparison_home/mass_filter_v15/confirmation_a9_uniform4d"
R=.008; OMEGA=2*math.pi*100_000.; E=1.602176634e-19; AMU=1.66053906660e-27
M0=3200.04652261404*AMU; C2=D2=.7718255558274212; C4=.5413395699439091
ALPHA,BETA,KAPPA=.00625,.075,-.015
MU=1.0
OCTUPOLE_SCALE=1.0
PHASES=(0.,math.pi/2); DELTAS=(0.,math.pi/2)
CONFIGS=(('Q','Q',0.),('SO_equal_harmonic','SO',1.),('SO_equal_rms','SO',math.sqrt(2)),('RO_PLUS','RO_PLUS',1.),('RO_MINUS','RO_MINUS',1.))
RTOL,ATOL,MAX_STEP=3e-9,3e-11,2*math.pi/64

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def voltage_peak(config,delta,mult):
    scale=M0*OMEGA**2*R**2/E; u2=BETA*scale/C2; dc=ALPHA*scale/D2; u4=KAPPA*mult*OCTUPOLE_SCALE*scale/C4
    peaks=[]
    for j in range(12):
      th=math.pi*j/6;q2=math.cos(2*th);c4=math.cos(4*th);s4=math.sin(4*th)
      if config=='Q':a=u2*q2;b=0.
      elif config.startswith('SO'):
       a=u2*q2+u4*c4*math.cos(delta);b=u4*c4*math.sin(delta)
      else:
       sig=1 if config=='RO_PLUS' else -1
       a=u2*q2+u4*(c4*math.cos(delta)-sig*s4*math.sin(delta))
       b=u4*(c4*math.sin(delta)+sig*s4*math.cos(delta))
      peaks.append(abs(dc*q2)+math.hypot(a,b))
    return max(peaks)

def integrate_one(y0,model,kappa,tau0,delta):
    u2=BETA*(M0*OMEGA**2*R**2/E)/C2;dc=ALPHA*(M0*OMEGA**2*R**2/E)/D2;u4=kappa*(M0*OMEGA**2*R**2/E)/C4
    scale=E/(M0*MU*OMEGA**2*R)
    def acceleration(t,state):
      x,y=state[:2]
      g4cx,g4cy,g4sx,g4sy=fem._grad4(x,y)
      gx2,gy2=fem._grad2(x,y)
      phase=t+tau0-delta
      if model=='Q': gx4=gy4=0.
      elif model=='SO': gx4=g4cx*math.cos(phase);gy4=g4cy*math.cos(phase)
      elif model=='RO_PLUS': gx4=g4cx*math.cos(phase)+g4sx*math.sin(phase);gy4=g4cy*math.cos(phase)+g4sy*math.sin(phase)
      elif model=='RO_MINUS': gx4=g4cx*math.cos(phase)-g4sx*math.sin(phase);gy4=g4cy*math.cos(phase)-g4sy*math.sin(phase)
      else: raise ValueError(model)
      qdrive=u2*math.cos(t+tau0)+dc
      ax=-scale*(u4*gx4+qdrive*gx2);ay=-scale*(u4*gy4+qdrive*gy2)
      torque=x*ay-y*ax
      return ax,ay,torque
    def rhs(t,z):
      ax,ay,torque=acceleration(t,z)
      return np.array([z[2],z[3],ax,ay,torque])
    def event(t,z):return math.hypot(z[0],z[1])-1.
    event.terminal=True;event.direction=1.
    def segment(state,t0,t1):
      sol=solve_ivp(rhs,(t0,t1),state,method='DOP853',rtol=RTOL,atol=ATOL,max_step=MAX_STEP,events=event,dense_output=True)
      if not sol.success:status='unresolved';exitp=None
      elif len(sol.t_events[0]):status='escaped';exitp=float(sol.t_events[0][0]/(2*math.pi))
      else:status='retained';exitp=None
      rho=np.hypot(sol.y[0],sol.y[1]);rmax=float(np.max(rho))
      if len(sol.t)>1:
       mids=sol.sol((sol.t[:-1]+sol.t[1:])/2);rmax=max(rmax,float(np.max(np.hypot(mids[0],mids[1]))))
      if status=='escaped':rmax=max(1.,rmax)
      return sol,status,exitp,rmax
    state=np.r_[np.asarray(y0,float),0.]
    start_l=float(state[0]*state[3]-state[1]*state[2])
    a,s100,e100,r100=segment(state,0.,2*math.pi*100)
    final100=a.y[:,-1]
    l100=float(final100[0]*final100[3]-final100[1]*final100[2])
    res100=float((l100-start_l)-final100[4])
    out=[(100,s100,e100,r100,final100,l100,float(final100[4]),res100,int(a.nfev))]
    if s100=='retained':
      b,s300,e300,r300=segment(final100,2*math.pi*100,2*math.pi*300)
      final300=b.y[:,-1];l300=float(final300[0]*final300[3]-final300[1]*final300[2]);r300=max(r100,r300)
      res300=float((l300-start_l)-final300[4])
      out.append((300,s300,e300,r300,final300,l300,float(final300[4]),res300,int(a.nfev+b.nfev)))
    else:
      out.append((300,s100,e100,r100,final100,l100,float(final100[4]),res100,int(a.nfev)))
    return out

def init_worker(mesh,mu,octupole_scale):
    """Initialize worker-local FEM tables and physical mass scaling.

    Windows uses spawn for ProcessPoolExecutor, so globals assigned in main()
    do not propagate to children unless included in initializer arguments.
    """
    global MU,OCTUPOLE_SCALE
    MU=float(mu)
    OCTUPOLE_SCALE=float(octupole_scale)
    fem.initialize(mesh)
def worker(task):
    mesh,config,model,mult,delta,tau0,starts=task
    kp=KAPPA*mult*OCTUPOLE_SCALE;peak=voltage_peak(config,delta,mult)
    out=[]
    for m in starts:
      result=integrate_one(m['state'],model,kp,tau0,delta)
      for horizon,status,exitp,rmax,end,lend,torque,resid,nfev in result:
        out.append({'mesh_um':mesh,'config':config,'model':model,'delta_rad':delta,'tau0_rad':tau0,
          'replica':m['replica'],'seed':m['seed'],'pair_index':m['pair_index'],'Lz_sign':m['Lz_sign'],
          'horizon_rf':horizon,'alpha':ALPHA/MU,'beta':BETA/MU,'kappa_applied':kp/MU,'mu':MU,
          'U2_V':BETA*(M0*OMEGA**2*R**2/E)/C2,'VDC_V':ALPHA*(M0*OMEGA**2*R**2/E)/D2,
          'U4_V':kp*(M0*OMEGA**2*R**2/E)/C4,'peak_electrode_V':peak,
          'status':status,'exit_rf':exitp,'rho_max':rmax,'Lz0':m['state'][0]*m['state'][3]-m['state'][1]*m['state'][2],
          'Lz_end':lend,'Delta_Lz':lend-(m['state'][0]*m['state'][3]-m['state'][1]*m['state'][2]),
          'torque_integral':torque,'torque_residual':resid,'nfev':nfev,
          'x0':m['state'][0],'y0':m['state'][1],'vx0':m['state'][2],'vy0':m['state'][3],
          'xend':end[0],'yend':end[1],'vxend':end[2],'vyend':end[3],
          'integrator':'DOP853','rtol':RTOL,'atol':ATOL,'max_step_tau':MAX_STEP})
    return out

def main():
    global OUT,PHASES,DELTAS,MU,OCTUPOLE_SCALE
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--pairs-per-replica',type=int,default=16)
    p.add_argument('--output-name',default='fem_confirmation_a9',help='directory below mass_filter_v15 for this protocol')
    p.add_argument('--delta-deg',nargs='+',type=float,default=[0.,90.]);p.add_argument('--tau0-deg',nargs='+',type=float,default=[0.,90.])
    p.add_argument('--mu',type=float,default=1.0,help='mass ratio at fixed electrode voltages')
    p.add_argument('--octupole-scale',type=float,default=1.0,help='multiply all requested octupole electrode amplitudes; useful for equal-voltage-cap studies')
    a=p.parse_args()
    if not 1<=a.workers<=6:p.error('workers must be 1..6')
    if not math.isfinite(a.mu) or a.mu<=0:p.error('--mu must be finite and positive')
    MU=float(a.mu)
    if not math.isfinite(a.octupole_scale) or a.octupole_scale<0:p.error('--octupole-scale must be finite and nonnegative')
    OCTUPOLE_SCALE=float(a.octupole_scale)
    if Path(a.output_name).name!=a.output_name:p.error('--output-name must be a single directory name')
    OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v15"/a.output_name
    DELTAS=tuple(math.radians(x) for x in a.delta_deg);PHASES=tuple(math.radians(x) for x in a.tau0_deg)
    OUT.mkdir(parents=True,exist_ok=True)
    source=[json.loads(x) for x in (CONFIRM/'checkpoint.jsonl').read_text(encoding='utf8').splitlines() if x]
    starts=[]
    for r in source:
      if r['config']=='Q' and r['delta_rad']==0 and r['tau0_rad']==0 and r['horizon_rf']==100 and r['pair_index']<a.pairs_per_replica:
       starts.append({'replica':r['replica'],'seed':r['seed'],'pair_index':r['pair_index'],'Lz_sign':r['Lz_sign'],
         'state':[r['x0'],r['y0'],r['vx0']/math.sqrt(MU),r['vy0']/math.sqrt(MU)]})
    starts.sort(key=lambda r:(r['replica'],r['pair_index'],-r['Lz_sign']))
    if len(starts)!=8*a.pairs_per_replica*2:raise RuntimeError(f'unexpected matched start count {len(starts)}')
    tasks=[]
    for mesh in (50,35):
     for delta in DELTAS:
      for tau0 in PHASES:
       for config,model,mult in CONFIGS:
        if voltage_peak(config,delta,mult)>250:continue
        tasks.append((mesh,config,model,mult,delta,tau0,starts))
    included_configs=sorted({t[1] for t in tasks})
    excluded_configs=sorted({c[0] for c in CONFIGS}-set(included_configs))
    hashes={}
    files=[ROOT/'outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz',
      ROOT/'outputs/octupole_comparison_home/mass_filter_v13/v13_1/fem/quadrupole_h50_P4_sector_solution.npz',
      ROOT/'outputs/abc_validation/direct_fem_validation/v2/h35_P4/sector_solution.npz',
      ROOT/'outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas/fem/quadrupole_h35_P4_sector_solution.npz',
      ROOT/'outputs/meshes/gmsh_50um/trap.msh',ROOT/'outputs/meshes/gmsh_35um/trap.msh',CONFIRM/'checkpoint.jsonl']
    hashes={'runner_script':sha(Path(__file__)),**{str(f.relative_to(ROOT)):sha(f) for f in files}}
    manifest={'study':'V15 A9 full h50/h35 P4 FEM matched chirality/torque validation','alpha':ALPHA/MU,'beta':BETA/MU,'kappa':KAPPA*OCTUPOLE_SCALE/MU,'mu':MU,'octupole_electrode_amplitude_scale':OCTUPOLE_SCALE,
      'R_m':R,'frequency_hz':100000.,'M0_Th':3200.04652261404,'c2_d2':C2,'c4':C4,'tau0_rad':PHASES,'delta_rad':DELTAS,
      'models':CONFIGS,'pairs_per_scramble':a.pairs_per_replica,'scrambled_replicas':8,'starts_per_group':len(starts),
      'included_configs_after_voltage_limit':included_configs,'excluded_configs_after_voltage_limit':excluded_configs,
      'horizons_rf':[100,300],'mass_scaling':'fixed physical voltages; initial dimensionless v scaled 1/sqrt(mu); alpha,beta,kappa scale as 1/mu',
      'method':'native saved P4 m2+m4 gradients; DOP853; 300 RF continuous extension of 100 RF survivors',
      'integrator':{'rtol':RTOL,'atol':ATOL,'max_step_tau':MAX_STEP},'voltage_limit_V':250,'source_hashes':hashes,
      'interpretation':'selected finite-time transverse FEM check; not finite-length transmission'}
    manifest['protocol_hash']=hashlib.sha256(json.dumps(manifest,sort_keys=True,default=float).encode()).hexdigest()
    mp=OUT/'manifest.json';cp=OUT/'checkpoint.jsonl'
    if mp.exists():
      saved=json.loads(mp.read_text(encoding='utf8'))
      canonical=json.loads(json.dumps(manifest,sort_keys=True,default=float))
      if saved!=canonical:raise RuntimeError('FEM manifest mismatch')
    if not mp.exists():mp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    done={}
    if cp.exists():
     for line in cp.read_text(encoding='utf8').splitlines():
      if line:
       r=json.loads(line)
       if r['protocol_hash']!=manifest['protocol_hash']:raise RuntimeError('FEM checkpoint hash mismatch')
       if r['key'] in done:raise RuntimeError('duplicate FEM checkpoint key')
       done[r['key']]=r
    pending=[]
    for task in tasks:
      mesh,config,model,mult,delta,tau0,ss=task
      need=8*a.pairs_per_replica*2*2
      found=sum(1 for r in done.values() if r['mesh_um']==mesh and r['config']==config and r['delta_rad']==delta and r['tau0_rad']==tau0)
      if found<need:pending.append(task)
      elif found>need:raise RuntimeError('FEM row count exceeds expected group size')
    print(f'V15 FEM groups={len(tasks)} pending={len(pending)} workers={a.workers} starts/group={len(starts)}',flush=True)
    for mesh in (50,35):
      mesh_tasks=[t for t in pending if t[0]==mesh]
      if not mesh_tasks:continue
      with ProcessPoolExecutor(max_workers=a.workers,initializer=init_worker,initargs=(mesh,MU,OCTUPOLE_SCALE)) as pool,cp.open('a',encoding='utf8') as f:
       futures=[pool.submit(worker,t) for t in mesh_tasks]
       for i,fut in enumerate(as_completed(futures),1):
        out=fut.result()
        for r in out:
          r['protocol_hash']=manifest['protocol_hash']
          r['key']='|'.join(str(r[k]) for k in ('mesh_um','config','delta_rad','tau0_rad','horizon_rf','replica','pair_index','Lz_sign'))
          if r['key'] in done:raise RuntimeError('duplicate FEM row')
          f.write(json.dumps(r,separators=(',',':'))+'\n');done[r['key']]=r
        f.flush();os.fsync(f.fileno());print(f'h{mesh} completed {i}/{len(mesh_tasks)} groups rows={len(done)}',flush=True)
    print('rows',len(done),'unresolved',sum(r['status']=='unresolved' for r in done.values()),'max_torque_residual',max(abs(r['torque_residual']) for r in done.values()))
if __name__=='__main__':main()
