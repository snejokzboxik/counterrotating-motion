"""Full native P4 FEM first-exit pilot for selected V13.1 cases."""
from __future__ import annotations
import argparse,csv,json,math
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
R=.008; FREQ=100e3; OMEGA=2*math.pi*FREQ
Q_E=1.602176634e-19; AMU=1.66053906660e-27
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1"
_F4=_F2=None; _M0=0.; _SCALE=0.

def initialize():
    global _F4,_F2,_M0,_SCALE
    import sys
    sys.path.insert(0,str(ROOT/"src"))
    from three_phase_trap.direct_fem import SectorSolution,DirectFEMRF
    from three_phase_trap.gmsh_mesh import load_gmsh_mesh
    fem=BASE/"fem"
    s4=SectorSolution.load(ROOT/"outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz")
    s2=SectorSolution.load(fem/"quadrupole_h50_P4_sector_solution.npz")
    mesh=load_gmsh_mesh(ROOT/"outputs/meshes/gmsh_50um/trap.msh")
    _F4=DirectFEMRF(s4,mesh,amplitude=1.,rotation_factor=complex(-.5,-math.sqrt(3)/2))
    _F2=DirectFEMRF(s2,mesh,amplitude=1.,rotation_factor=complex(.5,math.sqrt(3)/2))
    pars=json.loads((BASE/"physical_parameters.json").read_text(encoding="utf-8"))
    _M0=pars["M0_Th"]*AMU
    _SCALE=Q_E/(_M0*OMEGA**2*R)

def advance(task):
    mu,phase,model,eta,index,state,hmax,*tolerances=task
    rtol,atol=tolerances if tolerances else (3e-9,3e-11)
    beta=.175/mu; alpha=.027000013157894725/mu
    U2=190.; Vdc=30. if model=="Qopt" else 29.3143
    U4=0. if model in ("Q","Qopt") else eta*.175*_M0*OMEGA**2*R**2/(Q_E*.5413395699439091)
    standing=model=="SO"
    def rhs(t,z):
        x,y=z[:2]; xm,ym=R*x,R*y
        i,u,v=_F4.find(xm,ym); ecx,ecy,esx,esy=_F4.local_quadratures(i,u,v)
        gx4c,gy4c=-esx,-esy; gx4s,gy4s=ecx,ecy
        j,a,b=_F2.find(xm,ym); _,_,qsx,qsy=_F2.local_quadratures(j,a,b)
        gx2,gy2=-qsx,-qsy
        angle=t+phase
        if standing: gx4,gy4=gx4c*math.cos(angle),gy4c*math.cos(angle)
        else: gx4,gy4=gx4c*math.cos(angle)+gx4s*math.sin(angle),gy4c*math.cos(angle)+gy4s*math.sin(angle)
        qdrive=U2*math.cos(angle)+Vdc
        return np.array([z[2],z[3],-_SCALE/mu*(U4*gx4+qdrive*gx2),-_SCALE/mu*(U4*gy4+qdrive*gy2)])
    def event(t,z): return math.hypot(z[0],z[1])-1.
    event.terminal=True;event.direction=1
    y=np.asarray(state,float);rmax=math.hypot(*y[:2]);last_exit=None
    try:
        a=solve_ivp(rhs,(0,2*math.pi*100),y,method="DOP853",rtol=rtol,atol=atol,
                    max_step=2*math.pi*hmax,events=event)
        if not a.success: raise RuntimeError(a.message)
        rmax=max(rmax,float(np.hypot(a.y[0],a.y[1]).max()))
        if len(a.t_events[0]): last_exit=float(a.t_events[0][0]);rmax=max(rmax,1.)
        status100="escaped" if last_exit is not None else "retained"
        exit300=last_exit;rmax300=rmax;status300=status100
        if last_exit is None:
            b=solve_ivp(rhs,(2*math.pi*100,2*math.pi*300),a.y[:,-1],method="DOP853",rtol=rtol,atol=atol,
                        max_step=2*math.pi*hmax,events=event)
            if not b.success: raise RuntimeError(b.message)
            rmax300=max(rmax300,float(np.hypot(b.y[0],b.y[1]).max()))
            if len(b.t_events[0]): exit300=float(b.t_events[0][0]);rmax300=max(rmax300,1.);status300="escaped"
            else: status300="retained"
        return dict(mu=mu,phase_rad=phase,phase_deg=phase*180/math.pi,model=model,eta=eta,index=index,
                    U4_V=U4,U2_V=U2,Vdc_V=Vdc,rtol=rtol,atol=atol,max_step_tau=2*math.pi*hmax,status_100=status100,
                    exit_periods_100=last_exit/(2*math.pi) if last_exit is not None else None,max_rho_100=rmax,
                    status_300=status300,exit_periods_300=exit300/(2*math.pi) if exit300 is not None else None,
                    max_rho_300=rmax300,x0=float(state[0]),y0=float(state[1]),vx0=float(state[2]),vy0=float(state[3]))
    except Exception as exc:
        return dict(mu=mu,phase_rad=phase,phase_deg=phase*180/math.pi,model=model,eta=eta,index=index,
                    U4_V=U4,U2_V=U2,Vdc_V=Vdc,rtol=rtol,atol=atol,max_step_tau=2*math.pi*hmax,status_100="unresolved",status_300="unresolved",
                    exit_periods_100=None,exit_periods_300=None,max_rho_100=float("nan"),max_rho_300=float("nan"),
                    x0=float(state[0]),y0=float(state[1]),vx0=float(state[2]),vy0=float(state[3]),error=str(exc))

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--mu",nargs="+",type=float,required=True)
    p.add_argument("--etas",nargs="+",type=float,default=[-.1,-.05,.05,.1]);p.add_argument("--workers",type=int,default=2)
    p.add_argument("--step-rf",type=float,default=1/64);p.add_argument("--max-initial",type=int,default=32);p.add_argument("--qopt",action="store_true");args=p.parse_args()
    dest=BASE/args.run;dest.mkdir(parents=True,exist_ok=True)
    initials=np.genfromtxt(BASE/"ideal_pilot/initial_conditions.csv",delimiter=",",names=True)
    states=np.stack([initials[n] for n in ("x0_rho","y0_rho","vx0_rho_per_tau","vy0_rho_per_tau")],axis=1)
    states=states[:args.max_initial]
    phases=[0.,math.pi/2];models=[("Q",0.)]+([("Qopt",0.)] if args.qopt else [])+[(m,e) for e in args.etas for m in ("SO","RO")]
    config=dict(field="full native P4 FEM h50 m4+m2",mesh_sha256="8ea670e6bf5bada2c86756bc98718063fe38cbfd1692c872c4a4e77f59b1a284",
        mu=args.mu,etas=args.etas,models=models,phase_rad=phases,n_common_initial=len(states),initial_source="ideal pilot Sobol seed 131",
        aperture_rho=1.,horizons=[100,300],continuation="from 100-cycle final state, unreset phase",integrator="DOP853 rtol 3e-9 atol 3e-11",
        max_step_rf=args.step_rf,unresolved_policy="separate; never count as escaped",Q_C=Q_E,M0_Th=_load_M0(),R_m=R,Omega_rad_s=OMEGA)
    (dest/"parameters.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    ck=dest/"checkpoint.jsonl";old=[]
    if ck.exists():old=[json.loads(x) for x in ck.read_text(encoding="utf-8").splitlines() if x]
    done={(r["mu"],r["phase_rad"],r["model"],r["eta"],r["index"]) for r in old}
    tasks=[]
    for mu in args.mu:
      for phase in phases:
       for model,eta in models:
        for i,state in enumerate(states):
         key=(mu,phase,model,eta,i)
         if key not in done:tasks.append((*key,state,args.step_rf))
    if tasks:
      with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
       fs=[pool.submit(advance,t) for t in tasks]
       with ck.open("a",encoding="utf-8") as f:
        for fut in as_completed(fs):
         row=fut.result();old.append(row);f.write(json.dumps(row)+"\n");f.flush()
    if old:
      with (dest/"trajectory_results.csv").open("w",newline="",encoding="utf-8") as f:
       w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(old)

def _load_M0():
    return json.loads((BASE/"physical_parameters.json").read_text(encoding="utf-8"))["M0_Th"]

if __name__=="__main__":main()
