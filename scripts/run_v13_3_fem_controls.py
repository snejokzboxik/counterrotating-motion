"""Checkpointed h50 P4 FEM controls for selected V13.3 atlas points."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
V13_1=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1"
F4_PATH=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz"
F2_PATH=V13_1/"fem"/"quadrupole_h50_P4_sector_solution.npz"
MESH_PATH=ROOT/"outputs/meshes/gmsh_50um/trap.msh"
MESH_UM=50
R=.008;FREQ=100_000.;OMEGA=2*math.pi*FREQ
E_CHARGE=1.602176634e-19;AMU=1.66053906660e-27
M0=3200.04652261404*AMU;C2=.7718255558274212;D2=C2;C4=.5413395699439091
RTOL,ATOL=3e-9,3e-11;MAX_STEP=2*math.pi/64
PHASES=(0.,math.pi/2)
_F4=_F2=None;_SCALE=0.


def configure_mesh(mesh_um):
    global F4_PATH,F2_PATH,MESH_PATH,MESH_UM
    MESH_UM=int(mesh_um)
    if MESH_UM==50:
        F4_PATH=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz"
        F2_PATH=V13_1/"fem"/"quadrupole_h50_P4_sector_solution.npz"
    elif MESH_UM==35:
        F4_PATH=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h35_P4/sector_solution.npz"
        F2_PATH=BASE/"fem"/"quadrupole_h35_P4_sector_solution.npz"
    else:
        raise ValueError("Only saved h35/h50 P4 meshes are supported")
    MESH_PATH=ROOT/f"outputs/meshes/gmsh_{MESH_UM}um/trap.msh"


def initialize(mesh_um=50):
    global _F4,_F2,_SCALE
    configure_mesh(mesh_um)
    import sys
    sys.path.insert(0,str(ROOT/"src"))
    from three_phase_trap.direct_fem import SectorSolution,DirectFEMRF
    from three_phase_trap.gmsh_mesh import load_gmsh_mesh
    mesh=load_gmsh_mesh(MESH_PATH)
    s4=SectorSolution.load(F4_PATH);s2=SectorSolution.load(F2_PATH)
    _F4=DirectFEMRF(s4,mesh,amplitude=1.,rotation_factor=complex(-.5,-math.sqrt(3)/2))
    _F2=DirectFEMRF(s2,mesh,amplitude=1.,rotation_factor=complex(.5,math.sqrt(3)/2))
    _SCALE=E_CHARGE/(M0*OMEGA**2*R)


def _grad4(x,y):
    i,u,v=_F4.find(R*x,R*y)
    ecx,ecy,esx,esy=_F4.local_quadratures(i,u,v)
    return -esx,-esy,ecx,ecy


def _grad2(x,y):
    i,u,v=_F2.find(R*x,R*y)
    _,_,qsx,qsy=_F2.local_quadratures(i,u,v)
    return -qsx,-qsy


def simulate_segment(point,model,mu,phase,delta,state_i,initial_state,state,t0_cycles,t1_cycles,old_max):
    alpha0,beta0,kappa0=float(point["alpha"]),float(point["beta"]),float(point["kappa"])
    alpha,beta,kappa=alpha0/mu,beta0/mu,(kappa0/mu if model=="RO" else 0.)
    scale=_SCALE/mu
    u2=beta0*(M0*OMEGA**2*R**2/E_CHARGE)/C2
    vdc=alpha0*(M0*OMEGA**2*R**2/E_CHARGE)/D2
    u4=kappa0*(M0*OMEGA**2*R**2/E_CHARGE)/C4 if model=="RO" else 0.
    y0=np.asarray(state,float)
    def rhs(t,z):
        x,y=z[:2]
        g4cx,g4cy,g4sx,g4sy=_grad4(x,y)
        # Rotate the two octupole quadratures by the relative phase delta.
        phase_oct=t+phase-delta
        gx4=g4cx*math.cos(phase_oct)+g4sx*math.sin(phase_oct)
        gy4=g4cy*math.cos(phase_oct)+g4sy*math.sin(phase_oct)
        gx2,gy2=_grad2(x,y)
        qdrive=u2*math.cos(t+phase)+vdc
        return np.array([z[2],z[3],-scale*(u4*gx4+qdrive*gx2),
                         -scale*(u4*gy4+qdrive*gy2)])
    def event(t,z):return math.hypot(z[0],z[1])-1.0
    event.terminal=True;event.direction=1.0
    t0,t1=2*math.pi*t0_cycles,2*math.pi*t1_cycles
    applied_kappa=kappa if model=="RO" else 0.0
    row={"candidate_id":str(point["candidate_id"]),"state_i":state_i,
         "mesh_um":MESH_UM,
         "alpha_base":alpha0,"beta_base":beta0,"kappa_base":kappa0,
         "alpha":alpha,"beta":beta,"kappa":applied_kappa,
         "eta":applied_kappa/beta,"mu":mu,"model":model,
         "tau0_rad":phase,"tau0_deg":math.degrees(phase),"delta_rad":delta,
         "start_horizon":t0_cycles,"horizon":t1_cycles,
         "U2_V":u2,"VDC_V":vdc,"U4_V":u4,
         "max_electrode_peak_abs_V":max_electrode_peak(alpha0,beta0,applied_kappa),
         "x0_rho":float(initial_state[0]),"y0_rho":float(initial_state[1]),
         "vx0_rho_tau":float(initial_state[2]),"vy0_rho_tau":float(initial_state[3]),
         "x_start_rho":float(state[0]),"y_start_rho":float(state[1]),
         "vx_start_rho_tau":float(state[2]),"vy_start_rho_tau":float(state[3]),
         "integrator":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP}
    try:
        sol=solve_ivp(rhs,(t0,t1),y0,method="DOP853",rtol=RTOL,atol=ATOL,
                      max_step=MAX_STEP,events=event,dense_output=True)
        if not sol.success:status="unresolved";exit_periods=math.nan
        elif len(sol.t_events[0]):status="escaped";exit_periods=float(sol.t_events[0][0]/(2*math.pi))
        else:status="retained";exit_periods=math.nan
        rho_max=max(float(old_max),float(np.max(np.hypot(sol.y[0],sol.y[1]))))
        if len(sol.t)>1:
            mid=sol.sol((sol.t[:-1]+sol.t[1:])/2)
            rho_max=max(rho_max,float(np.max(np.hypot(mid[0],mid[1]))))
        if status=="escaped":rho_max=max(1.,rho_max)
        row.update({"status":status,"exit_periods":exit_periods,
                    "rho_max_cumulative":rho_max,
                    "final_state":json.dumps([float(v) for v in sol.y[:,-1]],separators=(",",":")),
                    "nfev":int(sol.nfev)})
    except Exception as exc:
        row.update({"status":"unresolved","exit_periods":math.nan,"rho_max_cumulative":math.nan,
                    "final_state":"","nfev":math.nan,"error":f"{type(exc).__name__}:{exc}"})
    row["key"]="|".join(str(v) for v in (row["candidate_id"],row["mu"],row["model"],row["tau0_rad"],
                                            state_i,row["horizon"]))
    return row


def worker(task):return simulate_segment(*task)


def max_electrode_peak(alpha,beta,kappa):
    with (V13_1/"electrode_voltage_peaks.csv").open(newline="",encoding="utf-8") as f:
        electrodes=[r for r in csv.DictReader(f) if r["scheme"]=="Q"]
    scale=M0*OMEGA**2*R**2/E_CHARGE
    u2=beta*scale/C2;vdc=alpha*scale/D2;u4=kappa*scale/C4
    return max(abs(vdc*float(r["Q2"]))+
               math.hypot(u2*float(r["Q2"])+u4*float(r["C4"]),u4*float(r["S4"]))
               for r in electrodes)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--points",default=str(BASE/"fem_control_points.csv"))
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--mesh-um",type=int,choices=(35,50),default=50)
    p.add_argument("--mu",nargs="+",type=float,default=[1.0])
    p.add_argument("--max-states",type=int,default=32)
    p.add_argument("--delta",type=float,default=0.)
    p.add_argument("--output-name",default=None,help="separate output subdirectory below the V13.3 atlas")
    args=p.parse_args()
    if not 1<=args.workers<=8:p.error("--workers must be between 1 and 8")
    configure_mesh(args.mesh_um)
    points=[]
    with Path(args.points).open(newline="",encoding="utf-8") as f:
        for i,row in enumerate(csv.DictReader(f)):
            row={**row,"alpha":float(row["alpha"]),"beta":float(row["beta"]),
                 "kappa":float(row["kappa"])}
            row["candidate_id"]=row.get("candidate_id",f"fem_{i:02d}")
            row["max_electrode_peak_abs_V"]=max_electrode_peak(row["alpha"],row["beta"],row["kappa"])
            if row["max_electrode_peak_abs_V"]>250.0:
                raise ValueError(f"Point {row['candidate_id']} exceeds the 250 V electrode peak limit")
            points.append(row)
    initials=np.genfromtxt(V13_1/"ideal_pilot"/"initial_conditions.csv",delimiter=",",names=True)
    states=np.stack([initials[k] for k in ("x0_rho","y0_rho","vx0_rho_per_tau","vy0_rho_per_tau")],axis=1)[:args.max_states]
    default_out="fem_controls" if args.mesh_um==50 else "fem_controls_h35"
    outname=args.output_name or default_out
    if Path(outname).name!=outname:raise ValueError("--output-name must be a single directory name")
    outdir=BASE/outname;outdir.mkdir(parents=True,exist_ok=True)
    ck=outdir/"checkpoint.jsonl";done={};
    if ck.exists():
        for line in ck.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row=json.loads(line)
                if row["key"] in done:raise RuntimeError(f"Duplicate FEM task key {row['key']}")
                done[row["key"]]=row
    tasks=[]
    for point in points:
      for mu in args.mu:
       for phase in PHASES:
        for model in ("Q","RO"):
         for i,state in enumerate(states):
          key="|".join(str(v) for v in (point["candidate_id"],mu,model,phase,i,100))
          old=done.get(key)
          if old is None:
              tasks.append((point,model,mu,phase,args.delta,i,state,state,0,100,0.0))
          elif old["status"]=="retained":
              key300="|".join(str(v) for v in (point["candidate_id"],mu,model,phase,i,300))
              if key300 not in done:
                  tasks.append((point,model,mu,phase,args.delta,i,state,json.loads(old["final_state"]),
                                100,300,float(old["rho_max_cumulative"])))
    if tasks:
      count=0
      # Persist each completed trajectory as it finishes. This makes a long FEM
      # control batch resumable instead of keeping all completed work in memory.
      with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.mesh_um,)) as pool, \
           ck.open("a",encoding="utf-8") as stream:
        futures=[pool.submit(worker,t) for t in tasks]
        for fut in as_completed(futures):
            row=fut.result()
            if row["key"] in done:
                raise RuntimeError(f"Duplicate FEM task key {row['key']}")
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
            done[row["key"]]=row
            count+=1
            if count%8==0:
                stream.flush();os.fsync(stream.fileno())
        stream.flush();os.fsync(stream.fileno())
    rows=list(done.values())
    if rows:
      fields=[]
      for row in rows:fields.extend(k for k in row if k not in fields)
      with (outdir/"trajectory_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (F4_PATH,F2_PATH,MESH_PATH)}
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config={"field":f"full h{args.mesh_um} P4 FEM superposition, m4 and m2 bases on shared full 12-electrode mesh",
            "script_sha256":script_sha,
            "mesh_um":args.mesh_um,
            "hashes":hashes,"c2":C2,"d2":D2,"c4":C4,"R_m":R,"f_Hz":FREQ,
            "M0_Th":3200.04652261404,"Umax_V":250,"delta_rad":args.delta,
            "mass_ratios":args.mu,"phases_rad":PHASES,"points":[{"candidate_id":p["candidate_id"],"alpha":p["alpha"],"beta":p["beta"],"kappa":p["kappa"],"max_electrode_peak_abs_V":p["max_electrode_peak_abs_V"]} for p in points],
            "initial_source":"first common V13.1 seed131 Sobol states",
            "initial_conditions_sha256":hashlib.sha256((V13_1/"ideal_pilot"/"initial_conditions.csv").read_bytes()).hexdigest(),
            "n_initial":len(states),"aperture_rho":1,"horizons":[100,300],
            "continuation":"300-cycle segment starts from exact 100-cycle endpoint; absolute tau retained",
            "integrator":{"method":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP},
            "unresolved_policy":"separate, never classified as lost"}
    (outdir/"parameters.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print(json.dumps({"tasks_completed_or_loaded":len(rows),"pending_before_run":len(tasks),"hashes":hashes,
                      "script_sha256":script_sha},indent=2))


if __name__=="__main__":main()
