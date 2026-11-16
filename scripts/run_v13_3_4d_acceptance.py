"""Paired scrambled-Sobol 4D acceptance for selected V13.3 fixed-voltage rays."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
OUT=BASE/"acceptance_4d"
E_CHARGE=1.602176634e-19;AMU=1.66053906660e-27
M0_TH=3200.04652261404;FREQ=100_000.;OMEGA=2*math.pi*FREQ;R=.008;RIN=.0075;EMAX_EV=5.
C2=D2=.7718255558274212;C4=.5413395699439091
PHASES=(0.,math.pi/2,math.pi,3*math.pi/2);SEEDS=(513217,883391,20261114,941029)
RTOL,ATOL=2e-10,2e-12;MAX_STEP=2*math.pi/64;APERTURE=1.
FEM4=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz"
FEM2=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1/fem/quadrupole_h50_P4_sector_solution.npz"
FEM_MESH=ROOT/"outputs/meshes/gmsh_50um/trap.msh"
sys.path.insert(0,str(ROOT/"scripts"))
from summarize_v13_3_fem_controls import read_csv

def write_csv(path,rows):
    if not rows:return
    names=[]
    for r in rows:names.extend(k for k in r if k not in names)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=names,extrasaction="ignore");w.writeheader();w.writerows(rows)

def key(row):return "|".join((row["field_mode"],row["point_id"],row["model"],str(row["mu"]),str(row["phase_i"]),str(row["replica"]),str(row["state_i"]),str(row["horizon"]),str(row["relative_delta_rad"])))

def initialize_worker(field_mode):
    if field_mode=="fem_h50_P4":
        import run_v13_3_fem_controls as fem
        fem.RTOL,fem.ATOL,fem.MAX_STEP=RTOL,ATOL,MAX_STEP
        fem.initialize(50)

def integrate(task):
    from three_phase_trap.mass_filter_v13_3 import acceleration
    field_mode,point,model,mu,phase_i,phase,delta,rep,seed,state_i,initial,initial_physical,start_horizon,stop_horizon,y_start,old_max,old_exit,carry=task
    a0,b0,k0=map(float,(point["alpha"],point["beta"],point["kappa"]))
    alpha,beta=a0/mu,b0/mu;kappa=k0/mu if model=="RO" else 0.
    z0=np.asarray(y_start,float)
    row={"field_mode":field_mode,"point_id":point["candidate_id"],"alpha_base":a0,"beta_base":b0,"kappa_base":k0,
         "eta":k0/b0,"mu":mu,"M_over_Q_Th_per_e":M0_TH*mu,"mass_kg":M0_TH*mu*AMU,
         "alpha":alpha,"beta":beta,"kappa":kappa,"model":model,"phase_i":phase_i,
         "tau0_rad":phase,"tau0_deg":math.degrees(phase),"relative_delta_rad":delta,
         "replica":rep,"sobol_seed":seed,"state_i":state_i,"horizon":stop_horizon,
         "start_horizon":start_horizon,"carried_status":False,"aperture_rho":APERTURE,
         "x0_rho":initial[0],"y0_rho":initial[1],"vx0_rho_tau":initial[2],"vy0_rho_tau":initial[3],
         "x0_m":initial_physical[0],"y0_m":initial_physical[1],"vx0_m_s":initial_physical[2],"vy0_m_s":initial_physical[3],
         "U2_V":point["U2_V"],"VDC_V":point["VDC_V"],
         "U4_setting_V":point["U4_setting_V"],"U4_applied_V":point["U4_setting_V"] if model=="RO" else 0.,
         "max_electrode_peak_abs_V":point["max_electrode_peak_abs_V" if model=="RO" else "max_electrode_peak_abs_V_Q"],
         "injection_radius_m":RIN,"max_transverse_energy_eV":EMAX_EV,
         "integrator":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP}
    if carry:
        row.update({"status":carry,"exit_periods":old_exit,"rho_max":old_max,"final_state":json.dumps(list(map(float,z0)),separators=(",",":")),"nfev":0,"carried_status":True})
    elif field_mode=="ideal_harmonic":
        def rhs(t,z):
            ax,ay=acceleration(z[0],z[1],t,kappa,beta,alpha,delta=delta,tau0=phase)
            return np.array([z[2],z[3],ax,ay])
        def event(t,z):return math.hypot(z[0],z[1])-APERTURE
        event.terminal=True;event.direction=1.
        try:
            t0,t1=2*math.pi*start_horizon,2*math.pi*stop_horizon
            sol=solve_ivp(rhs,(t0,t1),z0,method="DOP853",rtol=RTOL,atol=ATOL,max_step=MAX_STEP,events=event,dense_output=True)
            if not sol.success:status="unresolved";exitp=math.nan
            elif len(sol.t_events[0]):status="escaped";exitp=float(sol.t_events[0][0]/(2*math.pi))
            else:status="retained";exitp=math.nan
            rmax=max(float(old_max),float(np.max(np.hypot(sol.y[0],sol.y[1]))))
            if len(sol.t)>1:
                mid=sol.sol((sol.t[:-1]+sol.t[1:])/2);rmax=max(rmax,float(np.max(np.hypot(mid[0],mid[1]))))
            if status=="escaped":rmax=max(APERTURE,rmax)
            row.update({"status":status,"exit_periods":exitp,"rho_max":rmax,
                        "final_state":json.dumps([float(v) for v in sol.y[:,-1]],separators=(",",":")),"nfev":int(sol.nfev)})
        except Exception as exc:
            row.update({"status":"unresolved","exit_periods":math.nan,"rho_max":math.nan,"final_state":"","nfev":math.nan,"error":f"{type(exc).__name__}:{exc}"})
    else:
        import run_v13_3_fem_controls as fem
        try:
            fr=fem.simulate_segment(point,model,mu,phase,delta,state_i,initial,y_start,
                                    start_horizon,stop_horizon,old_max)
            row.update({"status":fr["status"],"exit_periods":fr["exit_periods"],
                        "rho_max":fr["rho_max_cumulative"],"final_state":fr["final_state"],
                        "nfev":fr["nfev"]})
        except Exception as exc:
            row.update({"status":"unresolved","exit_periods":math.nan,"rho_max":math.nan,
                        "final_state":"","nfev":math.nan,"error":f"{type(exc).__name__}:{exc}"})
    row["key"]=key(row);return row

def main():
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=8);p.add_argument("--mu",nargs="+",type=float,default=[.9,1.,1.1])
    p.add_argument("--points",default=str(BASE/"acceptance_points.csv"),help="CSV of selected fixed-voltage parameter rays")
    p.add_argument("--output-name",default="acceptance_4d",help="separate result directory below the V13.3 atlas")
    p.add_argument("--field-mode",choices=("ideal_harmonic","fem_h50_P4"),default="ideal_harmonic",
                   help="ideal model for broad QMC; full h50 P4 field for selected validation ensembles")
    p.add_argument("--delta-rad",type=float,default=0.,help="relative octupole phase; kept distinct from injection phase")
    p.add_argument("--horizons",nargs="+",type=int,default=[100]);p.add_argument("--states-per-replica",type=int,default=64)
    p.add_argument("--replicas",type=int,default=4);args=p.parse_args()
    if not 1<=args.workers<=8:p.error("workers must be 1..8")
    if args.replicas!=4:p.error("the documented acceptance protocol uses four independent scrambles")
    if args.states_per_replica<1 or args.states_per_replica&(args.states_per_replica-1):p.error("states-per-replica must be a power of two")
    horizons=sorted(set(args.horizons))
    if horizons[0]!=100 or any(b<=a for a,b in zip(horizons,horizons[1:])):p.error("horizons must start at 100 and increase")
    if Path(args.output_name).name!=args.output_name:p.error("output-name must be one directory name")
    OUT=BASE/args.output_name
    OUT.mkdir(parents=True,exist_ok=True)
    point_rows=read_csv(Path(args.points))
    point_rows=[{**r,"alpha":float(r["alpha"]),"beta":float(r["beta"]),"kappa":float(r["kappa"])} for r in point_rows]
    electrode_rows=read_csv(ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1/electrode_voltage_peaks.csv")
    electrodes=[{k:float(r[k]) for k in ("C4","S4","Q2")} for r in electrode_rows if r["scheme"]=="Q"]
    voltage_scale=M0_TH*AMU*OMEGA**2*R**2/E_CHARGE
    points={}
    for r in point_rows:
        u2=r["beta"]*voltage_scale/C2;vdc=r["alpha"]*voltage_scale/D2;u4=r["kappa"]*voltage_scale/C4
        def peak(octupole):
            u=octupole*u4
            return max(abs(vdc*e["Q2"])+math.hypot(u2*e["Q2"]+u*e["C4"],u*e["S4"]) for e in electrodes)
        if peak(1.)>250. or peak(0.)>250.:
            raise ValueError(f"{r['candidate_id']} Q or Q+RO exceeds the 250 V peak-electrode limit")
        points[r["candidate_id"]]={**r,"U2_V":u2,"VDC_V":vdc,"U4_setting_V":u4,
                                    "max_electrode_peak_abs_V":peak(1.),
                                    "max_electrode_peak_abs_V_Q":peak(0.)}
    muvals=[round(float(x),12) for x in args.mu]
    volume={}; initial={}; initial_rows=[]
    starts_by={}
    for mu in muvals:
        mass=M0_TH*mu*AMU;vmax=math.sqrt(2*EMAX_EV*E_CHARGE/mass)
        domain=math.pi**2*RIN**2*vmax**2
        volume[mu]={"mass_kg":mass,"vmax_m_s":vmax,"vmax_rho_per_tau":vmax/(R*OMEGA),"D_volume_m4_s2":domain,
                    "a4_domain":domain/(R**4*OMEGA**2)}
        for rep,seed in enumerate(SEEDS):
            u=qmc.Sobol(4,scramble=True,seed=seed).random_base2(int(math.log2(args.states_per_replica)))
            for i,q in enumerate(u):
                rad=RIN*math.sqrt(float(q[0]));ang=2*math.pi*float(q[1]);speed=vmax*math.sqrt(float(q[2]));vang=2*math.pi*float(q[3])
                x,y=rad*math.cos(ang)/R,rad*math.sin(ang)/R
                vx,vy=speed*math.cos(vang)/(R*OMEGA),speed*math.sin(vang)/(R*OMEGA)
                physical=[rad*math.cos(ang),rad*math.sin(ang),speed*math.cos(vang),speed*math.sin(vang)]
                initial[(mu,rep,i)]=[x,y,vx,vy];starts_by[(mu,rep,i)]=physical
                initial_rows.append({"mu":mu,"replica":rep,"sobol_seed":seed,"state_i":i,
                                     "x0_rho":x,"y0_rho":y,"vx0_rho_tau":vx,"vy0_rho_tau":vy,
                                     "x0_m":physical[0],"y0_m":physical[1],"vx0_m_s":physical[2],"vy0_m_s":physical[3],
                                     "r0_m":rad,"speed0_m_s":speed,"E0_eV":.5*mass*speed**2/E_CHARGE,
                                     "domain_velocity_max_m_s":vmax})
    write_csv(OUT/"initial_conditions.csv",initial_rows)
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_files=[ROOT/"src/three_phase_trap/mass_filter_v13_3.py"]
    if args.field_mode=="fem_h50_P4":
        source_files.extend((FEM4,FEM2,FEM_MESH,ROOT/"scripts/run_v13_3_fem_controls.py",
                             ROOT/"src/three_phase_trap/direct_fem.py",
                             ROOT/"src/three_phase_trap/gmsh_mesh.py"))
    source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    manifest={"script_sha256":script_hash,"field_mode":args.field_mode,"source_hashes":source_hashes,
              "point_definitions":list(points.values()),"mu_values":muvals,"phase_rad":PHASES,
              "relative_delta_rad":args.delta_rad,"sobol":"SciPy scrambled Sobol(4); independent scramble per replica, paired Q/RO and common phase/mass coordinates",
              "geometric_harmonic_coefficients":{"c2":C2,"d2":D2,"c4":C4},
              "replica_seeds":SEEDS,"states_per_replica":args.states_per_replica,"physical_domain":{"position":"uniform area disk r<=7.5 mm","velocity":"uniform area disk, transverse kinetic energy <=5 eV, recomputed at each M"},
              "domain_volumes":volume,"aperture_rho":APERTURE,
              "parameter_scaling":"fixed electrode voltages; alpha,beta,kappa scale as 1/mu; kappa=0 in Q",
              "solver":{"method":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP},
              "uncertainty":"four replicate-level scrambled-Sobol means and paired replicate differences; do not pool all points as independent Bernoulli samples",
              "unresolved_policy":"separate; report acceptance fraction lower and upper bounds"}
    protocol={**manifest,"initial_conditions_sha256":hashlib.sha256((OUT/"initial_conditions.csv").read_bytes()).hexdigest()}
    run_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(",",":"),default=float).encode()).hexdigest()
    manifest={**protocol,"horizons":horizons,"run_hash":run_hash}
    manifest_path=OUT/"manifest.json"
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("run_hash")!=run_hash:raise RuntimeError("acceptance checkpoint protocol/hash mismatch")
        manifest["horizons"]=sorted(set(old.get("horizons",[]))|set(horizons))
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    ck=OUT/"checkpoint.jsonl";done={}
    if ck.exists():
        for line in ck.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r=json.loads(line)
                if r.get("run_hash")!=run_hash:raise RuntimeError("acceptance checkpoint row hash mismatch")
                if r["key"] in done:raise RuntimeError(f"duplicate acceptance key {r['key']}")
                done[r["key"]]=r
    tasks_total=0
    batch_size=max(32,args.workers*16)
    for h_i,horizon in enumerate(horizons):
     n=0
     with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(args.field_mode,)) as pool,ck.open("a",encoding="utf-8") as stream:
      batch=[]
      for mu in muvals:
       for point in points.values():
        for phase_i,phase in enumerate(PHASES):
         for rep,seed in enumerate(SEEDS):
          for i in range(args.states_per_replica):
           for model in ("Q","RO"):
            test={"field_mode":args.field_mode,"point_id":point["candidate_id"],"model":model,"mu":mu,"phase_i":phase_i,"replica":rep,"state_i":i,"horizon":horizon,"relative_delta_rad":args.delta_rad}
            if key(test) in done:continue
            if h_i==0:
             start_h=0;y=initial[(mu,rep,i)];oldmax=0.;oldexit=math.nan;carry=None
            else:
             prev=done.get(key({**test,"horizon":horizons[h_i-1]}))
             if prev is None:continue
             start_h=horizons[h_i-1];oldmax=float(prev["rho_max"]);oldexit=prev["exit_periods"]
             carry=prev["status"] if prev["status"] in ("escaped","unresolved") else None
             y=json.loads(prev["final_state"]) if prev["final_state"] else initial[(mu,rep,i)]
            batch.append((args.field_mode,point,model,mu,phase_i,phase,args.delta_rad,rep,seed,i,initial[(mu,rep,i)],starts_by[(mu,rep,i)],start_h,horizon,y,oldmax,oldexit,carry))
            if len(batch)>=batch_size:
             tasks_total+=len(batch)
             futures=[pool.submit(integrate,t) for t in batch]
             batch=[]
             for future in as_completed(futures):
              r=future.result();r["run_hash"]=run_hash
              if r["key"] in done:raise RuntimeError(f"duplicate acceptance key {r['key']}")
              stream.write(json.dumps(r,separators=(",",":"))+"\n");done[r["key"]]=r;n+=1
              if n%20==0:stream.flush();os.fsync(stream.fileno())
      if batch:
       tasks_total+=len(batch)
       futures=[pool.submit(integrate,t) for t in batch]
       for future in as_completed(futures):
        r=future.result();r["run_hash"]=run_hash
        if r["key"] in done:raise RuntimeError(f"duplicate acceptance key {r['key']}")
        stream.write(json.dumps(r,separators=(",",":"))+"\n");done[r["key"]]=r;n+=1
        if n%20==0:stream.flush();os.fsync(stream.fileno())
      stream.flush();os.fsync(stream.fileno())
    rows=list(done.values());write_csv(OUT/"trajectory_results.csv",rows)
    print(json.dumps({"points":len(points),"mu":muvals,"horizons":horizons,"new_tasks":tasks_total,"total_rows":len(rows),"checkpoint":str(ck)},indent=2))

if __name__=="__main__":main()
