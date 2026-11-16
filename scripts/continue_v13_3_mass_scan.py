"""Continue selected paired mass-scan trajectories from 100 to 300/1000 RF cycles."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
sys.path.insert(0,str(ROOT/"src"))
RTOL,ATOL=2e-10,2e-12;MAX_STEP=2*math.pi/64

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write(path,rows):
    if not rows:return
    cols=[]
    for r in rows:cols.extend(k for k in r if k not in cols)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader();w.writerows(rows)
def key(r):
    return "|".join((r["candidate_id"],r["model"],str(r["mu"]),str(r["phase_i"]),str(r["replica"]),str(r["state_i"]),str(r["horizon_RF"])))

def integrate(task):
    from three_phase_trap.mass_filter_v13_3 import acceleration
    source,point,horizon,start_state,carry_status=task
    row={**source,"start_horizon_RF":int(source["horizon_RF"]),"horizon_RF":horizon,
         "integrator":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP,
         "carried_prior_status":bool(carry_status)}
    if carry_status:
        row.update({"status":source["status"],"exit_periods":source["exit_periods"],"rho_max":source["rho_max"],
                    "final_state":source.get("final_state",""),"nfev":0})
    else:
        mu=float(source["mu"]);phase=float(source["tau0_rad"])
        alpha=float(point["alpha"])/mu;beta=float(point["beta"])/mu
        kappa=float(point["kappa"])/mu if source["model"]=="RO" else 0.
        y0=np.asarray(start_state,float);t0=2*math.pi*int(source["horizon_RF"]);t1=2*math.pi*horizon
        def rhs(t,z):
            ax,ay=acceleration(z[0],z[1],t,kappa,beta,alpha,delta=0.,tau0=phase)
            return np.asarray((z[2],z[3],ax,ay))
        def event(t,z):return math.hypot(z[0],z[1])-1.
        event.terminal=True;event.direction=1.
        try:
            sol=solve_ivp(rhs,(t0,t1),y0,method="DOP853",rtol=RTOL,atol=ATOL,max_step=MAX_STEP,events=event,dense_output=True)
            if not sol.success:status="unresolved";exitp=math.nan
            elif len(sol.t_events[0]):status="escaped";exitp=float(sol.t_events[0][0]/(2*math.pi))
            else:status="retained";exitp=math.nan
            rmax=max(float(source["rho_max"]),float(np.max(np.hypot(sol.y[0],sol.y[1]))))
            if len(sol.t)>1:
                mid=sol.sol((sol.t[:-1]+sol.t[1:])/2);rmax=max(rmax,float(np.max(np.hypot(mid[0],mid[1]))))
            if status=="escaped":rmax=max(1.,rmax)
            row.update({"alpha":alpha,"beta":beta,"kappa":kappa,"status":status,"exit_periods":exitp,"rho_max":rmax,
                        "final_state":json.dumps([float(v) for v in sol.y[:,-1]],separators=(",",":")),"nfev":int(sol.nfev)})
        except Exception as exc:
            row.update({"status":"unresolved","exit_periods":math.nan,"rho_max":math.nan,"final_state":"","nfev":math.nan,
                        "error":f"{type(exc).__name__}:{exc}"})
    row["key"]=key(row);return row

def main():
    p=argparse.ArgumentParser();p.add_argument("--directory",required=True,help="mass-scan subdirectory below the V13.3 atlas")
    p.add_argument("--mu",nargs="+",type=float,required=True);p.add_argument("--workers",type=int,default=4)
    p.add_argument("--state-count",type=int,default=32,help="same initial states per replicate continued at each mass")
    args=p.parse_args()
    if Path(args.directory).name!=args.directory:p.error("directory must be one subdirectory below the V13.3 atlas")
    if not 1<=args.workers<=8:p.error("workers must be 1..8")
    out=BASE/args.directory;source_path=out/"trajectory_results.csv"
    source_rows=read(source_path);source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"));point=manifest["point"]
    mus=sorted(set(round(float(m),12) for m in args.mu));mu_src=sorted({round(float(r["mu"]),12) for r in source_rows})
    missing=set(mus)-set(mu_src)
    if missing:raise SystemExit(f"requested masses absent from scan: {sorted(missing)}")
    if args.state_count<1:p.error("state-count must be positive")
    selected=[r for r in source_rows if int(r["N_RF"])==100 and round(float(r["mu"]),12) in set(mus)
              and int(r["state_i"])<args.state_count]
    if not selected:raise SystemExit("no matching 100-cycle source rows")
    source_index={key({**r,"horizon_RF":100}):r for r in selected}
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run={"script_sha256":script_sha,"source_directory":args.directory,
         "source_trajectory_results_sha256":source_sha,"selected_mu":mus,
         "state_count_per_replicate":args.state_count,
         "continued_horizons_RF":[300,1000],"point":point,"replica_seeds":manifest["replica_seeds"],
         "solver":{"method":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP},
         "continuation":"exact saved endpoint and absolute RF phase; escaped/unresolved states carried without reintegration",
         "unresolved_policy":"explicit; never classified as lost"}
    run_hash=hashlib.sha256(json.dumps(run,sort_keys=True,separators=(",",":"),default=float).encode()).hexdigest();run["run_hash"]=run_hash
    (out/"continuation_manifest.json").write_text(json.dumps(run,indent=2)+"\n",encoding="utf-8")
    ck=out/"continuation_checkpoint.jsonl";done={}
    if ck.exists():
        for line in ck.read_text(encoding="utf-8").splitlines():
            if not line.strip():continue
            r=json.loads(line)
            if r.get("run_hash")!=run_hash:raise RuntimeError("continuation checkpoint/source hash mismatch")
            if r["key"] in done:raise RuntimeError(f"duplicate continuation key {r['key']}")
            done[r["key"]]=r
    for horizon in (300,1000):
        prev_h=100 if horizon==300 else 300;tasks=[]
        previous=source_index if prev_h==100 else done
        for source in selected:
            kprev=key({**source,"horizon_RF":prev_h})
            if kprev not in previous:continue
            prior={**previous[kprev],"horizon_RF":prev_h}
            target={**prior,"horizon_RF":horizon}
            if key(target) in done:continue
            carry=prior["status"] in ("escaped","unresolved")
            start=json.loads(prior["final_state"]) if prior.get("final_state") else [math.nan]*4
            tasks.append((prior,point,horizon,start,carry))
        if tasks:
            with ProcessPoolExecutor(max_workers=args.workers) as pool,ck.open("a",encoding="utf-8") as stream:
                futures=[pool.submit(integrate,t) for t in tasks]
                for future in as_completed(futures):
                    row=future.result();row["run_hash"]=run_hash
                    if row["key"] in done:raise RuntimeError(f"duplicate continuation key {row['key']}")
                    stream.write(json.dumps(row,separators=(",",":"))+"\n");done[row["key"]]=row
                    if len(done)%32==0:stream.flush();os.fsync(stream.fileno())
                stream.flush();os.fsync(stream.fileno())
    rows=list(done.values());write(out/"continuation_results.csv",rows)
    print(json.dumps({"selected_mu":mus,"source_rows":len(selected),"continued_rows":len(rows),"checkpoint":str(ck)},indent=2))
if __name__=="__main__":main()
