"""Adaptive paired refinement with independent scrambled Sobol starts and continuation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
PHASES=(0.0,math.pi/2)
N_PER_REPLICA=64
REPLICA_SEEDS=(731337,913729,20261114,8675309)
HORIZONS=(100,300,1000)
RTOL,ATOL=2e-10,2e-12
MAX_STEP=2*math.pi/64
APERTURE=1.0


def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    if not rows:return
    fields=[]
    for row in rows:fields.extend(k for k in row if k not in fields)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)


def select_candidates() -> list[dict]:
    atlas=read_csv(OUT/"atlas_retention_physical_summary.csv")
    idx={(int(r["alpha_i"]),int(r["beta_i"]),int(r["eta_i"]),int(r["phase_i"])):r for r in atlas}
    phases=[0,1]
    candidates=[]
    for eta_i in sorted({int(r["eta_i"]) for r in atlas if r["model"]=="RO"}):
        eta=float(next(r["eta"] for r in atlas if int(r["eta_i"])==eta_i and r["model"]=="RO"))
        scored=[]
        for ai in range(17):
          for bi in range(21):
            ro=[idx.get((ai,bi,eta_i,p)) for p in phases]
            q=[idx.get((ai,bi,4,p)) for p in phases]
            if any(r is None for r in ro+q) or not all(str(r["within_250V"]).lower()=="true" for r in ro):continue
            prs=[float(r["P_ret"]) for r in ro];pqs=[float(r["P_ret"]) for r in q]
            if not np.all(np.isfinite(prs+pqs)):continue
            delta=float(np.mean(prs)-np.mean(pqs))
            phase_sensitivity=abs(prs[1]-prs[0])
            grad=0.0
            for aj,bj in ((ai-1,bi),(ai+1,bi),(ai,bi-1),(ai,bi+1)):
                if 0<=aj<17 and 0<=bj<21:
                    neighbor=idx.get((aj,bj,eta_i,0))
                    if neighbor is not None and np.isfinite(float(neighbor["P_ret"])):
                        grad=max(grad,abs(prs[0]-float(neighbor["P_ret"])))
            floquet=str(ro[0]["floquet_status"])
            outside_nonlinear= floquet!="strictly_stable" and float(np.mean(prs))>=0.5
            inside_losses= floquet=="strictly_stable" and float(np.mean(prs))<=0.5
            score=max(abs(delta),grad,phase_sensitivity,0.75 if outside_nonlinear or inside_losses else 0.0)
            base={"alpha_i":ai,"beta_i":bi,"eta_i":eta_i,"alpha":float(ro[0]["alpha"]),
                  "beta":float(ro[0]["beta"]),"eta":eta,"kappa":float(ro[0]["kappa"]),
                  "P_RO_phase0":prs[0],"P_RO_phase90":prs[1],
                  "P_Q_phase0":pqs[0],"P_Q_phase90":pqs[1],
                  "mean_RO_minus_Q":delta,"RO_phase_sensitivity":phase_sensitivity,
                  "neighbor_gradient":grad,"floquet_status":floquet,
                  "nonlinear_retention_outside_Floquet":outside_nonlinear,
                  "nonlinear_loss_inside_Floquet":inside_losses,"score":score}
            scored.append(base)
        # Pick both signs of response per eta when present, then sharp boundaries.
        picked=[]
        positive=sorted((r for r in scored if r["mean_RO_minus_Q"]>0),key=lambda r:r["mean_RO_minus_Q"],reverse=True)
        negative=sorted((r for r in scored if r["mean_RO_minus_Q"]<0),key=lambda r:r["mean_RO_minus_Q"])
        gradients=sorted(scored,key=lambda r:r["neighbor_gradient"],reverse=True)
        special=sorted((r for r in scored if r["nonlinear_retention_outside_Floquet"] or r["nonlinear_loss_inside_Floquet"]),key=lambda r:r["score"],reverse=True)
        for group,label in ((positive,"largest_RO_gain"),(negative,"largest_RO_loss"),
                            (gradients,"sharp_local_change"),(special,"Floquet_nonlinear_mismatch")):
            for r in group:
                if not any((q["alpha_i"],q["beta_i"])==(r["alpha_i"],r["beta_i"]) for q in picked):
                    picked.append({**r,"selection_reason":label});break
        candidates.extend(picked)
    # Keep a manageable, diverse set; include the strongest phase-sensitive points.
    unique={}
    for row in candidates:
        key=(row["alpha_i"],row["beta_i"],row["eta_i"])
        if key not in unique or row["score"]>unique[key]["score"]:unique[key]=row
    selected=sorted(unique.values(),key=lambda r:r["score"],reverse=True)[:24]
    best_by_eta=[]
    for eta_i in sorted({r["eta_i"] for r in selected}):
        best_by_eta.append(max((r for r in selected if r["eta_i"]==eta_i),key=lambda r:r["score"]))
    long_selected=[]
    for sign in (-1,1):
        pool=[r for r in best_by_eta if np.sign(r["eta"])==sign]
        if pool:
            first=max(pool,key=lambda r:(abs(r["mean_RO_minus_Q"]),r["RO_phase_sensitivity"],r["score"]))
            long_selected.append(first)
            remaining=[r for r in pool if r["eta_i"]!=first["eta_i"]]
            if remaining:long_selected.append(max(remaining,key=lambda r:(r["score"],abs(r["mean_RO_minus_Q"]),r["neighbor_gradient"])))
    if len(long_selected)<4:
        remaining=[r for r in sorted(best_by_eta,key=lambda r:r["score"],reverse=True)
                   if r not in long_selected]
        long_selected.extend(remaining[:4-len(long_selected)])
    long_keys={(r["alpha_i"],r["beta_i"],r["eta_i"]) for r in long_selected}
    selected=[{**r,"long_horizon_selected":(r["alpha_i"],r["beta_i"],r["eta_i"]) in long_keys}
              for r in selected]
    selected.sort(key=lambda r:(r["eta_i"],r["alpha_i"],r["beta_i"]))
    return selected


def sobol_starts() -> list[dict]:
    rows=[]
    for rep,seed in enumerate(REPLICA_SEEDS):
        points=qmc.Sobol(4,scramble=True,seed=seed).random_base2(6)
        for i,p in enumerate(points):
            rows.append({"replica":rep,"sobol_seed":seed,"state_i":i,
                         "x0_rho":float((p[0]-.5)*.5),"y0_rho":float((p[1]-.5)*.5),
                         "vx0_rho_tau":float((p[2]-.5)*.08),"vy0_rho_tau":float((p[3]-.5)*.08)})
    return rows


def adaptive_midpoints(centers):
    """Add real half-cell parameter nodes around the strongest distinct-eta centers."""
    e=1.602176634e-19;amu=1.66053906660e-27
    m0=3200.04652261404*amu;omega=2*math.pi*100000.;radius=.008
    c2,d2,c4=.7718255558274212,.7718255558274212,.5413395699439091
    scale=m0*omega**2*radius**2/e
    with (OUT.parent/"v13_1"/"electrode_voltage_peaks.csv").open(newline="",encoding="utf-8") as f:
        electrodes=[{k:float(row[k]) for k in ("C4","S4","Q2")} for row in csv.DictReader(f) if row["scheme"]=="Q"]
    def feasible(alpha,beta,eta):
        u2,vdc,u4=beta*scale/c2,alpha*scale/d2,eta*beta*scale/c4
        peaks=[abs(vdc*r["Q2"])+math.hypot(u2*r["Q2"]+u4*r["C4"],u4*r["S4"])
               for r in electrodes]
        return max(peaks)<=250.0
    best_by_eta=[]
    for eta_i in sorted({r["eta_i"] for r in centers}):
        best_by_eta.append(max((r for r in centers if r["eta_i"]==eta_i),key=lambda r:r["score"]))
    chosen=[]
    for sign in (-1,1):
        pool=[r for r in best_by_eta if np.sign(r["eta"])==sign]
        if pool:
            first=max(pool,key=lambda r:(abs(r.get("mean_RO_minus_Q",0.)),r["RO_phase_sensitivity"],r["score"]))
            chosen.append(first)
            remaining=[r for r in pool if r["eta_i"]!=first["eta_i"]]
            if remaining:chosen.append(max(remaining,key=lambda r:(r["score"],abs(r.get("mean_RO_minus_Q",0.)),r["neighbor_gradient"])))
    if len(chosen)<4:
        chosen.extend([r for r in sorted(best_by_eta,key=lambda r:r["score"],reverse=True)
                       if r not in chosen][:4-len(chosen)])
    alpha_grid=np.linspace(-.05,.05,17);beta_grid=np.linspace(.05,.30,21)
    added=[];seen=set()
    for center in chosen:
        for da,db in ((-.5,0),(.5,0),(0,-.5),(0,.5)):
            ia=float(center["alpha_i"]+da);ib=float(center["beta_i"]+db)
            if not (0<=ia<=16 and 0<=ib<=20):continue
            alpha=float(np.interp(ia,np.arange(17),alpha_grid))
            beta=float(np.interp(ib,np.arange(21),beta_grid))
            eta=float(center["eta"])
            key=(round(alpha,12),round(beta,12),int(center["eta_i"]))
            if key in seen or not feasible(alpha,beta,eta):continue
            seen.add(key)
            added.append({"alpha_i":ia,"beta_i":ib,"eta_i":int(center["eta_i"]),
                          "alpha":alpha,"beta":beta,"eta":eta,"kappa":eta*beta,
                          "score":float(center["score"]*.8),
                          "selection_reason":"actual half-cell refinement adjacent to coarse candidate",
                          "parent_alpha_i":center["alpha_i"],"parent_beta_i":center["beta_i"],
                          "long_horizon_selected":False,"within_250V":True})
    return added


def _segment(task):
    point,model,rep,phase_i,phase,state_i,state,start_horizon,stop_horizon,old_max=task
    from three_phase_trap.mass_filter_v13_3 import acceleration
    alpha,beta,eta=point["alpha"],point["beta"],point["eta"]
    kappa=0.0 if model=="Q" else eta*beta
    y0=np.asarray(state,float)
    def rhs(t,y):
        ax,ay=acceleration(y[0],y[1],t,kappa,beta,alpha,delta=0.0,tau0=phase)
        return np.array([y[2],y[3],ax,ay])
    def event(t,y):return math.hypot(y[0],y[1])-APERTURE
    event.terminal=True;event.direction=1.0
    interval=2*math.pi*(stop_horizon-start_horizon)
    row={"candidate_id":f"a{point['alpha_i']}_b{point['beta_i']}_e{point['eta_i']}",
         "alpha_i":point["alpha_i"],"beta_i":point["beta_i"],"eta_i":point["eta_i"],
         "alpha":alpha,"beta":beta,"kappa":kappa,"model":model,
         "replica":rep,"sobol_seed":REPLICA_SEEDS[rep],"phase_i":phase_i,
         "tau0_rad":phase,"tau0_deg":math.degrees(phase),"state_i":state_i,
         "start_horizon":start_horizon,"horizon":stop_horizon,
         "x0_rho":float(state[0]),"y0_rho":float(state[1]),
         "vx0_rho_tau":float(state[2]),"vy0_rho_tau":float(state[3]),
         "integrator":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP}
    try:
        sol=solve_ivp(rhs,(0,interval),y0,method="DOP853",rtol=RTOL,atol=ATOL,
                      max_step=MAX_STEP,events=event,dense_output=True)
        if not sol.success: status="unresolved";exitp=math.nan
        elif len(sol.t_events[0]):status="escaped";exitp=start_horizon+float(sol.t_events[0][0]/(2*math.pi))
        else:status="retained";exitp=math.nan
        rmax=max(float(old_max),float(np.max(np.hypot(sol.y[0],sol.y[1]))))
        if len(sol.t)>1:
            mid=sol.sol((sol.t[:-1]+sol.t[1:])/2)
            rmax=max(rmax,float(np.max(np.hypot(mid[0],mid[1]))))
        if status=="escaped":rmax=max(rmax,APERTURE)
        final=[float(v) for v in sol.y[:,-1]]
        row.update({"status":status,"exit_periods":exitp,"rho_max_cumulative":rmax,
                    "final_state":json.dumps(final,separators=(",",":")),"nfev":int(sol.nfev)})
    except Exception as exc:
        row.update({"status":"unresolved","exit_periods":math.nan,"rho_max_cumulative":math.nan,
                    "final_state":"","nfev":math.nan,"error":f"{type(exc).__name__}:{exc}"})
    row["key"]="|".join(str(v) for v in (row["candidate_id"],model,rep,phase_i,state_i,stop_horizon))
    return row


def write_jsonl(path,rows):
    with Path(path).open("a",encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
        stream.flush();os.fsync(stream.fileno())


def run_and_checkpoint(tasks,ck,run_hash,workers,completed):
    if not tasks:return
    with ProcessPoolExecutor(max_workers=workers) as pool,Path(ck).open("a",encoding="utf-8") as stream:
        futures=[pool.submit(_segment,t) for t in tasks]
        count=0
        for fut in as_completed(futures):
            row=fut.result();row["run_hash"]=run_hash
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
            completed[row["key"]]=row;count+=1
            if count%20==0:
                stream.flush();os.fsync(stream.fileno())
        stream.flush();os.fsync(stream.fileno())


def refinement_summary(rows,candidates):
    from scipy.stats import t as student_t
    index={}
    for r in rows:
        key=(r["candidate_id"],r["model"],int(r["horizon"]),int(r["phase_i"]),int(r["replica"]))
        index.setdefault(key,[]).append(r)
    out=[]
    for point in candidates:
      ro_id=f"a{point['alpha_i']}_b{point['beta_i']}_e{point['eta_i']}"
      q_id=f"a{point['alpha_i']}_b{point['beta_i']}_e4"
      horizons=sorted({int(r["horizon"]) for r in rows if r["candidate_id"]==ro_id and r["model"]=="RO"})
      for horizon in horizons:
       for phase_i,phase in enumerate(PHASES):
        rep_ro=[];rep_q=[];rep_d=[];rep_common=[];unresolved=0
        for rep in range(4 if not len({r["replica"] for r in rows})==1 else 1):
            rr=index.get((ro_id,"RO",horizon,phase_i,rep),[])
            qr=index.get((q_id,"Q",horizon,phase_i,rep),[])
            if not rr or not qr:continue
            resolved_ro=[r for r in rr if r["status"] in ("retained","escaped")]
            resolved_q=[r for r in qr if r["status"] in ("retained","escaped")]
            unresolved+=len(rr)-len(resolved_ro)
            if resolved_ro:rep_ro.append(np.mean([r["status"]=="retained" for r in resolved_ro]))
            if resolved_q:rep_q.append(np.mean([r["status"]=="retained" for r in resolved_q]))
            qby={int(r["state_i"]):r for r in resolved_q}
            pairs=[(int(r["status"]=="retained")-int(qby[int(r["state_i"])]["status"]=="retained"))
                   for r in resolved_ro if int(r["state_i"]) in qby]
            if pairs:rep_d.append(float(np.mean(pairs)));rep_common.append(len(pairs))
        def estimate(v,lo,hi):
            if not v:return (math.nan,math.nan,math.nan,math.nan)
            mu=float(np.mean(v));sd=float(np.std(v,ddof=1)) if len(v)>1 else math.nan
            se=sd/math.sqrt(len(v)) if len(v)>1 else math.nan
            tc=float(student_t.ppf(.975,len(v)-1)) if len(v)>1 else math.nan
            return mu,sd,max(lo,mu-tc*se) if np.isfinite(se) else math.nan,min(hi,mu+tc*se) if np.isfinite(se) else math.nan
        p_ro=estimate(rep_ro,0,1);p_q=estimate(rep_q,0,1);d=estimate(rep_d,-1,1)
        out.append({"candidate_id":ro_id,"alpha_i":point["alpha_i"],"beta_i":point["beta_i"],
                    "eta_i":point["eta_i"],"alpha":point["alpha"],"beta":point["beta"],
                    "eta":point["eta"],"kappa":point["kappa"],"phase_deg":math.degrees(phase),
                    "horizon_RF":horizon,"n_replicates":len(rep_d),"n_per_replica":N_PER_REPLICA,
                    "RO_P_ret_rep_mean":p_ro[0],"RO_rep_sd":p_ro[1],
                    "RO_QMC_t95_low":p_ro[2],"RO_QMC_t95_high":p_ro[3],
                    "Q_P_ret_rep_mean":p_q[0],"Q_rep_sd":p_q[1],
                    "Q_QMC_t95_low":p_q[2],"Q_QMC_t95_high":p_q[3],
                    "paired_RO_minus_Q_rep_mean":d[0],"paired_diff_rep_sd":d[1],
                    "paired_QMC_t95_low":d[2],"paired_QMC_t95_high":d[3],
                    "mean_common_pairs_per_replica":float(np.mean(rep_common)) if rep_common else 0,
                    "unresolved_RO_total":unresolved,
                    "warning":"CI is based on independent scrambled-Sobol replicate means, not pooled Bernoulli points."})
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--max-candidates",type=int,default=24)
    p.add_argument("--pilot",action="store_true",help="Use first two candidates and one Sobol replicate (not a final estimate).")
    p.add_argument("--select-only",action="store_true",help="Write the adaptive parameter candidates without launching integrations.")
    args=p.parse_args()
    if not (1<=args.workers<=8):p.error("--workers must be in 1..8")
    coarse=select_candidates()[:args.max_candidates]
    if args.pilot:
        candidates=coarse[:2]
    else:
        candidates=coarse+adaptive_midpoints(coarse)
    if not candidates:raise SystemExit("No physically feasible refinement candidates were selected.")
    OUT.mkdir(parents=True,exist_ok=True)
    cand_path=OUT/"adaptive_candidate_points.csv"
    write_csv(cand_path,candidates)
    if args.select_only:
        print(json.dumps({"coarse_candidates":len(coarse),"half_step_refinements":len(candidates)-len(coarse),
                          "candidate_file":str(cand_path)},indent=2))
        return
    starts=sobol_starts()[:N_PER_REPLICA*(1 if args.pilot else len(REPLICA_SEEDS))]
    start_path=OUT/"adaptive_initial_conditions.csv"
    write_csv(start_path,starts)
    if args.pilot:
        seed_count=1
        active_seeds=REPLICA_SEEDS[:1]
    else:
        seed_count=len(REPLICA_SEEDS)
        active_seeds=REPLICA_SEEDS
    protocol={"candidates":candidates,"starts":starts,"phases":PHASES,"horizons":HORIZONS,
              "seeds":active_seeds,"rtol":RTOL,"atol":ATOL,"max_step":MAX_STEP,
              "aperture":APERTURE,"delta":0.0}
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol["script_sha256"]=script_sha
    run_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(",",":"),default=float).encode()).hexdigest()
    manifest_path=OUT/("adaptive_pilot_manifest.json" if args.pilot else "adaptive_run_manifest.json")
    if manifest_path.exists():
        old_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("run_hash")!=run_hash:
            raise RuntimeError("Adaptive checkpoint does not match selected points, Sobol starts, or integration protocol.")
    else:
        manifest_path.write_text(json.dumps({"run_hash":run_hash,"protocol":protocol},indent=2,default=float),encoding="utf-8")
    ck=OUT/("adaptive_pilot_checkpoint.jsonl" if args.pilot else "adaptive_checkpoint.jsonl")
    completed={}
    if ck.exists():
        with ck.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row=json.loads(line)
                    if row.get("run_hash")!=run_hash:raise RuntimeError("Adaptive checkpoint run-hash mismatch.")
                    if row["key"] in completed:raise RuntimeError(f"Duplicate adaptive key: {row['key']}")
                    completed[row["key"]]=row
    starts_by={(r["replica"],r["state_i"]):[r["x0_rho"],r["y0_rho"],r["vx0_rho_tau"],r["vy0_rho_tau"]]
               for r in starts}
    tasks=[]
    # Q is calculated once per alpha/beta and reused for every eta paired at that location.
    q_points={}
    for point in candidates:
        key=(point["alpha_i"],point["beta_i"])
        q_points.setdefault(key,{**point,"eta_i":4,"eta":0.0,"kappa":0.0})
    for model_points,model in ((candidates,"RO"),(list(q_points.values()),"Q")):
      for point in model_points:
        for rep in range(seed_count):
          for phase_i,phase in enumerate(PHASES):
            for state_i in range(N_PER_REPLICA):
              state=starts_by[(rep,state_i)]
              key=f"a{point['alpha_i']}_b{point['beta_i']}_e{point['eta_i']}|{model}|{rep}|{phase_i}|{state_i}|100"
              if key not in completed:
                tasks.append((point,model,rep,phase_i,phase,state_i,state,0,100,0.0))
    run_and_checkpoint(tasks,ck,run_hash,args.workers,completed)

    # Continue the same retained trajectories without reinitializing state or RF phase.
    long_candidates=[r for r in candidates if bool(r.get("long_horizon_selected",False))]
    long_q_points={}
    for point in long_candidates:
        key=(point["alpha_i"],point["beta_i"])
        long_q_points.setdefault(key,{**point,"eta_i":4,"eta":0.0,"kappa":0.0})
    for start_horizon,stop_horizon in ((100,300),(300,1000)):
        tasks=[];carried=[]
        long_models=((long_candidates,"RO"),(list(long_q_points.values()),"Q"))
        for model_points,model in long_models:
          for point in model_points:
            for rep in range(seed_count):
              for phase_i,phase in enumerate(PHASES):
                for state_i in range(N_PER_REPLICA):
                  key0=f"a{point['alpha_i']}_b{point['beta_i']}_e{point['eta_i']}|{model}|{rep}|{phase_i}|{state_i}|{start_horizon}"
                  key=f"a{point['alpha_i']}_b{point['beta_i']}_e{point['eta_i']}|{model}|{rep}|{phase_i}|{state_i}|{stop_horizon}"
                  if key in completed or key0 not in completed:continue
                  previous=completed[key0]
                  if previous["status"]!="retained":
                    row={**previous,"horizon":stop_horizon,"start_horizon":start_horizon,
                         "status":previous["status"],"carried_status":True,"key":key}
                    completed[key]=row;carried.append(row)
                    continue
                  state=json.loads(previous["final_state"])
                  tasks.append((point,model,rep,phase_i,phase,state_i,state,
                                start_horizon,stop_horizon,float(previous["rho_max_cumulative"])))
        if carried:write_jsonl(ck,carried)
        run_and_checkpoint(tasks,ck,run_hash,args.workers,completed)
        # Persist carried rows as well, with all terminal survivors/losses explicit.
        # Carried statuses are checkpointed separately and do not count as lost trajectories.

    all_rows=list(completed.values())
    write_csv(OUT/("adaptive_pilot_results.csv" if args.pilot else "adaptive_refinement_results.csv"),all_rows)
    if not args.pilot:
        write_csv(OUT/"adaptive_refinement_summary.csv",refinement_summary(all_rows,candidates))
    config={"candidate_count":len(candidates),"sobol_n_per_replica":N_PER_REPLICA,
            "replica_seeds":list(active_seeds),"replicas":seed_count,
            "long_horizon_candidate_count":len(long_candidates),
            "long_horizon_selection":"highest score at up to four distinct eta values",
            "initial_box":{"x_rho":[-.25,.25],"y_rho":[-.25,.25],
                           "vx_rho_tau":[-.04,.04],"vy_rho_tau":[-.04,.04]},
            "phases_rad":PHASES,"horizons":HORIZONS,"continuation":"state and max radius checkpointed at each integer-cycle horizon; drive phase is unchanged because all segment boundaries are integer RF periods",
            "delta_rad":0.0,"integrator":{"method":"DOP853","rtol":RTOL,"atol":ATOL,"max_step_tau":MAX_STEP},
            "aperture_rho":APERTURE,"unresolved_policy":"separate from retained/escaped",
            "selection":"top physically feasible nonlinear differences, sharp neighbor changes, and Floquet/retention mismatches from the completed coarse atlas",
            "run_hash":run_hash,"script_sha256":script_sha}
    (OUT/("adaptive_pilot_config.json" if args.pilot else "adaptive_refinement_config.json")).write_text(json.dumps(config,indent=2),encoding="utf-8")
    print(json.dumps({"candidates":len(candidates),"task_rows":len(all_rows),"checkpoint":str(ck)},indent=2))


if __name__=="__main__":main()
