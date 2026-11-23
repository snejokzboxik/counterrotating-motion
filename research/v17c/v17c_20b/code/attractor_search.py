from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = HERE.parents[3]
V12 = ROOT / "research" / "v17c" / "v17c_12b"
V17 = ROOT / "research" / "v17c" / "v17c_17"
META = V12 / "results" / "rescaled_legacy_candidate_validation" / "campaign_meta.json"
ROWS = V12 / "results" / "rescaled_legacy_candidate_validation" / "trajectories.jsonl"
LONG_ROWS = V12 / "results" / "long_horizon_refine12_240k_validation" / "trajectories.jsonl"
FIELD_PATH = V12 / "model" / "field_model.json"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

SIM = load_module("v12b_audited_integrator", V12 / "results" / "rescaled_legacy_candidate_validation" / "campaign_integrator_source.py")
PAIR = load_module("v17c17_pair_integrator", V17 / "run_pair_coulomb_scan.py")
TWOPI = 2.0 * math.pi

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load_field():
    c = read_json(FIELD_PATH); coeff = np.asarray(c["coeff"], float)
    return {"family": np.asarray(c["family"], np.int64), "k": np.asarray(c["k"], np.int64), "coeff": coeff[:,0] + 1j*coeff[:,1]}

def load_candidates():
    raw = read_json(META); out = {}
    for c in raw["candidates"]:
        phys = c["physical"]; p = np.asarray(c["p"], float).copy()
        cc = [float(phys["particle1"]["Cunningham"]), float(phys["particle2"]["Cunningham"])]
        lam_stokes = [float(p[6]), float(p[7])]
        p[6] /= cc[0]; p[7] /= cc[1]
        s = float(p[4]); qratio = float(c["qratio"])
        out[c["id"]] = {
            "id": c["id"], "legacy_candidate": c["legacy_candidate"], "p_unadjusted": c["p"],
            "p_corrected": p.tolist(), "qratio": qratio, "mu": 1.0/s, "alpha1": 1.0,
            "alpha2": qratio*s, "physical": phys, "Cunningham": cc,
            "lambda_Stokes": lam_stokes, "lambda_corrected_divide_Cc": [float(p[6]), float(p[7])],
            "f1_Hz": float(phys["frequency_f1_Hz"]), "f2_Hz": float(phys["frequency_f2_Hz"]),
            "a1_over_R": float(phys["a1_over_R"]), "a2_over_R": float(phys["a2_over_R"]),
            "re1_per_speed": float(phys["particle1"]["Re_per_dimensionless_speed"]),
            "re2_per_speed": float(phys["particle2"]["Re_per_dimensionless_speed"]),
            "gravity_z": float(phys["gravity_z_dimensionless"]),
            "mass_ratio_m2_m1": float(1.0/s),
            "diameters_um": [float(phys["particle1"]["diameter_um"]), float(phys["particle2"]["diameter_um"])],
            "densities_kg_m3": [float(phys["particle1"]["density_kg_m3"]), float(phys["particle2"]["density_kg_m3"])],
            "charges_e": [float(phys["particle1"]["charge_e"]), float(phys["particle2"]["charge_e"])],
        }
    return out

def choose_common_rest_start(cases):
    rows = [json.loads(x) for x in ROWS.read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = list(cases); by = {(r["configuration"], r["ensemble"], int(r["start_index"])): r for r in rows}
    candidates = []
    for i in sorted({k[2] for k in by if k[1] == "rest"}):
        rr = [by.get((cid, "rest", i)) for cid in ids]
        if any(x is None or not x["strict_late_category"].startswith("STRICT_CR") for x in rr): continue
        margins = []
        for cid, r in zip(ids, rr):
            c = cases[cid]
            margins += [1.0-max(r["r_max_over_R"])-max(c["a1_over_R"],c["a2_over_R"]),
                        2.0-max(r["z_max_over_R"])-max(c["a1_over_R"],c["a2_over_R"]),
                        r["distance_min_over_R"]-c["a1_over_R"]-c["a2_over_R"]]
        candidates.append((min(margins),-i,i,rr[0]))
    if not candidates: raise RuntimeError("No shared rest start has strict CR labels in all three V12B candidates.")
    _,_,idx,row=max(candidates)
    return {"ensemble":"rest","start_index":idx,"source":"V12B rescaled_legacy_candidate_validation; shared frozen start with strict CR in all candidates",
      "initial":row["initial"],"phase":row["phase"],"seed":row["seed"],
      "selection_rule":"maximize minimum radial-wall, axial, and finite-body pair clearance among shared strict-CR rest starts"}

def choose_refine_long_start():
    rows=[json.loads(x) for x in LONG_ROWS.read_text(encoding="utf-8").splitlines() if x.strip()]
    good=[r for r in rows if r.get("ensemble")=="rest" and r.get("strict_late_category")=="STRICT_CR_PLUS_MINUS" and int(r.get("completed_periods",0))==240000]
    if not good: raise RuntimeError("No completed 240k rest trajectory.")
    row=max(good,key=lambda r:(float(r["min_pair_distance_over_R"]),-int(r["start_index"])))
    return {"ensemble":row["ensemble"],"start_index":int(row["start_index"]),
      "source":"V12B 240000-period refine12_sobol17 selected start; archived record has initial/phase and metrics only, not a sampled state history",
      "initial":row["initial"],"phase":row["phase"],"seed":row.get("seed"),"source_turns":row["turns"],
      "source_min_pair_distance_over_R":row["min_pair_distance_over_R"],"source_completed_periods":row["completed_periods"]}

def build_config():
    cases=load_candidates(); shared=choose_common_rest_start(cases); refine_long=choose_refine_long_start(); field=load_field()
    src=V12/"results"/"rescaled_legacy_candidate_validation"/"campaign_integrator_source.py"
    cfg={"study":"V17C.20B","base_sha":"45866d0559db1615a5e2aebc521ddae90f3a0101",
      "branch":"research/v17c20b-particle-attractor-search",
      "model_source":{"field":str(FIELD_PATH.relative_to(ROOT)).replace("\\","/"),"field_sha256":sha256(FIELD_PATH),
       "integrator":str(src.relative_to(ROOT)).replace("\\","/"),"integrator_sha256":sha256(src),
       "verified_pair_integrator":str((V17/"run_pair_coulomb_scan.py").relative_to(ROOT)).replace("\\","/"),
       "verified_pair_integrator_sha256":sha256(V17/"run_pair_coulomb_scan.py"),
       "harmonics_k":field["k"].tolist(),"harmonics_family":field["family"].tolist(),
       "harmonics_coeff":[[float(x.real),float(x.imag)] for x in field["coeff"]]},
      "model_conventions":{"RF":"full 8-complex-harmonic transverse MFS spectrum; fixed carrier frequencies and independent initial phases",
       "DC":"ideal axisymmetric Laplace quadrupole in the published dimensionless model",
       "drag":"individual lambda_Stokes/Cunningham and Schiller-Naumann factor 1+0.15 Re^0.687; no interparticle hydrodynamic term",
       "gravity":"constant dimensionless vertical acceleration from candidate physical mapping",
       "contact":"finite spherical radii; segment-resolved pair contact and radial/axial escape bounds","polarization":"disabled"},
      "candidates":list(cases.values()),"matched_common_start":shared,"refine12_240k_seed":refine_long}
    write_json(OUT/"candidate_parameters.json",cfg)
    write_json(OUT/"start_conditions.json",{"shared_three_candidate_start":shared,"refine12_240k_replay_start":refine_long})
    return cfg

def arrays(case,field,alpha1=None,alpha2=None,mu=None,qratio=None,lam1=None,lam2=None,C=None):
    alpha1=case["alpha1"] if alpha1 is None else float(alpha1)
    alpha2=case["alpha2"] if alpha2 is None else float(alpha2)
    mu=case["mu"] if mu is None else float(mu)
    C=float(case["p_corrected"][5]) if C is None else float(C)
    lam1=case["p_corrected"][6] if lam1 is None else float(lam1)
    lam2=case["p_corrected"][7] if lam2 is None else float(lam2)
    p=np.asarray(case["p_corrected"],float).copy()
    p[0]*=alpha1; p[1]*=alpha1; p[3]*=alpha1
    p[4]=1.0/mu; p[5]=C; p[6]=lam1; p[7]=lam2
    qratio=mu*alpha2/alpha1 if qratio is None else float(qratio)
    return p,qratio

def run_summary(case,field,start,periods,steps,skip,**overrides):
    p,qratio=arrays(case,field,**overrides)
    raw=SIM.integrate_one(np.asarray(start["initial"],float),np.asarray(start["phase"],float),p,
      field["family"],field["k"],field["coeff"],int(periods),int(steps),int(skip),qratio,
      float(case["re1_per_speed"]),0,float(case["a1_over_R"]),float(case["a2_over_R"]),
      float(case["re2_per_speed"]),float(case["gravity_z"]),2,1.0)
    end,turns,windows,rmin,rmax,zmax,dtheta,dmin,completed,retained,escaped,singular,category,legacyfreq,vmax=raw
    signs=[[0 if abs(v)<0.05 else int(math.copysign(1,v)) for v in w] for w in windows]
    mode="UNRESOLVED"
    if singular: mode="SINGULAR"
    elif category=="finite_particle_contact": mode="PARTICLE_CONTACT"
    elif category=="finite_particle_wall_contact": mode="WALL_CONTACT"
    elif escaped: mode="ESCAPE"
    elif retained and all(s[0]*s[1]<0 for s in signs): mode="SCREEN_CR"
    elif retained and all(s[0]*s[1]>0 for s in signs): mode="SCREEN_CO"
    elif retained and np.max(np.abs(turns))<0.25: mode="NONROTATING_OR_SLOW"
    f=np.asarray(turns,float)*float(case["f1_Hz"])/max(1,int(completed)-int(skip))
    return {"end_state":np.asarray(end,float).tolist(),"turns":np.asarray(turns,float).tolist(),
      "window_turns":np.asarray(windows,float).tolist(),"window_signs":signs,"frequency_Hz":f.tolist(),
      "r_min_R":np.asarray(rmin,float).tolist(),"r_max_R":np.asarray(rmax,float).tolist(),"z_max_R":np.asarray(zmax,float).tolist(),
      "pair_min_R":float(dmin),"max_speed_dimless":np.asarray(vmax,float).tolist(),"completed_periods":int(completed),
      "retained":bool(retained),"escaped":bool(escaped),"singular":bool(singular),"integrator_category":str(category),
      "screen_class":mode,"ten_turn_requirement_met":bool(retained and min(abs(np.asarray(turns,float)))>=10 and all(s[0]*s[1]<0 for s in signs)),
      "screen_steps":int(steps),"screen_periods":int(periods),"skip_periods":int(skip),
      "parameters":{"alpha1":float(overrides.get("alpha1",case["alpha1"])),"alpha2":float(overrides.get("alpha2",case["alpha2"])),
        "mu_m2_m1":float(overrides.get("mu",case["mu"])),"Q_derived":float(qratio),"C12":float(overrides.get("C",case["p_corrected"][5])),
        "lambda1":float(overrides.get("lam1",case["p_corrected"][6])),"lambda2":float(overrides.get("lam2",case["p_corrected"][7]))}}

def run_baselines(cfg,periods=15000,steps=64,skip=1000):
    field=load_field(); start=cfg["matched_common_start"]; summary=[]
    for case in cfg["candidates"]:
        cid=case["id"]; p,qratio=arrays(case,field)
        raw=PAIR._integrate_trace(np.asarray(start["initial"],float),np.asarray(start["phase"],float),p,
          field["family"],field["k"],field["coeff"],qratio,float(case["a1_over_R"]),float(case["a2_over_R"]),
          float(case["re1_per_speed"]),float(case["re2_per_speed"]),float(case["gravity_z"]),
          int(periods),int(steps),int(skip),int(steps*10))
        end,states,idx,wh,turns,windows,rmin,rmax,zmax,vmax,dmin,avgr,avd,avuc,winrm,windm,wrk,completed,status=raw
        f=np.asarray(turns)*case["f1_Hz"]/max(1,completed-skip)
        row={"candidate":cid,"periods":periods,"steps_per_period":steps,"skip":skip,"completed_periods":completed,
          "status":int(status),"retained":bool(status==0),"turns":np.asarray(turns).tolist(),"window_turns":np.asarray(windows).tolist(),
          "frequency_Hz":f.tolist(),"mean_radius_R":np.asarray(avgr).tolist(),"r_min_R":np.asarray(rmin).tolist(),
          "r_max_R":np.asarray(rmax).tolist(),"max_abs_z_R":np.asarray(zmax).tolist(),"pair_min_R":float(dmin),
          "mean_pair_distance_R":float(avd),"mean_Coulomb_potential_dimensionless":float(avuc),
          "initial":start["initial"],"phase":start["phase"],
          "integrator":"V17C.17 Numba RK4 full MFS, Cunningham-corrected lambda and Schiller-Naumann self-drag",
          "ten_turn_CR":bool(status==0 and min(abs(np.asarray(turns)))>=10 and np.all(np.asarray(windows)[:,0]*np.asarray(windows)[:,1]<0))}
        np.savez_compressed(OUT/f"baseline_{cid}_{periods}p_rk4{steps}.npz",states=states,step_index=idx,work_history=wh,
          turns=turns,window_turns=windows,initial=start["initial"],phase=start["phase"],p=p,qratio=qratio,periods=periods,steps=steps,skip=skip)
        summary.append(row)
        print(f"{cid}: f={f[0]:+.6g},{f[1]:+.6g} Hz; turns={turns}; dmin={dmin:.5g}R; status={status}",flush=True)
    write_json(OUT/"baseline_reproduction.json",{"source_sha":"45866d0559db1615a5e2aebc521ddae90f3a0101","rows":summary})
    return summary

def parameter_tasks(cfg):
    scales=[0.6,0.8,1.0,1.2,1.4]
    pairs=[("alpha1","alpha2"),("alpha1","lambda1"),("alpha2","lambda2"),("lambda1","lambda2"),
      ("mu","C12"),("alpha1","C12"),("alpha2","C12")]
    for c in cfg["candidates"]:
      for a,b in pairs:
        for i,x in enumerate(scales):
          for j,y in enumerate(scales):
            if x==1.0 and y==1.0: continue
            yield c,a,b,x,y,i,j

def override_point(case,a,b,x,y):
    base={"alpha1":case["alpha1"],"alpha2":case["alpha2"],"lambda1":case["p_corrected"][6],
      "lambda2":case["p_corrected"][7],"mu":case["mu"],"C12":case["p_corrected"][5]}
    d={}
    for key,mult in ((a,x),(b,y)):
      if key=="C12": d["C"]=base[key]*mult
      elif key=="lambda1": d["lam1"]=base[key]*mult
      elif key=="lambda2": d["lam2"]=base[key]*mult
      else: d[key]=base[key]*mult
    return d

def run_parameter_grid(cfg,periods=3000,steps=32,workers=4):
    field=load_field(); start=cfg["matched_common_start"]; path=OUT/"parameter_grid_checkpoint.jsonl"
    done=set()
    if path.exists():
      for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): done.add(json.loads(line)["run_id"])
    tasks=list(parameter_tasks(cfg)); first=tasks[0]
    run_summary(first[0],field,start,2,steps,0,**override_point(first[0],first[1],first[2],1.0,1.0))
    def one(t):
      c,a,b,x,y,i,j=t; ov=override_point(c,a,b,x,y)
      r=run_summary(c,field,start,periods,steps,periods//5,**ov)
      r.update({"run_id":f"{c['id']}|{a}:{x:.1f}|{b}:{y:.1f}","candidate":c["id"],
        "slice":f"{a}__{b}","axis_x":a,"axis_y":b,"multiplier_x":x,"multiplier_y":y,
        "start_index":start["start_index"],"start_ensemble":start["ensemble"]})
      return r
    todo=[t for t in tasks if f"{t[0]['id']}|{t[1]}:{t[3]:.1f}|{t[2]}:{t[4]:.1f}" not in done]
    t0=time.perf_counter(); n=0
    with path.open("a",encoding="utf-8",buffering=1) as f:
      with ThreadPoolExecutor(max_workers=workers) as pool:
        futs=[pool.submit(one,t) for t in todo]
        for future in as_completed(futs):
          row=future.result(); f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n"); f.flush(); n+=1
          if n%25==0 or n==len(todo): print(f"parameter grid {n}/{len(todo)} new; elapsed {time.perf_counter()-t0:.1f}s",flush=True)
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    with (OUT/"parameter_grid.csv").open("w",newline="",encoding="utf-8") as f:
      keys=["run_id","candidate","slice","axis_x","axis_y","multiplier_x","multiplier_y","alpha1","alpha2","mu_m2_m1","Q_derived","C12","lambda1","lambda2","screen_class","ten_turn_requirement_met","retained","frequency_Hz","pair_min_R"]
      w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
      for r in rows:
        d={k:r.get(k) for k in keys};d.update(r.get("parameters",{}));w.writerow(d)
    write_json(OUT/"parameter_grid_audit.json",{"n":len(rows),"unique_ids":len({r['run_id'] for r in rows}),
      "candidate_counts":{c:sum(r["candidate"]==c for r in rows) for c in sorted({r["candidate"] for r in rows})},
      "periods":periods,"steps_per_rf":steps,"grid_multipliers":[0.6,0.8,1.0,1.2,1.4],
      "short_scan_interpretation":"Signs and retention only; this horizon does not satisfy ten-turn stability criterion for these slow orbits."})

def run_coupling_grid(cfg,periods=5000,steps=32,workers=4):
    field=load_field();start=cfg["matched_common_start"];path=OUT/"coulomb_sensitivity_checkpoint.jsonl"
    done=set()
    if path.exists():
      for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():done.add(json.loads(line)["run_id"])
    scales=[0,.1,.25,.5,.75,1,1.5,2,3,5]
    tasks=[(c,s) for c in cfg["candidates"] for s in scales]
    c,s=tasks[0];run_summary(c,field,start,2,steps,0,C=c["p_corrected"][5]*s)
    def one(t):
      c,s=t;r=run_summary(c,field,start,periods,steps,periods//5,C=c["p_corrected"][5]*s)
      r.update({"run_id":f"{c['id']}|sC={s:g}","candidate":c["id"],"sC":s,"start_index":start["start_index"]})
      return r
    todo=[t for t in tasks if f"{t[0]['id']}|sC={t[1]:g}" not in done];t0=time.perf_counter()
    with path.open("a",encoding="utf-8",buffering=1) as f:
      with ThreadPoolExecutor(max_workers=workers) as pool:
        futs=[pool.submit(one,t) for t in todo]
        for n,fu in enumerate(as_completed(futs),1):
          r=fu.result();f.write(json.dumps(r,separators=(",",":"),allow_nan=False)+"\n");f.flush()
          if n%10==0 or n==len(todo):print(f"C grid {n}/{len(todo)} new; elapsed {time.perf_counter()-t0:.1f}s",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["config","baseline","grid","coulomb","all"],default="config")
    ap.add_argument("--periods",type=int,default=15000);ap.add_argument("--steps",type=int,default=64);ap.add_argument("--skip",type=int,default=1000)
    ap.add_argument("--screen-periods",type=int,default=3000);ap.add_argument("--screen-steps",type=int,default=32);ap.add_argument("--workers",type=int,default=4)
    a=ap.parse_args();cfg=build_config()
    if a.stage in ("baseline","all"):run_baselines(cfg,a.periods,a.steps,a.skip)
    if a.stage in ("grid","all"):run_parameter_grid(cfg,a.screen_periods,a.screen_steps,a.workers)
    if a.stage in ("coulomb","all"):run_coupling_grid(cfg,max(a.screen_periods,5000),a.screen_steps,a.workers)

if __name__=="__main__": main()
