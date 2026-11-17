"""Checkpointed ideal-field A9 mass-ray scan on common physical 4D starts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import sem, t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v15/mass_ray_a9"
CONF = ROOT / "outputs/octupole_comparison_home/mass_filter_v15/confirmation_a9_uniform4d"
SRC = CONF / "checkpoint.jsonl"
ALPHA, BETA, KAPPA = 0.00625, 0.075, -0.015
M0_TH = 3200.04652261404
R_M = 0.008
F_HZ = 100_000.0
OMEGA = 2 * math.pi * F_HZ
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
M0_KG = M0_TH * AMU
MASS_RATIOS = (0.75, 0.85, 0.90, 1.00, 1.10, 1.20, 1.30)
MODELS = (("Q", "Q", 0.0), ("SO_equal_harmonic", "SO", 1.0),
          ("SO_equal_rms", "SO", math.sqrt(2)), ("RO_PLUS", "RO_PLUS", 1.0),
          ("RO_MINUS", "RO_MINUS", 1.0))
STEPS_PER_RF = 256


def common_starts():
    starts = {}
    for line in SRC.read_text(encoding="utf8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        if (r["config"] == "Q" and r["delta_rad"] == 0 and r["tau0_rad"] == 0
                and r["horizon_rf"] == 100):
            key = (int(r["replica"]), int(r["pair_index"]), int(r["Lz_sign"]))
            starts[key] = {"state": np.array([r[k] for k in ("x0", "y0", "vx0", "vy0")], float),
                            "seed": int(r["seed"])}
    expected = 8 * 128 * 2
    if len(starts) != expected:
        raise RuntimeError(f"expected {expected} unique starts, got {len(starts)}")
    return starts


def voltage_peak(config, mult):
    # A9 electrode basis at theta_j=j*pi/6, phase delta=tau0=0.
    scale = M0_KG * OMEGA**2 * R_M**2 / E_CHARGE
    u2 = BETA * scale / 0.7718255558274212
    dc = ALPHA * scale / 0.7718255558274212
    u4 = KAPPA * mult * scale / 0.5413395699439091
    peaks = []
    for j in range(12):
        th = math.pi*j/6
        q2, c4, s4 = math.cos(2*th), math.cos(4*th), math.sin(4*th)
        if config == "Q":
            a, b = u2*q2, 0.0
        elif config.startswith("SO"):
            a, b = u2*q2+u4*c4, 0.0
        else:
            sig = 1 if config == "RO_PLUS" else -1
            a, b = u2*q2+u4*c4, sig*u4*s4
        peaks.append(abs(dc*q2)+math.hypot(a,b))
    return max(peaks)


def summarise(rows):
    reps = sorted({int(r["replica"]) for r in rows})
    keys = [(r["mu"], r["config"], r["horizon_rf"]) for r in rows]
    summary = []
    for mu, cfg, horizon in sorted(set(keys)):
        group = [r for r in rows if (r["mu"], r["config"], r["horizon_rf"]) == (mu, cfg, horizon)]
        repstats = []
        for rep in reps:
            rr = [r for r in group if int(r["replica"]) == rep]
            if not rr:
                continue
            plus = [r for r in rr if int(r["Lz_sign"]) > 0]
            minus = [r for r in rr if int(r["Lz_sign"]) < 0]
            pp = np.mean([r["status"] == "retained" for r in plus])
            pm = np.mean([r["status"] == "retained" for r in minus])
            repstats.append((float(pp), float(pm), float(np.mean([r["status"] == "retained" for r in rr]))))
        for i, metric in enumerate(("P_plus", "P_minus", "P_total")):
            vals = np.array([x[i] for x in repstats])
            mean = float(np.mean(vals))
            half = float(t.ppf(.975, len(vals)-1)*sem(vals)) if len(vals) > 1 and np.std(vals,ddof=1)>0 else 0.
            summary.append({"mu":mu,"config":cfg,"horizon_rf":horizon,"metric":metric,
                            "mean":mean,"ci95_low":mean-half,"ci95_high":mean+half,
                            "n_scrambles":len(vals),"replicate_values":json.dumps(vals.tolist())})
        vals = np.array([x[0]-x[1] for x in repstats])
        mean = float(np.mean(vals)); half = float(t.ppf(.975,len(vals)-1)*sem(vals)) if len(vals)>1 and np.std(vals,ddof=1)>0 else 0.
        summary.append({"mu":mu,"config":cfg,"horizon_rf":horizon,"metric":"Delta_chi",
                        "mean":mean,"ci95_low":mean-half,"ci95_high":mean+half,
                        "n_scrambles":len(vals),"replicate_values":json.dumps(vals.tolist())})
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(ROOT/"src"))
    from three_phase_trap.v15_dynamics import integrate_rk4_batch

    starts = common_starts()
    manifest = {"study":"V15 ideal A9 physical-voltage mass ray", "candidate":"A9",
        "alpha0":ALPHA,"beta0":BETA,"kappa0":KAPPA,"mu_grid":MASS_RATIOS,
        "mass_scaling":"alpha,beta,kappa proportional to 1/mu at fixed electrode voltages",
        "initial_domain":"same 8 scrambled Sobol replicates x 128 mirrored pairs from A9 confirmation; r area-uniform <=7.5 mm; E_perp area-uniform <=5 eV",
        "initial_physical_energy":"fixed in eV across masses; dimensionless velocities are multiplied by mu^(-1/2)",
        "tau0_rad":0.0,"delta_rad":0.0,"models":MODELS,"horizons_rf":[100,300],
        "steps_per_rf":STEPS_PER_RF,"aperture_rho":1.0,"voltage_limit_V":250.0,
        "seed_data_sha256":hashlib.sha256(SRC.read_bytes()).hexdigest(),
        "field_model_sha256":hashlib.sha256((ROOT/"src/three_phase_trap/v15_dynamics.py").read_bytes()).hexdigest(),
        "interpretation":"finite-time transverse retention only; not finite-length filter transmission"}
    mp = OUT/"manifest.json"
    normalized = json.loads(json.dumps(manifest, sort_keys=True, default=float))
    if mp.exists() and json.loads(mp.read_text(encoding="utf8")) != normalized:
        raise RuntimeError("mass-ray manifest mismatch")
    if not mp.exists(): mp.write_text(json.dumps(normalized,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    manifest_hash = hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(",",":"),default=float).encode()).hexdigest()
    done = {}
    ck = OUT/"checkpoint.jsonl"
    if ck.exists():
        for line in ck.read_text(encoding="utf8").splitlines():
            if not line: continue
            r=json.loads(line)
            if r["manifest_hash"]!=manifest_hash: raise RuntimeError("checkpoint protocol mismatch")
            if r["key"] in done: raise RuntimeError("duplicate checkpoint key")
            done[r["key"]]=r

    with ck.open("a",encoding="utf8") as f:
      for mu in MASS_RATIOS:
        for name,model,mult in MODELS:
          complete = True
          for horizon in (100,300):
            for rep in range(8):
              for pair in range(128):
                for sign in (-1,1):
                  k="|".join(str(v) for v in (float(mu),name,horizon,rep,pair,sign))
                  if k not in done:
                    complete=False;break
                if not complete:break
              if not complete:break
            if not complete:break
          if complete: continue
          init=[]; meta=[]
          for (rep,pair,sign),row in sorted(starts.items()):
            s=row["state"].copy(); s[2:]*=1/math.sqrt(mu)
            init.append(s);meta.append((rep,pair,sign,row["seed"]))
          init=np.asarray(init)
          kap=KAPPA*mult/mu
          first=integrate_rk4_batch(init,ALPHA/mu,BETA/mu,kap,model,delta=0.,tau0=0.,
                                    horizon_rf=100,steps_per_rf=STEPS_PER_RF,aperture=1.)
          # Keep exact endpoints and absolute RF phase for the 100->300 extension.
          status300=first["status"].copy(); exit300=first["exit_rf_step_estimate"].copy()
          state300=first["final_state"].copy(); rho300=first["rho_max_stage_sampled"].copy()
          alive=np.flatnonzero(first["status"]=="retained")
          if len(alive):
            second=integrate_rk4_batch(state300[alive],ALPHA/mu,BETA/mu,kap,model,delta=0.,tau0=0.,
                    horizon_rf=200,steps_per_rf=STEPS_PER_RF,aperture=1.,tau_start_cycles=100.)
            status300[alive]=second["status"]
            exit300[alive]=second["exit_rf_step_estimate"]
            state300[alive]=second["final_state"]
            rho300[alive]=np.maximum(rho300[alive],second["rho_max_stage_sampled"])
          peak=voltage_peak(name,mult)
          for i,(rep,pair,sign,seed) in enumerate(meta):
            for horizon,status,exit_rf,state,rhomax in (
              (100,first["status"][i],first["exit_rf_step_estimate"][i],first["final_state"][i],first["rho_max_stage_sampled"][i]),
              (300,status300[i],exit300[i],state300[i],rho300[i])):
              row={"mu":float(mu),"mass_Th":float(M0_TH*mu),"alpha":ALPHA/mu,"beta":BETA/mu,"kappa":kap,
                "config":name,"model":model,"kappa_multiplier":mult,"delta_rad":0.,"tau0_rad":0.,
                "replica":rep,"seed":seed,"pair_index":pair,"Lz_sign":sign,"horizon_rf":horizon,
                "status":str(status),"exit_rf":float(exit_rf) if np.isfinite(exit_rf) else None,
                "rho_max":float(rhomax),"x0":float(init[i,0]),"y0":float(init[i,1]),
                "vx0":float(init[i,2]),"vy0":float(init[i,3]),"Eperp0_eV":float(.5*M0_KG*mu*(OMEGA*R_M)**2*np.sum(init[i,2:]**2)/E_CHARGE),
                "xend":float(state[0]),"yend":float(state[1]),"vxend":float(state[2]),"vyend":float(state[3]),
                "peak_electrode_V":peak,"integrator":"RK4","steps_per_rf":STEPS_PER_RF,
                "aperture_rho":1.,"manifest_hash":manifest_hash}
              row["key"]="|".join(str(row[k]) for k in ("mu","config","horizon_rf","replica","pair_index","Lz_sign"))
              if row["key"] in done: continue
              f.write(json.dumps(row,separators=(",",":"))+"\n");done[row["key"]]=row
          f.flush()
          os.fsync(f.fileno())
          print(f"mu={mu:.3f} config={name} completed; unique_rows={len(done)}",flush=True)

    all_rows=list(done.values())
    fields=list(all_rows[0])
    with (OUT/"trajectory_results.csv").open("w",newline="",encoding="utf8") as f:
      w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
    summary=summarise(all_rows)
    with (OUT/"retention_summary.csv").open("w",newline="",encoding="utf8") as f:
      w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    by_key={(r["mu"],r["config"],r["horizon_rf"],r["replica"],r["pair_index"],r["Lz_sign"]):float(r["status"]=="retained") for r in all_rows}
    pair_names=(("RO_PLUS","Q","RO_PLUS-Q"),("RO_PLUS","SO_equal_harmonic","RO_PLUS-SO_equal_harmonic"),
                ("RO_PLUS","SO_equal_rms","RO_PLUS-SO_equal_rms"))
    paired=[]
    for mu in MASS_RATIOS:
      for hor in (100,300):
       for a,b,label in pair_names:
        for rep in range(8):
         diffs=[]
         for pair in range(128):
          for sign in (-1,1):
           va=by_key.get((float(mu),a,hor,rep,pair,sign));vb=by_key.get((float(mu),b,hor,rep,pair,sign))
           if va is not None and vb is not None:diffs.append(va-vb)
         if diffs:paired.append((float(mu),hor,label,rep,float(np.mean(diffs))))
    pair_summary=[]
    for mu,hor,label in sorted({r[:3] for r in paired}):
      vals=np.asarray([r[4] for r in paired if r[:3]==(mu,hor,label)])
      mean=float(np.mean(vals));half=float(t.ppf(.975,len(vals)-1)*sem(vals)) if len(vals)>1 and np.std(vals,ddof=1)>0 else 0.
      pair_summary.append({"mu":mu,"horizon_rf":hor,"contrast":label,"mean_difference_pp":100*mean,
        "ci95_low_pp":100*(mean-half),"ci95_high_pp":100*(mean+half),"n_scrambles":len(vals),
        "replicate_differences_pp":json.dumps((100*vals).tolist())})
    with (OUT/"paired_retention_differences.csv").open("w",newline="",encoding="utf8") as f:
      w=csv.DictWriter(f,fieldnames=list(pair_summary[0]));w.writeheader();w.writerows(pair_summary)
    val={"rows":len(all_rows),"unique_keys":len({r["key"] for r in all_rows}),
      "unresolved":sum(r["status"]=="unresolved" for r in all_rows),
      "over_250V":sum(r["peak_electrode_V"]>250 for r in all_rows),
      "mu_values":MASS_RATIOS,"models":len(MODELS),"scrambled_replicates":8,"pairs_per_replica":128,
      "horizons_rf":[100,300],"manifest_hash":manifest_hash}
    (OUT/"validation.json").write_text(json.dumps(val,indent=2)+"\n",encoding="utf8")
    print(json.dumps(val,indent=2))


if __name__=="__main__": main()
