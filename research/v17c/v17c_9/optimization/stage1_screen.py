"""Deterministic Sobol P1 screening of common-field parameters on rest starts.

The objective is only a short-horizon basin proxy. Every configuration uses
the same 32 frozen, stratified starts and RF phases. Final P_CR is not inferred
from this screen; candidates require long-horizon RK4/DOP853 and a fresh holdout.
"""
import argparse,json,math,time,csv
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.stats import qmc

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/"baseline"))
from basin_scan import integrate_one,load_model
RUNS=BASE/"baseline"/"runs"
OUT=HERE/"stage1"
N_CONFIGS=128; N_STARTS=32; PERIODS=1800; STEPS=48; SKIP=400; SEED=20261123
# Archived MFS unit-voltage leading potential coefficients for k=+2 and -4.
PHI_RATIO_PER_VOLT=.7982172694357996/.8703168951521896
BASE_RATIO=.1875/.075  # B/A in normalized leading-potential amplitudes, not U2/U1.

def parameter_sets(basep):
    u=qmc.Sobol(d=9,scramble=True,seed=SEED).random_base2(7)
    out=[]
    for i,x in enumerate(u):
        rf=.4+1.1*x[0]; ratio=.5+4.5*x[1]; nu=1+3*x[2]
        dc=.5+1.5*x[3]; mu=2+6*x[4]; qr=np.exp(np.log(.25)+x[5]*np.log(16))
        l1=np.exp(np.log(.003)+x[6]*np.log(.12/.003)); l2=np.exp(np.log(.003)+x[7]*np.log(.12/.003))
        C=np.exp(np.log(5e-6)+x[8]*np.log(1e-3/5e-6))
        p=basep.copy(); p[0]=basep[0]*rf; p[1]=basep[1]*rf*ratio/BASE_RATIO
        p[2]=nu; p[3]=basep[3]*rf*rf*dc; p[4]=1/mu; p[5]=C; p[6]=l1; p[7]=l2
        out.append({"id":i,"anchor":"sobol","p":p,"rf1_scale":rf,
                    "harmonic_potential_ratio_B_over_A":ratio,
                    "drive_U2_U1_physical":ratio/PHI_RATIO_PER_VOLT,
                    "nu":nu,"dc_relative_to_rf_squared":dc,"mass_ratio":mu,"charge_ratio":qr,
                    "lambda1":l1,"lambda2":l2,"C_eff":C,"qratio":qr})
    # Explicit checks of the nominal point and the reported 0.85 RF / stronger-drag lead.
    anchors=[(1.,BASE_RATIO,np.sqrt(2),1.,4.0221695608,1.,.012,.012,1e-5,"nominal"),
       (.85,BASE_RATIO,np.sqrt(2),1.,4.0221695608,1.,.025,.025,1e-5,"rf085_drag025"),
       (.85,BASE_RATIO,np.sqrt(2),1.,4.0221695608,1.,.05,.05,1e-5,"rf085_drag050"),
       (1.,BASE_RATIO,np.sqrt(2),1.,4.0221695608,1.,.025,.025,1e-5,"rf100_drag025")]
    for rf,ratio,nu,dc,mu,qr,l1,l2,C,label in anchors:
        p=basep.copy(); p[0]=basep[0]*rf; p[1]=basep[1]*rf*ratio/BASE_RATIO
        p[2]=nu; p[3]=basep[3]*rf*rf*dc; p[4]=1/mu; p[5]=C; p[6]=l1; p[7]=l2
        out.append({"id":len(out),"anchor":label,"p":p,"rf1_scale":rf,
                    "harmonic_potential_ratio_B_over_A":ratio,
                    "drive_U2_U1_physical":ratio/PHI_RATIO_PER_VOLT,
                    "nu":nu,"dc_relative_to_rf_squared":dc,"mass_ratio":mu,"charge_ratio":qr,
                    "lambda1":l1,"lambda2":l2,"C_eff":C,"qratio":qr})
    return out

def is_proxy(retained,turns,windows,rmin):
    if not retained or min(rmin)<.02 or min(abs(turns))<.5: return False
    s=np.sign(windows)
    return bool(np.all(s[:,0]*s[:,1]<0) and np.all(np.abs(windows)>.05))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--threads",type=int,default=6); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    basep,family,k,coeff=load_model()
    starts=np.load(RUNS/"rest_initial_conditions_64s.npz")
    idx=np.linspace(0,len(starts["initial"])-1,N_STARTS,dtype=int)
    init=starts["initial"][idx]; phases=starts["phases"][idx]
    cfgs=parameter_sets(basep)
    np.savez_compressed(OUT/"configurations.npz",parameters=np.array([x["p"] for x in cfgs]),
       rf1_scale=np.array([x["rf1_scale"] for x in cfgs]),
       harmonic_potential_ratio_B_over_A=np.array([x["harmonic_potential_ratio_B_over_A"] for x in cfgs]),
       drive_U2_U1_physical=np.array([x["drive_U2_U1_physical"] for x in cfgs]),
       nu=np.array([x["nu"] for x in cfgs]),dc_relative=np.array([x["dc_relative_to_rf_squared"] for x in cfgs]),
       mass_ratio=np.array([x["mass_ratio"] for x in cfgs]),charge_ratio=np.array([x["charge_ratio"] for x in cfgs]),
       lambda1=np.array([x["lambda1"] for x in cfgs]),lambda2=np.array([x["lambda2"] for x in cfgs]),C_eff=np.array([x["C_eff"] for x in cfgs]),
       initial=init,phases=phases,initial_indices=idx)
    meta={"seed":SEED,"configs":len(cfgs),"starts_per_config":N_STARTS,"periods":PERIODS,"steps":STEPS,"skip":SKIP,
          "common_start_set":"32 evenly spaced indices from frozen V17C.5 rest ensemble seed 20261119",
          "note":"amplitude-ratio variable is normalized leading-potential B/A; actual electrode-voltage ratio reconstructed from archived unit-voltage MFS coefficients. qratio, lambda, and C are varied independently here; physical feasibility is a later gate"}
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    result=OUT/"runs.jsonl"; done=set()
    if result.exists():
        for line in result.read_text(encoding="utf-8").splitlines():
            if line.strip(): done.add((json.loads(line)["config_id"],json.loads(line)["start_id"]))
    cfgs=cfgs[:a.limit or len(cfgs)]
    p0=cfgs[0]["p"] if cfgs else basep
    integrate_one(init[0],phases[0],p0,family,k,coeff,2,STEPS,0,1.)
    tasks=[(cfg,si) for cfg in cfgs for si in range(N_STARTS) if (cfg["id"],si) not in done]
    print(f"P1 Sobol stage1: configs={len(cfgs)}, starts/config={N_STARTS}, pending={len(tasks)}, threads={a.threads}",flush=True)
    t0=time.perf_counter(); totals={c["id"]:{"retained":0,"proxy":0,"done":0} for c in cfgs}
    if done:
        # Resume totals from the immutable checkpoint stream.
        for line in result.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            r=json.loads(line); q=totals.get(r["config_id"])
            if q is not None: q["retained"]+=r["retained"]; q["proxy"]+=r["cr_proxy"]; q["done"]+=1
    def one(task):
        c,si=task; res=integrate_one(init[si],phases[si],c["p"],family,k,coeff,PERIODS,STEPS,SKIP,c["qratio"])
        end,turns,windows,rmin,rmax,zmax,maxd,dmin,completed,retained,escaped,singular,cat,freq=res
        proxy=is_proxy(retained,turns,windows,rmin)
        return {"config_id":c["id"],"start_id":si,"anchor":c["anchor"],"turns":turns.tolist(),"window_turns":windows.tolist(),
         "frequency_Hz":freq.tolist(),"r_min":rmin.tolist(),"r_max":rmax.tolist(),"max_abs_z":zmax.tolist(),
         "max_inner_dtheta_rad":maxd.tolist(),"minimum_separation":float(dmin),"completed_periods":int(completed),
         "retained":bool(retained),"cr_proxy":proxy,"category":str(cat),"final_state":end.tolist()}
    with result.open("a",encoding="utf-8",buffering=1) as f, ThreadPoolExecutor(max_workers=a.threads) as pool:
        futures={pool.submit(one,t):t for t in tasks}
        for j,fut in enumerate(as_completed(futures),1):
            row=fut.result(); f.write(json.dumps(row,separators=(",",":"))+"\n")
            q=totals[row["config_id"]]; q["done"]+=1; q["retained"]+=row["retained"]; q["proxy"]+=row["cr_proxy"]
            if j%128==0 or j==len(tasks):
                summary=[]
                for cid,v in totals.items():
                    if v["done"]==N_STARTS:
                        c=cfgs[cid]; summary.append({"config_id":cid,"anchor":c["anchor"],"n":v["done"],"proxy":v["proxy"],"retained":v["retained"],"p_proxy":v["proxy"]/N_STARTS,"p_retained":v["retained"]/N_STARTS,"params":{k:c[k] for k in ("rf1_scale","harmonic_potential_ratio_B_over_A","drive_U2_U1_physical","nu","dc_relative_to_rf_squared","mass_ratio","charge_ratio","lambda1","lambda2","C_eff")}})
                summary.sort(key=lambda x:(x["p_proxy"],x["p_retained"]),reverse=True)
                (OUT/"ranked.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
                print(f"P1 checkpoint {j}/{len(tasks)} elapsed={time.perf_counter()-t0:.1f}s top="
                      f"{[(x['config_id'],x['p_proxy'],x['p_retained']) for x in summary[:3]]}",flush=True)

if __name__=="__main__":main()
