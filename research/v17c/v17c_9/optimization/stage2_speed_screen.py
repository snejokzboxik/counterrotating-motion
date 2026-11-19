"""Sobol refinement that makes the 10-turn threshold part of the screening objective."""
import argparse,json,math,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.stats import qmc
HERE=Path(__file__).resolve().parent; BASE=HERE.parent
sys.path.insert(0,str(BASE/"baseline"))
from basin_scan import integrate_one,load_model

OUT=HERE/"stage2_speed"; NCFG=64; NPER=1800; NSTEPS=48; NSKIP=400; SEED=20261134
GAMMA=.7982172694357996/.8703168951521896

def make_configs(basep,q0):
    u=qmc.Sobol(d=9,scramble=True,seed=SEED).random_base2(6)
    out=[]
    for i,x in enumerate(u):
        rf=.78+.44*x[0]; phi=2.45+1.3*x[1]; nu=2.15+1.10*x[2]; dc=.72+1.18*x[3]
        mu=5.5+4.5*x[4]; qr=2.0+2.5*x[5]
        l1=np.exp(np.log(.025)+x[6]*np.log(.12/.025)); l2=np.exp(np.log(.035)+x[7]*np.log(.14/.035))
        C=np.exp(np.log(4e-6)+x[8]*np.log(2e-5/4e-6))
        p=basep.copy(); p[0]=basep[0]*rf; p[1]=p[0]*phi
        p[2]=nu; p[3]=basep[3]*rf*rf*dc; p[4]=1/mu; p[5]=C; p[6]=l1; p[7]=l2
        out.append({"id":i,"p":p,"qratio":qr,"rf1_scale":rf,
          "harmonic_potential_ratio_B_over_A":phi,"drive_U2_U1_physical":phi/GAMMA,
          "nu":nu,"dc_relative_to_rf_squared":dc,"mass_ratio":mu,"charge_ratio":qr,
          "lambda1":l1,"lambda2":l2,"C_eff":C})
    return out

def proxy(ret,turns,wins,rmin):
    # For 1,400 post-transient screening cycles, 1.05 turns extrapolates to >10 turns
    # in the 14,800-cycle validation window. This is a screen, not final evidence.
    if not ret or min(rmin)<.02 or min(abs(turns))<1.05:return False
    s=np.sign(wins)
    return bool(np.all(s[:,0]*s[:,1]<0) and np.all(np.abs(wins)>.05))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--threads",type=int,default=6); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); basep,family,k,coeff=load_model()
    # Center on the fast local candidate; screening starts are a fixed subset of
    # the independent candidate-81 holdout, reused only for optimization.
    old=np.load(HERE/"stage2_local"/"configurations.npz"); basep=old["p"][53].copy(); q0=float(old["qratio"][53])
    src=HERE/"stage2"/"runs"; starts={}; phases={}; srcids={}
    for g in ("rest","weak","wide"):
        z=np.load(src/f"cfg81_{g}_holdout_128_15000p_64s_initials.npz")
        take=np.linspace(0,127,16,dtype=int); starts[g]=z["initial"][take].copy(); phases[g]=z["phases"][take].copy(); srcids[g]=take
    configs=make_configs(basep,q0)[:a.limit or NCFG]
    np.savez_compressed(OUT/"configurations.npz",p=np.array([c["p"] for c in configs]),qratio=np.array([c["qratio"] for c in configs]),
        initial_rest=starts["rest"],phase_rest=phases["rest"],initial_weak=starts["weak"],phase_weak=phases["weak"],
        initial_wide=starts["wide"],phase_wide=phases["wide"])
    (OUT/"meta.json").write_text(json.dumps({"n_configs":len(configs),"n_per_ensemble":16,"periods":NPER,"steps":NSTEPS,
       "transient_cycles":NSKIP,"seed":SEED,"criterion":"retained, min radius >=0.02, opposite signs in every late window, >=1.05 turns per particle during the 1400-cycle post-transient screen",
       "limitation":"short-horizon screening proxy only; independent long holdout is required"},indent=2)+"\n",encoding="utf-8")
    result=OUT/"runs.jsonl"; done=set()
    if result.exists(): done={(json.loads(s)["config_id"],json.loads(s)["ensemble"],json.loads(s)["start_id"]) for s in result.read_text().splitlines() if s.strip()}
    tasks=[(c,g,i) for c in configs for g in ("rest","weak","wide") for i in range(16) if (c["id"],g,i) not in done]
    if tasks: integrate_one(starts[tasks[0][1]][tasks[0][2]],phases[tasks[0][1]][tasks[0][2]],tasks[0][0]["p"],family,k,coeff,2,NSTEPS,0,tasks[0][0]["qratio"])
    print(f"speed-gated screen: {len(configs)} configs, {len(tasks)} trials, {a.threads} workers",flush=True)
    tic=time.perf_counter()
    def one(t):
        c,g,i=t; r=integrate_one(starts[g][i],phases[g][i],c["p"],family,k,coeff,NPER,NSTEPS,NSKIP,c["qratio"])
        end,tr,w,rmin,rmax,zmax,maxd,dmin,complete,ret,esc,sing,cat,freq=r
        return {"config_id":c["id"],"ensemble":g,"start_id":i,"turns":tr.tolist(),"window_turns":w.tolist(),
          "frequency_Hz":freq.tolist(),"r_min":rmin.tolist(),"retained":bool(ret),"proxy_ge10_turns":proxy(ret,tr,w,rmin),
          "category":str(cat),"min_distance":float(dmin),"completed_periods":int(complete)}
    with result.open("a",encoding="utf-8",buffering=1) as f,ThreadPoolExecutor(max_workers=a.threads) as pool:
        futures={pool.submit(one,t):t for t in tasks}
        for j,fu in enumerate(as_completed(futures),1):
            f.write(json.dumps(fu.result(),separators=(",",":"))+"\n")
            if j%128==0 or j==len(tasks):
                rows=[json.loads(s) for s in result.read_text().splitlines() if s.strip()]
                agg={}
                for row in rows:
                    agg.setdefault(row["config_id"],{}).setdefault(row["ensemble"],[]).append(row["proxy_ge10_turns"])
                rank=[]
                for c in configs:
                    ss=agg.get(c["id"],{}); rates=[float(np.mean(ss[g])) if g in ss and ss[g] else 0. for g in ("rest","weak","wide")]
                    if all(g in ss and len(ss[g])==16 for g in ("rest","weak","wide")):
                        rank.append({"config_id":c["id"],"balanced_min":min(rates),"balanced_mean":float(np.mean(rates)),"rates":rates,
                                     "parameters":{kk:vv for kk,vv in c.items() if kk not in ("id","p")}})
                rank.sort(key=lambda r:(r["balanced_min"],r["balanced_mean"]),reverse=True)
                (OUT/"ranked.json").write_text(json.dumps(rank,indent=2)+"\n",encoding="utf-8")
                print(f"checkpoint {j}/{len(tasks)} wall={time.perf_counter()-tic:.1f}s leaders={[(x['config_id'],x['balanced_min'],x['balanced_mean']) for x in rank[:5]]}",flush=True)

if __name__=="__main__":main()
