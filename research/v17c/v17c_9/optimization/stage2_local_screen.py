"""Balanced local Sobol refinement around P1 candidate 81 across all start classes."""
import argparse,json,math,time,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.stats import qmc
HERE=Path(__file__).resolve().parent; BASE=HERE.parent
sys.path.insert(0,str(BASE/"baseline"))
from basin_scan import integrate_one,load_model

OUT=HERE/"stage2_local"; NCFG=64; NPER=1800; NSTEPS=48; NSKIP=400; SEED=20261127
GAMMA=.7982172694357996/.8703168951521896
BASE_PHI_RATIO=.1875/.075

def cfg_set(basep,anchor):
    u=qmc.Sobol(d=9,scramble=True,seed=SEED).random_base2(6)
    configs=[]
    for i,x in enumerate(u):
        rf=.42+.35*x[0]; phi=2.7+1.3*x[1]; nu=2.1+1.0*x[2]; dc=.8+1.2*x[3]
        mu=5+4*x[4]; qr=2+2*x[5]
        l1=np.exp(np.log(.02)+x[6]*np.log(.08/.02)); l2=np.exp(np.log(.05)+x[7]*np.log(.16/.05))
        C=np.exp(np.log(5e-6)+x[8]*np.log(2e-5/5e-6))
        p=basep.copy(); p[0]=basep[0]*rf; p[1]=basep[1]*rf*phi/BASE_PHI_RATIO
        p[2]=nu; p[3]=basep[3]*rf*rf*dc; p[4]=1/mu; p[5]=C; p[6]=l1; p[7]=l2
        configs.append({"id":i,"source":"sobol_local","p":p,"qratio":qr,"rf1_scale":rf,
           "harmonic_potential_ratio_B_over_A":phi,"drive_U2_U1_physical":phi/GAMMA,
           "nu":nu,"dc_relative_to_rf_squared":dc,"mass_ratio":mu,"charge_ratio":qr,
           "lambda1":l1,"lambda2":l2,"C_eff":C})
    p=anchor["parameters"].copy(); qr=anchor["charge_ratio"]
    phi=p[1]/p[0]
    configs.append({"id":NCFG,"source":"anchor_config81","p":p,"qratio":qr,
       "rf1_scale":p[0]/basep[0],"harmonic_potential_ratio_B_over_A":phi,
       "drive_U2_U1_physical":phi/GAMMA,"nu":p[2],
       "dc_relative_to_rf_squared":p[3]/(basep[3]*(p[0]/basep[0])**2),
       "mass_ratio":1/p[4],"charge_ratio":qr,"lambda1":p[6],"lambda2":p[7],"C_eff":p[5]})
    return configs

def proxy(ret,turns,wins,rmin):
    if not ret or min(rmin)<.02 or min(abs(turns))<.5:return False
    s=np.sign(wins)
    return bool(np.all(s[:,0]*s[:,1]<0) and np.all(np.abs(wins)>.05))

def rank_results(path,configs):
    results={c["id"]:{g:{"n":0,"proxy":0,"ret":0} for g in ("rest","weak","wide")} for c in configs}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():continue
            r=json.loads(line); s=results[r["config_id"]][r["ensemble"]]; s["n"]+=1; s["proxy"]+=r["cr_proxy"]; s["ret"]+=r["retained"]
    ranked=[]
    for c in configs:
        stats={g:{"p_proxy":v["proxy"]/v["n"] if v["n"] else 0.,"p_retained":v["ret"]/v["n"] if v["n"] else 0.,"n":v["n"]} for g,v in results[c["id"]].items()}
        if min(x["n"] for x in stats.values())==0: continue
        rates=[stats[g]["p_proxy"] for g in ("rest","weak","wide")]
        rets=[stats[g]["p_retained"] for g in ("rest","weak","wide")]
        ranked.append({"config_id":c["id"],"source":c["source"],"p_proxy_by_ensemble":stats,
          "balanced_min_proxy":min(rates),"balanced_mean_proxy":float(np.mean(rates)),
          "balanced_min_retention":min(rets),"parameters":{k:v for k,v in c.items() if k not in ("p","id","source")}})
    ranked.sort(key=lambda r:(r["balanced_min_proxy"],r["balanced_mean_proxy"],r["balanced_min_retention"]),reverse=True)
    (OUT/"ranked.json").write_text(json.dumps(ranked,indent=2)+"\n",encoding="utf-8")
    return ranked

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--threads",type=int,default=6);ap.add_argument("--limit",type=int,default=0);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); basep,family,k,coeff=load_model()
    src=HERE/"stage2"/"runs"; init={}; phase={}; ids={}
    for g,seed in (("rest",20261124),("weak",20261125),("wide",20261126)):
        file=src/f"cfg81_{g}_holdout_128_15000p_64s_initials.npz"; z=np.load(file)
        take=np.linspace(0,len(z["initial"])-1,16,dtype=int)
        init[g]=z["initial"][take].copy(); phase[g]=z["phases"][take].copy(); ids[g]=take
    prev=np.load(HERE/"stage1"/"configurations.npz")
    configs=cfg_set(basep,{"parameters":prev["parameters"][81],"charge_ratio":prev["charge_ratio"][81]})
    configs=configs[:a.limit or len(configs)]
    np.savez_compressed(OUT/"configurations.npz",p=np.array([c["p"] for c in configs]),qratio=np.array([c["qratio"] for c in configs]),
      initial_rest=init["rest"],phase_rest=phase["rest"],initial_weak=init["weak"],phase_weak=phase["weak"],
      initial_wide=init["wide"],phase_wide=phase["wide"],source_indices=np.array([ids[g] for g in ("rest","weak","wide")]))
    meta={"n_configs":len(configs),"initials_per_ensemble":16,"ensembles":["rest","weak","wide"],
      "periods":NPER,"steps":NSTEPS,"skip":NSKIP,"seed":SEED,"objective":"maximize worst and mean short-horizon CR proxy across the three explicit distributions",
      "note":"screening only; new candidate needs fresh holdout and long/independent verification"}
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    result=OUT/"runs.jsonl"; done=set()
    if result.exists(): done={(json.loads(x)["config_id"],json.loads(x)["ensemble"],json.loads(x)["start_id"]) for x in result.read_text().splitlines() if x.strip()}
    rank_results(result,configs)
    tasks=[(c,g,i) for c in configs for g in ("rest","weak","wide") for i in range(16) if (c["id"],g,i) not in done]
    if tasks: integrate_one(init["rest"][0],phase["rest"][0],tasks[0][0]["p"],family,k,coeff,2,NSTEPS,0,tasks[0][0]["qratio"])
    print(f"local P1 screen {len(configs)} configs; tasks={len(tasks)} on {a.threads} workers",flush=True)
    t0=time.perf_counter()
    def one(task):
        c,g,i=task; q=integrate_one(init[g][i],phase[g][i],c["p"],family,k,coeff,NPER,NSTEPS,NSKIP,c["qratio"])
        end,tr,w,rmin,rmax,zmax,maxd,dmin,complete,ret,esc,sing,cat,freq=q
        return {"config_id":c["id"],"ensemble":g,"start_id":i,"source_index":int(ids[g][i]),"turns":tr.tolist(),
          "window_turns":w.tolist(),"frequency_Hz":freq.tolist(),"r_min":rmin.tolist(),"r_max":rmax.tolist(),
          "retained":bool(ret),"cr_proxy":proxy(ret,tr,w,rmin),"category":str(cat),"completed_periods":int(complete),
          "min_distance":float(dmin),"final_state":end.tolist()}
    with result.open("a",encoding="utf-8",buffering=1) as f,ThreadPoolExecutor(max_workers=a.threads) as pool:
        futures={pool.submit(one,t):t for t in tasks}
        for j,fu in enumerate(as_completed(futures),1):
            r=fu.result();f.write(json.dumps(r,separators=(",",":"))+"\n")
            if j%128==0 or j==len(tasks):
                ranked=rank_results(result,configs)
                print(f"local checkpoint {j}/{len(tasks)} {time.perf_counter()-t0:.1f}s top={[(x['config_id'],x['balanced_min_proxy'],x['balanced_mean_proxy']) for x in ranked[:3]]}",flush=True)

if __name__=="__main__":main()
