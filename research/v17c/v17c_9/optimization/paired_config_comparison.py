"""Run the prior candidate on candidate-48's exact fresh starts for paired comparison."""
import argparse,json,sys,time,math
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/"baseline"))
from basin_scan import integrate_one,load_model

def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]

def strict(row):
    return bool(row["retained"] and min(row["r_min"])>=.02 and min(map(abs,row["turns"]))>=10 and
       all(abs(x)>.05 and abs(y)>.05 and x*y<0 for x,y in row["window_turns"]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--ensemble",choices=("rest","weak","wide"),required=True)
    ap.add_argument("--periods",type=int,default=15000);ap.add_argument("--threads",type=int,default=2);a=ap.parse_args()
    startsfile=HERE/"stage3"/"runs"/f"cfg48_{a.ensemble}_holdout_128_{a.periods}p_64s_initials.npz"
    z=np.load(startsfile);initial=z["initial"];phases=z["phases"]
    newfile=HERE/"stage3"/"runs"/f"cfg48_{a.ensemble}_holdout_128_{a.periods}p_64s.jsonl"
    rows48={int(r["index"]):r for r in (json.loads(s) for s in newfile.read_text().splitlines() if s.strip())}
    cfg1=np.load(HERE/"stage1"/"configurations.npz");p=cfg1["parameters"][81].copy();qr=float(cfg1["charge_ratio"][81])
    _,family,k,coeff=load_model();dst=HERE/"stage3"/"runs"/f"paired_cfg81_vs48_{a.ensemble}_{a.periods}p.jsonl"
    done=set()
    if dst.exists():done={int(json.loads(s)["index"]) for s in dst.read_text().splitlines() if s.strip()}
    tasks=[i for i in range(len(initial)) if i not in done]
    if tasks:integrate_one(initial[tasks[0]],phases[tasks[0]],p,family,k,coeff,2,64,0,qr)
    def one(i):
        r=integrate_one(initial[i],phases[i],p,family,k,coeff,a.periods,64,200,qr)
        end,tr,w,rmin,rmax,zmax,maxd,dmin,complete,ret,esc,sing,cat,freq=r
        return {"index":i,"category":str(cat),"turns":tr.tolist(),"window_turns":w.tolist(),"frequency_Hz":freq.tolist(),
          "r_min":rmin.tolist(),"r_max":rmax.tolist(),"minimum_separation":float(dmin),"retained":bool(ret),
          "strict_CR":bool(cat=="CR_screen"),"paired_candidate48_category":rows48[i]["category"],
          "paired_candidate48_strict_CR":strict(rows48[i])}
    t0=time.perf_counter()
    with dst.open("a",encoding="utf-8",buffering=1) as f,ThreadPoolExecutor(max_workers=a.threads) as pool:
        fs={pool.submit(one,i):i for i in tasks}
        for j,fu in enumerate(as_completed(fs),1):
            f.write(json.dumps(fu.result(),separators=(",",":"))+"\n")
            if j%16==0 or j==len(tasks):print(f"{a.ensemble} paired compare {j}/{len(tasks)} wall={time.perf_counter()-t0:.1f}s",flush=True)
    rows=[json.loads(s) for s in dst.read_text().splitlines() if s.strip()]
    n=len(rows);s1=sum(x["strict_CR"] for x in rows);s2=sum(x["paired_candidate48_strict_CR"] for x in rows)
    # strict_CR is candidate 81; paired_candidate48_strict_CR is candidate 48.
    candidate81_only=sum(x["strict_CR"] and not x["paired_candidate48_strict_CR"] for x in rows)
    candidate48_only=sum(not x["strict_CR"] and x["paired_candidate48_strict_CR"] for x in rows)
    # Exact two-sided McNemar test (binomial under equal discordant probabilities).
    small=min(candidate81_only,candidate48_only); total=candidate81_only+candidate48_only
    pexact=min(1.,2*sum(math.comb(total,i) for i in range(small+1))/(2**total)) if total else 1.
    summary={"ensemble":a.ensemble,"N_total":n,"candidate81_strict_CR":s1,"P_CR_candidate81":s1/n,
      "Wilson95_candidate81":wilson(s1,n),"candidate48_strict_CR":s2,"P_CR_candidate48":s2/n,
      "Wilson95_candidate48":wilson(s2,n),"candidate81_only_success":candidate81_only,
      "candidate48_only_success":candidate48_only,
      "two_sided_exact_McNemar_p":pexact}
    (HERE/"stage3"/f"paired_cfg81_vs48_{a.ensemble}.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
