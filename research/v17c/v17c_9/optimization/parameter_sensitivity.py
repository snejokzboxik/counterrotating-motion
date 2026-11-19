"""Small local parameter perturbations around a long-lived candidate trajectory."""
import argparse,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/"baseline"))
from basin_scan import integrate_one,load_model

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config-id",type=int,default=48);ap.add_argument("--ensemble",default="rest");ap.add_argument("--index",type=int,default=0);ap.add_argument("--count",type=int,default=64);ap.add_argument("--periods",type=int,default=15000);ap.add_argument("--threads",type=int,default=6);a=ap.parse_args()
    source="stage2_speed"; z=np.load(HERE/source/"configurations.npz");p0=z["p"][a.config_id].copy();q0=float(z["qratio"][a.config_id])
    file=HERE/"stage3"/"runs"/f"cfg{a.config_id}_{a.ensemble}_holdout_{a.count}_{a.periods}p_64s.jsonl"
    row=next(json.loads(s) for s in file.read_text().splitlines() if s.strip() and int(json.loads(s)["index"])==a.index)
    initial=np.asarray(row["initial"],float);phase=np.asarray(row["phase"],float);_,fam,k,c=load_model()
    cases=[]
    for name,index,fac in [("baseline",None,1.),("RF+1%",0,1.01),("RF-1%",0,.99),
       ("DC+1%",3,1.01),("DC-1%",3,.99),("mass2+1%",4,1.01),("qratio+1%",8,1.01),
       ("lambda_both+1%",9,1.01),("C+10%",5,1.10)]:
        p=p0.copy();qr=q0
        if index in (0,1):p[0]*=fac;p[1]*=fac
        elif index==3:p[3]*=fac
        elif index==4:p[4]/=fac;p[7]/=fac
        elif index==5:p[5]*=fac
        elif index==8:qr*=fac;p[5]*=fac
        elif index==9:p[6]*=fac;p[7]*=fac
        cases.append((name,p,qr))
    def one(task):
        name,p,q=task;tic=time.perf_counter()
        r=integrate_one(initial,phase,p,fam,k,c,a.periods,64,200,q)
        end,tr,w,rmin,rmax,zmax,maxd,dmin,complete,ret,esc,sing,cat,freq=r
        return {"case":name,"category":str(cat),"frequency_Hz":freq.tolist(),"turns":tr.tolist(),
          "window_turns":w.tolist(),"r_min":rmin.tolist(),"r_max":rmax.tolist(),"minimum_separation":float(dmin),
          "retained":bool(ret),"wall_s":time.perf_counter()-tic}
    out=[]
    with ThreadPoolExecutor(max_workers=a.threads) as pool:
        fs=[pool.submit(one,t) for t in cases]
        for f in as_completed(fs):
            r=f.result();out.append(r);print(r["case"],r["frequency_Hz"],r["category"],flush=True)
    result={"config_id":a.config_id,"start":f"{a.ensemble}:{a.index}","periods":a.periods,
      "perturbations":"one at a time; RF amplitude scales both carriers together","results":sorted(out,key=lambda r:r["case"])}
    dst=HERE/"stage3"/"parameter_sensitivity_cfg48.json";dst.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
