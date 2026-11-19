"""Same-start fixed-step refinement for selected long-horizon P1 candidates."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; BASE=HERE.parent
sys.path.insert(0,str(BASE/"baseline"))
from basin_scan import integrate_one,load_model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate-source",choices=("stage1","stage2_local","stage2_speed"),required=True)
    ap.add_argument("--config-id",type=int,required=True)
    ap.add_argument("--ensemble",choices=("rest","weak","wide"),required=True)
    ap.add_argument("--index",type=int,required=True)
    ap.add_argument("--count",type=int,default=64)
    ap.add_argument("--periods",type=int,default=15000)
    ap.add_argument("--perturb-radii-percent",type=float,default=0.)
    ap.add_argument("--threads",type=int,default=1)
    a=ap.parse_args()
    srcnpz=np.load(HERE/a.candidate_source/"configurations.npz")
    if a.candidate_source=="stage1": p=srcnpz["parameters"][a.config_id]; qr=float(srcnpz["charge_ratio"][a.config_id])
    else: p=srcnpz["p"][a.config_id]; qr=float(srcnpz["qratio"][a.config_id])
    file=HERE/"stage3"/"runs"/f"cfg{a.config_id}_{a.ensemble}_holdout_{a.count}_{a.periods}p_64s.jsonl"
    row=next(json.loads(s) for s in file.read_text(encoding="utf-8").splitlines() if s.strip() and int(json.loads(s)["index"])==a.index)
    init=np.asarray(row["initial"],float).copy(); phase=np.asarray(row["phase"],float)
    if a.perturb_radii_percent:
        factor=1.+a.perturb_radii_percent/100.
        init[[0,1,6,7]]*=factor
    _,family,k,coeff=load_model(); runs=[]
    for steps in (64,128):
        tic=time.perf_counter()
        r=integrate_one(init,phase,p,family,k,coeff,a.periods,steps,200,qr)
        end,turns,wins,rmin,rmax,zmax,maxd,dmin,complete,ret,esc,sing,cat,freq=r
        runs.append({"steps_per_primary_cycle":steps,"wall_s":time.perf_counter()-tic,"turns":turns.tolist(),
          "window_turns":wins.tolist(),"frequency_Hz":freq.tolist(),"r_min":rmin.tolist(),"r_max":rmax.tolist(),
          "max_abs_z":zmax.tolist(),"max_inner_step_dtheta_rad":maxd.tolist(),"min_separation":float(dmin),
          "retained":bool(ret),"escaped":bool(esc),"singular":bool(sing),"category":str(cat),"final_state":end.tolist()})
        print(steps,runs[-1]["frequency_Hz"],runs[-1]["wall_s"],flush=True)
    out={"candidate_source":a.candidate_source,"config_id":a.config_id,"ensemble":a.ensemble,"index":a.index,
      "periods":a.periods,"initial":init.tolist(),"phase":phase.tolist(),"radial_perturb_percent":a.perturb_radii_percent,
      "run_pair":runs,"frequency_delta_Hz":(np.asarray(runs[1]["frequency_Hz"])-np.asarray(runs[0]["frequency_Hz"])).tolist(),
      "turn_delta":(np.asarray(runs[1]["turns"])-np.asarray(runs[0]["turns"])).tolist()}
    dest=HERE/"stage3"/"runs"/f"resolution_cfg{a.config_id}_{a.ensemble}_{a.index}_p{a.perturb_radii_percent:g}.json"
    dest.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(dest)

if __name__=="__main__":main()
