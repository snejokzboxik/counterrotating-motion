"""Long-horizon validation/holdout for a P1 configuration."""
import argparse,json,math,time,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.special import ndtri
HERE=Path(__file__).resolve().parent; BASE=HERE.parent
sys.path.insert(0,str(BASE/"baseline"))
from basin_scan import integrate_one,make_initials,load_model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config-id",type=int,required=True)
    ap.add_argument("--ensemble",choices=["rest","weak","wide"],default="rest")
    ap.add_argument("--count",type=int,default=128); ap.add_argument("--seed",type=int,default=20261124)
    ap.add_argument("--periods",type=int,default=5000); ap.add_argument("--steps",type=int,default=64)
    ap.add_argument("--threads",type=int,default=6); ap.add_argument("--search-starts",action="store_true")
    ap.add_argument("--candidate-source",choices=["stage1","stage2_local","stage2_speed","stage4_candidate48"],default="stage1")
    ap.add_argument("--initials-from",type=Path,default=None,
                    help="reuse exact initial/phase arrays from a saved holdout NPZ")
    ap.add_argument("--oseen-Re-per-speed",type=float,default=0.0,
                    help="dimensionless Re=chi*|v| for first-order Oseen drag")
    ap.add_argument("--oseen-Re2-per-speed",type=float,default=None,
                    help="particle-2 chi; defaults to particle-1 chi")
    ap.add_argument("--hydro-mode",choices=["none","oseen","rpy"],default="none")
    ap.add_argument("--a1-over-R",type=float,default=0.0); ap.add_argument("--a2-over-R",type=float,default=0.0)
    ap.add_argument("--coupling-scale",type=float,default=1.0,
                    help="multiply the fitted dimensionless Coulomb coefficient for a matched-start sensitivity scan")
    ap.add_argument("--run-tag",default="",
                    help="optional unique label for a new seed or finite-size holdout; avoids overwriting earlier runs")
    a=ap.parse_args()
    source_npz=np.load(HERE/a.candidate_source/"configurations.npz")
    if a.candidate_source=="stage1":
        p=source_npz["parameters"][a.config_id].copy(); qratio=float(source_npz["charge_ratio"][a.config_id])
    else:
        p=source_npz["p"][a.config_id].copy(); qratio=float(source_npz["qratio"][a.config_id])
    p[5]*=a.coupling_scale
    if a.search_starts:
        if a.candidate_source=="stage1":
            init=source_npz["initial"].copy(); phases=source_npz["phases"].copy(); seed=20261123
        else:
            init=source_npz[f"initial_{a.ensemble}"].copy(); phases=source_npz[f"phase_{a.ensemble}"].copy(); seed=20261127
        label="search_starts_reused"
    elif a.initials_from is not None:
        saved=np.load(a.initials_from)
        init=np.asarray(saved["initial"][:a.count],float).copy()
        phases=np.asarray(saved["phases"][:a.count],float).copy()
        if init.shape[0] != min(a.count,len(saved["initial"])) or phases.shape[0] != init.shape[0]:
            raise ValueError("initial conditions and phases have inconsistent lengths")
        seed=None;label="matched_holdout"
    else:
        init,phases=make_initials(a.ensemble,a.count,a.seed); seed=a.seed; label="holdout"
    hydro_mode={"none":0,"oseen":1,"rpy":2}[a.hydro_mode]
    chi2=a.oseen_Re_per_speed if a.oseen_Re2_per_speed is None else a.oseen_Re2_per_speed
    if a.oseen_Re_per_speed or chi2 or hydro_mode or a.a1_over_R or a.a2_over_R:
        label += f"_{a.hydro_mode}_chi{a.oseen_Re_per_speed:g}-{chi2:g}_a{a.a1_over_R:g}-{a.a2_over_R:g}"
    if a.coupling_scale != 1.0:
        label += f"_Cscale{a.coupling_scale:g}"
    if a.run_tag:
        label += f"_{a.run_tag}"
    _,family,k,coeff=load_model()
    if a.candidate_source=="stage1":
        dest=HERE/"stage2"/"runs"
    elif a.candidate_source=="stage4_candidate48":
        dest=HERE/"stage4_candidate48"/"holdout_runs"
    else:
        dest=HERE/"stage3"/"runs"
    dest.mkdir(parents=True,exist_ok=True)
    stem=f"cfg{a.config_id}_{a.ensemble}_{label}_{len(init)}_{a.periods}p_{a.steps}s"
    result=dest/f"{stem}.jsonl"; meta=result.with_suffix(".meta.json")
    np.savez_compressed(dest/f"{stem}_initials.npz",initial=init,phases=phases,seed=seed,p=p,qratio=qratio)
    metadata={"config_id":a.config_id,"parameters":p.tolist(),"qratio":qratio,"ensemble":a.ensemble,
      "n":len(init),"seed":seed,"periods":a.periods,"steps":a.steps,"transient":200,
      "starts":(f"matched to {a.initials_from}" if a.initials_from is not None else
                ("search subset" if a.search_starts else "fresh scrambled Sobol ensemble")),
      "candidate_source":a.candidate_source,"coupling_scale":a.coupling_scale}
    if a.run_tag:
        metadata["run_tag"] = a.run_tag
    if a.oseen_Re_per_speed or chi2 or hydro_mode or a.a1_over_R or a.a2_over_R:
        metadata.update(oseen_Re_per_speed=a.oseen_Re_per_speed,hydro_mode=a.hydro_mode,
                       oseen_Re2_per_speed=chi2,a1_over_R=a.a1_over_R,a2_over_R=a.a2_over_R)
    if meta.exists() and json.loads(meta.read_text())!=metadata: raise RuntimeError("checkpoint metadata mismatch")
    meta.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    done=set()
    if result.exists(): done={json.loads(x)["index"] for x in result.read_text().splitlines() if x.strip()}
    todo=[i for i in range(len(init)) if i not in done]
    if todo: integrate_one(init[todo[0]],phases[todo[0]],p,family,k,coeff,2,a.steps,0,qratio,
                           a.oseen_Re_per_speed,hydro_mode,a.a1_over_R,a.a2_over_R,chi2)
    print(f"candidate {a.config_id}: {len(init)} conditions, {len(todo)} pending; q2/q1={qratio:.6g}",flush=True)
    t0=time.perf_counter()
    def one(i):
        return i,integrate_one(init[i],phases[i],p,family,k,coeff,a.periods,a.steps,200,qratio,
                               a.oseen_Re_per_speed,hydro_mode,a.a1_over_R,a.a2_over_R,chi2)
    with result.open("a",encoding="utf-8",buffering=1) as f,ThreadPoolExecutor(max_workers=a.threads) as pool:
        fs={pool.submit(one,i):i for i in todo}
        for j,fut in enumerate(as_completed(fs),1):
            i,r=fut.result(); end,turns,wins,rmin,rmax,zmax,maxd,dmin,complete,retained,escaped,sing,cat,freq=r
            row={"index":i,"initial":init[i].tolist(),"phase":phases[i].tolist(),"final_state":end.tolist(),
              "turns":turns.tolist(),"window_turns":wins.tolist(),"frequency_Hz":freq.tolist(),
              "r_min":rmin.tolist(),"r_max":rmax.tolist(),"z_max":zmax.tolist(),
              "max_abs_inner_step_dtheta_rad":maxd.tolist(),"minimum_separation":float(dmin),
              "completed_periods":int(complete),"retained":bool(retained),"escaped":bool(escaped),
              "singular_unresolved":bool(sing),"category":str(cat),"hydro_model":a.hydro_mode,
              "oseen_Re_per_speed":a.oseen_Re_per_speed,"oseen_Re2_per_speed":chi2,
              "a1_over_R":a.a1_over_R,"a2_over_R":a.a2_over_R}
            f.write(json.dumps(row,separators=(",",":"))+"\n")
            if j%16==0 or j==len(todo):
                print(f"{stem}: checkpoint {j}/{len(todo)} ({time.perf_counter()-t0:.1f}s)",flush=True)

if __name__=="__main__": main()
