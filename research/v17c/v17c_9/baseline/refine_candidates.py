"""Repeat every P0 CR_screen hit at RK4 64 steps per primary RF cycle."""
import json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from basin_scan import integrate_one,load_model

HERE=Path(__file__).resolve().parent
RUNS=HERE/"runs"
OUT=RUNS/"candidate_refine64_5000p.jsonl"
META=OUT.with_suffix(".meta.json")

def main():
    inputs=[]
    for group in ("rest","weak","wide"):
        path=RUNS/f"{group}_5000p_32s.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            r=json.loads(line)
            if r["category"]=="CR_screen":
                inputs.append((group,r))
    meta={"candidates":len(inputs),"periods":5000,"steps":64,"transient":200}
    if META.exists() and json.loads(META.read_text())!=meta: raise RuntimeError("checkpoint metadata mismatch")
    META.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    done=set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip(): done.add(json.loads(line)["key"])
    todo=[x for x in inputs if f"{x[0]}:{x[1]['index']}" not in done]
    p,family,k,coeff=load_model()
    if todo: integrate_one(np.array(todo[0][1]["initial"]),np.array(todo[0][1]["phase"]),p,family,k,coeff,2,64,0)
    print(f"RK4-64 validation: {len(inputs)} screen hits; {len(done)} already checkpointed; {len(todo)} remain",flush=True)
    t0=time.perf_counter()
    with OUT.open("a",encoding="utf-8",buffering=1) as f, ThreadPoolExecutor(max_workers=6) as pool:
        fs={pool.submit(integrate_one,np.array(row["initial"]),np.array(row["phase"]),p,family,k,coeff,5000,64,200):(group,row) for group,row in todo}
        for j,fut in enumerate(as_completed(fs),1):
            group,row=fs[fut]
            (end,turns,windows,rmin,rmax,zmax,maxd,dmin,completed,retained,escaped,singular,cat,freq)=fut.result()
            record={"key":f"{group}:{row['index']}","group":group,"index":row["index"],
                "category32":row["category"],"category64":str(cat),"turns":turns.tolist(),
                "window_turns":windows.tolist(),"frequency_Hz":freq.tolist(),"r_min":rmin.tolist(),
                "r_max":rmax.tolist(),"max_abs_z":zmax.tolist(),"max_abs_inner_step_dtheta_rad":maxd.tolist(),
                "distance_min":float(dmin),"completed_periods":int(completed),"retained":bool(retained),
                "escaped":bool(escaped),"singular_unresolved":bool(singular),"final_state":end.tolist()}
            f.write(json.dumps(record,separators=(",",":"))+"\n")
            if j%8==0 or j==len(todo): print(f"refined {j}/{len(todo)} ({time.perf_counter()-t0:.1f}s)",flush=True)

if __name__=="__main__":main()
