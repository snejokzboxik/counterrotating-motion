"""Long-horizon strict counterrotation validation on fresh independent starts."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib, json, math, os
from pathlib import Path
import sys, time

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"v17c_9"/"baseline"
sys.path.insert(0,str(BASE))
import basin_scan  # noqa: E402
import screen_loading_protocols as screen  # noqa: E402
import loading_dynamics as ld  # noqa: E402

OUT=HERE/"strict_long_validation"
GROUPS=("rest","weak","wide","edge")
SEEDS={"rest":20261151,"weak":20261152,"wide":20261153,"edge":20261154}
PROTOCOL_IDS=(0,505,353)


def sha(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def starts_frozen(n=64):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"long_starts.npz"; mp=OUT/"long_starts.json"
    if path.exists() and mp.exists():
        m=json.loads(mp.read_text(encoding="utf-8"))
        if m["n"]!=n:raise RuntimeError("long start set frozen at different N")
        z=np.load(path);return {g:(z[f"{g}_initial"],z[f"{g}_phase"]) for g in GROUPS},m
    old=screen.SEEDS.copy()
    try:
        screen.SEEDS=SEEDS.copy()
        data=screen.make_training_starts(n)
    finally:screen.SEEDS=old
    arrays={};m={"n":n,"seeds":SEEDS,"groups":{},
             "definition":"Fresh independent scrambled Sobol starts from existing V17C.10 distributions; edge has one area-uniform radius in [0.70,0.90]R."}
    for g,(x,ph) in data.items():
        arrays[f"{g}_initial"]=x;arrays[f"{g}_phase"]=ph
        m["groups"][g]={"initial_sha256":sha(x),"phase_sha256":sha(ph)}
    np.savez_compressed(path,**arrays);mp.write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")
    return data,m


def block(task):
    pid,g,ids,initials,phases,prot,p,fam,k,coeff,qratio,periods,steps=task
    trans=int(math.ceil(ld.protocol_duration_cycles(prot)))
    skip=min(periods,trans+200)  # discard the complete common load and 200-cycle relaxation
    out=[]
    for idx,initial,phase in zip(ids,initials,phases):
        r=ld.integrate(initial,phase,p,prot,fam,k,coeff,periods,steps,skip,qratio,
                       a1_over_R=.0002,a2_over_R=.00189)
        out.append({
            "protocol_id":pid,"ensemble":g,"start_id":int(idx),
            "periods":periods,"steps_per_period":steps,
            "transition_cycles":ld.protocol_duration_cycles(prot),
            "post_transition_relaxation_cycles":200,"classification_skip_cycles":skip,
            "initial":np.asarray(initial).tolist(),"phase":np.asarray(phase).tolist(),
            "final_state":np.asarray(r["final_state"]).tolist(),
            "category":r["category"],"retained":bool(r["retained"]),
            "completed_periods":int(r["completed_periods"]),
            "turns_after_relaxation":np.asarray(r["turns"]).tolist(),
            "window_turns":np.asarray(r["window_turns"]).tolist(),
            "frequency_Hz":np.asarray(r["frequency_Hz"]).tolist(),
            "r_min":np.asarray(r["r_min"]).tolist(),"r_max":np.asarray(r["r_max"]).tolist(),
            "abs_z_max":np.asarray(r["abs_z_max"]).tolist(),
            "minimum_separation":float(r["minimum_separation"]),
        })
    return out


def main():
    n,periods,steps,workers,chunk_size=64,15000,96,4,4
    OUT.mkdir(parents=True,exist_ok=True)
    starts,sm=starts_frozen(n)
    desc=json.loads((HERE/"loading_screen"/"protocols.json").read_text(encoding="utf-8"))
    protocols={pid:np.asarray(desc[pid]["protocol"],dtype=np.float64) for pid in PROTOCOL_IDS}
    fixed=json.loads((HERE/"local15_parameters.json").read_text(encoding="utf-8"))
    _p0,fam,k,coeff=basin_scan.load_model()
    p=np.asarray(fixed["candidate"]["p"],dtype=np.float64)
    qratio=float(fixed["candidate"]["q2_over_q1"])
    outpath=OUT/"strict_runs.jsonl";done=set()
    if outpath.exists():
        for line in outpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x=json.loads(line);done.add((x["protocol_id"],x["ensemble"],x["start_id"]))
    tasks=[]
    for pid,prot in protocols.items():
        for g in GROUPS:
            ii,ph=starts[g]
            for beg in range(0,n,chunk_size):
                ids=[j for j in range(beg,min(n,beg+chunk_size)) if (pid,g,j) not in done]
                if ids:tasks.append((pid,g,ids,np.asarray([ii[j] for j in ids]),np.asarray([ph[j] for j in ids]),prot,p,fam,k,coeff,qratio,periods,steps))
    if tasks:
        ld.integrate(starts["rest"][0][0],starts["rest"][1][0],p,ld.constant_protocol(),fam,k,coeff,1,steps,0,qratio,a1_over_R=.0002,a2_over_R=.00189)
    meta={"status":"running","protocol_ids":list(PROTOCOL_IDS),"N_per_ensemble":n,
          "ensembles":list(GROUPS),"periods":periods,"steps_per_period":steps,
          "workers":workers,"chunk_size":chunk_size,"seeds":SEEDS,
          "starts_sha256":sm["groups"],"terminal_p":p.tolist(),"q2_over_q1":qratio,
          "strict_success":"category CR_screen only: retained, no contact/axis ambiguity, >=10 turns each after transition+200 cycle relaxation, opposite signs in all three late windows",
          "note":"Selected from prior training and validation; this fresh long sample is independent of both. Record-level initial and final states are included."}
    (OUT/"strict_metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(f"strict long protocols={len(protocols)}, N={n}*4, horizon={periods}, steps={steps}, tasks={len(tasks)}, workers={workers}",flush=True)
    t0=time.time();blocks=0
    with outpath.open("a",encoding="utf-8",buffering=1) as out,ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(block,t) for t in tasks]
        for f in as_completed(futures):
            for row in f.result():out.write(json.dumps(row,separators=(",",":"))+"\n")
            out.flush();os.fsync(out.fileno());blocks+=1
            if blocks%8==0:print(f"checkpoint {blocks}/{len(tasks)} starts={blocks*chunk_size} elapsed={time.time()-t0:.0f}s",flush=True)
    meta["status"]="complete";meta["elapsed_seconds"]=time.time()-t0
    (OUT/"strict_metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(f"complete blocks={blocks}; elapsed={meta['elapsed_seconds']:.1f}s",flush=True)


if __name__=="__main__":main()
