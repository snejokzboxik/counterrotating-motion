"""Fresh-start adaptive envelope screen around strict leaders #505 and #353."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,math,os,time
from pathlib import Path
import sys

import numpy as np
from scipy.stats import qmc
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"v17c_9"/"baseline"
sys.path.insert(0,str(BASE))
import basin_scan
import loading_dynamics as ld
import screen_loading_protocols as first

OUT=HERE/"adaptive_loading"
GROUPS=("rest","weak","wide","edge")
SEEDS={"rest":20261161,"weak":20261162,"wide":20261163,"edge":20261164}
N=32;PERIODS=48;STEPS=96;WORKERS=4

def sha(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def logu(u,lo,hi):return float(math.exp(math.log(lo)+u*math.log(hi/lo)))

def gen_protocols():
    desc0=json.loads((HERE/"loading_screen"/"protocols.json").read_text(encoding="utf-8"))
    out=[]
    for pid in (0,505,353):
        d=desc0[pid];out.append({"id":len(out),"family":"anchor_"+str(pid),
            "duration_cycles":float(d.get("duration_cycles",0)),"protocol":d["protocol"]})
    u=qmc.Sobol(d=12,scramble=True,seed=20261165).random_base2(7)
    shape_ids=[0,1,2,3,4];powers=[.5,1.,2.,3.,4.]
    for a in u[:125]:
        family=min(4,int(a[0]*5))
        shape=shape_ids[min(4,int(a[1]*5))];power=powers[min(4,int(a[2]*5))]
        if family==0:  # #505 neighborhood: delayed DC, fast B and slow A
            ga=float(.05+.55*a[3]);gb=float(.05+.55*a[4]);gd=float(.10+.55*a[5])
            ta=logu(a[6],1.0,20.0);tb=logu(a[7],.10,1.5);td=logu(a[8],.10,4.0)
            delay=float(a[9]*5.0)
            prot=ld.ramp_protocol(duration=ta,duration_b=tb,duration_dc=td,
                delay_dc=delay,shape=shape,power=power,g0=ga,g0_b=gb,g0_dc=gd,dc_mode=0)
            label="adaptive_delayed_dc"
        elif family==1:  # #353 neighborhood: high nonzero RF start, soft ramps
            ga=float(.25+.65*a[3]);gb=float(.25+.65*a[4]);gd=float(.05+.65*a[5])
            ta=logu(a[6],3.0,25.0);tb=logu(a[7],.10,4.0);td=logu(a[8],.10,8.0)
            delay=float(a[9]*2.0)
            prot=ld.ramp_protocol(duration=ta,duration_b=tb,duration_dc=td,
                delay_dc=delay,shape=shape,power=power,g0=ga,g0_b=gb,g0_dc=gd,dc_mode=0)
            label="adaptive_soft_start"
        elif family==2:  # synchronized RF/DC-square branch
            g0=float(.05+.80*a[3]);ta=logu(a[6],.10,20.0)
            prot=ld.ramp_protocol(duration=ta,shape=shape,power=power,g0=g0,
                                  g0_dc=g0*g0,dc_mode=1)
            label="adaptive_dc_rf_squared"
        elif family==3:  # varied RF rates with common nonzero initial level
            g0=float(.10+.75*a[3]);ta=logu(a[6],.10,25.0);tb=logu(a[7],.10,8.0)
            td=logu(a[8],.10,10.0);delay=float(a[9]*5.0)
            prot=ld.ramp_protocol(duration=ta,duration_b=tb,duration_dc=td,
                delay_b=float(a[10]*3.0),delay_dc=delay,shape=shape,power=power,g0=g0,
                g0_dc=float(.05+.75*a[5]),dc_mode=0)
            label="adaptive_independent_rates"
        else:  # capture level and smooth transition, with moderate excursions only
            prot=ld.capture_protocol(capture_a=float(.75+.65*a[3]),
                capture_b=float(.75+.65*a[4]),capture_dc=float(.55+.45*a[5]),
                ramp_cycles=logu(a[6],.25,5.0),hold_cycles=float(a[7]*4.0),
                transition_cycles=logu(a[8],.25,8.0),shape=shape,
                transition_shape=min(3,shape),power=power,
                start_a=float(.1+.55*a[9]),start_b=float(.1+.55*a[10]),start_dc=float(.1+.5*a[11]))
            label="adaptive_capture"
        out.append({"id":len(out),"family":label,
                    "duration_cycles":ld.protocol_duration_cycles(prot),"protocol":prot.tolist()})
    if len(out)!=128:raise RuntimeError("expected 128 protocols")
    return out

def starts_frozen():
    OUT.mkdir(parents=True,exist_ok=True)
    npz=OUT/"training_starts.npz";mp=OUT/"training_starts.json"
    if npz.exists() and mp.exists():
        m=json.loads(mp.read_text(encoding="utf-8"))
        if m["n"]!=N:raise RuntimeError("adaptive training starts already frozen at different N")
        z=np.load(npz);return {g:(z[f"{g}_initial"],z[f"{g}_phase"]) for g in GROUPS},m
    old=first.SEEDS.copy()
    try:first.SEEDS=SEEDS.copy();data=first.make_training_starts(N)
    finally:first.SEEDS=old
    arrays={};m={"n":N,"seeds":SEEDS,"groups":{},"definition":"Fresh V17C.10 distributions with independent scrambled Sobol seeds; edge has area-uniform one-particle radius [0.70,0.90]R."}
    for g,(initial,phase) in data.items():
        arrays[f"{g}_initial"]=initial;arrays[f"{g}_phase"]=phase
        m["groups"][g]={"initial_sha256":sha(initial),"phase_sha256":sha(phase)}
    np.savez_compressed(npz,**arrays);mp.write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")
    return data,m

def block(args):
    pid,g,label,ids,initials,phases,prot,p,fam,k,coeff,qratio=args
    skip=min(PERIODS,int(math.ceil(ld.protocol_duration_cycles(prot))))
    out=[]
    for j,initial,phase in zip(ids,initials,phases):
        r=ld.integrate(initial,phase,p,prot,fam,k,coeff,PERIODS,STEPS,skip,qratio,
                       a1_over_R=.0002,a2_over_R=.00189)
        out.append({"protocol_id":pid,"ensemble":g,"start_id":j,
          "family":label,"transition_cycles":ld.protocol_duration_cycles(prot),
          "transition_complete":ld.protocol_duration_cycles(prot)<=PERIODS,
          "category":r["category"],"retained":bool(r["retained"]),
          "completed_periods":int(r["completed_periods"]),"r_min":r["r_min"].tolist(),
          "minimum_separation":float(r["minimum_separation"]),
          "window_turns":r["window_turns"].tolist(),"turns":r["turns"].tolist()})
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    starts,sm=starts_frozen();protocols=gen_protocols()
    (OUT/"adaptive_protocols.json").write_text(json.dumps(protocols,indent=2)+"\n",encoding="utf-8")
    fixed=json.loads((HERE/"local15_parameters.json").read_text(encoding="utf-8"))
    p=np.asarray(fixed["candidate"]["p"],float);q=float(fixed["candidate"]["q2_over_q1"])
    _p0,fam,k,coeff=basin_scan.load_model()
    outpath=OUT/"adaptive_runs.jsonl";done=set()
    if outpath.exists():
        for s in outpath.read_text(encoding="utf-8").splitlines():
            if s.strip():
                try:
                    r=json.loads(s)
                except json.JSONDecodeError:
                    # A power loss may leave a partial final JSONL row. Earlier
                    # rows are durable and can be resumed without redoing them.
                    continue
                done.add((r["protocol_id"],r["ensemble"],r["start_id"]))
    tasks=[]
    for d in protocols:
        pid=d["id"];prot=np.asarray(d["protocol"],float)
        for g in GROUPS:
            ii,ph=starts[g];ids=[j for j in range(N) if (pid,g,j) not in done]
            if ids:tasks.append((pid,g,d["family"],ids,ii[ids],ph[ids],prot,p,fam,k,coeff,q))
    if tasks:ld.integrate(starts["rest"][0][0],starts["rest"][1][0],p,ld.constant_protocol(),fam,k,coeff,1,STEPS,0,q,a1_over_R=.0002,a2_over_R=.00189)
    meta={"status":"running","protocols":128,"N_per_ensemble":N,"periods":PERIODS,"steps_per_period":STEPS,
       "workers":WORKERS,"ensembles":list(GROUPS),"seeds":SEEDS,"start_hashes":sm["groups"],
       "terminal_parameters":p.tolist(),"q2_over_q1":q,"screen_semantics":"early retention only; never interpret as strict P_CR"}
    (OUT/"adaptive_metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(f"adaptive protocols=128, N={N}*4, horizon={PERIODS}, tasks={len(tasks)}, workers={WORKERS}",flush=True)
    t0=time.time();blocks=0
    with outpath.open("a",encoding="utf-8",buffering=1) as out,ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures=[pool.submit(block,t) for t in tasks]
        for f in as_completed(futures):
            for r in f.result():out.write(json.dumps(r,separators=(",",":"))+"\n")
            out.flush();os.fsync(out.fileno());blocks+=1
            if blocks%16==0:print(f"checkpoint blocks={blocks}/{len(tasks)} rows={blocks*N} elapsed={time.time()-t0:.0f}s",flush=True)
    meta["status"]="complete";meta["elapsed_seconds"]=time.time()-t0
    (OUT/"adaptive_metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(f"complete blocks={blocks}; elapsed={meta['elapsed_seconds']:.1f}s",flush=True)

if __name__=="__main__":main()
