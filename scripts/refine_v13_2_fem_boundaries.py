"""Step-size checks for V13.2 h50/P4 trajectories near the finite aperture."""
from __future__ import annotations

import argparse,csv,json,math,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from run_v13_1_fem_pilot import advance,initialize

OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability/stage_a_full_fem_h50_p4/step_refinement"
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability/stage_a_full_fem_h50_p4/trajectory_results.csv"


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=2);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    base=list(csv.DictReader(BASE.open(newline="",encoding="utf-8")))
    selected=[]
    for phase in (0.,90.):
      for model in ("Q","SO","RO"):
        rr=[r for r in base if abs(float(r["phase_deg"])-phase)<1e-8 and r["model"]==model]
        retained=sorted([r for r in rr if r["status_300"]=="retained"],
                        key=lambda r:float(r["max_rho_300"]),reverse=True)[:2]
        escaped=sorted([r for r in rr if r["status_300"]=="escaped" and r.get("exit_periods_300") not in (None,"")],
                       key=lambda r:float(r["exit_periods_300"]),reverse=True)[:2]
        for r in retained+escaped:
            if not any(int(q["index"])==int(r["index"]) and q["phase_deg"]==r["phase_deg"] and q["model"]==r["model"] for q in selected):
                selected.append(r)
    tasks=[]
    for r in selected:
        state=[float(r[k]) for k in ("x0","y0","vx0","vy0")]
        tasks.append((float(r["mu"]),float(r["phase_rad"]),r["model"],float(r["eta"]),int(r["index"]),state,1/128))
    config={"source":"selected V13.2 full FEM h50/P4 boundary trajectories",
        "selection":"up to two largest-radius retained and two latest-exit escaped trajectories per phase/model",
        "n_selected":len(tasks),"horizons_rf":[100,300],"baseline_step_rf":1/64,"refined_step_rf":1/128,
        "integrator":"DOP853 rtol=3e-9 atol=3e-11; same initial states, RF phase and FEM bases",
        "unresolved_policy":"status changes are marked step_sensitive, not counted as lost"}
    (OUT/"parameters.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    ck=OUT/"checkpoint.jsonl";old=[json.loads(x) for x in ck.read_text(encoding="utf-8").splitlines() if x.strip()] if ck.exists() else []
    done={(round(float(r["mu"]),8),round(float(r["phase_rad"]),12),r["model"],round(float(r["eta"]),8),int(r["index"])) for r in old}
    todo=[t for t in tasks if (round(t[0],8),round(t[1],12),t[2],round(t[3],8),t[4]) not in done]
    print(f"step refinement cases={len(tasks)} pending={len(todo)}, workers={args.workers}",flush=True)
    if todo:
      with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool,ck.open("a",encoding="utf-8") as f:
        futures=[pool.submit(advance,t) for t in todo]
        for i,fu in enumerate(as_completed(futures),1):
            row=fu.result();old.append(row);f.write(json.dumps(row,separators=(",",":"))+"\n");f.flush()
            if i%8==0 or i==len(futures):print(f"refined {i}/{len(futures)}",flush=True)
    selected_keys={(round(float(r["mu"]),8),round(float(r["phase_rad"]),12),r["model"],round(float(r["eta"]),8),int(r["index"])) for r in selected}
    old=[r for r in old if (round(float(r["mu"]),8),round(float(r["phase_rad"]),12),r["model"],round(float(r["eta"]),8),int(r["index"])) in selected_keys]
    old=list({(round(float(r["mu"]),8),round(float(r["phase_rad"]),12),r["model"],round(float(r["eta"]),8),int(r["index"])):r for r in old}.values())
    base_by={(round(float(r["mu"]),8),round(float(r["phase_rad"]),12),r["model"],round(float(r["eta"]),8),int(r["index"])):r for r in base}
    out=[]
    for r in old:
        k=(round(float(r["mu"]),8),round(float(r["phase_rad"]),12),r["model"],round(float(r["eta"]),8),int(r["index"]))
        b=base_by[k]
        out.append({**r,"baseline_status_100":b["status_100"],"baseline_status_300":b["status_300"],
            "step_sensitive_100":r["status_100"]!=b["status_100"],"step_sensitive_300":r["status_300"]!=b["status_300"],
            "delta_exit_periods_300":(float(r["exit_periods_300"])-float(b["exit_periods_300"]))
                if r.get("exit_periods_300") not in (None,"") and b.get("exit_periods_300") not in (None,"") else None})
    fields=[]
    for r in out:fields.extend(k for k in r if k not in fields)
    with (OUT/"trajectory_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(out)
    with ck.open("w",encoding="utf-8") as f:
        for r in old:f.write(json.dumps(r,separators=(",",":"))+"\n")
    print(f"classified step-sensitive 100={sum(bool(r['step_sensitive_100']) for r in out)} "
          f"300={sum(bool(r['step_sensitive_300']) for r in out)} of {len(out)}",flush=True)


if __name__=="__main__":main()
