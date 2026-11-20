"""Matched representative C_eff controls for the strict loading protocols."""
import json,math,time
from pathlib import Path
import sys

import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"v17c_9"/"baseline"))
import basin_scan
import loading_dynamics as ld

OUT=HERE/"strict_long_validation"
FACTORS=(0.0,0.5,1.0,2.0)

def main():
    d=json.loads((HERE/"local15_parameters.json").read_text(encoding="utf-8"))
    p0=np.asarray(d["candidate"]["p"],float);q=float(d["candidate"]["q2_over_q1"])
    _p,fam,k,coeff=basin_scan.load_model()
    protos=json.loads((HERE/"loading_screen"/"protocols.json").read_text(encoding="utf-8"))
    dop=json.loads((OUT/"dop853_energy_validation.json").read_text(encoding="utf-8"))
    rows=[json.loads(s) for s in (OUT/"strict_runs.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    outputs=[]
    for rep in dop:
        pid=rep["protocol_id"]
        row=next(r for r in rows if r["protocol_id"]==pid and r["ensemble"]==rep["ensemble"] and r["start_id"]==rep["start_id"])
        protocol=np.asarray(protos[pid]["protocol"],float)
        skip=int(math.ceil(ld.protocol_duration_cycles(protocol)))+200
        case=[]
        for factor in FACTORS:
            p=p0.copy();p[5]=p0[5]*factor
            start=time.perf_counter()
            r=ld.integrate(np.asarray(row["initial"],float),np.asarray(row["phase"],float),p,protocol,
                 fam,k,coeff,15000,96,skip,q,a1_over_R=.0002,a2_over_R=.00189)
            case.append({"coupling_factor":factor,"C_eff":float(p[5]),"category":r["category"],
              "turns":r["turns"].tolist(),"window_turns":r["window_turns"].tolist(),
              "frequency_Hz":r["frequency_Hz"].tolist(),"r_min":r["r_min"].tolist(),
              "r_max":r["r_max"].tolist(),"minimum_separation":float(r["minimum_separation"]),
              "final_state":r["final_state"].tolist(),"runtime_s":time.perf_counter()-start})
        nominal=next(x for x in case if x["coupling_factor"]==1.0)
        for x in case:
            x["frequency_delta_vs_nominal_Hz"]=(np.asarray(x["frequency_Hz"])-nominal["frequency_Hz"]).tolist()
            x["final_state_max_abs_delta_vs_nominal"]=float(np.max(np.abs(np.asarray(x["final_state"])-nominal["final_state"])))
        outputs.append({"protocol_id":pid,"ensemble":rep["ensemble"],"start_id":rep["start_id"],
                        "same_initial_and_phase_all_C":True,"coupling_cases":case})
        print(pid,[(x["coupling_factor"],x["category"],x["frequency_Hz"],x["final_state_max_abs_delta_vs_nominal"]) for x in case],flush=True)
    (OUT/"coupling_controls.json").write_text(json.dumps(outputs,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
