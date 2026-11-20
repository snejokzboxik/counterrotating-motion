"""RK4 step refinement on the exact representatives used by DOP853."""
import json, math, time
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"v17c_9"/"baseline"))
import basin_scan
import loading_dynamics as ld
OUT=HERE/"strict_long_validation"
PERIODS=15000

def main():
    params=json.loads((HERE/"local15_parameters.json").read_text(encoding="utf-8"))
    p=np.asarray(params["candidate"]["p"],float);qratio=float(params["candidate"]["q2_over_q1"])
    _p0,fam,k,coeff=basin_scan.load_model()
    protos=json.loads((HERE/"loading_screen"/"protocols.json").read_text(encoding="utf-8"))
    dop=json.loads((OUT/"dop853_energy_validation.json").read_text(encoding="utf-8"))
    baseline=[json.loads(s) for s in (OUT/"strict_runs.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    result=[]
    for d in dop:
        pid=d["protocol_id"];key=(pid,d["ensemble"],d["start_id"])
        row=next(r for r in baseline if (r["protocol_id"],r["ensemble"],r["start_id"])==key)
        protocol=np.asarray(protos[pid]["protocol"],float)
        skip=int(math.ceil(ld.protocol_duration_cycles(protocol)))+200
        levels=[]
        for steps in (64,96,128):
            t0=time.perf_counter()
            r=ld.integrate(np.asarray(row["initial"],float),np.asarray(row["phase"],float),p,protocol,
                 fam,k,coeff,qratio= qratio, periods=PERIODS,steps=steps,skip=skip,
                 a1_over_R=.0002,a2_over_R=.00189)
            levels.append({"steps":steps,"category":r["category"],"turns":r["turns"].tolist(),
              "window_turns":r["window_turns"].tolist(),"frequency_Hz":r["frequency_Hz"].tolist(),
              "r_min":r["r_min"].tolist(),"r_max":r["r_max"].tolist(),
              "minimum_separation":r["minimum_separation"],"final_state":r["final_state"].tolist(),
              "elapsed_seconds":time.perf_counter()-t0})
        base=levels[1]
        for x in levels:
            x["max_abs_final_state_delta_vs_RK4_96"]=float(np.max(np.abs(np.asarray(x["final_state"])-np.asarray(base["final_state"]))))
            x["max_abs_frequency_delta_Hz_vs_RK4_96"]=float(np.max(np.abs(np.asarray(x["frequency_Hz"])-np.asarray(base["frequency_Hz"]))))
        result.append({"protocol_id":pid,"ensemble":d["ensemble"],"start_id":d["start_id"],
          "DOP853_category":d["strict_dop853"]["category"],"DOP853_frequency_Hz":d["frequency_Hz"],
          "levels":levels})
        print(pid,{x["steps"]:(x["category"],x["frequency_Hz"],x["max_abs_final_state_delta_vs_RK4_96"]) for x in levels},flush=True)
    (OUT/"step_convergence.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
