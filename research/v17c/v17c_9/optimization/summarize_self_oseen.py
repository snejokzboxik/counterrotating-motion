"""Matched-start single-particle Oseen-drag correction summary."""
import json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUNS=HERE/"stage3"/"runs"
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]
def main():
    out={"config_id":48,"N_per_ensemble":64,"periods":15000,"steps_per_RF_period":64,
      "radii_over_R":[.0002,.00189],"Re_per_dimensionless_speed":[.2,1.89],"ensembles":{}}
    for g in ("rest","weak","wide"):
        path=RUNS/f"cfg48_{g}_matched_holdout_none_chi0.2-1.89_a0.0002-0.00189_64_15000p_64s.jsonl"
        rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        bpath=RUNS/f"cfg48_{g}_holdout_128_15000p_64s.jsonl"
        base={int(r["index"]):r for r in (json.loads(s) for s in bpath.read_text(encoding="utf-8").splitlines() if s.strip())}
        cr=[r for r in rows if r["category"]=="CR_screen"]
        cats={}
        for r in rows:cats[r["category"]]=cats.get(r["category"],0)+1
        new=sum(r["category"]=="CR_screen" and base[r["index"]]["category"]!="CR_screen" for r in rows)
        lost=sum(r["category"]!="CR_screen" and base[r["index"]]["category"]=="CR_screen" for r in rows)
        shifts=[np.asarray(r["frequency_Hz"])-np.asarray(base[r["index"]]["frequency_Hz"]) for r in cr]
        out["ensembles"][g]={"N_total":len(rows),"category_counts":cats,"N_CR":len(cr),
          "P_CR":len(cr)/len(rows),"Wilson95":wilson(len(cr),len(rows)),
          "matched_changes_vs_point":{"new_CR":new,"lost_point_CR":lost},
          "median_frequency_Hz_CR":np.median([r["frequency_Hz"] for r in cr],axis=0).tolist() if cr else None,
          "median_frequency_change_vs_point_CR_Hz":np.median(shifts,axis=0).tolist() if shifts else None,
          "minimum_surface_gap_CR_over_R":min((r["minimum_separation"]-.0002-.00189 for r in cr),default=None)}
    dst=HERE/"stage3"/"self_oseen_summary.json";dst.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
