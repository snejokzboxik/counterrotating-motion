"""Matched-start finite hard-sphere boundary/contact check for candidate 48."""
import json, math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RUNS=HERE/"stage3"/"runs"
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-h),min(1.,c+h)]

def main():
    n=64; periods=15000; a1=.0002; a2=.00189
    out={"candidate":48,"N_per_ensemble":n,"periods":periods,
      "radii_over_R":[a1,a2],"models":"hard spherical exclusion and radial/axial domain-contact classification; same Stokes drag as point control",
      "ensemble_results":{}}
    for g in ("rest","weak","wide"):
        f=RUNS/f"cfg48_{g}_matched_holdout_none_chi0-0_a{a1:g}-{a2:g}_{n}_{periods}p_64s.jsonl"
        rows=[json.loads(s) for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]
        bfile=RUNS/f"cfg48_{g}_holdout_128_{periods}p_64s.jsonl"
        base={int(r["index"]):r for r in (json.loads(s) for s in bfile.read_text(encoding="utf-8").splitlines() if s.strip())}
        k=sum(r["category"]=="CR_screen" for r in rows)
        new=sum(r["category"]=="CR_screen" and base[r["index"]]["category"]!="CR_screen" for r in rows)
        lost=sum(r["category"]!="CR_screen" and base[r["index"]]["category"]=="CR_screen" for r in rows)
        cr=[r for r in rows if r["category"]=="CR_screen"]
        cats={}
        for r in rows:cats[r["category"]]=cats.get(r["category"],0)+1
        out["ensemble_results"][g]={"N_total":len(rows),"category_counts":cats,
          "strict_CR_N":k,"P_CR":k/len(rows),"Wilson95":wilson(k,len(rows)),
          "matched_status_changes_vs_point":{"new_CR":new,"lost_CR":lost},
          "median_frequency_Hz_CR":np.median([r["frequency_Hz"] for r in cr],axis=0).tolist() if cr else None,
          "maximum_abs_frequency_change_vs_point_CR_Hz":max((float(np.max(np.abs(np.asarray(r["frequency_Hz"])-np.asarray(base[r["index"]]["frequency_Hz"])))) for r in cr),default=None),
          "minimum_center_separation_CR_over_R":min((r["minimum_separation"] for r in cr),default=None),
          "minimum_surface_gap_CR_over_R":min((r["minimum_separation"]-a1-a2 for r in cr),default=None)}
    dst=HERE/"stage3"/"finite_sphere_summary.json";dst.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
