"""Matched-start Coulomb-strength scan for finite candidate-48 spheres."""
import json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUNS=HERE/"stage3"/"runs"
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-h),min(1.,c+h)]
def main():
    n=64;periods=15000;a1=.0002;a2=.00189
    scales=(0.,.5,1.,2.)
    out={"candidate_id":48,"N_per_ensemble":n,"periods":periods,"radii_over_R":[a1,a2],
      "C_scale_definition":"scale applied to fitted Coulomb coefficient C; external RF/DC, q/m ratios and drag held fixed",
      "ensembles":{}}
    for g in ("rest","weak","wide"):
        out["ensembles"][g]={}
        baseline=None
        for scale in scales:
            suffix=(f"_Cscale{scale:g}" if scale!=1. else "")
            f=RUNS/f"cfg48_{g}_matched_holdout_none_chi0-0_a{a1:g}-{a2:g}{suffix}_{n}_{periods}p_64s.jsonl"
            rows=[json.loads(s) for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]
            cr=[r for r in rows if r["category"]=="CR_screen"]
            cats={}
            for r in rows:cats[r["category"]]=cats.get(r["category"],0)+1
            if scale==1.:baseline={int(r["index"]):r for r in rows}
            out["ensembles"][g][f"C_scale_{scale:g}"]={"N_total":len(rows),"category_counts":cats,
              "N_CR":len(cr),"P_CR":len(cr)/len(rows),"Wilson95":wilson(len(cr),len(rows)),
              "median_frequency_Hz_CR":np.median([r["frequency_Hz"] for r in cr],axis=0).tolist() if cr else None,
              "minimum_center_separation_CR_over_R":min((r["minimum_separation"] for r in cr),default=None),
              "minimum_surface_gap_CR_over_R":min((r["minimum_separation"]-a1-a2 for r in cr),default=None)}
        for scale in (0.,.5,2.):
            suffix=f"_Cscale{scale:g}"
            f=RUNS/f"cfg48_{g}_matched_holdout_none_chi0-0_a{a1:g}-{a2:g}{suffix}_{n}_{periods}p_64s.jsonl"
            rows=[json.loads(s) for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]
            cr=[r for r in rows if r["category"]=="CR_screen"]
            ccr=[r for r in baseline.values() if r["category"]=="CR_screen"]
            new=sum(r["category"]=="CR_screen" and baseline[int(r["index"])]["category"]!="CR_screen" for r in rows)
            lost=sum(r["category"]!="CR_screen" and baseline[int(r["index"])]["category"]=="CR_screen" for r in rows)
            shifts=[np.asarray(r["frequency_Hz"])-np.asarray(baseline[int(r["index"])]["frequency_Hz"]) for r in rows
                    if r["category"]=="CR_screen" and baseline[int(r["index"])]["category"]=="CR_screen"]
            item=out["ensembles"][g][f"C_scale_{scale:g}"]
            item["matched_changes_vs_C1"]={"new_CR":new,"lost_C1_CR":lost}
            item["median_frequency_change_vs_C1_Hz_for_both_CR"]=np.median(shifts,axis=0).tolist() if shifts else None
    dst=HERE/"stage3"/"coulomb_coupling_summary.json";dst.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
