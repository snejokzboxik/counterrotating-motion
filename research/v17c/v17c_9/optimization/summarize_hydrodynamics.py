"""Matched-start point/finite-sphere/Stokes-Oseen/RPY comparison."""
import argparse,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; RUNS=HERE/"stage3"/"runs"

def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-h),min(1.,c+h)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--count",type=int,default=64);ap.add_argument("--periods",type=int,default=15000)
    ap.add_argument("--a1-over-R",type=float,default=.0002);ap.add_argument("--a2-over-R",type=float,default=.00189)
    ap.add_argument("--chi1",type=float,default=.2);ap.add_argument("--chi2",type=float,default=1.89);a=ap.parse_args()
    groups=("rest","weak","wide")
    models={
      "point_linear":None,
      "finite_sphere_stokes":f"none_chi0-0_a{a.a1_over_R:g}-{a.a2_over_R:g}",
      "single_particle_oseen":f"none_chi{a.chi1:g}-{a.chi2:g}_a{a.a1_over_R:g}-{a.a2_over_R:g}",
      "pair_oseen_tensor":f"oseen_chi0-0_a{a.a1_over_R:g}-{a.a2_over_R:g}",
      "pair_rpy_tensor":f"rpy_chi0-0_a{a.a1_over_R:g}-{a.a2_over_R:g}"}
    out={"config_id":48,"count":a.count,"periods":a.periods,"all_initials_in_N_total":True,
      "finite_radii_over_R":[a.a1_over_R,a.a2_over_R],"models":{}}
    for model,suffix in models.items():
        out["models"][model]={}
        for g in groups:
            if suffix is None:
                f=RUNS/f"cfg48_{g}_holdout_{a.count}_{a.periods}p_64s.jsonl"
            else:
                f=RUNS/f"cfg48_{g}_holdout_{suffix}_{a.count}_{a.periods}p_64s.jsonl"
            rows=[json.loads(s) for s in f.read_text(encoding="utf-8").splitlines() if s.strip()]
            basefile=RUNS/f"cfg48_{g}_holdout_{a.count}_{a.periods}p_64s.jsonl"
            base={int(r["index"]):r for r in (json.loads(s) for s in basefile.read_text(encoding="utf-8").splitlines() if s.strip())}
            cr=[r for r in rows if r["category"]=="CR_screen"]
            n=len(rows);k=len(cr)
            b=sum(r["category"]=="CR_screen" and base[int(r["index"])]["category"]!="CR_screen" for r in rows)
            c=sum(r["category"]!="CR_screen" and base[int(r["index"])]["category"]=="CR_screen" for r in rows)
            rec={"N_total":n,"counts":{},"N_CR":k,"P_CR":k/n,"Wilson95":wilson(k,n),
              "N_retained":sum(bool(r["retained"]) for r in rows),"P_retained":sum(bool(r["retained"]) for r in rows)/n,
              "median_frequency_Hz_CR":np.median([r["frequency_Hz"] for r in cr],axis=0).tolist() if cr else None,
              "minimum_center_separation_CR":min((r["minimum_separation"] for r in cr),default=None),
              "minimum_surface_gap_over_R_CR":min((r["minimum_separation"]-a.a1_over_R-a.a2_over_R for r in cr),default=None),
              "matched_changes_vs_point":{"new_CR":b,"lost_point_CR":c},
              "paired_McNemar_exact_p":1.0}
            for r in rows:rec["counts"][r["category"]]=rec["counts"].get(r["category"],0)+1
            d=b+c
            if d:
                m=min(b,c);rec["paired_McNemar_exact_p"]=min(1.,2*sum(math.comb(d,i) for i in range(m+1))/(2**d))
            out["models"][model][g]=rec
    dst=HERE/"stage3"/"hydrodynamics_summary.json";dst.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
