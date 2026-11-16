"""Compare matched h35/h50 P4 trajectory controls on common initial states."""
from __future__ import annotations
import csv, math
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"

def read(p):
    with Path(p).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def main():
    a=read(BASE/"fem_controls"/"trajectory_results.csv")
    b=read(BASE/"fem_controls_h35"/"trajectory_results.csv")
    def ix(rows):
        out={}
        for r in rows:
            k=(r["candidate_id"],r["model"],round(float(r["tau0_rad"]),12),int(r["state_i"]),int(r["horizon"]))
            if k in out:raise RuntimeError(f"duplicate trajectory key {k}")
            out[k]=r
        return out
    ia,ib=ix(a),ix(b);common=sorted(ia.keys()&ib.keys());rows=[];counts=Counter();by=defaultdict(list)
    for k in common:
        x,y=ia[k],ib[k];same=x["status"]==y["status"]
        dx=abs(float(x["exit_periods"])-float(y["exit_periods"])) if x["exit_periods"] not in ("","nan") and y["exit_periods"] not in ("","nan") else math.nan
        dr=abs(float(x["rho_max_cumulative"])-float(y["rho_max_cumulative"]))
        counts[(k[4],"agree" if same else "different")]+=1
        by[k[4]].append((same,dx,dr))
        rows.append({"candidate_id":k[0],"model":k[1],"tau0_rad":float(x["tau0_rad"]),"state_i":k[3],"horizon_RF":k[4],
                     "h35_status":y["status"],"h50_status":x["status"],"status_agree":same,
                     "h35_exit_RF":y["exit_periods"],"h50_exit_RF":x["exit_periods"],"abs_exit_difference_RF":dx,
                     "h35_rho_max":y["rho_max_cumulative"],"h50_rho_max":x["rho_max_cumulative"],"abs_rho_max_difference":dr})
    out=BASE/"fem_mesh_trajectory_comparison.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for h,vals in sorted(by.items()):
        exits=[x[1] for x in vals if math.isfinite(x[1])];rhos=[x[2] for x in vals]
        summary.append({"horizon_RF":h,"matched_pairs":len(vals),"status_agreements":sum(x[0] for x in vals),
                        "status_disagreements":sum(not x[0] for x in vals),"agreement_fraction":sum(x[0] for x in vals)/len(vals),
                        "median_abs_exit_difference_RF":float(__import__('numpy').median(exits)) if exits else math.nan,
                        "max_abs_exit_difference_RF":max(exits) if exits else math.nan,
                        "median_abs_rho_max_difference":float(__import__('numpy').median(rhos)),"max_abs_rho_max_difference":max(rhos),
                        "interpretation":"matched full-P4 h35 versus h50 finite-time trajectories; same physical voltages and starts"})
    with (BASE/"fem_mesh_trajectory_comparison_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print({"matched_rows":len(rows),"summary":summary})
if __name__=="__main__":main()
