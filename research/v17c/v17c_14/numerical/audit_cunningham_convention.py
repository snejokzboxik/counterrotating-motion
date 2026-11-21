"""Quantify the published Cunningham convention mismatch without reruns."""
from __future__ import annotations
import json
from pathlib import Path

HERE=Path(__file__).resolve()
REPO=HERE.parents[4]
SRC=REPO/"research/v17c/v17c_12b/best_physical_configuration.json"
OUT=HERE.parents[1]/"results/cunningham_convention_audit.json"


def main():
    data=json.loads(SRC.read_text(encoding="utf-8"))
    entries={
        "best_discovery":data["best_discovery_candidate"]["physical"],
        "fresh_physical_screen_leader":data["final_comparative_validation"]["fresh_physical_screen_leader"]["configuration"]["physical"],
    }
    candidates=data["final_comparative_validation"]["additional_physical_finalists"]["candidates"]
    for key in ("phys0352","phys0109","phys0526"):
        entries[key]=candidates[key]["configuration"]["physical"]
    rows=[]
    individual=[]
    for name,physical in entries.items():
        c1=float(physical["particle1"]["Cunningham"])
        c2=float(physical["particle2"]["Cunningham"])
        mean=0.5*(c1+c2)
        row={"configuration":name,"Cc_particle1":c1,"Cc_particle2":c2,
             "Cc_mean_used_in_additional_finalist_validation":mean,
             "implemented_over_standard_drag_ratio_for_mean_Cc":mean**2,
             "mean_drag_overstatement_percent":100*(mean**2-1)}
        rows.append(row)
        individual.extend((c1,c2))
    summary={"source_file":"research/v17c/v17c_12b/best_physical_configuration.json",
             "old_rhs":"drag_factor = Cc * (1 + 0.15 Re^0.687)",
             "standard_slip_convention":"drag_factor = (1/Cc) * (same finite-Re factor)",
             "ratio_used_to_standard_for_same_Cc":"Cc^2",
             "individual_Cc_range":[min(individual),max(individual)],
             "individual_drag_overstatement_percent_range":[100*(min(individual)**2-1),100*(max(individual)**2-1)],
             "configurations":rows,
             "trajectory_reruns":0,
             "interpretation":"Coefficient-level correction only; old capture outcomes are not recalculated and may be sensitive to the changed damping."}
    OUT.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
