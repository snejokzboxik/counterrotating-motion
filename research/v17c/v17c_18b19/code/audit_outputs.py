"""Integrity audit for generated V17C.18B-19 data and figure deliverables."""
from __future__ import annotations
import csv, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read_csv(name):
    with (ROOT/"data"/name).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    summary=json.loads((ROOT/"results"/"screening_summary.json").read_text(encoding="utf-8"))
    assert summary["total_trajectories"]==1000
    assert len(summary["maps"])==5
    expected={"alpha_lambda.csv":169,"alpha_nu.csv":169,"alpha_b_over_a.csv":169,
              "alpha_pol_PS2.csv":169,"alpha_pair_starts.csv":324,
              "alpha_pair_summary.csv":81}
    checks=[]
    for name,n in expected.items():
        rows=read_csv(name)
        assert len(rows)==n,(name,len(rows),n)
        checks.append({"file":name,"rows":len(rows),"expected":n})
    pair=read_csv("alpha_pair_starts.csv")
    outcomes={}
    for row in pair: outcomes[row["outcome"]]=outcomes.get(row["outcome"],0)+1
    assert sum(outcomes.values())==324
    assert outcomes.get("CR_PLUS_MINUS_SCREEN",0)+outcomes.get("CR_MINUS_PLUS_SCREEN",0)==75
    fem=json.loads((ROOT/"results"/"finite_3d_rf_fem.json").read_text(encoding="utf-8"))
    assert len(fem["mesh_levels"])==3
    assert all(len(r["midplane_leading_transfer_sample"])==2 for r in fem["mesh_levels"])
    endcap=json.loads((ROOT/"results"/"continuous_rods_endcap_fem.json").read_text(encoding="utf-8"))
    assert len(endcap["mesh_levels"])==5
    assert endcap["convergence_gate"]["passed"] is True
    assert abs(endcap["mesh_levels"][-1]["mesh_hmax_mm"]-0.24)<1e-10
    endcap_audit=json.loads((ROOT/"results"/"endcap_geometry_audit.json").read_text(encoding="utf-8"))
    assert endcap_audit["audit_passed"] is True
    assert endcap_audit["two_consecutive_refinement_gate_passed"] is True
    assert endcap_audit["final_trajectories_run"] is False
    validation=json.loads((ROOT/"results"/"selected_numerical_validation.json").read_text(encoding="utf-8"))
    assert len(validation["pair_RK4_refinement"])==3 and validation["pair_DOP853"]["n_output_periods"]==1000
    index=json.loads((ROOT/"data"/"figure_index.json").read_text(encoding="utf-8"))
    figs=[]
    for stem in index["named_figure_stems"]:
        for ext in ("png","pdf","svg"):
            p=ROOT/"figures"/f"{stem}.{ext}"
            assert p.exists() and p.stat().st_size>1000,p
        figs.append(stem)
    audit={"trajectory_total":summary["total_trajectories"],"csv_checks":checks,
           "pair_outcome_counts":outcomes,"fem_mesh_levels":len(fem["mesh_levels"]),
           "continuous_rods_endcap_fem":{"solved_levels":len(endcap["mesh_levels"]),
             "scalar_convergence_gate":endcap["convergence_gate"]["passed"],
             "finest_hmax_mm":endcap["mesh_levels"][-1]["mesh_hmax_mm"],
             "final_trajectories_run":endcap_audit["final_trajectories_run"]},
           "selected_validation":"RK4 64/128/256 and DOP853; 1000 periods",
           "figure_stems":figs,"all_assertions":"PASS"}
    (ROOT/"results"/"output_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(audit,indent=2))

if __name__=="__main__": main()
