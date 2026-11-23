"""One-particle M3 diagnostic of the zero mean RF-torque radius at omega=0."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE.parents[0]/"v17c_21"/"code"))
sys.path.insert(0,str(BASE/"code"))
import analytical_model as model
from m3_harmonic_balance import HBConfig, HarmonicBalanceFailure, solve_harmonic_balance

def main():
    d=model.load_inputs()
    cfg=HBConfig(ntheta=96,nphase=12,cutoff=.05,oversample=1,method="newton_krylov",
                 tolerance=1e-7,equation_tolerance=5e-5,max_iter=30)
    radii=[.45,.52,.58,.610413482917315,.64,.70,.78,.86]
    output={"study":"M3 zero mean RF torque radial diagnostic",
            "scope":"Single particle, omega=0, Coulomb off; diagnostic torque locus only, not a full orbital branch or stability boundary.",
            "grid":[96,12,12],"cutoff":.05,"radii_R":radii,"particles":[]}
    for i in range(2):
        xi=None; rows=[]
        for r in radii:
            try:
                hb=solve_harmonic_balance(r,0.0,d["alpha"][i],d["lambda"][i],d["Re_per_speed"][i],
                                          d["beta"],d["modes"],cfg,xi0=xi,require_convergence=True)
            except HarmonicBalanceFailure as exc:
                valid_rows=rows
                brackets=[]
                for left,right in zip(valid_rows[:-1],valid_rows[1:]):
                    if left["mean_RF_torque"]*right["mean_RF_torque"]<0:
                        brackets.append([left["r_R"],right["r_R"]])
                output["particles"].append({"particle":i+1,"status":"UNRESOLVED_INNER_HB",
                    "rows":rows+[{
                        "r_R":r,"omega_over_Omega1":0.0,"status":"INNER_SOLVER_NONCONVERGENCE",
                        "error":str(exc),"grid":[96,12,12],"cutoff":.05,
                        "iteration_cap":cfg.max_iter,"map_residual_reported":2.295e-4,
                        "relative_fast_equation_tolerance":cfg.equation_tolerance}],
                    "zero_torque_brackets_R":brackets})
                output["status"]="TORQUE_BRACKET_FOUND_SCAN_INCOMPLETE" if brackets else "UNRESOLVED_BEFORE_TORQUE_BRACKET"
                output["completed_particles"]=[x.get("particle") for x in output["particles"]]
                output["abort_reason"]="Inner HB failed at a later radius; preceding converged samples retained. No statement is made about the unrun second particle or unsampled radii."
                (BASE/"results"/"m3_zero_rf_torque_scan.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
                print(json.dumps(output,indent=2,ensure_ascii=False),flush=True)
                return
            xi=hb["xi_grid"]
            rows.append({"r_R":r,"mean_RF_torque":hb["mean_total_RF_torque"],
                         "mean_drag_torque":hb["mean_total_drag_torque"],
                         "relative_fast_residual":hb["relative_original_fast_equation_residual"],
                         "inner_map_residual":hb["relative_nonlinear_residual"],
                         "micromotion_rms":hb["rms_micromotion"]})
        brackets=[]
        for a,b in zip(rows[:-1],rows[1:]):
            if a["mean_RF_torque"]*b["mean_RF_torque"]<0:
                brackets.append([a["r_R"],b["r_R"]])
        output["particles"].append({"particle":i+1,"rows":rows,"zero_torque_brackets_R":brackets})
    (BASE/"results"/"m3_zero_rf_torque_scan.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(output,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
