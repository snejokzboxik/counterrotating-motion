"""Small M3 control solves: isolated leading drives vs their competition."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research/v17c/v17c_21/code"))
sys.path.insert(0,str(ROOT/"research/v17c/v17c_21a/code"))
from analytical_model import load_inputs
from m3_harmonic_balance import HBConfig
from m3_system import solve_m3_root

def run():
    data=load_inputs()
    campaign=json.loads((ROOT/"research/v17c/v17c_21a/results/full_grid_campaign.json").read_text())
    seed=campaign["stages"][-1]["root"]
    cfg=HBConfig(ntheta=96,nphase=8,cutoff=.05,oversample=2,method="newton_krylov",
                 tolerance=2e-7,equation_tolerance=8e-5,max_iter=40,relaxation=.5,history=6)
    tests=[
        ("plus2_only",[m for m in data["modes"] if m.k==2],data["exact_p"][5]),
        ("minus4_only",[m for m in data["modes"] if m.k==-4],data["exact_p"][5]),
        ("both_leading",[m for m in data["modes"] if m.k in (2,-4)],data["exact_p"][5]),
        ("both_leading_C0",[m for m in data["modes"] if m.k in (2,-4)],0.0),
    ]
    rows=[]
    x0=np.array([seed["radii_R"][0],seed["omega_over_Omega1"][0],
                 seed["radii_R"][1],seed["omega_over_Omega1"][1],
                 seed["mean_z_R"][0],seed["mean_z_R"][1]],float)
    for name,modes,C in tests:
        d=dict(data); d["modes"]=modes; d["C"]=C
        try:
            ans=solve_m3_root(d,x0,cfg,max_nfev=40,scaled_balance_tolerance=1e-5,
                              require_convergence=False)
            root=ans["root"]
            rows.append({"case":name,"k_modes":[m.k for m in modes],"C12":C,
                         "status":ans["status"],"root":root,
                         "scaled_residual":ans["scaled_balance_residual"],
                         "inner_all_converged":ans["inner_all_converged"],
                         "grid":[96,8,8],"wall_s":ans["wall_time_s"],
                         "counterrotation_signs":bool(root["omega_over_Omega1"][0]*root["omega_over_Omega1"][1]<0),
                         "interpretation":"Exploratory M3 control, coarser than nominal; not a branch continuation."})
            x0=np.array([root["radii_R"][0],root["omega_over_Omega1"][0],
                         root["radii_R"][1],root["omega_over_Omega1"][1],
                         root["mean_z_R"][0],root["mean_z_R"][1]])
            print(name,ans["status"],root["omega_over_Omega1"],flush=True)
        except Exception as exc:
            rows.append({"case":name,"k_modes":[m.k for m in modes],"C12":C,
                         "status":"SOLVER_FAILURE","error_type":type(exc).__name__,"error":repr(exc),"grid":[96,8,8]})
            print(name,"SOLVER_FAILURE",str(exc),flush=True)
    return {"study":"M3 leading-family controls","parameters":data["exact_p"],
            "config":cfg.__dict__,"rows":rows,
            "caveat":"Independent RF families are removed from MFS drive; this is a mathematical control, not a realizable electrode voltage change preserving all field components."}

if __name__=="__main__":
    out=run(); p=ROOT/"data/rf_component_controls.json"
    p.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")

