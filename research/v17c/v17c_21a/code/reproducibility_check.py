"""Freshly re-solve the coarsest complete M3 root as a reproducibility check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[0]
sys.path.insert(0,str(ROOT/"v17c_21"/"code"))
sys.path.insert(0,str(BASE/"code"))
import analytical_model as model  # noqa:E402
from m3_harmonic_balance import HBConfig  # noqa:E402
from m3_system import initial_root, solve_m3_root  # noqa:E402


def main():
    archived=json.loads((BASE/"results"/"full_grid_campaign.json").read_text(encoding="utf-8"))
    stage=archived["stages"][0]
    c=stage["config"]
    config=HBConfig(**c)
    with np.load(BASE/"results"/"xi_seed_96_8.npz") as z:
        seed_xi=[z["xi1"],z["xi2"]]
    source_seed=initial_root(stage["root"])
    perturbed_seed=source_seed+np.array([1e-3,1e-4,-1e-3,-1e-4,1e-3,-1e-3])
    fresh=solve_m3_root(model.load_inputs(),perturbed_seed,config,
                        max_nfev=40,xi_seed=seed_xi,require_convergence=True)
    root=fresh["root"]
    old=stage["root"]
    result={
        "study":"V17C.21.1 independent root re-solve",
        "grid":stage["grid"],
        "method":"fresh JFNK harmonic-balance solves and fresh outer six-balance solve from a perturbed slow-root seed; converged xi checkpoint used only as warm start",
        "slow_root_seed_perturbation": [1e-3,1e-4,-1e-3,-1e-4,1e-3,-1e-3],
        "status":fresh["status"],
        "fresh_root":root,
        "original_root":old,
        "relative_root_change_percent":{
            "radii_R":[100*(a/b-1) for a,b in zip(root["radii_R"],old["radii_R"])],
            "frequency_Hz":[100*(a/b-1) for a,b in zip(root["frequency_Hz"],old["frequency_Hz"])],
            "mean_z_R":[100*(a/b-1) for a,b in zip(root["mean_z_R"],old["mean_z_R"])],
        },
        "scaled_balance_max_norm":fresh["scaled_balance_max_norm"],
        "inner_all_converged":fresh["inner_all_converged"],
        "inner_residuals":[{
            "map":p["relative_nonlinear_residual"],
            "original_fast_equation":p["relative_original_fast_equation_residual"],
            "slow_unresolved_rms":p["rms_unresolved_slow_force"],
        } for p in fresh["inner"]],
        "outer_nfev":fresh["outer_nfev"],
        "balance_evaluations":fresh["balance_evaluations"],
        "wall_time_s":fresh["wall_time_s"],
    }
    (BASE/"results"/"reproducibility_check.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
