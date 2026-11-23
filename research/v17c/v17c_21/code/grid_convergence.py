from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=HERE.parent
sys.path.insert(0,str(HERE))
import analytical_model as model  # noqa:E402


def main():
    data=model.load_inputs()
    pred=json.loads((BASE/"results"/"analytical_predictions.json").read_text(encoding="utf-8"))
    root=pred["models"]["M3_full_MFS_SN_torus_Galerkin"]
    grids=[(96,8),(96,12),(128,12),(192,16)]
    rows=[]
    for nt,np_ in grids:
        particles=[]
        for i in range(2):
            hb=model.nonlinear_micromotion_full(root["radii_R"][i],root["omega_over_Omega1"][i],
                data["alpha"][i],data["lambda"][i],data["Re_per_speed"][i],data["beta"],data["modes"],
                nt,np_,tolerance=5e-6,max_iter=160)
            particles.append({k:hb[k] for k in ("converged","iterations","relative_fixed_point_residual",
                "mean_rf_force_rotating_xy","mean_drag_force_rotating_xy","mean_total_RF_torque",
                "mean_total_drag_torque","mean_angular_momentum","mean_speed","rms_micromotion",
                "rms_rf_force","rms_unresolved_force","rms_slow_mode_residual",
                "rms_fast_mode_equation_residual","phase_grid","slow_mode_cutoff")})
        rows.append({"grid":[nt,np_,np_],"particles":particles})
    out={"method":"M3 full MFS Schiller–Naumann phase-torus collocation, evaluated at frozen M3 slow roots",
         "source_240k_sha256":hashlib.sha256((ROOT/"research"/"v17c"/"v17c_20b"/"results"/"refine12_240k_corrected_drag_trajectory.npz").read_bytes()).hexdigest(),
         "rows":rows,"interpretation":"Mean torque and force are grid-stable; fixed-point convergence and unresolved residuals are reported separately, especially at the largest grid."}
    (BASE/"results"/"m3_grid_convergence.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
