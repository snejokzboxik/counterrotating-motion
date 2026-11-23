from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(HERE))
import analytical_model as model  # noqa:E402


def main():
    data=model.load_inputs()
    # Predeclared hold-out: A +5%; its full RK4 trajectory is in heldout_A_1.050.npz.
    data["modes"]=[replace(m,amplitude=m.amplitude*1.05) if m.family==0 else m for m in data["modes"]]
    data["exact_p"][0]*=1.05
    seed=model.solve_m12(data,list(range(len(data["modes"]))))
    m3=model.solve_m3(data,seed,(96,12))
    rk=json.loads((BASE/"results"/"heldout_validation.json").read_text(encoding="utf-8"))
    ref=next(x for x in rk["rows"] if x["parameter"]=="A" and x["factor"]==1.05)
    out={"heldout_case":"A*1.05, all other V20B parameters held fixed",
      "analytical_M3":{"radii_R":m3["radii_R"],"mean_z_R":m3["mean_z_R"],
          "frequency_Hz":m3["orbital_frequency_Hz"],"solver_residual_scaled":m3["solver_residual_scaled"],
          "torus_grid":m3["phase_grid"],"particle_hb":[{k:p[k] for k in
             ("converged","relative_fixed_point_residual","rms_unresolved_force","rms_rf_force",
              "mean_total_RF_torque","mean_total_drag_torque","rms_micromotion")} for p in m3["inner_harmonic_balance"]["particles"]]},
      "full_RK4_reference":{"horizon_periods":ref["horizon_periods"],"frequency_Hz":ref["frequency_Hz"],
          "mean_radius_R":ref["mean_radius_R"],"late_window_frequency_Hz":ref["late_window_frequency_Hz"]},
      "frequency_relative_error_percent":[float(100*(a-b)/abs(b)) for a,b in zip(m3["orbital_frequency_Hz"],ref["frequency_Hz"])],
      "radius_relative_error_percent":[float(100*(a-b)/b) for a,b in zip(m3["radii_R"],ref["mean_radius_R"])],
      "qualification":"One continued-state finite-horizon hold-out. Good agreement is a numerical validation point, not a general accuracy guarantee."}
    (BASE/"results"/"heldout_M3_A105.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
