from __future__ import annotations

"""Four predeclared one-factor hold-outs against the V17C.20B RK4 pair RHS."""
import json
import hashlib
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "v17c" / "v17c_17"))
import analytical_model as model  # noqa: E402
import run_pair_coulomb_scan as pair  # noqa: E402

BASE = HERE.parent
TRAJECTORY = ROOT / "research" / "v17c" / "v17c_20b" / "results" / "refine12_240k_corrected_drag_trajectory.npz"


def point_data(base, key, factor):
    d = dict(base)
    d["modes"] = list(base["modes"])
    d["exact_p"] = list(base["exact_p"])
    if key in ("A", "B"):
        fam = 0 if key == "A" else 1
        d["modes"] = [replace(m, amplitude=m.amplitude*factor) if m.family == fam else m for m in base["modes"]]
        d["exact_p"][fam] *= factor
    elif key == "nu":
        d["modes"] = [replace(m, frequency=m.frequency*factor) if m.family == 1 else m for m in base["modes"]]
        d["exact_p"][2] *= factor
    elif key == "beta":
        d["beta"] *= factor
        d["exact_p"][3] *= factor
    return d


def solve_reduced(d, x0):
    nmode=len(d["modes"])
    sol=least_squares(lambda x:model.m12_balance(x,d,list(range(nmode))),x0,
                      bounds=([.1,-.05,.1,-.05,-2,-2],[.99,.05,.99,.05,2,2]),
                      xtol=2e-12,ftol=2e-12,gtol=2e-12,max_nfev=500)
    return sol.x


def run():
    base=model.load_inputs()
    root=model.solve_m12(base,list(range(len(base["modes"]))))
    x0=np.array([root["radii_R"][0],root["omega_over_Omega1"][0],root["radii_R"][1],
                 root["omega_over_Omega1"][1],root["mean_z_R"][0],root["mean_z_R"][1]])
    data=np.load(TRAJECTORY)
    y0=np.asarray(data["states"][-1],float)
    tau_end=float(data["step_index"][-1])*2*np.pi/float(data["steps"])
    phase0=np.asarray(data["phase"],float)
    p0=np.asarray(base["exact_p"],float)
    family=np.asarray([m.family for m in base["modes"]],np.int64)
    modes=np.asarray([m.k for m in base["modes"]],np.int64)
    coeff=np.asarray([m.coefficient for m in base["modes"]],np.complex128)
    phys=[base["candidate"]["physical_parameter_mapping_from_V20B"]["particle1"],
          base["candidate"]["physical_parameter_mapping_from_V20B"]["particle2"]]
    mapping=base["candidate"]["physical_parameter_mapping_from_V20B"]
    a1=float(mapping["a1_over_R"]); a2=float(mapping["a2_over_R"])
    re1,re2=base["Re_per_speed"]
    cases=[("A",1.05),("B",0.95),("nu",1.02),("beta",1.05)]
    rows=[]
    for key,factor in cases:
        d=point_data(base,key,factor)
        xp=solve_reduced(d,x0)
        p=np.asarray(d["exact_p"],float)
        # Preserve the absolute external phases at the archived terminal time.
        phase=np.array([phase0[0]+tau_end,phase0[1]+p[2]*tau_end])%(2*np.pi)
        raw=pair._integrate_trace(y0.copy(),phase,p,family,modes,coeff,
             d["charge_ratio_q2_q1"],a1,a2,re1,re2,d["gravity_z"],
             15000,64,1000,6400)
        end,states,step_idx,work,turns,windows,rmin,rmax,zmax,vmax,dmin,avgr,avd,avuc,winrm,windm,wrk,completed,status=raw
        frequencies=np.asarray(turns)*d["f1_Hz"]/max(1,int(completed)-1000)
        bounds=[1000,1000+(completed-1000)//3,1000+2*(completed-1000)//3,completed]
        window_frequency=np.asarray([windows[j]*d["f1_Hz"]/max(1,bounds[j+1]-bounds[j]) for j in range(3)])
        rows.append({"parameter":key,"factor":factor,"p":p.tolist(),
          "initial_state_source":"terminal state of archived nominal 240k run; RF phases continued at absolute time then translated to local t=0",
          "initial_RF_phases_rad":phase.tolist(),"horizon_periods":int(completed),"steps_per_RF1_period":64,
          "skip_periods":1000,"status":int(status),"turns":np.asarray(turns).tolist(),
          "late_window_frequency_Hz":window_frequency.tolist(),"window_period_bounds":bounds,
          "frequency_Hz":frequencies.tolist(),"mean_radius_R":np.asarray(avgr).tolist(),
          "r_min_R":np.asarray(rmin).tolist(),"r_max_R":np.asarray(rmax).tolist(),
          "max_abs_z_R":np.asarray(zmax).tolist(),"min_pair_distance_R":float(dmin),
          "analytical_M2_radii_R":[float(xp[0]),float(xp[2])],
          "analytical_M2_frequency_Hz":[float(xp[1]*d["f1_Hz"]),float(xp[3]*d["f1_Hz"])],
          "radius_error_percent":[float(100*(xp[j]-avgr[i])/avgr[i]) for i,j in ((0,0),(1,2))],
          "frequency_error_percent":[float(100*(xp[j]*d["f1_Hz"]-frequencies[i])/abs(frequencies[i])) for i,j in ((0,1),(1,3))],
          "classification":"finite-horizon held-out CR check" if status==0 and frequencies[0]*frequencies[1]<0 else "UNRESOLVED/FAIL",
          "qualification":"Finite horizon and one continued late nominal state; not an attractor or basin/broad-window proof."})
        np.savez_compressed(BASE/"results"/f"heldout_{key}_{factor:.3f}.npz",states=states,
            step_index=step_idx,work_history=work,initial=y0,phase=phase,p=p,turns=turns,
            window_turns=windows,periods=completed,steps=64,skip=1000)
        print(key,factor,"periods",completed,"status",status,"f",frequencies,"r",avgr,flush=True)
    out={"study":"V17C.21 held-out validation","selection":"four one-factor points selected before full-model runs; none used to estimate a fitted coefficient",
         "source_240k_trajectory_sha256":hashlib.sha256(TRAJECTORY.read_bytes()).hexdigest(),
         "rows":rows}
    (BASE/"results"/"heldout_validation.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    run()
