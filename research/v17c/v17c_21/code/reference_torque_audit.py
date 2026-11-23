from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=HERE.parent
ARCHIVE=ROOT/"research"/"v17c"/"v17c_20b"/"results"/"refine12_240k_corrected_drag_trajectory.npz"
SUMMARY=ROOT/"research"/"v17c"/"v17c_20b"/"results"/"refine12_240k_corrected_drag_summary.json"


def main():
    z=np.load(ARCHIVE)
    states=z["states"]; wh=z["work_history"]
    period=z["step_index"]/int(z["steps"])
    mu=1.0/float(z["p"][4]); f1=50.0
    rows=[]
    for lo,hi in ((200000,220000),(220000,240000),(200000,240000)):
        i=np.searchsorted(period,lo); j=np.searchsorted(period,hi)
        dt=2*np.pi*(period[j]-period[i]); dw=wh[j]-wh[i]
        # Work-history torque columns: RF, DC, Coulomb, drag, gravity for each body.
        torques={"particle1":{k:float(dw[idx]/dt) for k,idx in
                 (("RF",10),("DC",11),("Coulomb",12),("drag",13),("gravity",14))},
                 "particle2_own_mass_units":{k:float(dw[idx]/dt/mu) for k,idx in
                 (("RF",15),("DC",16),("Coulomb",17),("drag",18),("gravity",19))}}
        obs=[]
        for q,j0 in enumerate((0,6)):
            rr=np.hypot(states[i:j,j0],states[i:j,j0+1])
            zz=states[i:j,j0+2]
            angle=np.unwrap(np.arctan2(states[i:j,j0+1],states[i:j,j0]))
            hz=(angle[-1]-angle[0])/(2*np.pi*(period[j-1]-period[i]))*f1
            obs.append({"mean_radius_R":float(rr.mean()),"std_radius_R":float(rr.std()),
                        "mean_z_R":float(zz.mean()),"std_z_R":float(zz.std()),"frequency_Hz":float(hz)})
        rows.append({"period_bounds":[lo,hi],"mean_torques":torques,"observables_from_100_period_samples":obs})
    out={"source":"V20B corrected 240k refine12 NPZ, sampled every 100 RF1 periods",
         "source_sha256":hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),"mass_ratio_m2_m1":mu,
         "sampling_caveat":"Torque averages are integrated from RK4 work ledger; orbit/radius/z moments use sparse 100-period samples.",
         "windows":rows}
    (BASE/"results"/"reference_torque_audit.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__=="__main__": main()
