"""Reproduce the 12,000-period G0 64/128-step long-horizon comparison."""
import json
import numpy as np
from rf_dc_core import RESULTS, beta_equilibrium, load_geometry, run_rk4, torque_omega_estimate

meta,modes,coeff=load_geometry("G0_h50")
geom={**meta,"modes":modes,"coeff":coeff}
r,theta,alpha,lam=.55,.371,.025,.0005
beta=beta_equilibrium(modes,coeff,alpha,r)
omega=torque_omega_estimate(modes,coeff,alpha,r,lam)
ref=np.load(RESULTS/"G0_h50_positive_12000RF.npz")
out=run_rk4(geom,[r],[theta],[omega],[alpha],[beta],[lam],periods=12000,
            steps_per_period=128,save_per_period=1,phase0=.23)
delta=np.abs(out["state"][:,:6]-ref["state"][:,:6])
p=out["particles"][0]
report={"steps64_vs128_max_state":float(delta.max()),
        "steps64_vs128_max_position":float(delta[:,:3].max()),
        "turns128":p["turns"],
        "turns64":float(np.unwrap(np.arctan2(ref["state"][:,1],ref["state"][:,0]))[-1]/(2*np.pi)
                        -np.unwrap(np.arctan2(ref["state"][:,1],ref["state"][:,0]))[0]/(2*np.pi)),
        "radius_min128":p["radius_min"],"radius_max128":p["radius_max"],
        "kinetic_balance_residual128":float(p["K_end"]-p["K_start"]-p["work_rf"]-
            p["work_dc"]-p["work_coulomb"]-p["work_drag"])}
np.savez_compressed(RESULTS/"G0_h50_positive_12000RF_steps128.npz",
                    t=out["t"],state=out["state"])
(RESULTS/"G0_h50_long_step_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
