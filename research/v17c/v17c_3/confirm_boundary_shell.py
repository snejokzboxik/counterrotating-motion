"""Long check of the small-amplitude G0 state inside the predicted shell."""
import json
import numpy as np
from rf_dc_core import RESULTS, beta_equilibrium, load_geometry, run_rk4, torque_omega_estimate

meta,modes,coeff=load_geometry("G0_h50")
geom={**meta,"modes":modes,"coeff":coeff}
r,theta,alpha,lam=.999,.371,.0004,.0005
beta=beta_equilibrium(modes,coeff,alpha,r)
omega=torque_omega_estimate(modes,coeff,alpha,r,lam)
out=run_rk4(geom,[r],[theta],[omega],[alpha],[beta],[lam],periods=100000,
            steps_per_period=64,save_per_period=1,phase0=.23,escape_radius=1.0)
p=out["particles"][0]
angle=np.unwrap(np.arctan2(out["state"][:,1],out["state"][:,0]))/(2*np.pi)
periods=out["t"]/(2*np.pi); windows=[]
for lo,hi in ((0,.1),(.1,.2),(.5,.6),(.8,.9),(.9,1.)):
    i=np.searchsorted(periods,lo*100000);j=np.searchsorted(periods,hi*100000)
    windows.append({"window":[lo,hi],"turns":float(angle[j]-angle[i])})
report={"periods":out["duration_rf_periods"],"escaped":out["escaped"],
    "turns":p["turns"],"mean_omega":p["mean_omega"],"late_windows":windows,
    "radius_mean":p["radius_mean"],"radius_min":p["radius_min"],"radius_max":p["radius_max"],
    "work_rf":p["work_rf"],"work_drag":p["work_drag"],
    "kinetic_balance_residual":float(p["K_end"]-p["K_start"]-p["work_rf"]-
        p["work_dc"]-p["work_coulomb"]-p["work_drag"]),
    "parameters":{"r0":r,"alpha":alpha,"beta":beta,"lambda":lam,"omega0":omega}}
np.savez_compressed(RESULTS/"G0_h50_thin_negative_shell_100000RF.npz",
    t=out["t"],state=out["state"],initial=np.array([r,theta,omega,alpha,beta,lam]))
(RESULTS/"G0_h50_thin_negative_shell_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
