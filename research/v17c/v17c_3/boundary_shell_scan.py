"""Check whether the very thin G0 averaged negative-torque shell is usable."""
import json
from rf_dc_core import RESULTS, beta_equilibrium, load_geometry, run_rk4, torque_omega_estimate

meta,modes,coeff=load_geometry("G0_h50")
geom={**meta,"modes":modes,"coeff":coeff}
r=.999;lam=.0005;rows=[]
for alpha in (.025,.01,.003,.001,.0004,.0002):
    beta=beta_equilibrium(modes,coeff,alpha,r)
    omega=torque_omega_estimate(modes,coeff,alpha,r,lam)
    for phase in (0.,.23,1.57,3.14):
        out=run_rk4(geom,[r],[.371],[omega],[alpha],[beta],[lam],periods=3000,
            steps_per_period=128,save_per_period=1,phase0=phase,escape_radius=1.0)
        p=out["particles"][0]
        rows.append({"alpha":alpha,"beta":beta,"omega_est":omega,"phase":phase,
            "duration_RF_periods":out["duration_rf_periods"],"escaped":out["escaped"],
            "turns":p["turns"],"r_min":p["radius_min"],"r_max":p["radius_max"],
            "r_mean":p["radius_mean"]})
(RESULTS/"G0_boundary_negative_shell_scan.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
print(json.dumps(rows,indent=2))
