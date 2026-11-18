"""RF--DC, q/m--DC and DC--drag maps for the native G0 field."""
from __future__ import annotations
import csv,json
import numpy as np
from scipy.optimize import brentq
from rf_dc_core import (RESULTS,beta_equilibrium,load_geometry,radial_shape,
                        run_rk4,torque_omega_estimate)


def predicted_radius(modes,coeff,alpha,beta):
    B=beta/max(alpha*alpha,1e-30)
    vals=np.array([radial_shape(modes,coeff,r)-B for r in np.linspace(.03,.99,200)])
    grid=np.linspace(.03,.99,200)
    roots=[]
    for i in np.flatnonzero(vals[:-1]*vals[1:]<0):
        roots.append(brentq(lambda r:radial_shape(modes,coeff,r)-B,grid[i],grid[i+1]))
    return roots[-1] if roots else None


def trial(geom,r,alpha,beta,lam,periods=1000):
    if r is None: r=.85
    om=torque_omega_estimate(geom["modes"],geom["coeff"],alpha,r,lam)
    out=run_rk4(geom,[r],[.371],[om],[alpha],[beta],[lam],periods=periods,
                steps_per_period=48,save_per_period=1,phase0=.23)
    p=out["particles"][0]
    return {"r_prediction":r,"r_mean":p["radius_mean"],"r_min":p["radius_min"],
        "r_max":p["radius_max"],"turns":p["turns"],"mean_omega":p["mean_omega"],
        "z_rms":p["z_rms"],"work_rf":p["work_rf"],"work_drag":p["work_drag"],
        "escaped":out["escaped"],"duration_RF_periods":out["duration_rf_periods"]}


def write(name,rows):
    with (RESULTS/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def main():
    meta,modes,coeff=load_geometry("G0_h50");geom={**meta,"modes":modes,"coeff":coeff}
    alpha0=.025;r0=.55;beta0=beta_equilibrium(modes,coeff,alpha0,r0);lam0=.0005
    rows=[]
    # Two-dimensional direct dynamic alpha--beta map.
    for alpha in (.015,.025,.04):
        for B in (2.0,4.5,8.0,12.0):
            beta=alpha*alpha*B;r=predicted_radius(modes,coeff,alpha,beta)
            rows.append({"alpha":alpha,"beta":beta,"B_beta_over_alpha2":B,
                         **trial(geom,r,alpha,beta,lam0)})
    write("G0_alpha_beta_map.csv",rows)

    # Physically linked q/m and V_DC changes at fixed U_RF and Omega.
    qm_rows=[]
    for xi in (.5,.75,1.,1.25,1.5):
        for dc_factor in (.5,1.,1.5):
            alpha=alpha0*xi;beta=beta0*xi*dc_factor
            r=predicted_radius(modes,coeff,alpha,beta)
            qm_rows.append({"q_over_m_scale":xi,"VDC_scale":dc_factor,
                "alpha":alpha,"beta":beta,"B_beta_over_alpha2":beta/alpha**2,
                **trial(geom,r,alpha,beta,lam0)})
    write("G0_qm_vdc_map.csv",qm_rows)

    # V_DC--friction continuation around the 0.55R orbit.
    drag_rows=[]
    for dc_factor in (.75,1.,1.25):
        for lam in (0.0001,0.0005,0.002,0.01):
            beta=beta0*dc_factor;r=predicted_radius(modes,coeff,alpha0,beta)
            drag_rows.append({"VDC_scale":dc_factor,"lambda":lam,
                "alpha":alpha0,"beta":beta,**trial(geom,r,alpha0,beta,lam)})
    write("G0_vdc_friction_map.csv",drag_rows)

    # Conservative limit and drag continuation at fixed orbit, with the same
    # prepared positive angular velocity to distinguish stored energy from drive.
    continuation=[]
    w0=torque_omega_estimate(modes,coeff,alpha0,r0,lam0)
    for lam in (0.,1e-4,5e-4,0.002,0.01):
        w=w0 if lam==0 else torque_omega_estimate(modes,coeff,alpha0,r0,lam)
        out=run_rk4(geom,[r0],[.371],[w],[alpha0],[beta0],[lam],periods=12000,
                    steps_per_period=64,save_per_period=1,phase0=.23)
        p=out["particles"][0]
        continuation.append({"lambda":lam,"periods":out["duration_rf_periods"],
            "turns":p["turns"],"mean_omega":p["mean_omega"],
            "late_windows":p["window_turns"],"r_mean":p["radius_mean"],
            "r_min":p["radius_min"],"r_max":p["radius_max"],
            "work_rf":p["work_rf"],"work_drag":p["work_drag"],
            "escaped":out["escaped"]})
        if lam in (0.,.0005):
            np.savez_compressed(RESULTS/f"G0_h50_lambda{lam:g}_12000RF.npz",
                t=out["t"],state=out["state"],
                initial=np.array([r0,.371,w,alpha0,beta0,lam]))
    (RESULTS/"G0_h50_drag_continuation.json").write_text(
        json.dumps(continuation,indent=2),encoding="utf-8")
    print(json.dumps({"alpha_beta_rows":len(rows),"qm_dc_rows":len(qm_rows),
        "vdc_drag_rows":len(drag_rows),"drag_continuation":continuation},indent=2),flush=True)

if __name__=="__main__":main()
