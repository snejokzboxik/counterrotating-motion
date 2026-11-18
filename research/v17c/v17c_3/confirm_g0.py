"""Long, perturbed, and parameter-continuation checks for the G0 control."""
from __future__ import annotations
import csv, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from rf_dc_core import (RESULTS, beta_equilibrium, load_geometry, radial_shape,
                        run_rk4, torque_omega_estimate)


def main():
    meta,modes,coeff=load_geometry("G0_h50")
    geom={**meta,"modes":modes,"coeff":coeff}
    alpha=.025;r0=.55;theta=.371;lam=.0005
    beta=beta_equilibrium(modes,coeff,alpha,r0)
    omega=torque_omega_estimate(modes,coeff,alpha,r0,lam)
    base=run_rk4(geom,[r0],[theta],[omega],[alpha],[beta],[lam],
                 periods=50000,steps_per_period=64,save_per_period=1,phase0=.23)
    p=base["particles"][0]
    np.savez_compressed(RESULTS/"G0_h50_positive_50000RF.npz",
        t=base["t"],state=base["state"],
        initial=np.array([r0,theta,omega,alpha,beta,lam]))
    lz=base["state"][:,0]*base["state"][:,4]-base["state"][:,1]*base["state"][:,3]
    torque_sum=p["torque_rf_impulse"]+p["torque_coulomb_impulse"]+p["torque_drag_impulse"]
    summary={"geometry":"native P4 FEM h50; two harmonics A4 and A-8 only",
        "periods":base["duration_rf_periods"],"escaped":base["escaped"],
        "turns":p["turns"],"mean_omega":p["mean_omega"],
        "late_window_turns":p["window_turns"],"radius_mean":p["radius_mean"],
        "radius_std":p["radius_std"],"radius_min":p["radius_min"],
        "radius_max":p["radius_max"],"initial_angular_momentum":float(lz[0]),
        "final_angular_momentum":float(lz[-1]),
        "torque_impulse_sum":float(torque_sum),
        "angular_momentum_balance_residual":float(lz[-1]-lz[0]-torque_sum),
        "energy":{"K_initial":p["K_start"],"K_final":p["K_end"],
            "W_RF":p["work_rf"],"W_DC":p["work_dc"],"W_Coulomb":p["work_coulomb"],
            "W_drag":p["work_drag"],
            "kinetic_work_residual":float(p["K_end"]-p["K_start"]-p["work_rf"]-
                p["work_dc"]-p["work_coulomb"]-p["work_drag"]),
            "RF_plus_drag":float(p["work_rf"]+p["work_drag"])},
        "parameters":{"alpha":alpha,"beta":beta,"lambda":lam,
            "r0":r0,"theta0":theta,"omega0":omega}}
    (RESULTS/"G0_h50_long_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    # Small state and model perturbations; beta remains common when RF changes.
    trials=[("r0_plus_1pct",r0*1.01,alpha,beta,lam,omega,theta,.0,.0),
            ("alpha_plus_0p5pct",r0,alpha*1.005,beta,lam,omega*1.005,theta,.0,.0),
            ("lambda_plus_5pct",r0,alpha,beta,lam*1.05,omega,theta,.0,.0),
            ("omega_plus_1pct",r0,alpha,beta,lam,omega*1.01,theta,.0,.0),
            ("phase_plus_0p01",r0,alpha,beta,lam,omega,theta,.0,.01),
            ("axial_seed",r0,alpha,beta,lam,omega,theta,.01,.001)]
    pert=[]
    for name,rr,aa,bb,ll,ww,th,z0,vz in trials:
        out=run_rk4(geom,[rr],[th],[ww],[aa],[bb],[ll],z0=[z0],vz=[vz],
                    periods=8000,steps_per_period=64,save_per_period=1,
                    phase0=.23+(0.01 if name=="phase_plus_0p01" else 0.0))
        q=out["particles"][0]
        pert.append({"name":name,"escaped":out["escaped"],"turns":q["turns"],
            "window_turns":q["window_turns"],"radius_mean":q["radius_mean"],
            "radius_min":q["radius_min"],"radius_max":q["radius_max"],
            "z_rms":q["z_rms"],"work_rf":q["work_rf"],"work_drag":q["work_drag"]})
    (RESULTS/"G0_h50_perturbations.json").write_text(json.dumps(pert,indent=2),encoding="utf-8")

    # Step doubling at matched sample times.
    conv=[]; tracks={}
    for steps in (32,64,128):
        out=run_rk4(geom,[r0],[theta],[omega],[alpha],[beta],[lam],
                    periods=2000,steps_per_period=steps,save_per_period=1,phase0=.23)
        conv.append(out);tracks[steps]=out
    rows=[]
    for steps in (32,64):
        delta=np.abs(tracks[steps]["state"][:,:6]-tracks[128]["state"][:,:6])
        rows.append({"steps_per_period":steps,"reference_steps":128,
                     "max_abs_state_difference":float(np.max(delta)),
                     "max_abs_position_difference":float(np.max(delta[:,:3])),
                     "turns":tracks[steps]["particles"][0]["turns"],
                     "reference_turns":tracks[128]["particles"][0]["turns"]})
    (RESULTS/"G0_h50_step_convergence.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")

    # DC continuation: vary B=beta/alpha^2; recover the axisymmetric mean ring
    # prediction, then integrate from that radius at a fixed RF voltage.
    dc_rows=[]
    for target_r in (.35,.50,.70,.85,.93,.97):
        b=beta_equilibrium(modes,coeff,alpha,target_r)
        def f(r):return alpha*alpha*radial_shape(modes,coeff,r)-b
        rpred=brentq(f,.05,.999)
        om=torque_omega_estimate(modes,coeff,alpha,rpred,lam)
        out=run_rk4(geom,[rpred],[theta],[om],[alpha],[b],[lam],
                    periods=4000,steps_per_period=64,save_per_period=1,phase0=.23)
        q=out["particles"][0]
        dc_rows.append({"r_predicted":target_r,"beta":b,"r_integrated_mean":q["radius_mean"],
            "r_min":q["radius_min"],"r_max":q["radius_max"],"turns":q["turns"],
            "mean_omega":q["mean_omega"],"escaped":out["escaped"]})
    (RESULTS/"G0_h50_dc_switch_scan.json").write_text(json.dumps(dc_rows,indent=2),encoding="utf-8")

    # Compact trajectory and balance visualization.
    period=base["t"]/(2*np.pi)
    states=base["state"]
    angle=np.unwrap(np.arctan2(states[:,1],states[:,0]))/(2*np.pi)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    axes[0].plot(period,np.hypot(states[:,0],states[:,1]),lw=.7)
    axes[0].set_ylabel("r/R");axes[0].grid(alpha=.25)
    axes[1].plot(period,angle,lw=.7,label="unwrapped turns")
    axes[1].set_ylabel("turns");axes[1].set_xlabel("RF periods");axes[1].grid(alpha=.25)
    fig.suptitle("G0 h50: long positive RF--DC orbit")
    fig.tight_layout();fig.savefig(RESULTS/"G0_h50_positive_50000RF.png",dpi=170);plt.close(fig)
    print(json.dumps({"long":summary,"perturbations":pert,"convergence":rows,
                      "dc_scan":dc_rows},indent=2),flush=True)

if __name__=="__main__":main()
