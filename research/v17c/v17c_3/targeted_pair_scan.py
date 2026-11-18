"""Focused G1 two-particle Coulomb tests around the six-electrode proposal."""
from __future__ import annotations
import csv,json
import numpy as np
from rf_dc_core import (RESULTS,beta_equilibrium,load_geometry,radial_shape,
                        run_rk4,torque_omega_estimate)


def total_balance(out,mass,charge,chi):
    p=out["particles"]
    dk=sum(x["K_end"]-x["K_start"] for x in p)
    wrf=sum(x["work_rf"] for x in p);wdc=sum(x["work_dc"] for x in p)
    wc=sum(x["work_coulomb"] for x in p);wdrag=sum(x["work_drag"] for x in p)
    y0=out["state"][0];yf=out["state"][-1]
    d0=np.linalg.norm(y0[:3]-y0[13:16]);df=np.linalg.norm(yf[:3]-yf[13:16])
    uc0=chi*charge[0]*charge[1]/d0;ucf=chi*charge[0]*charge[1]/df
    torque_c=sum(x["torque_coulomb_impulse"] for x in p)
    return {"delta_K":dk,"work_rf":wrf,"work_dc":wdc,"work_coulomb":wc,
            "work_drag":wdrag,"kinetic_balance_residual":dk-wrf-wdc-wc-wdrag,
            "U_coulomb_initial":uc0,"U_coulomb_final":ucf,
            "coulomb_energy_balance_residual":wc+(ucf-uc0),
            "weighted_coulomb_torque_exchange_residual":torque_c}


def main():
    name="G1";meta,modes,coeff=load_geometry(name);geom={**meta,"modes":modes,"coeff":coeff}
    alpha1=.025;beta1=beta_equilibrium(modes,coeff,alpha1,.4);lam1=.0005
    specs=[]
    for init_name,r1,r2,mu,theta1,theta2,z1,z2 in [
        ("V17C3_canonical_mu2p7",.4,.75,2.7,.371,2.11,-.05,.05),
        ("full_spectrum_matched",.4,.75,
            radial_shape(modes,coeff,.75)/radial_shape(modes,coeff,.4),
            .371,2.11,-.05,.05),
        ("local_minimum_control_mu2p7",.478,.61,2.7,np.pi/6,0.0,-.04,.04)]:
        alpha=np.array([alpha1,alpha1/mu]);beta=np.array([beta1,beta1/mu])
        lam=np.array([lam1,lam1/mu]);mass=np.array([1.,mu]);charge=np.ones(2)
        if init_name.startswith("local_minimum"):
            om1=torque_omega_estimate(modes,coeff,alpha[0],r1,lam[0])
            base2=torque_omega_estimate(modes,coeff,alpha[1],r2,lam[1])
            opposite=np.array([om1,-max(abs(base2),.01)])
        else:
            opposite=np.array([torque_omega_estimate(modes,coeff,alpha[0],r1,lam[0]),
                               -abs(torque_omega_estimate(modes,coeff,alpha[1],r2,lam[1]))])
        for speed_case,omegas in (("opposite",opposite),
                                  ("same",np.abs(opposite)),
                                  ("rest",np.zeros(2))):
            for chi in (0.,1e-5,1e-4,1e-3,1e-2,3e-2,1e-1):
                out=run_rk4(geom,[r1,r2],[theta1,theta2],omegas,alpha,beta,lam,mass,charge,chi,
                    z0=[z1,z2],vr=[0.,0.],vz=[0.,0.],periods=1000,
                    steps_per_period=64,save_per_period=1,phase0=.23)
                p1,p2=out["particles"]
                bal=total_balance(out,mass,charge,chi)
                row={"initial_set":init_name,"spin_case":speed_case,"chi":chi,
                    "duration_RF_periods":out["duration_rf_periods"],"escaped":out["escaped"],
                    "collided":out["collided"],"mass_ratio":mu,
                    "r1_mean":p1["radius_mean"],"r2_mean":p2["radius_mean"],
                    "r1_min":p1["radius_min"],"r2_min":p2["radius_min"],
                    "r1_max":p1["radius_max"],"r2_max":p2["radius_max"],
                    "turns1":p1["turns"],"turns2":p2["turns"],
                    "window1":json.dumps(p1["window_turns"]),"window2":json.dumps(p2["window_turns"]),
                    "min_distance":out["minimum_pair_distance"],
                    "mean_distance":out["mean_pair_distance"],**bal}
                specs.append(row)
                if chi in (0.,1e-3,.1) and speed_case in ("opposite","rest"):
                    np.savez_compressed(RESULTS/f"targeted_G1_{init_name}_{speed_case}_chi{chi:g}.npz",
                        t=out["t"],state=out["state"],
                        initial=np.array([r1,r2,theta1,theta2,*omegas,chi,mu,*alpha,*beta]))
                print(init_name,speed_case,chi,"dur",round(out["duration_rf_periods"],2),
                      "turns",round(p1["turns"],3),round(p2["turns"],3),
                      "r",round(p1["radius_mean"],3),round(p2["radius_mean"],3),
                      "escape",out["escaped"],flush=True)
    with (RESULTS/"targeted_G1_pair_scan.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=specs[0].keys());w.writeheader();w.writerows(specs)
    (RESULTS/"targeted_G1_pair_scan.json").write_text(json.dumps(specs,indent=2),encoding="utf-8")

if __name__=="__main__":main()
