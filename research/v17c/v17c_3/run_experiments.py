"""Reproducible RF--DC single- and two-particle parameter scans."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys

import numpy as np

from rf_dc_core import (RESULTS, beta_equilibrium, complex_dc_check,
                        load_geometry, pseudopotential, radial_shape,
                        run_rk4, torque_omega_estimate, torque_sign_roots)


def geom_bundle(name):
    meta, modes, coeff = load_geometry(name)
    return {**meta, "modes": modes, "coeff": coeff}


def selected_radii(name, root):
    # Search positions are kept inside the free aperture and away from the
    # predicted torque boundary by at least 0.04R where possible.
    table = {
        "G0_h50": (.55, .94),
        "G1": (.40, .75),          # explicit V17C.3 analytic starting point
        "G2": (.38, .67),
        "G3": (.73, .84),
        "G5": (.81, .91),
    }
    if name in table:
        return table[name]
    return (.62, .78)


def common_dc_candidate(geom, r1, r2, alpha_ref=.025, lam_ref=.0005):
    modes, coeff = geom["modes"], geom["coeff"]
    s1, s2 = radial_shape(modes, coeff, r1), radial_shape(modes, coeff, r2)
    mu2 = s2/s1  # equal positive charges; common U_RF and V_DC
    beta_ref = alpha_ref**2*s1
    alpha = np.array([alpha_ref, alpha_ref/mu2])
    beta = np.array([beta_ref, beta_ref/mu2])
    mass = np.array([1.0, mu2])
    charge = np.ones(2)
    # Equal hydrodynamic drag coefficients imply lambda_i proportional to 1/m_i.
    lam = np.array([lam_ref, lam_ref/mu2])
    omega = np.array([
        torque_omega_estimate(modes, coeff, alpha[0], r1, lam[0]),
        torque_omega_estimate(modes, coeff, alpha[1], r2, lam[1]),
    ])
    return {"radii": [r1, r2], "alpha": alpha.tolist(), "beta": beta.tolist(),
            "mass_ratio": mu2, "mass": mass.tolist(), "charge": charge.tolist(),
            "lambda": lam.tolist(), "omega_estimate": omega.tolist(),
            "beta_ref": beta_ref, "alpha_ref": alpha_ref,
            "full_spectrum_S1": s1, "full_spectrum_S2": s2}


def static_maps():
    names = ["G0_h50", "G0_h35", "G1", "G2", "G3", "G3thin", "G5"]
    rows=[]
    roots_doc={}
    for name in names:
        geom=geom_bundle(name)
        roots=torque_sign_roots(geom["modes"],geom["coeff"])
        roots_doc[name]=roots
        lead=int(geom["m"])
        for r in np.linspace(.08,.96,441):
            s=radial_shape(geom["modes"],geom["coeff"],r)
            om=torque_omega_estimate(geom["modes"],geom["coeff"],.025,r)
            beta=.025**2*s
            # Axisymmetric mean of RF pseudopotential; angular ripple is sampled
            # separately at each radius from the exact complex spectrum.
            vv=np.array([pseudopotential(.025,beta,r*np.cos(th),r*np.sin(th),
                                         geom["modes"],geom["coeff"])
                         for th in np.linspace(0,2*np.pi,96,endpoint=False)])
            rfmean=float(np.mean(vv)+.5*beta*r*r)
            rows.append({"geometry":name,"r_over_R":r,"S_full":s,
                         "beta_eq_alpha0p025":beta,"omega_avg_estimate":om,
                         "pseudo_angular_ripple_peak_to_peak":float(np.ptp(vv)),
                         "pseudo_rf_mean":rfmean,
                         "ripple_over_mean":float(np.ptp(vv)/(abs(rfmean)+1e-30))})
    with (RESULTS/"radial_torque_dc_maps.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (RESULTS/"torque_sign_roots.json").write_text(json.dumps(roots_doc,indent=2),encoding="utf-8")
    return roots_doc


def run_single_particle_scan(roots, periods=1400):
    rows=[]
    for name in ("G0_h50","G1","G2","G3","G5"):
        geom=geom_bundle(name)
        root=roots[name][0] if roots[name] else None
        r1,r2=selected_radii(name,root)
        cand=common_dc_candidate(geom,r1,r2)
        for which,idx in (("inner",0),("outer",1)):
            r=cand["radii"][idx]; alpha=cand["alpha"][idx]; beta=cand["beta"][idx]
            lam=cand["lambda"][idx]
            w0=cand["omega_estimate"][idx]
            for spin_label,spin in (("rest",0.0),("predicted",w0),("opposite",-w0)):
                out=run_rk4(geom,[r],[.371],[spin],[alpha],[beta],[lam],
                            periods=periods,steps_per_period=48,save_per_period=1,
                            phase0=.23)
                p=out["particles"][0]
                rows.append({"geometry":name,"branch":which,"r_target":r,
                    "beta":beta,"alpha":alpha,"lambda":lam,
                    "initial_spin_kind":spin_label,"initial_omega":spin,
                    "predicted_omega":w0,"periods":out["duration_rf_periods"],
                    **{k:p[k] for k in ("turns","mean_omega","window_turns",
                      "radius_mean","radius_std","radius_min","radius_max",
                      "z_rms","work_rf","work_dc","work_drag",
                      "torque_rf_impulse","torque_drag_impulse")},
                    "escaped":out["escaped"],"collided":out["collided"]})
                if name in ("G3","G5") and spin_label=="predicted":
                    np.savez_compressed(RESULTS/f"single_{name}_{which}_{periods}RF.npz",
                        t=out["t"],state=out["state"],
                        initial=np.array([r,.371,spin,alpha,beta,lam]))
    with (RESULTS/"single_particle_rf_dc_scan.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader()
        for row in rows: row["window_turns"]=json.dumps(row["window_turns"]);w.writerow(row)
    return rows


def run_pair_coupling_scan(roots, periods=2400):
    rows=[]
    specs={
        "G1": (.40,.75),
        "G3": (.73,.84),
        "G5": (.81,.91),
    }
    coupling=[0.0,1e-5,1e-4,1e-3,1e-2]
    for name,(r1,r2) in specs.items():
        geom=geom_bundle(name)
        cand=common_dc_candidate(geom,r1,r2)
        for chi in coupling:
            # Opposite tangential directions, nonidentical azimuth and z starts.
            out=run_rk4(geom,[r1,r2],[.371,2.11],cand["omega_estimate"],
                        cand["alpha"],cand["beta"],cand["lambda"],
                        cand["mass"],cand["charge"],chi,
                        z0=[-.035,.065],vr=[2e-4,-1e-4],vz=[1e-4,-2e-4],
                        periods=periods,steps_per_period=40,save_per_period=1,
                        phase0=.23)
            p1,p2=out["particles"]
            row={"geometry":name,"chi":chi,"periods":out["duration_rf_periods"],
                 "initial_r1":r1,"initial_r2":r2,"mass_ratio":cand["mass_ratio"],
                 "alpha1":cand["alpha"][0],"alpha2":cand["alpha"][1],
                 "beta1":cand["beta"][0],"beta2":cand["beta"][1],
                 "lambda1":cand["lambda"][0],"lambda2":cand["lambda"][1],
                 "r1_mean":p1["radius_mean"],"r2_mean":p2["radius_mean"],
                 "r1_min":p1["radius_min"],"r2_min":p2["radius_min"],
                 "r1_max":p1["radius_max"],"r2_max":p2["radius_max"],
                 "z1_rms":p1["z_rms"],"z2_rms":p2["z_rms"],
                 "turns1":p1["turns"],"turns2":p2["turns"],
                 "omega1":p1["mean_omega"],"omega2":p2["mean_omega"],
                 "window1":json.dumps(p1["window_turns"]),
                 "window2":json.dumps(p2["window_turns"]),
                 "min_pair_distance":out["minimum_pair_distance"],
                 "mean_pair_distance":out["mean_pair_distance"],
                 "coulomb_energy_end":out["coulomb_energy_end_unit_chi"],
                 "coulomb_work_total":p1["work_coulomb"]+p2["work_coulomb"],
                 "coulomb_torque_total":p1["torque_coulomb_impulse"]+p2["torque_coulomb_impulse"],
                 "rf_work_total":p1["work_rf"]+p2["work_rf"],
                 "drag_work_total":p1["work_drag"]+p2["work_drag"],
                 "escaped":out["escaped"],"collided":out["collided"]}
            rows.append(row)
            if chi in (0.0,1e-3):
                np.savez_compressed(RESULTS/f"pair_{name}_chi{chi:g}_{periods}RF.npz",
                                    t=out["t"],state=out["state"],
                                    params=np.array([chi,cand["mass_ratio"],*cand["alpha"],*cand["beta"]]))
            print("pair",name,"chi",chi,"turns",p1["turns"],p2["turns"],
                  "r",p1["radius_mean"],p2["radius_mean"],"esc",out["escaped"],flush=True)
    with (RESULTS/"pair_coulomb_coupling_scan.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    return rows


def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    audit={"DC_laplace_and_force":complex_dc_check(),
           "model":"full signed transverse RF harmonics + 3D harmonic DC + 3D Coulomb + linear drag",
           "scaling":{"tau":"Omega*t","u":"r/R","alpha":"kappa*|A_m|",
                      "beta":"q*V_DC/(m*R^2*Omega^2)",
                      "chi":"q_ref^2/(4*pi*eps0*m_ref*Omega^2*R^3)"}}
    roots=static_maps()
    audit["torque_sign_roots"]=roots
    (RESULTS/"model_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    singles=run_single_particle_scan(roots)
    pairs=run_pair_coupling_scan(roots)
    (RESULTS/"scan_summary.json").write_text(json.dumps({"single_rows":len(singles),
          "pair_rows":len(pairs),"single_positive":sum(r["mean_omega"]>0 for r in singles),
          "single_negative":sum(r["mean_omega"]<0 for r in singles),
          "pair_opposite":sum(r["omega1"]*r["omega2"]<0 for r in pairs)},indent=2),encoding="utf-8")
    print("completed",len(singles),"single and",len(pairs),"pair runs",flush=True)

if __name__=="__main__":main()
