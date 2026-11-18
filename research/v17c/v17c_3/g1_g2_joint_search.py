"""Adaptive broad search for 3D Coulomb-coupled G1/G2 counter-rotation.

All particles see common RF and DC fields: alpha_i and beta_i scale together
with q_i/m_i. The screening uses the complete supplied transverse spectra.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from rf_dc_core import RESULTS, _rf_accel, load_geometry, pseudopotential, radial_shape, run_rk4

HERE = Path(__file__).resolve().parent
OUT = RESULTS / "g1_g2_joint"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 17032026
FIELDS = [
    "run_id", "geometry", "r1_0", "r2_0", "theta1_0", "theta2_0",
    "z1_0", "z2_0", "vz1_0", "vz2_0", "omega1_0", "omega2_0",
    "spin_class", "rf_alpha1", "dc_scale", "beta1", "q_over_m_ratio",
    "mass2_over_mass1", "charge2_over_charge1", "lambda1", "lambda2",
    "chi", "phase0", "periods", "escaped", "collided", "turns1",
    "turns2", "mean_omega1", "mean_omega2", "late_windows1",
    "late_windows2", "r1_mean", "r2_mean", "r1_min", "r2_min",
    "r1_max", "r2_max", "z1_rms", "z2_rms", "min_distance",
    "mean_distance", "coulomb_energy_end", "coulomb_force_rf_ratio",
    "barrier1", "barrier2", "initial_ke_barrier_ratio1",
    "initial_ke_barrier_ratio2", "work_rf", "work_dc", "work_coulomb",
    "work_drag", "energy_balance_residual", "late_opposite_sign_only",
    "counter_rotating_10turn_screen", "score",
]

def log_uniform(x, lo, hi):
    return math.exp(math.log(lo) + float(x) * math.log(hi / lo))

def write_checkpoint(path: Path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)

def angular_barrier_unit(geom, r, theta, direction):
    """Pseudopotential barrier per alpha^2 to the next sixfold sector maximum."""
    samples = np.linspace(0.0, 2*np.pi/6.0, 241)
    u = [pseudopotential(1.0, 0.0, r*np.cos(theta+direction*s),
                         r*np.sin(theta+direction*s), geom["modes"], geom["coeff"])
         for s in samples]
    return max(0.0, float(max(u)-u[0]))

def spin_category(u):
    if u < .70: return "opposite"
    if u < .90: return "same"
    return "rest"

def sample_one(geometry, i, row, geom, modes, coeff, periods, steps):
    r1 = .22+.35*row[0] if geometry == "G1" else .18+.34*row[0]
    r2 = .65+.29*row[1] if geometry == "G1" else .59+.34*row[1]
    theta1, theta2 = row[2]*np.pi/3.0, row[3]*np.pi/3.0
    alpha1 = log_uniform(row[4], .0015, .045)
    dc_scale = .55+row[5]
    S1, S2 = radial_shape(modes, coeff, r1), radial_shape(modes, coeff, r2)
    beta1 = dc_scale*alpha1**2*S1
    qom_ratio = dc_scale*S1/S2*log_uniform(row[6], .4, 2.5)
    q2 = log_uniform(row[7], .4, 2.5)
    m2 = q2/qom_ratio
    alpha = np.array([alpha1, alpha1*qom_ratio])
    beta = np.array([beta1, beta1*qom_ratio])
    lam1 = log_uniform(row[8], 1e-5, 8e-3)
    drag_ratio = log_uniform(row[9], .25, 4.0)
    lam = np.array([lam1, lam1*drag_ratio/m2])
    charge, mass = np.array([1.0, q2]), np.array([1.0, m2])
    chi = 0.0 if row[10] < .06 else log_uniform((row[10]-.06)/.94, 1e-8, 3e-3)
    w1, w2 = log_uniform(row[11], 1e-5, .035), log_uniform(row[12], 1e-5, .035)
    sc = spin_category(row[13])
    if sc == "opposite": omegas = np.array([w1, -w2])
    elif sc == "same":
        direction = 1.0 if row[14] < .5 else -1.0
        omegas = np.array([direction*w1, direction*w2])
    else: omegas = np.zeros(2)
    z0 = np.array([-.12+.24*row[15], -.12+.24*row[16]])
    vz = np.array([-.004+.008*row[17], -.004+.008*row[18]])
    phase0 = 2*np.pi*row[19]
    out = run_rk4(geom, [r1,r2], [theta1,theta2], omegas, alpha,beta,lam,
                  mass,charge,float(chi),z0=z0,vz=vz,periods=periods,
                  steps_per_period=steps,save_per_period=1,phase0=phase0,
                  escape_radius=.995,z_limit=1.0)
    p1,p2 = out["particles"]
    w1late,w2late = np.asarray(p1["window_turns"]),np.asarray(p2["window_turns"])
    s1,s2 = np.sign(np.sum(w1late[-2:])),np.sign(np.sum(w2late[-2:]))
    late_opposite = bool(not out["escaped"] and not out["collided"] and s1*s2<0
                         and np.all(np.sign(w1late[-2:])==s1)
                         and np.all(np.sign(w2late[-2:])==s2))
    ten_turn_screen = bool(late_opposite
                           and min(abs(p1["turns"]),abs(p2["turns"])) >= 10.0
                           and abs(w1late[-1]) >= 1.0
                           and abs(w2late[-1]) >= 1.0)
    late_min_turns = min(abs(w1late[-1]),abs(w2late[-1]))
    if ten_turn_screen:
        score = 20.0+late_min_turns
    elif late_opposite:
        score = 10.0+late_min_turns
    elif not out["escaped"] and not out["collided"]:
        score = 1.0+late_min_turns
    else:
        score = min(out["duration_rf_periods"]/periods,1.0)
    states=out["state"]
    d=states[:,:3]-states[:,13:16]
    ds=np.linalg.norm(d,axis=1)
    mean_fc=float(np.mean(chi*q2/(mass[0]*ds**2)))
    rf1=[_rf_accel(st[0],st[1],tt+phase0,modes,coeff,alpha[0])
         for tt,st in zip(out["t"],states)]
    mean_rf=float(np.mean([math.hypot(a[0],a[1]) for a in rf1]))
    force_ratio=mean_fc/max(mean_rf,1e-30)
    b1=angular_barrier_unit(geom,r1,theta1,float(np.sign(omegas[0]) or 1.0))*alpha[0]**2
    b2=angular_barrier_unit(geom,r2,theta2,float(np.sign(omegas[1]) or -1.0))*alpha[1]**2
    ke1,ke2=.5*r1*r1*omegas[0]**2,.5*r2*r2*omegas[1]**2
    energy_res=sum(x["K_end"]-x["K_start"]-x["work_rf"]-x["work_dc"]-
                   x["work_coulomb"]-x["work_drag"] for x in (p1,p2))
    result={
      "run_id":f"{geometry}_{i:05d}","geometry":geometry,
      "r1_0":r1,"r2_0":r2,"theta1_0":theta1,"theta2_0":theta2,
      "z1_0":z0[0],"z2_0":z0[1],"vz1_0":vz[0],"vz2_0":vz[1],
      "omega1_0":omegas[0],"omega2_0":omegas[1],"spin_class":sc,
      "rf_alpha1":alpha1,"dc_scale":dc_scale,"beta1":beta1,
      "q_over_m_ratio":qom_ratio,"mass2_over_mass1":m2,
      "charge2_over_charge1":q2,"lambda1":lam[0],"lambda2":lam[1],
      "chi":chi,"phase0":phase0,"periods":out["duration_rf_periods"],
      "escaped":out["escaped"],"collided":out["collided"],
      "turns1":p1["turns"],"turns2":p2["turns"],
      "mean_omega1":p1["mean_omega"],"mean_omega2":p2["mean_omega"],
      "late_windows1":json.dumps(w1late.tolist()),"late_windows2":json.dumps(w2late.tolist()),
      "r1_mean":p1["radius_mean"],"r2_mean":p2["radius_mean"],
      "r1_min":p1["radius_min"],"r2_min":p2["radius_min"],
      "r1_max":p1["radius_max"],"r2_max":p2["radius_max"],
      "z1_rms":p1["z_rms"],"z2_rms":p2["z_rms"],
      "min_distance":out["minimum_pair_distance"],"mean_distance":out["mean_pair_distance"],
      "coulomb_energy_end":out["coulomb_energy_end_unit_chi"],
      "coulomb_force_rf_ratio":force_ratio,"barrier1":b1,"barrier2":b2,
      "initial_ke_barrier_ratio1":ke1/max(b1,1e-30),
      "initial_ke_barrier_ratio2":ke2/max(b2,1e-30),
      "work_rf":sum(x["work_rf"] for x in (p1,p2)),
      "work_dc":sum(x["work_dc"] for x in (p1,p2)),
      "work_coulomb":sum(x["work_coulomb"] for x in (p1,p2)),
      "work_drag":sum(x["work_drag"] for x in (p1,p2)),
      "energy_balance_residual":energy_res,
      "late_opposite_sign_only":late_opposite,
      "counter_rotating_10turn_screen":ten_turn_screen,"score":score}
    return result,out

def run_geometry(geometry,n,periods,steps,seed):
    meta,modes,coeff=load_geometry(geometry)
    geom={**meta,"modes":modes,"coeff":coeff}
    samples=qmc.LatinHypercube(d=20,seed=seed).random(n)
    rows=[]
    csvpath=OUT/f"{geometry}_screening_checkpoint.csv"
    for i,sample in enumerate(samples):
        result,trajectory=sample_one(geometry,i,sample,geom,modes,coeff,periods,steps)
        rows.append(result)
        if (i+1)%50==0 or i+1==n:
            write_checkpoint(csvpath,rows)
            print(geometry,i+1,"/",n,"best_score",round(max(r["score"] for r in rows),6),
                  "late_opposite_sign_only",sum(r["late_opposite_sign_only"] for r in rows),
                  "10turn_screen",sum(r["counter_rotating_10turn_screen"] for r in rows),flush=True)
        if result["counter_rotating_10turn_screen"] and not result["escaped"]:
            np.savez_compressed(OUT/f"{result['run_id']}_screen.npz",
                t=trajectory["t"],state=trajectory["state"],
                parameters=np.array([result[k] for k in
                  ("r1_0","r2_0","omega1_0","omega2_0","rf_alpha1","beta1",
                   "q_over_m_ratio","mass2_over_mass1","charge2_over_charge1",
                   "lambda1","lambda2","chi","phase0")]))
    return rows

def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples",type=int,default=900)
    ap.add_argument("--periods",type=int,default=320)
    ap.add_argument("--steps",type=int,default=32)
    ap.add_argument("--seed",type=int,default=SEED)
    args=ap.parse_args()
    OUT = OUT/f"screen_{args.periods}RF_n{args.samples}_seed{args.seed}"
    OUT.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for j,geometry in enumerate(("G1","G2")):
        all_rows.extend(run_geometry(geometry,args.samples,args.periods,args.steps,args.seed+j))
    all_rows.sort(key=lambda r:r["score"],reverse=True)
    write_checkpoint(OUT/"G1_G2_screening_all.csv",all_rows)
    summary={
      "seed":args.seed,"samples_per_geometry":args.samples,"periods":args.periods,
      "steps_per_period":args.steps,"total_runs":len(all_rows),
      "late_opposite_sign_only":sum(bool(r["late_opposite_sign_only"]) for r in all_rows),
      "counter_rotating_10turn_screen":sum(bool(r["counter_rotating_10turn_screen"]) for r in all_rows),
      "survived_screen_horizon":sum(not bool(r["escaped"]) and not bool(r["collided"]) for r in all_rows),
      "best_score":all_rows[0]["score"] if all_rows else None,"top_30":all_rows[:30],
      "parameter_definition":{
        "alpha_i":"alpha1*(q_i/m_i)/(q1/m1)",
        "beta_i":"beta1*(q_i/m_i)/(q1/m1), common RF and DC fields",
        "chi":"shared point-Coulomb scale; zero controls included",
        "lambda_i":"gamma_i/(m_i*Omega), lambda2 sampled through gamma2/gamma1",
        "escape_radius":.995}}
    (OUT/"G1_G2_screening_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("total_runs","late_opposite_sign_only",
          "counter_rotating_10turn_screen","survived_screen_horizon","best_score")},
          indent=2),flush=True)

if __name__=="__main__": main()
