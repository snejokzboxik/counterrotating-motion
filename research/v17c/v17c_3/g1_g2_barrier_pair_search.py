"""Barrier-targeted 3D two-particle sweep for six-electrode G1/G2."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from rf_dc_core import RESULTS,_rf_accel,load_geometry,pseudopotential,radial_shape,run_rk4

ROOT=RESULTS/"g1_g2_joint"
SEED=17032027
FIELDS=[
 "run_id","geometry","r1_0","r2_0","theta1_0","theta2_0","z1_0","z2_0",
 "vz1_0","vz2_0","omega1_0","omega2_0","spin_case","rf_alpha1","dc_scale",
 "beta1","q_over_m_ratio","mass2_over_mass1","charge2_over_charge1",
 "lambda1","lambda2","chi","barrier1","barrier2","barrier_ratio1","barrier_ratio2",
 "phase0","periods","escaped","collided","turns1","turns2","windows1","windows2",
 "r1_mean","r2_mean","r1_min","r2_min","r1_max","r2_max","z1_rms","z2_rms",
 "min_distance","mean_distance","mean_coulomb_energy","coulomb_force_rf_ratio",
 "work_rf","work_dc","work_coulomb","work_drag","energy_balance_residual",
 "counter_rotating_late","counter_rotating_10turn_screen","score"]

def logu(x,a,b): return math.exp(math.log(a)+float(x)*math.log(b/a))

def profile(geom,r):
    theta=np.linspace(0,np.pi/3,361)
    U=np.array([pseudopotential(1.,0.,r*np.cos(t),r*np.sin(t),
                                geom["modes"],geom["coeff"]) for t in theta])
    return float(np.ptp(U)),float(theta[int(np.argmin(U))])

def save_csv(path,rows):
    temp=path.with_suffix(".tmp")
    with temp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    temp.replace(path)

def one(name,i,u,geom,modes,coeff,periods,steps):
    r1=(.22+.34*u[0]) if name=="G1" else (.18+.34*u[0])
    r2=(.65+.28*u[1]) if name=="G1" else (.59+.34*u[1])
    alpha1=logu(u[2],.002,.035)
    dcscale=.85+.30*u[3]
    lam1=logu(u[4],1e-5,2e-3)
    drag_ratio=logu(u[5],.3,3.0)
    q2=logu(u[6],.5,2.0)
    chi=0.0 if u[7]<.04 else logu((u[7]-.04)/.96,1e-6,.1)
    S1,S2=radial_shape(modes,coeff,r1),radial_shape(modes,coeff,r2)
    B1,thmin1=profile(geom,r1);B2,thmin2=profile(geom,r2)
    maxg1=max(.01,.9*dcscale*.5*r1*r1*S1/max(B1,1e-30))
    g1=.15+u[8]*(min(1.6,maxg1)-.15)
    alpha2_placeholder=alpha1
    omega1=alpha1/r1*math.sqrt(max(2*g1*B1,0.0))
    beta1=dcscale*alpha1*alpha1*S1-omega1*omega1
    maxg2=.9*.5*r2*r2*S2/max(B2,1e-30)
    g2=.15+u[9]*(min(1.6,maxg2)-.15)
    denom=S2-2*g2*B2/(r2*r2)
    if beta1<=0 or denom<=0 or not np.isfinite(denom):
        return None
    s=beta1/(alpha1*alpha1*denom)*logu(u[10],.7,1.3)
    if not (.003<s<4.0): return None
    alpha2=s*alpha1; beta2=s*beta1
    omega2=-s*alpha1/r2*math.sqrt(max(2*g2*B2,0.0))
    qomfactor=logu(u[10],.7,1.3)
    m2=q2/s
    lam=np.array([lam1,lam1*drag_ratio/m2])
    alpha=np.array([alpha1,alpha2]);beta=np.array([beta1,beta2])
    mass=np.array([1.0,m2]);charge=np.array([1.0,q2])
    spin_case="opposite"
    if u[11]>.82:
        spin_case="same"
        omega2=abs(omega2)
    if u[11]>.95:
        spin_case="rest";omega1=omega2=0.0
    theta1=(thmin1+.025*(u[12]-.5))%(2*np.pi)
    theta2=(thmin2+.025*(u[13]-.5))%(2*np.pi)
    z0=np.array([-.06+.12*u[14],-.06+.12*u[15]])
    vz=np.array([-.002+.004*u[16],-.002+.004*u[17]])
    phase0=2*np.pi*u[18]
    omegas=np.array([omega1,omega2])
    out=run_rk4(geom,[r1,r2],[theta1,theta2],omegas,alpha,beta,lam,
        mass,charge,float(chi),z0=z0,vz=vz,periods=periods,
        steps_per_period=steps,save_per_period=1,phase0=phase0,
        escape_radius=.995,z_limit=1.0)
    p1,p2=out["particles"]
    w1,w2=np.asarray(p1["window_turns"]),np.asarray(p2["window_turns"])
    signs=np.sign(np.sum(w1[-2:])),np.sign(np.sum(w2[-2:]))
    stable=bool(not out["escaped"] and not out["collided"])
    counter=bool(stable and signs[0]*signs[1]<0 and
        np.all(np.sign(w1[-2:])==signs[0]) and np.all(np.sign(w2[-2:])==signs[1]))
    ten_turn=bool(counter and abs(p1["turns"])>=10 and abs(p2["turns"])>=10 and
                  abs(w1[-1])>=1.0 and abs(w2[-1])>=1.0)
    states=out["state"]
    sep=np.linalg.norm(states[:,:3]-states[:,13:16],axis=1)
    mean_uc=float(np.mean(chi*q2/sep))
    fcr=chi*q2/(mass[0]*sep**2)
    frf=[]
    for tt,st in zip(out["t"],states):
        a=_rf_accel(st[0],st[1],tt+phase0,modes,coeff,alpha[0])
        frf.append(math.hypot(a[0],a[1]))
    frc_ratio=float(np.mean(fcr)/max(np.mean(frf),1e-30))
    enres=sum(x["K_end"]-x["K_start"]-x["work_rf"]-x["work_dc"]-
              x["work_coulomb"]-x["work_drag"] for x in (p1,p2))
    late_min=min(abs(w1[-1]),abs(w2[-1]))
    score=(20+late_min if ten_turn else
           10+late_min if counter else
           1+min(out["duration_rf_periods"]/periods,1.0)+
             min(abs(w1[-1]),abs(w2[-1])) if stable else
             min(out["duration_rf_periods"]/periods,1.0))
    row={
      "run_id":f"{name}_{i:05d}","geometry":name,"r1_0":r1,"r2_0":r2,
      "theta1_0":theta1,"theta2_0":theta2,"z1_0":z0[0],"z2_0":z0[1],
      "vz1_0":vz[0],"vz2_0":vz[1],"omega1_0":omega1,"omega2_0":omega2,
      "spin_case":spin_case,"rf_alpha1":alpha1,"dc_scale":dcscale,"beta1":beta1,
      "q_over_m_ratio":s,"mass2_over_mass1":m2,"charge2_over_charge1":q2,
      "lambda1":lam[0],"lambda2":lam[1],"chi":chi,"barrier1":alpha1**2*B1,
      "barrier2":alpha2**2*B2,"barrier_ratio1":g1,"barrier_ratio2":g2,
      "phase0":phase0,"periods":out["duration_rf_periods"],
      "escaped":out["escaped"],"collided":out["collided"],
      "turns1":p1["turns"],"turns2":p2["turns"],
      "windows1":json.dumps(w1.tolist()),"windows2":json.dumps(w2.tolist()),
      "r1_mean":p1["radius_mean"],"r2_mean":p2["radius_mean"],
      "r1_min":p1["radius_min"],"r2_min":p2["radius_min"],
      "r1_max":p1["radius_max"],"r2_max":p2["radius_max"],
      "z1_rms":p1["z_rms"],"z2_rms":p2["z_rms"],
      "min_distance":out["minimum_pair_distance"],
      "mean_distance":out["mean_pair_distance"],"mean_coulomb_energy":mean_uc,
      "coulomb_force_rf_ratio":frc_ratio,
      "work_rf":sum(x["work_rf"] for x in (p1,p2)),
      "work_dc":sum(x["work_dc"] for x in (p1,p2)),
      "work_coulomb":sum(x["work_coulomb"] for x in (p1,p2)),
      "work_drag":sum(x["work_drag"] for x in (p1,p2)),
      "energy_balance_residual":enres,"counter_rotating_late":counter,
      "counter_rotating_10turn_screen":ten_turn,"score":score}
    return row,out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples",type=int,default=600)
    ap.add_argument("--periods",type=int,default=600)
    ap.add_argument("--steps",type=int,default=48)
    ap.add_argument("--seed",type=int,default=SEED)
    args=ap.parse_args()
    outdir=ROOT/f"barrier_targeted_{args.periods}RF_n{args.samples}_seed{args.seed}"
    outdir.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for gj,name in enumerate(("G1","G2")):
        meta,modes,coeff=load_geometry(name)
        geom={**meta,"modes":modes,"coeff":coeff}
        samples=qmc.LatinHypercube(d=19,seed=args.seed+gj).random(args.samples)
        rows=[]
        for i,u in enumerate(samples):
            trial=one(name,i,u,geom,modes,coeff,args.periods,args.steps)
            if trial is None: continue
            row,trajectory=trial;rows.append(row);allrows.append(row)
            if row["counter_rotating_late"]:
                np.savez_compressed(outdir/f"{row['run_id']}_screen.npz",
                    t=trajectory["t"],state=trajectory["state"],
                    parameters=np.array([row[k] for k in
                    ("r1_0","r2_0","omega1_0","omega2_0","rf_alpha1","beta1",
                     "q_over_m_ratio","mass2_over_mass1","charge2_over_charge1",
                     "lambda1","lambda2","chi","phase0")]))
            if (i+1)%25==0 or i+1==args.samples:
                temp=outdir/f"{name}_checkpoint.csv"
                save_csv(temp,rows)
                print(name,i+1,"/",args.samples,"valid",len(rows),
                    "survivors",sum(not r["escaped"] and not r["collided"] for r in rows),
                    "counter",sum(r["counter_rotating_late"] for r in rows),
                    "10turn",sum(r["counter_rotating_10turn_screen"] for r in rows),flush=True)
    allrows.sort(key=lambda r:r["score"],reverse=True)
    save_csv(outdir/"G1_G2_barrier_targeted_all.csv",allrows)
    summary={"seed":args.seed,"samples_per_geometry":args.samples,
        "periods":args.periods,"steps_per_period":args.steps,
        "valid_runs":len(allrows),
        "survivors":sum(not r["escaped"] and not r["collided"] for r in allrows),
        "late_counter_rotating":sum(bool(r["counter_rotating_late"]) for r in allrows),
        "ten_turn_screen_candidates":sum(bool(r["counter_rotating_10turn_screen"]) for r in allrows),
        "top_40":allrows[:40],
        "controls":"chi=0 included; opposite, same-direction and rest starts; angular barriers estimated from full available MFS spectra."}
    (outdir/"G1_G2_barrier_targeted_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("valid_runs","survivors",
        "late_counter_rotating","ten_turn_screen_candidates")},indent=2),flush=True)

if __name__=="__main__":main()
