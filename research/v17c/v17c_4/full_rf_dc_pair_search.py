"""Checkpointed full RF--DC two-particle search for V17C.4 ADCFEB.

The RF force uses the complete complex MFS spectrum exported alongside the
independent P1 FEM solution. DC, Coulomb, and drag accelerations are applied
in all three dimensions. RF and DC coefficients obey one common field.
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v17c_3"))
from rf_dc_core import NORM, _rf_accel, radial_shape, run_rk4  # noqa: E402
from v17c4_core import angular_profile, get_coefficients, torque_shape  # noqa: E402

SEED = 20261118
FIELDS = [
    "run_id","sobol_index","geometry","r1_0","r2_0","r1_mean","r2_mean","r1_std","r2_std",
    "r1_min","r2_min","r1_max","r2_max","theta1_0","theta2_0","z1_0","z2_0","vr1_0","vr2_0",
    "vz1_0","vz2_0","omega1_0","omega2_0","spin_mode","phase0","alpha1","alpha2","beta1","beta2",
    "vdc_over_u0","qom2_over_qom1","mass2_over_mass1","charge2_over_charge1","lambda1","lambda2","chi",
    "barrier1","barrier2","omega_barrier1","omega_barrier2","periods","steps_per_period","escaped",
    "collided","min_distance","mean_distance","mean_coulomb_energy","coulomb_force_rf_ratio",
    "turns1_total","turns2_total","turns1_after_relax","turns2_after_relax","late_windows1","late_windows2",
    "relax_periods","relaxed_horizon","counter_late","counter_10turn_screen","work_rf","work_dc",
    "work_coulomb","work_drag","individual_work_residual_max","total_energy_residual",
    "coulomb_torque_impulse1","coulomb_torque_impulse2","rf_torque_impulse1","rf_torque_impulse2",
    "drag_torque_impulse1","drag_torque_impulse2","delta_Lz1","delta_Lz2",
    "angular_momentum_residual1","angular_momentum_residual2","internal_coulomb_torque_sum","score"
]

def log_uniform(u, lo, hi):
    return math.exp(math.log(lo) + float(u) * math.log(hi / lo))

def save_csv(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows); f.flush()
    tmp.replace(path)

def load_candidate_spectrum(path, result_index=0, candidate_list="strict_candidates_only"):
    doc=json.loads(Path(path).read_text(encoding="utf-8"))
    if "results" in doc:
        rec=doc["results"][int(result_index)]["best"]
    elif candidate_list in doc:
        rec=doc[candidate_list][int(result_index)]
    elif "full_spectrum" in doc:
        rec=doc
    else:
        raise ValueError("spectrum JSON must contain results[].best.full_spectrum or full_spectrum")
    rows=rec["full_spectrum"]
    modes=np.asarray([int(x[0]) for x in rows],dtype=np.int64)
    coeff=np.asarray([complex(float(x[1]),float(x[2])) for x in rows],dtype=np.complex128)
    return modes,coeff,rec

def angular_minimum(r, modes, coeff):
    theta, pot = angular_profile(1.0, r, modes, coeff, samples=2160)
    i = int(np.argmin(pot))
    return float(theta[i]), float(np.ptp(pot))

def draw_case(u, idx, modes, coeff, focused=False, strong_coupling=False, barrier_challenge=False):
    # Target the inner positive-torque annulus and the outer negative annulus,
    # while avoiding the edge region with strong angular corrugation.
    if barrier_challenge:
        r1=.34+.06*u[0]; r2=.57+.08*u[1]
        alpha1=log_uniform(u[2],.5,6.0)
        dc_scale=.90+.20*u[3]
        qom_shape=log_uniform(u[4],.82,1.22)
        lam1=log_uniform(u[5],.002,.02)
        drag_ratio=log_uniform(u[6],.5,2.0)
    else:
        r1 = .20 + .20*u[0]
        r2 = .47 + .18*u[1]
        alpha1 = log_uniform(u[2], .08, 1.4) if focused else log_uniform(u[2], .025, 1.8)
        dc_scale = (.85 + .30*u[3]) if focused else (.70 + .60*u[3])
        qom_shape = log_uniform(u[4], .78, 1.28) if (focused or strong_coupling) else log_uniform(u[4], .55, 1.8)
        lam1 = (log_uniform(u[5], .002, .03) if strong_coupling else
                (log_uniform(u[5], .003, .025) if focused else log_uniform(u[5], .003, .05)))
        drag_ratio = log_uniform(u[6], .5, 2.0) if strong_coupling else log_uniform(u[6], .25, 4.0)
    q2 = log_uniform(u[7], .4, 2.5)
    zero_cut = .04 if barrier_challenge else (.03 if strong_coupling else (.04 if focused else .08))
    if u[8] < zero_cut:
        chi = 0.0
    elif barrier_challenge:
        chi=log_uniform((u[8]-zero_cut)/(1-zero_cut),1e-4,.2)
    elif strong_coupling:
        chi = log_uniform((u[8]-zero_cut)/(1-zero_cut), .005, 1.0)
    else:
        chi = log_uniform((u[8]-zero_cut)/(1-zero_cut), 1e-5, .5)

    S1 = radial_shape(modes, coeff, r1)
    S2 = radial_shape(modes, coeff, r2)
    # Near the pseudo-ring ratio for the requested two radii, with independent
    # variation in q/m and common Vdc/U0.
    qom_ratio = dc_scale*(S1/S2)*qom_shape
    alpha2 = alpha1*qom_ratio
    vdc_ratio = alpha1*S1*dc_scale
    beta1, beta2 = alpha1*vdc_ratio, alpha2*vdc_ratio
    mass2 = q2/qom_ratio
    lam2 = lam1*drag_ratio if (strong_coupling or barrier_challenge) else lam1*drag_ratio/mass2

    th1min, b1unit = angular_minimum(r1, modes, coeff)
    th2min, b2unit = angular_minimum(r2, modes, coeff)
    barrier1, barrier2 = alpha1**2*b1unit, alpha2**2*b2unit
    wb1 = math.sqrt(max(2*barrier1, 0.0))/r1
    wb2 = math.sqrt(max(2*barrier2, 0.0))/r2
    cdict = {int(k): complex(c) for k,c in zip(modes,coeff)}
    t1, t2 = torque_shape(cdict,r1), torque_shape(cdict,r2)
    drift1 = alpha1**2*t1/max(r1*r1,1e-30)/(1+lam1*lam1)
    drift2 = alpha2**2*t2/max(r2*r2,1e-30)/(1+lam2*lam2)

    speed_factor = (.15+2.85*u[9]) if barrier_challenge else (
        (.25 + 4.75*u[9]) if strong_coupling else (
        (.35 + 3.65*u[9]) if focused else (.10 + 5.90*u[9])))
    su = u[10]
    if su < .72:
        spin_mode="opposite"
        omega1=drift1+speed_factor*wb1
        omega2=drift2-speed_factor*wb2
    elif su < .88:
        spin_mode="same"
        omega1=drift1+speed_factor*wb1
        omega2=drift2+speed_factor*wb2
    else:
        spin_mode="rest"; omega1=omega2=0.0

    if (strong_coupling or barrier_challenge) and u[11] < .72:
        theta1=th1min+2*np.pi*(u[12]>.5)
        theta2=th2min+2*np.pi*(u[12]>.5)
    elif (strong_coupling or barrier_challenge) and u[11] < .88:
        theta1=th1min+2*np.pi*(u[12]>.5)
        theta2=theta1+np.pi+.10*(u[13]-.5)
    elif u[11] < .4:
        theta1=th1min+2*np.pi*(u[12]>.5)
        theta2=theta1+np.pi+.10*(u[13]-.5)
    elif u[11] < .8:
        theta1=th1min+2*np.pi*(u[12]>.5)
        theta2=th2min+2*np.pi*(u[13]>.5)
    else:
        theta1=2*np.pi*u[12]; theta2=2*np.pi*u[13]
    theta1%=2*np.pi; theta2%=2*np.pi
    phase0=2*np.pi*u[14]
    zscale1=min(.04,.20/math.sqrt(max(beta1,1e-5)))
    zscale2=min(.04,.20/math.sqrt(max(beta2,1e-5)))
    z1,z2=zscale1*(2*u[15]-1),zscale2*(2*u[16]-1)
    vr1,vr2=.04*alpha1*(2*u[17]-1),.04*alpha2*(2*u[18]-1)
    vz1=.20*math.sqrt(max(2*beta1,1e-8))*(2*u[19]-1)
    vz2=.20*math.sqrt(max(2*beta2,1e-8))*(2*u[20]-1)
    return locals()

def turns_and_windows(t, data, i, lam):
    b=NORM*i; p=data[:,b:b+6]
    theta=np.unwrap(np.arctan2(p[:,1],p[:,0]))
    periods=t/(2*np.pi)
    total=float((theta[-1]-theta[0])/(2*np.pi))
    relax_periods=5.0/(2*np.pi*lam)
    j=min(max(int(np.searchsorted(periods,relax_periods,side="left")),0),len(t)-1)
    after=float((theta[-1]-theta[j])/(2*np.pi))
    start=max(float(periods[j]),.55*float(periods[-1]))
    edges=np.linspace(start,float(periods[-1]),4)
    wins=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        il=max(0,int(np.searchsorted(periods,lo,side="left")))
        ih=min(len(t)-1,max(il+1,int(np.searchsorted(periods,hi,side="right"))-1))
        wins.append(float((theta[ih]-theta[il])/(2*np.pi)))
    return total,after,wins,relax_periods,float(periods[-1])>=relax_periods+80.0

def full_energy(data,mass,beta,charge,chi):
    p1=data[:,0:6]; p2=data[:,NORM:NORM+6]
    K=.5*mass[0]*np.sum(p1[:,3:6]**2,axis=1)+.5*mass[1]*np.sum(p2[:,3:6]**2,axis=1)
    dc=np.zeros(len(data))
    for i,p in enumerate((p1,p2)):
        dc+=mass[i]*beta[i]*(p[:,2]**2-.5*(p[:,0]**2+p[:,1]**2))
    d=np.linalg.norm(p1[:,:3]-p2[:,:3],axis=1)
    return K+dc+chi*charge[0]*charge[1]/np.maximum(d,1e-15)

def analyze(out,case,steps):
    t,data=out["t"],out["state"]
    mass=np.array([1.,case["mass2"]]); charge=np.array([1.,case["q2"]])
    betas=np.array([case["beta1"],case["beta2"]])
    pp=[]; ti=[]
    for i,lam in enumerate((case["lam1"],case["lam2"])):
        b=NORM*i; pos=data[:,b:b+3]; rad=np.hypot(pos[:,0],pos[:,1])
        total,after,wins,relax,relaxed=turns_and_windows(t,data,i,lam)
        pp.append({"mean":float(rad.mean()),"std":float(rad.std()),"min":float(rad.min()),"max":float(rad.max())})
        ti.append((total,after,wins,relax,relaxed))
    sep=np.linalg.norm(data[:,0:3]-data[:,NORM:NORM+3],axis=1)
    uc=case["chi"]*case["q2"]/np.maximum(sep,1e-15)
    work=np.array([sum(mass[i]*data[-1,NORM*i+6+k] for i in range(2)) for k in range(4)])
    kres=[]
    lz=[]
    torque_parts=[]
    for i in range(2):
        b=NORM*i
        k0=.5*mass[i]*np.dot(data[0,b+3:b+6],data[0,b+3:b+6])
        k1=.5*mass[i]*np.dot(data[-1,b+3:b+6],data[-1,b+3:b+6])
        kres.append(float(k1-k0-mass[i]*np.sum(data[-1,b+6:b+10])))
        L0=mass[i]*(data[0,b]*data[0,b+4]-data[0,b+1]*data[0,b+3])
        L1=mass[i]*(data[-1,b]*data[-1,b+4]-data[-1,b+1]*data[-1,b+3])
        tor=np.asarray([mass[i]*data[-1,b+10],mass[i]*data[-1,b+11],mass[i]*data[-1,b+12]])
        lz.append(float(L1-L0));torque_parts.append(tor)
    E=full_energy(data,mass,betas,charge,case["chi"])
    eres=float(E[-1]-E[0]-work[0]-work[3])
    totals=[x[0] for x in ti]; after=[x[1] for x in ti]; wins=[x[2] for x in ti]
    relaxed=all(x[4] for x in ti)
    signs=[np.sign(np.mean(w)) if w else 0 for w in wins]
    no_stall=all(min(abs(v) for v in w)>=.08 for w in wins)
    counter=bool(not out["escaped"] and not out["collided"] and relaxed and signs[0]*signs[1]<0 and
                 all(np.all(np.sign(w)==signs[i]) for i,w in enumerate(wins)) and no_stall)
    in_regions=.15<=pp[0]["mean"]<=.45 and .44<=pp[1]["mean"]<=.68
    ten=bool(counter and in_regions and abs(after[0])>=10 and abs(after[1])>=10)
    ratios=[]
    for i in range(2):
        b=NORM*i; ob=NORM*(1-i)
        dist=np.linalg.norm(data[:,b:b+3]-data[:,ob:ob+3],axis=1)
        ac=case["chi"]*case["q2"]/mass[i]/np.maximum(dist**2,1e-15)
        arf=np.empty(len(t))
        for j,(tt,st) in enumerate(zip(t,data)):
            a=_rf_accel(st[b],st[b+1],tt+case["phase0"],case["modes"],case["coeff"],
                        case["alpha1"] if i==0 else case["alpha2"])
            arf[j]=math.hypot(a[0],a[1])
        ratios.append(float(np.mean(ac/np.maximum(arf,1e-15))))
    signs_op=signs[0]*signs[1]<0
    score=100+min(abs(after[0]),abs(after[1])) if ten else (20+min(abs(after[0]),abs(after[1])) if counter and in_regions else
        10+min(abs(after[0]),abs(after[1])) if counter else (4 if not out["escaped"] and not out["collided"] and signs_op else
        min(float(np.mean(np.abs(totals))),4.0)))
    return {
      "run_id":case["run_id"],"sobol_index":case["idx"],"geometry":case.get("geometry_label","ADCFEBADCFEB"),
      "r1_0":case["r1"],"r2_0":case["r2"],"r1_mean":pp[0]["mean"],"r2_mean":pp[1]["mean"],
      "r1_std":pp[0]["std"],"r2_std":pp[1]["std"],"r1_min":pp[0]["min"],"r2_min":pp[1]["min"],
      "r1_max":pp[0]["max"],"r2_max":pp[1]["max"],"theta1_0":case["theta1"],"theta2_0":case["theta2"],
      "z1_0":case["z1"],"z2_0":case["z2"],"vr1_0":case["vr1"],"vr2_0":case["vr2"],"vz1_0":case["vz1"],"vz2_0":case["vz2"],
      "omega1_0":case["omega1"],"omega2_0":case["omega2"],"spin_mode":case["spin_mode"],"phase0":case["phase0"],
      "alpha1":case["alpha1"],"alpha2":case["alpha2"],"beta1":case["beta1"],"beta2":case["beta2"],
      "vdc_over_u0":case["vdc_ratio"],"qom2_over_qom1":case["qom_ratio"],"mass2_over_mass1":case["mass2"],
      "charge2_over_charge1":case["q2"],"lambda1":case["lam1"],"lambda2":case["lam2"],"chi":case["chi"],
      "barrier1":case["barrier1"],"barrier2":case["barrier2"],"omega_barrier1":case["wb1"],"omega_barrier2":case["wb2"],
      "periods":float(t[-1]/(2*np.pi)),"steps_per_period":steps,"escaped":out["escaped"],"collided":out["collided"],
      "min_distance":float(sep.min()),"mean_distance":float(sep.mean()),"mean_coulomb_energy":float(uc.mean()),
      "coulomb_force_rf_ratio":float(np.mean(ratios)),"turns1_total":totals[0],"turns2_total":totals[1],
      "turns1_after_relax":after[0],"turns2_after_relax":after[1],"late_windows1":json.dumps(wins[0]),
      "late_windows2":json.dumps(wins[1]),"relax_periods":max(x[3] for x in ti),"relaxed_horizon":relaxed,
      "counter_late":counter,"counter_10turn_screen":ten,"work_rf":float(work[0]),"work_dc":float(work[1]),
      "work_coulomb":float(work[2]),"work_drag":float(work[3]),
      "individual_work_residual_max":max(abs(x) for x in kres),"total_energy_residual":eres,
      "coulomb_torque_impulse1":float(mass[0]*data[-1,11]),"coulomb_torque_impulse2":float(mass[1]*data[-1,NORM+11]),
      "rf_torque_impulse1":float(torque_parts[0][0]),"rf_torque_impulse2":float(torque_parts[1][0]),
      "drag_torque_impulse1":float(torque_parts[0][2]),"drag_torque_impulse2":float(torque_parts[1][2]),
      "delta_Lz1":lz[0],"delta_Lz2":lz[1],
      "angular_momentum_residual1":float(lz[0]-np.sum(torque_parts[0])),
      "angular_momentum_residual2":float(lz[1]-np.sum(torque_parts[1])),
      "internal_coulomb_torque_sum":float(torque_parts[0][1]+torque_parts[1][1]),
      "score":float(score)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples",type=int,default=512)
    ap.add_argument("--periods",type=int,default=800)
    ap.add_argument("--steps",type=int,default=48)
    ap.add_argument("--save-per-period",type=int,default=8)
    ap.add_argument("--seed",type=int,default=SEED)
    ap.add_argument("--run-dir",type=Path,default=None)
    ap.add_argument("--focused",action="store_true",
                    help="bias q/m and Vdc/U0 toward the requested .20-.40R and .47-.65R pseudo-rings")
    ap.add_argument("--strong-coupling",action="store_true",
                    help="sample chi=.01..10 and align angular minima more often to test Coulomb-assisted collective motion")
    ap.add_argument("--barrier-challenge",action="store_true",
                    help="sample RF strengths above the weak-micromotion barrier threshold at r1=.34-.40R, r2=.57-.65R")
    ap.add_argument("--spectrum-json",type=Path,default=None,
                    help="optional optimized full complex MFS spectrum JSON")
    ap.add_argument("--spectrum-result",type=int,default=0,
                    help="index in the spectrum JSON results[] list")
    ap.add_argument("--spectrum-candidate-list",type=str,default="strict_candidates_only",
                    help="candidate list key for optimized top_candidates_full_spectrum.json")
    args=ap.parse_args()
    run_dir=args.run_dir or HERE/"results"/"full_dynamics"/f"ADCFEB_sobol_{args.periods}p_n{args.samples}_s{args.seed}"
    run_dir.mkdir(parents=True,exist_ok=True)
    if args.spectrum_json is None:
        modes,coeff=get_coefficients("ADCFEBADCFEB","mfs",-1)
        geometry_label="ADCFEBADCFEB"
        phase_candidate=None
    else:
        modes,coeff,phase_candidate=load_candidate_spectrum(args.spectrum_json,args.spectrum_result,
                                                             args.spectrum_candidate_list)
        geometry_label=f"optimized_phase_candidate_{args.spectrum_candidate_list}_{args.spectrum_result}"
    geom={"label":geometry_label,"N":12,"modes":modes,"coeff":coeff}
    engine=qmc.Sobol(d=21,scramble=True,seed=args.seed)
    draws=engine.random_base2(int(math.ceil(math.log2(max(args.samples,1)))))[:args.samples]
    csv_path=run_dir/"checkpoint.csv"; rows=[]; done=set()
    if csv_path.exists():
        with csv_path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
        done={int(r["sobol_index"]) for r in rows}
    np.savez_compressed(run_dir/"sobol_design.npz",points=draws,seed=args.seed,periods=args.periods,steps_per_period=args.steps)
    runmeta={"geometry":geometry_label,"spectrum_source":str(args.spectrum_json) if args.spectrum_json else "saved_adcfeB_mfs_p64_k72",
        "spectrum_result_index":args.spectrum_result if args.spectrum_json else None,
        "samples":args.samples,"seed":args.seed,"periods":args.periods,"steps_per_period":args.steps,
        "save_per_period":args.save_per_period,"focused":args.focused,"strong_coupling":args.strong_coupling,
        "barrier_challenge":args.barrier_challenge,"modes_count":len(modes),
        "coefficient_max_abs":float(np.max(np.abs(coeff)))}
    (run_dir/"run_manifest.json").write_text(json.dumps(runmeta,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"run_dir":str(run_dir),"samples":args.samples,"already_done":len(done),
        "periods":args.periods,"steps_per_period":args.steps,"modes":len(modes)},ensure_ascii=False),flush=True)
    for i,u in enumerate(draws):
        if i in done: continue
        c=draw_case(u,i,modes,coeff,focused=args.focused or args.strong_coupling or args.barrier_challenge,
                    strong_coupling=args.strong_coupling,barrier_challenge=args.barrier_challenge)
        c.update({"idx":i,"modes":modes,"coeff":coeff,"geometry_label":geometry_label,
                  "run_id":f"RFDC_{i:06d}" if args.spectrum_json is not None else f"ADCFEB_{i:06d}"})
        out=run_rk4(geom,[c["r1"],c["r2"]],[c["theta1"],c["theta2"]],[c["omega1"],c["omega2"]],
            [c["alpha1"],c["alpha2"]],[c["beta1"],c["beta2"]],[c["lam1"],c["lam2"]],
            [1.,c["mass2"]],[1.,c["q2"]],c["chi"],z0=[c["z1"],c["z2"]],
            vr=[c["vr1"],c["vr2"]],vz=[c["vz1"],c["vz2"]],periods=args.periods,
            steps_per_period=args.steps,save_per_period=args.save_per_period,phase0=c["phase0"],
            escape_radius=.995,z_limit=1.5)
        row=analyze(out,c,args.steps); rows.append(row); done.add(i)
        opposite=(not out["escaped"] and not out["collided"] and
                  float(row["turns1_after_relax"])*float(row["turns2_after_relax"])<0)
        if row["counter_late"] or (opposite and min(abs(float(row["turns1_after_relax"])),
                                                      abs(float(row["turns2_after_relax"])))>=1.0):
            np.savez_compressed(run_dir/(row["run_id"]+".npz"),t=out["t"],state=out["state"],
                parameters=np.array([c[k] for k in ("r1","r2","alpha1","alpha2","beta1","beta2","lam1","lam2",
                    "mass2","q2","chi","omega1","omega2","phase0","vdc_ratio")]),
                parameter_names=np.array(["r1","r2","alpha1","alpha2","beta1","beta2","lambda1","lambda2",
                    "mass2","charge2","chi","omega1","omega2","phase0","Vdc_over_U0"]))
        if len(rows)%8==0 or i==args.samples-1:
            rows.sort(key=lambda r:float(r["score"]),reverse=True); save_csv(csv_path,rows)
            summary={"seed":args.seed,"requested_samples":args.samples,"completed":len(done),
                "periods":args.periods,"steps_per_period":args.steps,
                "stable_counter_late":sum(str(r["counter_late"]).lower()=="true" for r in rows),
                "counter_10turn_screen":sum(str(r["counter_10turn_screen"]).lower()=="true" for r in rows),
                "opposite_after_relax":sum(float(r["turns1_after_relax"])*float(r["turns2_after_relax"])<0 and str(r["escaped"]).lower()=="false" for r in rows),
                "top20":rows[:20]}
            tmp=run_dir/"summary.json.tmp"; tmp.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"); tmp.replace(run_dir/"summary.json")
            (run_dir/"checkpoint.json").write_text(json.dumps({"next_index":i+1,"completed_indices":sorted(done),
                "seed":args.seed,"samples":args.samples,"periods":args.periods,"steps":args.steps},indent=2),encoding="utf-8")
            print(f"done={len(done)}/{args.samples} opposite={summary['opposite_after_relax']} counter={summary['stable_counter_late']} "
                  f"ten_turn={summary['counter_10turn_screen']} top_score={float(rows[0]['score']):.4g}",flush=True)
    rows.sort(key=lambda r:float(r["score"]),reverse=True); save_csv(run_dir/"all_runs.csv",rows)
    print(json.dumps({"completed":len(done),"counter_late":sum(str(r["counter_late"]).lower()=="true" for r in rows),
        "counter_10turn":sum(str(r["counter_10turn_screen"]).lower()=="true" for r in rows),"best":rows[:5]},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
