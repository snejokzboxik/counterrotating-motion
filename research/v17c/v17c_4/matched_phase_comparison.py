"""Matched ADCFEB vs AFCBED RF--DC pair comparison at a saved initial state."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"v17c_3"))
from rf_dc_core import run_rk4
from v17c4_core import get_coefficients
from full_rf_dc_pair_search import analyze,load_candidate_spectrum

SRC=HERE/"results"/"full_dynamics"/"ADCFEB_focused_1000p_n384_s20261119"/"checkpoint.csv"
PHASES=HERE/"results"/"adaptive_phases"/"global_v2"/"phase_equal_amplitude"/"top_candidates_full_spectrum.json"
OUT=HERE/"results"/"phase_controls"/"ADCFEB_AFCBED_matched_000228"
OUT.mkdir(parents=True,exist_ok=True)

def main():
    with SRC.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    src=next(r for r in rows if int(r["sobol_index"])==228)
    n=lambda k:float(src[k])
    base_modes,base_coeff=get_coefficients("ADCFEBADCFEB","mfs",-1)
    alt_modes,alt_coeff,altrec=load_candidate_spectrum(PHASES,1,"strict_candidates_only")
    schedules=[("ADCFEB",base_modes,base_coeff),("AFCBED",alt_modes,alt_coeff)]
    results=[]
    for label,modes,coeff in schedules:
        geom={"label":label,"N":12,"modes":modes,"coeff":coeff}
        for chi in (0.0,n("chi")):
            idx=len(results)
            c={"idx":idx,"run_id":f"{label}_chi_{chi:.6g}","geometry_label":label,
              "r1":n("r1_0"),"r2":n("r2_0"),"theta1":n("theta1_0"),"theta2":n("theta2_0"),
              "z1":n("z1_0"),"z2":n("z2_0"),"vr1":n("vr1_0"),"vr2":n("vr2_0"),
              "vz1":n("vz1_0"),"vz2":n("vz2_0"),"omega1":n("omega1_0"),"omega2":n("omega2_0"),
              "phase0":n("phase0"),"alpha1":n("alpha1"),"alpha2":n("alpha2"),
              "beta1":n("beta1"),"beta2":n("beta2"),"vdc_ratio":n("vdc_over_u0"),
              "qom_ratio":n("qom2_over_qom1"),"mass2":n("mass2_over_mass1"),
              "q2":n("charge2_over_charge1"),"lam1":n("lambda1"),"lam2":n("lambda2"),
              "chi":chi,"barrier1":n("barrier1"),"barrier2":n("barrier2"),
              "wb1":n("omega_barrier1"),"wb2":n("omega_barrier2"),"modes":modes,"coeff":coeff,
              "spin_mode":src["spin_mode"]}
            out=run_rk4(geom,[c["r1"],c["r2"]],[c["theta1"],c["theta2"]],[c["omega1"],c["omega2"]],
              [c["alpha1"],c["alpha2"]],[c["beta1"],c["beta2"]],[c["lam1"],c["lam2"]],
              [1.,c["mass2"]],[1.,c["q2"]],chi,z0=[c["z1"],c["z2"]],vr=[c["vr1"],c["vr2"]],
              vz=[c["vz1"],c["vz2"]],periods=1200,steps_per_period=192,save_per_period=8,
              phase0=c["phase0"],escape_radius=.995,z_limit=1.5)
            row=analyze(out,c,192);row["optimized_schedule"]=label
            row["source_initial_condition"]="focused_sobol_000228"
            results.append(row)
            np.savez_compressed(OUT/(c["run_id"]+".npz"),t=out["t"],state=out["state"],
                parameters=np.array([c[k] for k in ("r1","r2","alpha1","alpha2","beta1","beta2","lam1","lam2",
                  "mass2","q2","chi","omega1","omega2","phase0","vdc_ratio")]))
            print(label,"chi",chi,"escaped",row["escaped"],"r",row["r1_mean"],row["r2_mean"],
                "turns_after_relax",row["turns1_after_relax"],row["turns2_after_relax"],flush=True)
    with (OUT/"matched_phase_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0].keys()));w.writeheader();w.writerows(results)
    summary={"purpose":"Same initial state and particle/field amplitudes; compare ADCFEB and conjugate-phase AFCBED.",
      "source":"ADCFEB focused Sobol 000228; exact initial state copied from saved checkpoint row.",
      "phase_candidate_rank":1,"candidate_voltage_phasors":altrec.get("voltage_phasors"),
      "periods":1200,"steps_per_period":192,"saved_full_trajectories":True,"rows":results}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")

if __name__=="__main__":main()
