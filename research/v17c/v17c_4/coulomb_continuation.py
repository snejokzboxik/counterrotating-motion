"""Matched-initial-condition Coulomb continuation for an in-aperture ADCFEB pair."""
from __future__ import annotations
import csv, json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"v17c_3"))
from rf_dc_core import run_rk4
from v17c4_core import get_coefficients
from full_rf_dc_pair_search import analyze

HERE=Path(__file__).resolve().parent
BASE=HERE/"results"/"full_dynamics"/"ADCFEB_focused_1000p_n384_s20261119"/"checkpoint.csv"
OUT=HERE/"results"/"coulomb_continuation"/"ADCFEB_000228"
OUT.mkdir(parents=True,exist_ok=True)

def main():
    with BASE.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    src=next(r for r in rows if int(r["sobol_index"])==228)
    num=lambda key:float(src[key])
    modes,coeff=get_coefficients("ADCFEBADCFEB","mfs",-1)
    geom={"label":"ADCFEBADCFEB","N":12,"modes":modes,"coeff":coeff}
    chi_values=[0.0,1e-5,3e-5,1e-4,2.5e-4,4.339213146e-4,7.5e-4,1e-3,2e-3,5e-3,1e-2,2e-2]
    result=[]
    for j,chi in enumerate(chi_values):
        c={"idx":j,"run_id":f"chi_{chi:.4g}","geometry_label":"ADCFEBADCFEB",
           "r1":num("r1_0"),"r2":num("r2_0"),"theta1":num("theta1_0"),"theta2":num("theta2_0"),
           "z1":num("z1_0"),"z2":num("z2_0"),"vr1":num("vr1_0"),"vr2":num("vr2_0"),
           "vz1":num("vz1_0"),"vz2":num("vz2_0"),"omega1":num("omega1_0"),"omega2":num("omega2_0"),
           "phase0":num("phase0"),"alpha1":num("alpha1"),"alpha2":num("alpha2"),
           "spin_mode":src["spin_mode"],
           "beta1":num("beta1"),"beta2":num("beta2"),"vdc_ratio":num("vdc_over_u0"),
           "qom_ratio":num("qom2_over_qom1"),"mass2":num("mass2_over_mass1"),
           "q2":num("charge2_over_charge1"),"lam1":num("lambda1"),"lam2":num("lambda2"),
           "chi":chi,"barrier1":num("barrier1"),"barrier2":num("barrier2"),
           "wb1":num("omega_barrier1"),"wb2":num("omega_barrier2"),"modes":modes,"coeff":coeff}
        out=run_rk4(geom,[c["r1"],c["r2"]],[c["theta1"],c["theta2"]],[c["omega1"],c["omega2"]],
            [c["alpha1"],c["alpha2"]],[c["beta1"],c["beta2"]],[c["lam1"],c["lam2"]],
            [1.,c["mass2"]],[1.,c["q2"]],chi,z0=[c["z1"],c["z2"]],vr=[c["vr1"],c["vr2"]],
            vz=[c["vz1"],c["vz2"]],periods=1200,steps_per_period=192,save_per_period=8,
            phase0=c["phase0"],escape_radius=.995,z_limit=1.5)
        row=analyze(out,c,192);row["source_initial_condition"]="focused_sobol_000228"
        row["original_chi"]=num("chi")
        result.append(row)
        np.savez_compressed(OUT/(f"chi_{chi:.4g}.npz"),t=out["t"],state=out["state"],
                            chi=chi,initial_row=np.array([num(k) for k in ("r1_0","r2_0","alpha1","alpha2",
                                "beta1","beta2","lambda1","lambda2","mass2_over_mass1",
                                "charge2_over_charge1","phase0")]))
        tmp=OUT/"checkpoint.csv.tmp"
        with tmp.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(result[0].keys()));w.writeheader();w.writerows(result)
        tmp.replace(OUT/"checkpoint.csv")
        summary={"source_initial_condition":"ADCFEB focused Sobol 000228",
            "fixed_parameters":{k:num(k) for k in ("r1_0","r2_0","theta1_0","theta2_0","z1_0","z2_0",
                "omega1_0","omega2_0","alpha1","alpha2","beta1","beta2","lambda1","lambda2",
                "mass2_over_mass1","charge2_over_charge1","phase0")},
            "chi_values":chi_values,"completed":len(result),"rows":result}
        tmp=OUT/"summary.json.tmp";tmp.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8");tmp.replace(OUT/"summary.json")
        print(f"{j+1}/{len(chi_values)} chi={chi:g} escaped={row['escaped']} "
              f"r=({row['r1_mean']:.3f},{row['r2_mean']:.3f}) "
              f"turns_after_relax=({row['turns1_after_relax']:.4g},{row['turns2_after_relax']:.4g}) "
              f"coulomb/RF={row['coulomb_force_rf_ratio']:.3g}",flush=True)

if __name__=="__main__":main()
