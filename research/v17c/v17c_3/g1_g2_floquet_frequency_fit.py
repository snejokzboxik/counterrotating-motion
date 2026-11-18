"""Fit full-RF translational Floquet frequencies to pseudopotential Hessians."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from rf_dc_core import RESULTS,integrate_rk4,load_geometry,run_rk4
from g1_g2_translation_spectrum import hessian

HERE=Path(__file__).resolve().parent
OUT=RESULTS/"g1_g2_joint"
OUT.mkdir(parents=True,exist_ok=True)


def map_period(v, modes, coeff, alpha, beta, lam, steps=128, save=False):
    y0=np.zeros(13,dtype=float)
    y0[[0,1,3,4]]=v
    times,data,escaped,collided=integrate_rk4(
        y0,1,steps,1 if save else steps,modes,coeff,
        np.array([alpha]),np.array([beta]),np.array([lam]),
        np.array([1.0]),np.array([1.0]),0.0,0.0,5.0,5.0)
    if escaped or collided:
        return np.full(4,np.nan),times,data
    return data[-1,[0,1,3,4]],times,data


def fit_one(name,alpha=.01,lam=1e-5):
    meta,modes,coeff=load_geometry(name)
    fit=json.loads((OUT/"G1_G2_translation_spectrum_fit.json").read_text(encoding="utf-8"))
    minimum=fit["stationary_points"][name]["selected_for_ringdown"]
    B=float(minimum["B_beta_over_alpha2"])
    beta=alpha*alpha*B
    r=float(minimum["r"]);theta=float(minimum["theta"])
    # The RF phase is fixed at tau=0. Use a micromotion-corrected initial
    # guess, then solve the exact stroboscopic fixed-point equation.
    probe=run_rk4({**meta,"modes":modes,"coeff":coeff},[r],[theta],[0.0],
        [alpha],[beta],[lam],periods=1,steps_per_period=128,
        save_per_period=1,phase0=0.0,add_micromotion=True,escape_radius=5.0)
    guess=probe["state"][0,[0,1,3,4]]
    def residual(v):
        end,_,_=map_period(v,modes,coeff,alpha,beta,lam)
        return end-v
    sol=root(residual,guess,method="hybr",options={"xtol":1e-11,"maxfev":300})
    end,t,orbit=map_period(sol.x,modes,coeff,alpha,beta,lam,save=True)
    mean_xy=np.mean(orbit[:,0:2],axis=0)
    H=hessian(mean_xy,B,modes,coeff)
    pseudo_eigs=np.linalg.eigvalsh(H)
    pseudo_freq=alpha*np.sqrt(np.maximum(pseudo_eigs,0.0))
    # Central-difference the exact one-period map to form its 4x4 monodromy.
    scales=np.array([1e-6,1e-6,1e-7,1e-7])
    M=np.empty((4,4))
    for j,h in enumerate(scales):
        e=np.zeros(4);e[j]=h
        plus=map_period(sol.x+e,modes,coeff,alpha,beta,lam)[0]
        minus=map_period(sol.x-e,modes,coeff,alpha,beta,lam)[0]
        M[:,j]=(plus-minus)/(2*h)
    mu=np.linalg.eigvals(M)
    omega=np.abs(np.angle(mu))/(2*np.pi)
    damping=np.log(np.maximum(np.abs(mu),1e-300))/(2*np.pi)
    # Collapse conjugate pairs into two unique positive frequencies.
    unique=[]
    for f in sorted(float(x) for x in omega):
        if not unique or abs(f-unique[-1])>2e-5:
            unique.append(f)
        else:
            unique[-1]=.5*(unique[-1]+f)
    full=sorted(unique)[:2]
    beta_z=beta
    omega_z=float(np.sqrt(max(2*beta_z-lam*lam/4,0.0)))
    match=[]
    for pf in pseudo_freq:
        nearest=min(full,key=lambda ff:abs(ff-pf)) if full else None
        match.append({"pseudopotential_omega_tau":float(pf),
                      "full_rf_floquet_omega_tau":nearest,
                      "relative_error":None if nearest is None else float((nearest-pf)/pf)})
    result={
      "geometry":name,"alpha":alpha,"beta":beta,"lambda":lam,"B":B,
      "pseudopotential_minimum_xy":minimum["x"],"pseudopotential_minimum_y":minimum["y"],
      "exact_rf_stroboscopic_fixed_point_xyvxvy":sol.x.tolist(),
      "poincare_residual_norm":float(np.linalg.norm(end-sol.x)),
      "root_success":bool(sol.success),"root_message":str(sol.message),
      "guiding_center_xy_period_average":mean_xy.tolist(),
      "guiding_center_radius":float(np.linalg.norm(mean_xy)),
      "pseudopotential_hessian_at_rf_orbit_mean":H.tolist(),
      "pseudopotential_frequencies_omega_tau":pseudo_freq.tolist(),
      "floquet_multipliers_real_imag":[[float(x.real),float(x.imag)] for x in mu],
      "floquet_multipliers_abs":[float(abs(x)) for x in mu],
      "floquet_frequencies_omega_tau":[float(x) for x in omega],
      "floquet_damping_per_tau":[float(x) for x in damping],
      "matched_transverse_frequencies":match,
      "axial_dc_omega_tau":omega_z,
      "axial_damped_exact_multiplier_abs":float(np.exp(-lam*np.pi)),
      "interpretation":"This is a numerical single-particle translation spectrum around an RF-periodic orbit; no experimental frequency data were supplied.",
    }
    np.savez_compressed(OUT/f"{name}_rf_periodic_orbit_monodromy.npz",
        initial=sol.x,orbit_times=t,orbit_states=orbit,monodromy=M,
        multipliers=mu,mean_xy=mean_xy)
    return result


def main():
    data=[fit_one("G1"),fit_one("G2")]
    (OUT/"G1_G2_floquet_frequency_fit.json").write_text(
        json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(data,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
