"""Small deterministic checks of the frozen-position RF averaging identities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from make_figures import P, harmonic_force_hessian


def direct_mean(f: np.ndarray, Df: np.ndarray, nu: float, lam: float, nphase: int = 65536):
    phase = np.linspace(0.0, 2*np.pi, nphase, endpoint=False)
    carrier = np.exp(-1j*phase)
    xihat = f/(-nu**2-1j*lam*nu)
    fast_force = np.real(carrier[:,None]*f[None,:])
    fast_jac = np.real(carrier[:,None,None]*Df[None,:,:])
    xi = np.real(carrier[:,None]*xihat[None,:])
    return np.mean(np.einsum("nij,nj->ni",fast_jac,xi),axis=0)


def analytic_mean(f: np.ndarray, Df: np.ndarray, nu: float, lam: float):
    xihat = f/(-nu**2-1j*lam*nu)
    return 0.5*np.real(Df @ np.conjugate(xihat))


def main():
    rng=np.random.default_rng(1410)
    max_time_average_rel=0.0
    max_leading_rel=0.0
    rows=[]
    # Check the complex averaging identity against explicit phase quadrature.
    for _ in range(12):
        f=rng.normal(size=2)+1j*rng.normal(size=2)
        Df=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
        nu=float(rng.uniform(.7,4.5)); lam=float(rng.uniform(.02,.5))
        a=analytic_mean(f,Df,nu,lam); q=direct_mean(f,Df,nu,lam)
        err=np.linalg.norm(a-q)/max(np.linalg.norm(a),1e-30)
        max_time_average_rel=max(max_time_average_rel,err)
        rows.append({"check":"phase quadrature vs complex mean","nu":nu,"lambda":lam,"relative_error":err})
    # Independently evaluate a single leading spatial mode and compare with the
    # closed-form radial/tangential expression, varying angle and radius.
    for mode,family,amp,nu,sigma,n in [(2,0,P["A"],1.0,1,2),(-4,1,P["B"],P["nu"],-1,4)]:
        for species,alpha,lam in [(1,1.0,P["lambda1"]),(2,P["alpha2"],P["lambda2"])]:
            for rho in np.linspace(.12,.82,17):
                for theta in np.linspace(0,2*np.pi,13,endpoint=False):
                    f,Df=harmonic_force_hessian(rho,theta,alpha,mode,amp,1+0j)
                    avg=analytic_mean(f,Df,nu,lam)
                    er=np.array([np.cos(theta),np.sin(theta)]); et=np.array([-np.sin(theta),np.cos(theta)])
                    fr=avg@er; ft=avg@et
                    K=alpha**2*n**2*(n-1)*amp**2/(nu**2+lam**2)
                    fr_ref=-K*rho**(2*n-3)
                    ft_ref=sigma*K*lam/nu*rho**(2*n-3)
                    err=max(abs(fr-fr_ref),abs(ft-ft_ref))/max(abs(fr_ref),abs(ft_ref),1e-30)
                    max_leading_rel=max(max_leading_rel,err)
    result={"status":"passed" if max(max_time_average_rel,max_leading_rel)<1e-11 else "failed",
            "random_seed":1410,"phase_quadrature_points_per_case":65536,"phase_quadrature_cases":12,
            "leading_mode_cases":2*2*17*13,"max_complex_mean_vs_phase_quadrature_relative_error":max_time_average_rel,
            "max_leading_harmonic_formula_relative_error":max_leading_rel,
            "scope":"Frozen-position algebra only; this does not validate long-time orbit averaging or Floquet stability.",
            "checks":rows}
    out=Path(__file__).resolve().parents[1]/"results/averaging_validation.json"
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"]!="passed": raise SystemExit(1)


if __name__=="__main__": main()
