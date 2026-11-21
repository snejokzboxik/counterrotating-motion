"""Analytic leading-mode comparison for the requested (p,s) multipole pairs.

The amplitudes A and B are held equal to the local15 leading-potential
coefficients at rho=1. This is not equal electrode-voltage or equal field
normalization across different multipole orders.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from make_figures import P


PAIRS = ((2, 2), (2, 3), (2, 4), (2, 6), (3, 4), (3, 6), (4, 6))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
TABLES = OUT / "figure_data"
CANDIDATES = {
    # Parameter vectors use the V17C.12B order (A,B,nu,beta,m1/m2,C,lambda1,lambda2).
    "candidate52": {"p":[0.03956095746691348,0.15032976498886072,2.8996277751401065,
                         0.015917322789905167,0.11307613545330587,9.281679945656495e-6,
                         0.1868668374144136,0.10359181855828002],"qratio":3.5274259874597194},
    "local15": {"p":[0.03716194907442204,0.1626938199343468,3.1118029452860356,
                      0.017273303124515367,0.10944955727644175,9.463121589050818e-6,
                      0.20975541409005094,0.12617311138613094],"qratio":3.1179673274978996},
    "refine12": {"p":[0.036536559669839035,0.17246401028140085,3.2239347513765098,
                        0.014742040731828608,0.12340655648341282,1.2076805896660412e-5,
                        0.20865112927753002,0.13669668777275765],"qratio":2.852663503959775},
    "refine12_sobol17": {"p":[0.03492413282400757,0.15631667332941065,3.2196773638017477,
                                0.015388559140477825,0.10963763906505944,9.747265456858355e-6,
                                0.24575362874758605,0.17750147818833362],"qratio":3.4490536225028334},
}


def species_roots(p: int, s: int, alpha: float, lam: float,
                  amplitudes: tuple[float,float], nu2: float, beta: float) -> list[dict]:
    """Find circular roots and reduced averaged-Jacobian eigenvalues."""
    # Leading-potential coefficients A and B are held at local15.
    terms = [
        (p, +1, 1.0, amplitudes[0]),
        (s, -1, nu2, amplitudes[1]),
    ]
    # K_n rho^(2n-4) is the radial drift scale after dividing force by rho.
    poly_s = np.zeros(max(p, s) - 1, dtype=float)
    poly_w = np.zeros_like(poly_s)
    for n, sigma, nu, amp in terms:
        kcoef = alpha**2 * n**2 * (n - 1) * amp**2 / (nu**2 + lam**2)
        poly_s[n - 2] += kcoef
        poly_w[n - 2] += sigma * kcoef / nu
    # H(d)=W(d)^2 + alpha*beta - S(d), d=rho^2.
    h = np.polynomial.polynomial.polymul(poly_w, poly_w)
    if len(h) < len(poly_s):
        h = np.pad(h, (0, len(poly_s)-len(h)))
    h[:len(poly_s)] -= poly_s
    h[0] += alpha * beta
    roots_d = np.polynomial.polynomial.polyroots(h)
    out=[]
    for z in roots_d:
        if abs(z.imag) > 2e-8 or not (0.0 < z.real < 1.0):
            continue
        d=float(z.real); rho=float(np.sqrt(d))
        omega=float(np.polynomial.polynomial.polyval(d, poly_w))

        def rhs(y: np.ndarray) -> np.ndarray:
            r, vr, om = y
            if r <= 0.0:
                r=1e-14
            fr=0.0; ft=0.0
            for n, sigma, nu, amp in terms:
                kc=alpha**2*n**2*(n-1)*amp**2/(nu**2+lam**2)
                fr -= kc*r**(2*n-3)
                ft += sigma*kc*lam/nu*r**(2*n-3)
            return np.array((vr,
                r*om**2 + alpha*beta*r + fr - lam*vr,
                ft/r - 2.0*vr*om/r - lam*om))

        y=np.array((rho, 0.0, omega))
        step=2e-6
        eye=np.eye(3)
        jac=np.column_stack([(rhs(y+step*eye[j])-rhs(y-step*eye[j]))/(2*step)
                             for j in range(3)])
        eig=np.linalg.eigvals(jac)
        out.append({
            "rho":rho,"omega_over_Omega1":omega,
            "stable_reduced_averaged_model":bool(np.max(eig.real)<1e-7),
            "eigenvalues_real_imag":[[float(q.real),float(q.imag)] for q in eig],
        })
    return sorted(out,key=lambda a:a["rho"])


def main() -> None:
    rows=[]
    roots_by_case={}
    configs={}
    for name,source in CANDIDATES.items():
        A,B,nu,beta,m1m2,C,lam1,lam2=source["p"]
        configs[name]={"A":A,"B":B,"nu":nu,"beta":beta,"lambda1":lam1,"lambda2":lam2,
                       "alpha1":1.0,"alpha2":m1m2*source["qratio"],"C":C,
                       "source":"V17C.12B PHYSICAL_PARAMETERIZATION_RU.md / model/physical_parameterization.py"}
    for p,s in PAIRS:
      for candidate,cfg in configs.items():
        case=f"{candidate}:{p},{s}"
        roots_by_case[case]={}
        for species,alpha,lam in ((1,1.0,cfg["lambda1"]),
                                  (2,cfg["alpha2"],cfg["lambda2"])):
            roots=species_roots(p,s,alpha,lam,(cfg["A"],cfg["B"]),cfg["nu"],cfg["beta"])
            roots_by_case[case][str(species)]=roots
            for index,root in enumerate(roots):
                rows.append({
                    "candidate":candidate,"p_positive_order":p,"s_negative_order":s,
                    "amplitude_normalization":"same candidate leading-potential coefficients A,B at rho=1",
                    "A":cfg["A"],"B":cfg["B"],"nu":cfg["nu"],"beta":cfg["beta"],
                    "species":species,"alpha_Q_over_mu":alpha,"lambda":lam,
                    "root_index_by_radius":index,"rho":root["rho"],
                    "omega_over_Omega1":root["omega_over_Omega1"],
                    "stable_reduced_averaged_model":root["stable_reduced_averaged_model"],
                    "trajectory_initial_conditions":"not applicable (algebraic averaged roots)",
                    "trajectory_horizon":"not applicable",
                })
    stable_signs={}
    for p,s in PAIRS:
      for candidate in configs:
        case=f"{candidate}:{p},{s}"
        r1=[r for r in roots_by_case[case]["1"] if r["stable_reduced_averaged_model"]]
        r2=[r for r in roots_by_case[case]["2"] if r["stable_reduced_averaged_model"]]
        stable_signs[case]={
            "species1_stable_root_count":len(r1),"species2_stable_root_count":len(r2),
            "stable_opposite_sign_pair_exists":any(a["omega_over_Omega1"]*b["omega_over_Omega1"]<0 for a in r1 for b in r2),
            "species1_stable_omega": [r["omega_over_Omega1"] for r in r1],
            "species2_stable_omega": [r["omega_over_Omega1"] for r in r2],
        }
    result={
        "method":"leading single-mode frozen-position average; distinct RF frequencies so long-time cross terms vanish",
        "parameter_sources":{"V17C.13_local15":"5438b1cdd79c47abab45c6cbaf99c6daec9fbb60",
            "V17C.12B_candidates":"801ceba8b8555443bd81a9591b2e532f7c36a0e6"},
        "candidate_parameters":configs,
        "normalization":"For each candidate, retain its own leading-potential coefficients A and B at rho=1; equal electrode voltage and equal field are different comparisons",
        "root_domain":"0 < rho < 1",
        "stability":"eigenvalues of the reduced 3D polar averaged vector-field Jacobian; not RF Floquet stability",
        "pairs":list([int(a),int(b)] for a,b in PAIRS),
        "stable_root_summary_by_candidate_and_orders":stable_signs,
        "roots":roots_by_case,
        "trajectory_count":0,
    }
    OUT.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    (OUT/"generalized_multipole_sweep.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    with (TABLES/"generalized_multipole_sweep.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(stable_signs,indent=2))


if __name__ == "__main__":
    main()
