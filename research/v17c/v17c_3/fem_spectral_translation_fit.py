"""Direct Floquet comparison using the independently audited G1/G2 FEM modes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from rf_dc_core import RESULTS,integrate_rk4,load_geometry,pseudopotential,run_rk4
from g1_g2_translation_spectrum import grad,hessian
from g1_g2_floquet_frequency_fit import map_period

HERE=Path(__file__).resolve().parent
OUT=RESULTS/"g1_g2_joint"
FEM_AUDIT=HERE.parent/"v17c_2"/"input"/"independent_fem_audit.json"

def solve(name,record,alpha=.01,lam=1e-5):
    meta,modes_mfs,coeff_mfs=load_geometry(name)
    fem=json.loads(FEM_AUDIT.read_text(encoding="utf-8"))
    rec=next(x for x in fem if x["name"]==record)
    fine=rec["levels"][-1]
    plus=complex(*fine["leading_FEM"])
    minus=complex(*fine["counter_FEM"])
    modes=np.array([2,-4],dtype=np.int64)
    coeff=np.array([1+0j,minus/plus],dtype=np.complex128)
    source=json.loads((OUT/"G1_G2_translation_spectrum_fit.json").read_text(encoding="utf-8"))
    B=float(source["stationary_points"][name]["selected_for_ringdown"]["B_beta_over_alpha2"])
    mfsmin=source["stationary_points"][name]["selected_for_ringdown"]
    seed=np.array([mfsmin["x"],mfsmin["y"]])
    sol=root(lambda xy:grad(xy,B,modes,coeff),seed,tol=1e-11)
    xy=sol.x
    H=hessian(xy,B,modes,coeff)
    pred=alpha*np.sqrt(np.maximum(np.linalg.eigvalsh(H),0.0))
    beta=alpha*alpha*B
    r=float(np.linalg.norm(xy));theta=float(np.arctan2(xy[1],xy[0]))
    geom={**meta,"modes":modes,"coeff":coeff}
    probe=run_rk4(geom,[r],[theta],[0.0],[alpha],[beta],[lam],
        periods=1,steps_per_period=128,save_per_period=1,phase0=0.0,
        add_micromotion=True,escape_radius=5.0)
    guess=probe["state"][0,[0,1,3,4]]
    def residual(v): return map_period(v,modes,coeff,alpha,beta,lam)[0]-v
    orbit_root=root(residual,guess,method="hybr",options={"xtol":1e-11,"maxfev":300})
    end,t,orbit=map_period(orbit_root.x,modes,coeff,alpha,beta,lam,save=True)
    center=np.mean(orbit[:,0:2],axis=0)
    Hc=hessian(center,B,modes,coeff)
    predc=alpha*np.sqrt(np.maximum(np.linalg.eigvalsh(Hc),0.0))
    scales=np.array([1e-6,1e-6,1e-7,1e-7]);M=np.empty((4,4))
    for j,h in enumerate(scales):
        e=np.zeros(4);e[j]=h
        M[:,j]=(map_period(orbit_root.x+e,modes,coeff,alpha,beta,lam)[0]-
                map_period(orbit_root.x-e,modes,coeff,alpha,beta,lam)[0])/(2*h)
    mu=np.linalg.eigvals(M)
    f=np.sort(np.abs(np.angle(mu))/(2*np.pi))
    uniq=[]
    for v in f:
        if not uniq or abs(v-uniq[-1])>2e-5: uniq.append(float(v))
        else: uniq[-1]=.5*(uniq[-1]+float(v))
    full=sorted(uniq)[:2]
    match=[]
    for p in predc:
        q=min(full,key=lambda a:abs(a-p))
        match.append({"pseudo_omega_tau":float(p),"full_rf_floquet_omega_tau":q,
                      "relative_error":float((q-p)/p)})
    return {
      "geometry":name,"FEM_source":record,"mesh_triangles":fine["triangles"],
      "leading_FEM":fine["leading_FEM"],"counter_FEM":fine["counter_FEM"],
      "counter_to_leading_amplitude":float(abs(minus/plus)),
      "counter_to_leading_phase_rad":float(np.angle(minus/plus)),
      "modes_used":[2,-4],"retained_harmonics":2,
      "alpha":alpha,"beta":beta,"B":B,"lambda":lam,
      "pseudopotential_stationary_root_success":bool(sol.success),
      "pseudopotential_minimum_xy":xy.tolist(),"hessian_at_initial_min":H.tolist(),
      "pseudopotential_frequencies_initial":pred.tolist(),
      "rf_periodic_fixed_point_success":bool(orbit_root.success),
      "rf_periodic_center_xy":center.tolist(),
      "poincare_residual_norm":float(np.linalg.norm(end-orbit_root.x)),
      "hessian_at_rf_center":Hc.tolist(),
      "pseudopotential_frequencies_at_rf_center":predc.tolist(),
      "floquet_multipliers":[[float(v.real),float(v.imag)] for v in mu],
      "floquet_frequencies_omega_tau":full,
      "matched_frequency_errors":match,
      "scope":"FEM harmonic coefficients are independent 2D P1 results, truncated here to the leading and counter-rotating modes; not a finite-electrode 3D FEM."
    }

def main():
    rows=[solve("G1","6_ABC_moderate"),solve("G2","6_ABC_thin")]
    (OUT/"G1_G2_fem_translation_frequency_fit.json").write_text(
        json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(rows,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
