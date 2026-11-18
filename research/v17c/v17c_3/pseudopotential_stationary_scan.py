"""Stationary-point audit of the complete RF pseudopotential plus DC."""
from __future__ import annotations
import json
import numpy as np
from scipy.optimize import root
from rf_dc_core import RESULTS, beta_equilibrium, load_geometry, pseudopotential


def gradient(xy, B, modes, coeff, h=2e-6):
    x,y=xy;r=np.hypot(x,y)
    if r>.98 or r<.005:
        return 1e3*np.asarray(xy)/(r+1e-30)
    f=lambda a,b:pseudopotential(1.0,B,a,b,modes,coeff)
    return np.array([(f(x+h,y)-f(x-h,y))/(2*h),
                     (f(x,y+h)-f(x,y-h))/(2*h)])


def hessian(xy,B,modes,coeff,h=1e-4):
    x,y=xy
    return np.column_stack([
        (gradient([x+h,y],B,modes,coeff)-gradient([x-h,y],B,modes,coeff))/(2*h),
        (gradient([x,y+h],B,modes,coeff)-gradient([x,y-h],B,modes,coeff))/(2*h)])


def main():
    targets={"G1":.4,"G2":.38,"G3":.73,"G5":.81}
    results={}
    for name,target in targets.items():
        meta,modes,coeff=load_geometry(name)
        alpha=.025;beta=beta_equilibrium(modes,coeff,alpha,target);B=beta/alpha**2
        pts=[]
        for r0 in np.linspace(max(.12,target-.2),min(.94,target+.12),5):
            for theta in np.linspace(0,2*np.pi,48,endpoint=False):
                s=root(lambda v:gradient(v,B,modes,coeff),
                       [r0*np.cos(theta),r0*np.sin(theta)],tol=1e-9)
                x,y=s.x;r=np.hypot(x,y)
                if not s.success or not .03<r<.975:continue
                if any(np.hypot(x-p["x"],y-p["y"])<.015 for p in pts):continue
                H=hessian(s.x,B,modes,coeff);ev=np.linalg.eigvalsh(H)
                pts.append({"x":float(x),"y":float(y),"r":float(r),
                    "theta":float(np.arctan2(y,x)%(2*np.pi)),
                    "pseudo_energy_scaled":float(pseudopotential(1.0,B,x,y,modes,coeff)),
                    "hessian_eigenvalues_scaled":ev.tolist(),
                    "classification":"minimum" if np.all(ev>1e-6) else
                        ("maximum" if np.all(ev<-1e-6) else "saddle")})
        results[name]={"target_radius_mean_model":target,"alpha":alpha,
            "beta":beta,"B_beta_over_alpha2":B,
            "stationary_points_inside_r0975":pts,
            "stable_minima":[p for p in pts if p["classification"]=="minimum"]}
    (RESULTS/"pseudopotential_stationary_points.json").write_text(
        json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps({k:{"target":v["target_radius_mean_model"],
        "minima":[[round(p["r"],5),round(p["theta"],4),
                    np.round(p["hessian_eigenvalues_scaled"],6).tolist()]
                   for p in v["stable_minima"]]}
        for k,v in results.items()},indent=2),flush=True)

if __name__=="__main__":main()
