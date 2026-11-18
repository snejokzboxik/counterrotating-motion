"""Full-spectrum RF pseudopotential maps and local minima for ADCFEB."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from v17c4_core import (HERE, angular_barrier, chirality, get_coefficients,
                        pseudo_gradient, pseudo_hessian)
from rf_dc_core import radial_shape

OUT = HERE / "results" / "pseudopotential"
OUT.mkdir(parents=True, exist_ok=True)


def potential_grid(x, y, beta, modes, coeff, alpha=1.0):
    z = np.asarray(x)+1j*np.asarray(y)
    fx = np.zeros_like(z, dtype=np.complex128)
    fy = np.zeros_like(z, dtype=np.complex128)
    for k, c in zip(modes, coeff):
        n = abs(int(k)); base = z if k > 0 else np.conjugate(z)
        f = -n*c*base**(n-1)
        fx += f
        fy += (-1j*f) if k > 0 else (1j*f)
    return alpha*alpha*(np.abs(fx)**2+np.abs(fy)**2)/4.0 - beta*(np.asarray(x)**2+np.asarray(y)**2)/2.0


def stationary_points(B, modes, coeff):
    found = []
    def grad(xy):
        return pseudo_gradient(1.0, B, xy, modes, coeff)
    for r0 in np.linspace(.12, .80, 15):
        for theta in np.linspace(0, 2*np.pi, 36, endpoint=False):
            seed = [r0*np.cos(theta), r0*np.sin(theta)]
            sol = root(grad, seed, tol=1e-10)
            if not sol.success:
                continue
            x, y = sol.x; r = float(np.hypot(x,y))
            if not .05 < r < .95:
                continue
            if any(np.hypot(x-q["x"],y-q["y"]) < 1e-4 for q in found):
                continue
            H = pseudo_hessian(1.0, B, sol.x, modes, coeff)
            eig = np.linalg.eigvalsh(H)
            found.append({"x":float(x),"y":float(y),"r":r,
                          "theta":float(np.arctan2(y,x)%(2*np.pi)),
                          "potential":float(potential_grid(x,y,B,modes,coeff)),
                          "hessian":H.tolist(),"hessian_eigenvalues":eig.tolist(),
                          "classification":"minimum" if np.all(eig>1e-7) else
                              ("maximum" if np.all(eig<-1e-7) else "saddle")})
    return found


def main():
    modes, coeff = get_coefficients("ADCFEBADCFEB", "mfs")
    s = np.abs(modes); c = {int(k): complex(v) for k,v in zip(modes,coeff)}
    target = [.30, .40, .47, .50, .55, .60, .65]
    values = []
    for r in target:
        B = radial_shape(modes,coeff,r)
        bar = angular_barrier(1.0,r,modes,coeff)
        T = sum(np.sign(k)*abs(k)**2*(abs(k)-1)*abs(a)**2*r**(2*abs(k)-2) for k,a in c.items())
        Ubar = bar["barrier_max_minus_min"]
        # Leading-order balance: steady RF-driven angular speed is alpha^2*T/r^2;
        # the angular barrier crossing speed scales as alpha*sqrt(2*Ubar)/r.
        alpha_critical = float(r*np.sqrt(2*Ubar)/abs(T)) if abs(T)>1e-30 else None
        values.append({"r_over_R":r,"beta_over_alpha2_radial_mean":B,
                       "chirality":chirality(c,r),"torque_shape":float(T),
                       "angular_barrier_per_alpha2":Ubar,
                       "angular_corrugation":bar["corrugation"],
                       "alpha_equal_steady_drift_and_barrier_speed":alpha_critical,
                       "angular_profile":bar})
    minima_by_target = {}
    for r in (.30,.40,.50,.55,.60):
        B = radial_shape(modes,coeff,r)
        points = stationary_points(B,modes,coeff)
        minima_by_target[str(r)] = {"B_beta_over_alpha2":B,
            "count":len(points),"minima":[p for p in points if p["classification"]=="minimum"],
            "saddles":[p for p in points if p["classification"]=="saddle"]}
    result={"geometry":"ADCFEBADCFEB, 12 electrodes, six equal-amplitude phases",
            "model":"full MFS p=64, K=72 spectrum; quadrupole DC pseudo-potential",
            "radial_metrics":values,"stationary_points":minima_by_target}
    (OUT/"ADCFEB_full_spectrum_pseudopotential.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")

    grid=np.linspace(-.78,.78,501); xx,yy=np.meshgrid(grid,grid)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,r in zip(axes,(.30,.55)):
        beta=radial_shape(modes,coeff,r)
        U=potential_grid(xx,yy,beta,modes,coeff)
        mask=np.hypot(xx,yy)>.94
        U=np.ma.array(U,mask=mask)
        lev=np.linspace(float(U.min()),float(np.quantile(U.compressed(),.90)),28)
        im=ax.contourf(xx,yy,U,levels=lev,cmap="viridis")
        ax.contour(xx,yy,U,levels=lev[::3],colors="white",linewidths=.35,alpha=.45)
        ax.add_patch(plt.Circle((0,0),r,fill=False,color="red",linestyle="--",linewidth=1.1))
        ax.set_aspect("equal");ax.set_title(f"β/α² = S({r:.2f}R), χ={chirality(c,r):+.3f}")
        ax.set_xlabel("x/R");ax.set_ylabel("y/R")
    fig.colorbar(im,ax=axes,label="U_eff / α²")
    fig.suptitle("Полный RF-псевдопотенциал ADCFEB с DC-квадруполем")
    fig.savefig(OUT/"ADCFEB_pseudopotential_inner_outer.png",dpi=190)
    plt.close(fig)
    print(json.dumps({"radial_metrics":[{k:v for k,v in x.items() if k!="angular_profile"} for x in values],
                      "minima_counts":{k:v['count'] for k,v in minima_by_target.items()}},
                     indent=2,ensure_ascii=False),flush=True)


if __name__ == "__main__": main()
