from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analytical_model as model  # noqa: E402

BASE = HERE.parent


def solve_near(data, x0):
    lower = [0.1, -0.05, 0.1, -0.05, -2, -2]
    upper = [0.99, 0.05, 0.99, 0.05, 2, 2]
    return least_squares(lambda x: model.m12_balance(x, data, list(range(len(data["modes"]))),),
                         x0, bounds=(lower, upper), xtol=2e-12, ftol=2e-12,
                         gtol=2e-12, max_nfev=500).x


def vary(data, key, factor):
    d = copy.deepcopy(data)
    if key in ("A", "B"):
        fam = 0 if key == "A" else 1
        for j, mode in enumerate(d["modes"]):
            if mode.family == fam:
                d["modes"][j] = replace(mode, amplitude=mode.amplitude*factor)
    elif key == "nu":
        for j, mode in enumerate(d["modes"]):
            if mode.family == 1:
                d["modes"][j] = replace(mode, frequency=mode.frequency*factor)
    elif key == "alpha2":
        d["alpha"][1] *= factor
    elif key == "lambda1":
        d["lambda"][0] *= factor
    elif key == "lambda2":
        d["lambda"][1] *= factor
    else:
        d[key] *= factor
    return d


def sensitivity(data, xbase):
    keys = ["A", "B", "nu", "beta", "alpha2", "lambda1", "lambda2", "C"]
    records = {}
    for key in keys:
        h = 0.01
        xp = solve_near(vary(data, key, 1+h), xbase)
        xm = solve_near(vary(data, key, 1-h), xbase)
        deriv = (xp-xm)/(2*h)
        base_scale = np.array([xbase[0], abs(xbase[1]), xbase[2], abs(xbase[3]),
                              max(abs(xbase[4]), 1e-4), max(abs(xbase[5]), 1e-4)])
        records[key] = {"central_difference_fraction": h,
                        "dx_dlnp_approx": deriv.tolist(),
                        "elasticity_r1": float(deriv[0]/base_scale[0]),
                        "elasticity_absomega1": float(deriv[1]/base_scale[1]),
                        "elasticity_r2": float(deriv[2]/base_scale[2]),
                        "elasticity_absomega2": float(deriv[3]/base_scale[3]),
                        "warning": "Formal one-at-a-time mathematical derivative; physical parameters are coupled."}
    return records


def slow_state_rhs(y, data):
    # Reduced secular flow in (r, radial speed, omega, z, axial speed) per particle.
    # RF averages are evaluated on instantaneous circular reference orbits.
    alpha, lam = data["alpha"], data["lambda"]
    beta, C, s, gz = data["beta"], data["C"], data["mass_ratio_m1_m2"], data["gravity_z"]
    states = [y[0:5], y[5:10]]
    (r1,u1,w1,z1,vz1),(r2,u2,w2,z2,vz2)=states
    fc1,fc2=model.coulomb_ring_average(r1,r2,z1,z2,C,s)
    out=[]
    for i,(r,u,w,z,vz,fc) in enumerate(((r1,u1,w1,z1,vz1,fc1),(r2,u2,w2,z2,vz2,fc2))):
        rf,_=model.mfs_rf_means(r,w,alpha[i],lam[i],beta,data["modes"],list(range(len(data["modes"]))))
        fr=alpha[i]*beta*r+fc[0]+rf[0]-lam[i]*u
        ft=fc[1]+rf[1]-lam[i]*r*w
        fz=-2*alpha[i]*beta*z+gz+fc[2]-lam[i]*vz
        out.extend([u,r*w*w+fr,(ft-2*u*w)/r,vz,fz])
    return np.asarray(out)


def stability(data, root):
    x=root[0]
    y=np.array([x[0],0,x[1],x[4],0,x[2],0,x[3],x[5],0.0])
    f0=slow_state_rhs(y,data)
    eps=2e-6
    jac=np.empty((10,10))
    for j in range(10):
        d=np.zeros(10); d[j]=eps*max(1.0,abs(y[j]))
        jac[:,j]=(slow_state_rhs(y+d,data)-slow_state_rhs(y-d,data))/(2*d[j])
    ev=np.linalg.eigvals(jac)
    return {"state_order": ["r1,u1,omega1,z1,vz1,r2,u2,omega2,z2,vz2"],
            "fixed_point_residual":f0.tolist(),"jacobian_per_tau":jac.tolist(),
            "eigenvalues_per_tau":[[float(e.real),float(e.imag)] for e in ev],
            "interpretation":"Conditional local spectrum of a reduced secular flow; RF response is circular-orbit closure and Coulomb is torus averaged, so not a Floquet/Lyapunov spectrum of the full forced system."}


def main():
    data=model.load_inputs()
    seed=model.solve_m12(data,list(range(len(data["modes"]))))
    xbase=np.array([seed[k] for k in ("radii_R", "omega_over_Omega1")]) if False else np.array([
        seed["radii_R"][0],seed["omega_over_Omega1"][0],seed["radii_R"][1],
        seed["omega_over_Omega1"][1],seed["mean_z_R"][0],seed["mean_z_R"][1]])
    out={"model":"M2 all-eight-mode moving-orbit, linear-Stokes torus-averaged reduction",
         "baseline_state":xbase.tolist(),"sensitivity":sensitivity(data,xbase),
         "reduced_slow_stability":stability(data,(xbase,))}
    (BASE/"results"/"sensitivity_and_stability.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"baseline":xbase.tolist(),"sensitivity":out["sensitivity"],
                      "eigenvalues":out["reduced_slow_stability"]["eigenvalues_per_tau"]},indent=2))


if __name__=="__main__":
    main()
