"""Pointwise numerical check of explicit RF power versus torque identity."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def main():
    rng=np.random.default_rng(1415)
    errors=[]
    for _ in range(1000):
        k=int(rng.choice([-12,-8,-4,-3,-2,-1,1,2,3,4,8,12]))
        nu=float(rng.uniform(0.1,8.0)); rho=float(rng.uniform(0.05,0.99))
        theta=float(rng.uniform(-np.pi,np.pi)); tau=float(rng.uniform(-10,10))
        q=float(rng.uniform(-4,4)); n=abs(k)
        a=rng.normal()+1j*rng.normal(); adot=rng.normal()+1j*rng.normal()
        phase=np.exp(1j*(k*theta-nu*tau))
        g=rho**n*phase
        # phi = Re[a(tau) rho^n exp(i(k theta - nu tau))]
        dtheta=np.real(a*(1j*k)*g)
        dtau=np.real((adot-1j*nu*a)*g)
        moment=-q*dtheta
        lhs=q*dtau
        rhs=nu/k*moment+q*np.real(adot*g)
        errors.append(abs(lhs-rhs)/max(1.0,abs(lhs),abs(rhs)))
    result={"seed":1415,"random_point_cases":len(errors),
            "max_scaled_absolute_residual":float(max(errors)),
            "status":"passed" if max(errors)<2e-15 else "failed",
            "scope":"Exact pointwise identity with complex amplitude derivative; no mechanical-orbit power inference."}
    path=Path(__file__).resolve().parents[1]/"results/power_torque_identity_validation.json"
    path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"]!="passed": raise SystemExit(1)


if __name__=="__main__": main()
