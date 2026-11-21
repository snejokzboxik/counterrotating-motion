"""Check the finite-window cross-frequency averaging formula directly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def analytic_average(jacs, xis, nus, t0, T):
    center=t0+0.5*T
    out=np.zeros(jacs[0].shape[0])
    sinc=lambda x: np.sinc(x/np.pi)
    for a,ja in enumerate(jacs):
        for b,xb in enumerate(xis):
            dw=float(nus[a]-nus[b]); sw=float(nus[a]+nus[b])
            phase_d=np.exp(-1j*dw*center)*sinc(dw*T/2)
            phase_s=np.exp(-1j*sw*center)*sinc(sw*T/2)
            out += 0.5*np.real((ja@np.conjugate(xb))*phase_d +
                                (ja@xb)*phase_s)
    return out


def direct_average(jacs, xis, nus, t0, T, count=131072):
    ts=np.linspace(t0,t0+T,count,endpoint=True)
    force_jac=sum(np.real(j[None,:,:]*np.exp(-1j*nu*ts[:,None,None]))
                   for j,nu in zip(jacs,nus))
    xi=sum(np.real(x[None,:]*np.exp(-1j*nu*ts[:,None]))
           for x,nu in zip(xis,nus))
    product=np.einsum("nij,nj->ni",force_jac,xi)
    return np.trapezoid(product,ts,axis=0)/T


def main():
    rng=np.random.default_rng(1414)
    cases=[]
    for label,nus in (("distinct",[1.0,3.1118029453]),
                      ("equal",[2.4,2.4]),
                      ("near_beat",[1.0,1.02])):
        jacs=[rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)) for _ in nus]
        xis=[rng.normal(size=2)+1j*rng.normal(size=2) for _ in nus]
        t0=0.731; T=80.0
        exact=analytic_average(jacs,xis,nus,t0,T)
        direct=direct_average(jacs,xis,nus,t0,T)
        rel=float(np.linalg.norm(exact-direct)/max(np.linalg.norm(exact),1e-30))
        cases.append({"case":label,"frequencies":nus,"window_start":t0,
                      "window_length":T,"quadrature_samples":131072,
                      "analytic_average":exact.tolist(),"direct_average":direct.tolist(),
                      "relative_error":rel})
    out={"seed":1414,"cases":cases,
         "max_relative_error":max(x["relative_error"] for x in cases),
         "status":"passed" if max(x["relative_error"] for x in cases)<2e-8 else "failed",
         "scope":"Finite-window quadratic averaging identity; not trajectory averaging accuracy."}
    path=Path(__file__).resolve().parents[1]/"results/mixed_frequency_average_validation.json"
    path.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))
    if out["status"]!="passed": raise SystemExit(1)


if __name__=="__main__": main()
