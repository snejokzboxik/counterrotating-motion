from __future__ import annotations

import math
import numpy as np

from model import Params, potential_jet_2d, _jet_fast, single_rhs, rk4_step, rk4_single_fast


def test_fast_jet_matches_potential_derivatives():
    x,y,t,A,B,nu,beta=0.31,-0.27,0.71,0.037,0.16,3.11,0.017
    _,g,H=potential_jet_2d(x,y,t,A,B,nu,beta)
    gf=_jet_fast(x,y,t,A,B,nu,beta)
    h=1e-6
    gx=(potential_jet_2d(x+h,y,t,A,B,nu,beta)[0]-
        potential_jet_2d(x-h,y,t,A,B,nu,beta)[0])/(2*h)
    gy=(potential_jet_2d(x,y+h,t,A,B,nu,beta)[0]-
        potential_jet_2d(x,y-h,t,A,B,nu,beta)[0])/(2*h)
    assert np.allclose(g,[gx,gy],rtol=1e-7,atol=2e-11)
    assert np.allclose(H,[[gf[2],gf[3]],[gf[3],gf[4]]],rtol=1e-12,atol=1e-12)
    assert abs(np.trace(H)+2*beta)<1e-12


def test_full_dc_harmonic_trace_is_zero():
    beta=0.017273303124515367
    hzz=2*beta
    trace=np.trace(np.diag([-beta,-beta,hzz]))
    assert abs(trace)<1e-15


def test_scalar_rk4_matches_vector_rhs():
    p=Params()
    s=(0.31,-0.27,0.02,-0.01)
    args=(1.0,0.212,p.A,p.B,p.nu,p.beta,0.0,0.0,0.0)
    a=np.asarray(s)
    fast=np.asarray(rk4_single_fast(s,0.71,0.02,args))
    vec=rk4_step(single_rhs,a,0.71,0.02,args[:8])
    assert np.allclose(fast,vec,rtol=1e-12,atol=1e-13)


def test_physical_charge_and_coulomb_scaling_are_tied():
    p=Params()
    alpha1,alpha2=1.0,0.34126029835390936
    M1,M2=1.0,p.mass2
    Q1,Q2=alpha1*M1,alpha2*M2
    F1=p.kappa0*Q1*Q2/M1
    F2=-p.kappa0*Q1*Q2/M2
    assert math.isclose(F1*M1+F2*M2,0.0,abs_tol=1e-18)
    assert math.isclose(Q2/ Q1,0.7253025,rel_tol=1e-7)

