import math
import numpy as np
import pytest

from three_phase_trap.v14_dynamics import (
    acceleration_and_jacobian,effective_potential,octupole_rms_field_factor,rhs,
)


def independent_potential(x,y,t,alpha,beta,kappa,model,delta):
    z=complex(x,y)
    q=(alpha+beta*math.cos(t))*(z*z).real
    if model=='Q':return q
    if model=='SO':return q+kappa*(z**4).real*math.cos(t-delta)
    if model=='RO':return q+kappa*(z**4*complex(math.cos(delta-t),math.sin(delta-t))).real
    raise ValueError(model)


@pytest.mark.parametrize('model',('Q','SO','RO'))
def test_force_and_jacobian_against_finite_differences(model):
    z=np.array([.17,-.24,.07,-.03]);a,b,k=.00625,.075,-.015;delta=.4;t=.7
    f,j=acceleration_and_jacobian(z,t,a,b,k,model,delta)
    h=1e-6
    grad=[]
    for i in (0,1):
        v=z.copy();v[i]+=h;vp=independent_potential(*v[:2],t,a,b,k,model,delta)
        v[i]-=2*h;vm=independent_potential(*v[:2],t,a,b,k,model,delta)
        grad.append((vp-vm)/(2*h))
    np.testing.assert_allclose(f,-np.array(grad),atol=1e-10,rtol=1e-8)
    fd=[]
    for i in (0,1):
        v=z.copy();v[i]+=h;fp=acceleration_and_jacobian(v,t,a,b,k,model,delta)[0]
        v[i]-=2*h;fm=acceleration_and_jacobian(v,t,a,b,k,model,delta)[0]
        fd.append((fp-fm)/(2*h))
    np.testing.assert_allclose(j,np.stack(fd,axis=1),atol=1e-10,rtol=1e-8)


@pytest.mark.parametrize('model',('Q','SO','RO'))
def test_pseudopotential_by_independent_time_quadrature(model):
    x,y=.23,-.19;a,b,k=.00625,.075,-.015;d=.37;h=1e-5
    fields=[]
    for t in np.arange(128)*2*math.pi/128:
        grad=[]
        for i in (0,1):
            v=np.array([x,y]);v[i]+=h
            p=independent_potential(*v,t,0.,b,k,model,d)
            v[i]-=2*h
            m=independent_potential(*v,t,0.,b,k,model,d)
            grad.append((p-m)/(2*h))
        fields.append(sum(v*v for v in grad))
    expected=a*(x*x-y*y)+.5*np.mean(fields)
    assert abs(effective_potential(x,y,a,b,k,model,d)-expected)<3e-10


def test_standing_vs_rotating_and_equal_rms_normalization():
    k=.015
    assert octupole_rms_field_factor('SO',math.sqrt(2)*k)==pytest.approx(
        octupole_rms_field_factor('RO',k))
    z=np.array([.2,.3,0.,0.])
    so=acceleration_and_jacobian(z,math.pi/2,0.,0.,k,'SO')[0]
    ro=acceleration_and_jacobian(z,math.pi/2,0.,0.,k,'RO')[0]
    np.testing.assert_allclose(so,0,atol=1e-15)
    assert np.linalg.norm(ro)>1e-4


def test_continuation_preserves_absolute_rf_phase():
    from scipy.integrate import solve_ivp
    z0=np.array([.01,.2,.001,0.])
    f=lambda t,z:rhs(t,z,.00625,.075,-.015,'RO',delta=.31,tau0=.74)
    settings=dict(method='DOP853',rtol=2e-11,atol=2e-13,max_step=2*math.pi/64)
    one=solve_ivp(f,(0,40*math.pi),z0,**settings)
    a=solve_ivp(f,(0,20*math.pi),z0,**settings)
    b=solve_ivp(f,(20*math.pi,40*math.pi),a.y[:,-1],**settings)
    assert one.success and a.success and b.success
    np.testing.assert_allclose(one.y[:,-1],b.y[:,-1],atol=2e-9,rtol=2e-8)
