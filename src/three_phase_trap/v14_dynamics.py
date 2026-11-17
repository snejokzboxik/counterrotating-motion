"""V14 ideal Q/SO/RO dynamics; dimensionless position and RF time."""
from __future__ import annotations

import math
import numpy as np


def acceleration_and_jacobian(state, tau, alpha, beta, kappa=0., model="Q",
                              delta=0., tau0=0.):
    """Return acceleration and its position Jacobian for one or many states.

    SO is the standing cos(4 theta) cos(tau+tau0-delta) field. RO is
    cos(4 theta-tau-tau0+delta). SO therefore has only one spatial RF
    quadrature, whereas RO has two.
    """
    a = np.asarray(state)
    z = a[..., 0] + 1j*a[..., 1]
    q = alpha + beta*math.cos(tau+tau0)
    if model == "Q":
        w = 2*q*z
        dw = 2*q + np.zeros_like(z)
    elif model == "SO":
        c = math.cos(tau+tau0-delta)
        w = 2*q*z + 4*kappa*z**3*c
        dw = 2*q + 12*kappa*z**2*c
    elif model == "RO":
        e = complex(math.cos(tau+tau0-delta),-math.sin(tau+tau0-delta))
        w = 2*q*z + 4*kappa*z**3*e
        dw = 2*q + 12*kappa*z**2*e
    else:
        raise ValueError(model)
    force = np.stack((-w.real,w.imag),axis=-1)
    jac = np.stack((np.stack((-dw.real,dw.imag),axis=-1),
                    np.stack((dw.imag,dw.real),axis=-1)),axis=-2)
    return force,jac


def rhs(t, state, alpha, beta, kappa=0., model="Q", delta=0., tau0=0.):
    force,_=acceleration_and_jacobian(state,t,alpha,beta,kappa,model,delta,tau0)
    return np.concatenate((np.asarray(state)[...,2:4],force),axis=-1)


def effective_potential(x, y, alpha, beta, kappa=0., model="Q", delta=0.):
    """Leading RF pseudopotential, one RF time period averaged."""
    z=np.asarray(x)+1j*np.asarray(y)
    r2=np.abs(z)**2
    base=beta**2*r2+alpha*(z*z).real
    if model=="Q":return base
    if model=="RO":
        return base+8*kappa**2*r2**3+4*beta*kappa*(z*z*r2*np.exp(1j*delta)).real
    if model=="SO":
        return base+4*kappa**2*r2**3+4*beta*kappa*math.cos(delta)*(z*z*r2).real
    raise ValueError(model)


def octupole_rms_field_factor(model, kappa):
    """RF-time mean |E_oct|² = factor * r^6 in dimensionless units."""
    return {"Q":0.,"SO":8*kappa*kappa,"RO":16*kappa*kappa}[model]
