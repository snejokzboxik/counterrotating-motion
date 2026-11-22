"""Dimensionless two-frequency multipole model for V17C.18B-19.

The RF part is the leading k=+2 and k=-4 harmonic model.  It is not a
finite-length electrode solution.  The dimensional map is the one in
V17C.16; all constants are explicit so mathematical and physical scans can be
kept separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Params:
    A: float = 0.03716194907442204
    B: float = 0.1626938199343468
    nu: float = 3.1118029452860356
    beta: float = 0.017273303124515367
    lam1: float = 0.21217402878354874
    lam2: float = 0.12835219025177635
    mass2: float = 2.1253644314868816
    q2_over_q1: float = 0.7253025
    kappa0: float = 1.7580955e-5
    gravity: float = 0.02201931048345998
    densities_kg_m3: tuple[float, float] = (1050.0, 1050.0)
    gas_density_kg_m3: float = 1.2
    particle_radii_R: tuple[float, float] = (0.007, 0.009)
    oseen_re_per_speed: tuple[float, float] = (6.925, 8.907)
    pol_coeff: tuple[float, float] = (0.0067859882792384195,
                                      0.00678598827923842)


def potential_jet_2d(x: float, y: float, t: float, A: float, B: float,
                     nu: float, beta: float = 0.0,
                     phase1: float = 0.0, phase2: float = 0.0):
    """Potential, gradient and Hessian in x,y for k=+2,-4 plus DC.

    Phi_RF=A Re[(x+i y)^2 exp(-it)]+B Re[(x-i y)^4 exp(-i nu t)].
    Phi_DC=beta*(-(x^2+y^2)/2); it is the transverse slice at z=0 of a
    harmonic quadrupole beta*(z^2-r^2/2).
    """
    c1, s1 = math.cos(t-phase1), math.sin(t-phase1)
    c2, s2 = math.cos(nu*t-phase2), math.sin(nu*t-phase2)
    r2 = x*x + y*y
    phi = A*((x*x-y*y)*c1 + 2.0*x*y*s1)
    gx = 2.0*A*(x*c1+y*s1)
    gy = 2.0*A*(x*s1-y*c1)
    hxx = 2.0*A*c1
    hxy = 2.0*A*s1
    hyy = -2.0*A*c1

    # q=(x-i y)^3 exp(-i nu t), p=(x-i y)^2 exp(-i nu t).
    qx = x*x*x - 3.0*x*y*y
    qy = y*y*y - 3.0*x*x*y
    # Multiplication by exp(-i nu t).
    qre, qim = qx*c2 + qy*s2, qy*c2 - qx*s2
    phi += B*((x**4-6*x*x*y*y+y**4)*c2 - (4*x*y*(x*x-y*y))*s2)
    gx += 4.0*B*qre
    gy += 4.0*B*qim
    px, py = x*x-y*y, -2.0*x*y
    pre, pim = px*c2 + py*s2, py*c2 - px*s2
    hxx += 12.0*B*pre
    hxy += 12.0*B*pim
    hyy -= 12.0*B*pre

    phi -= 0.5*beta*r2
    gx -= beta*x
    gy -= beta*y
    hxx -= beta
    hyy -= beta
    return phi, np.array([gx, gy]), np.array([[hxx, hxy], [hxy, hyy]])


def single_rhs(state: Sequence[float], t: float, alpha: float, lam: float,
               A: float, B: float, nu: float, beta: float = 0.0,
               pol_coeff: float = 0.0, oseen_re_per_speed: float = 0.0):
    x, y, vx, vy = map(float, state)
    _, grad, hess = potential_jet_2d(x, y, t, A, B, nu, beta)
    v = np.array([vx, vy])
    accel = -alpha*grad - lam*v
    if oseen_re_per_speed:
        Re = oseen_re_per_speed*math.hypot(vx, vy)
        accel -= lam*(3.0/16.0)*Re*v
    if pol_coeff:
        accel += pol_coeff*(hess@grad)
    return np.array([vx, vy, accel[0], accel[1]])


def pair_rhs(state: Sequence[float], t: float, alpha1: float, alpha2: float,
             p: Params, *, coulomb: bool = True, gravity: bool = False,
             pol_scale: float = 0.0, dc_scale: float = 1.0,
             phase1: float = 0.0, phase2: float = 0.0):
    """Coupled 3-D pair RHS with physical mass/charge coupling.

    alpha_i=(q_i/m_i)/(q1_ref/m1_ref), M_i=m_i/m1_ref and therefore
    Q_i=q_i/q1_ref=alpha_i*M_i. Coulomb accelerations are not independently
    tuned. The numerical maps currently use the planar (z=0) projection.
    """
    s = np.asarray(state, dtype=float).reshape(2, 6)
    out = np.zeros_like(s)
    M = np.array([1.0, p.mass2])
    alpha = np.array([alpha1, alpha2])
    lam = np.array([p.lam1, p.lam2])
    Q = alpha*M
    sep = s[0, :3]-s[1, :3]
    d2 = float(sep@sep)
    if d2 < (sum(p.particle_radii_R))**2:
        raise FloatingPointError("sphere contact")
    pair = np.zeros((2, 3))
    if coulomb:
        pair[0] = p.kappa0*Q[0]*Q[1]/M[0]*sep/(d2*math.sqrt(d2))
        pair[1] = -p.kappa0*Q[0]*Q[1]/M[1]*sep/(d2*math.sqrt(d2))
    for i in range(2):
        x, y, z, vx, vy, vz = s[i]
        # Extend the polynomial jet to the harmonic DC potential in z.
        _, gradxy, hxy = potential_jet_2d(x, y, t, p.A, p.B, p.nu, 0.0,
                                          phase1, phase2)
        grad = np.array([gradxy[0]-p.beta*dc_scale*x,
                         gradxy[1]-p.beta*dc_scale*y,
                         2*p.beta*dc_scale*z])
        hess3 = np.zeros((3, 3))
        hess3[:2, :2] = hxy-p.beta*dc_scale*np.eye(2)
        hess3[2, 2] = 2*p.beta*dc_scale
        vel = s[i, 3:]
        Re = p.oseen_re_per_speed[i]*float(np.linalg.norm(vel))
        acc = -alpha[i]*grad-lam[i]*(1.0+3.0*Re/16.0)*vel+pair[i]
        if gravity:
            acc[2] -= p.gravity*(1.0-p.gas_density_kg_m3/p.densities_kg_m3[i])
        if pol_scale:
            # 2-D RF+DC proxy only; z RF derivatives are identically omitted.
            acc += pol_scale*p.pol_coeff[i]*(hess3@grad)
        out[i, :3] = vel
        out[i, 3:] = acc
    return out.reshape(-1)


def rk4_step(rhs, state: np.ndarray, t: float, dt: float, args: tuple):
    k1 = rhs(state, t, *args)
    k2 = rhs(state+0.5*dt*k1, t+0.5*dt, *args)
    k3 = rhs(state+0.5*dt*k2, t+0.5*dt, *args)
    k4 = rhs(state+dt*k3, t+dt, *args)
    return state+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4)


def _jet_fast(x: float, y: float, t: float, A: float, B: float,
              nu: float, beta: float, phase1: float=0.0,
              phase2: float=0.0):
    """Scalar version of the 2-D potential jet, avoiding array allocation."""
    c1=math.cos(t-phase1); s1=math.sin(t-phase1)
    c2=math.cos(nu*t-phase2); s2=math.sin(nu*t-phase2)
    gx=2*A*(x*c1+y*s1); gy=2*A*(x*s1-y*c1)
    hxx=2*A*c1; hxy=2*A*s1
    qx=x*x*x-3*x*y*y; qy=y*y*y-3*x*x*y
    qre=qx*c2+qy*s2; qim=qy*c2-qx*s2
    gx+=4*B*qre; gy+=4*B*qim
    px=x*x-y*y; py=-2*x*y
    pre=px*c2+py*s2; pim=py*c2-px*s2
    hxx+=12*B*pre; hxy+=12*B*pim
    hyy=-2*A*c1-12*B*pre
    gx-=beta*x; gy-=beta*y
    hxx-=beta; hyy-=beta
    return gx,gy,hxx,hxy,hyy


def _single_fast_rhs(s, t, alpha, lam, A, B, nu, beta, pol_coeff,
                     phase1=0.0, phase2=0.0):
    x,y,vx,vy=s
    gx,gy,hxx,hxy,hyy=_jet_fast(x,y,t,A,B,nu,beta,phase1,phase2)
    ax=-alpha*gx-lam*vx+pol_coeff*(hxx*gx+hxy*gy)
    ay=-alpha*gy-lam*vy+pol_coeff*(hxy*gx+hyy*gy)
    return vx,vy,ax,ay


def rk4_single_fast(s, t, dt, args):
    k1=_single_fast_rhs(s,t,*args)
    q=tuple(s[i]+0.5*dt*k1[i] for i in range(4))
    k2=_single_fast_rhs(q,t+0.5*dt,*args)
    q=tuple(s[i]+0.5*dt*k2[i] for i in range(4))
    k3=_single_fast_rhs(q,t+0.5*dt,*args)
    q=tuple(s[i]+dt*k3[i] for i in range(4))
    k4=_single_fast_rhs(q,t+dt,*args)
    return tuple(s[i]+dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0 for i in range(4))


def _pair_fast_rhs(s, t, alpha1, alpha2, p: Params,
                   phase1=0.0, phase2=0.0):
    x1,y1,vx1,vy1,x2,y2,vx2,vy2=s
    g1x,g1y,h1xx,h1xy,h1yy=_jet_fast(x1,y1,t,p.A,p.B,p.nu,p.beta,phase1,phase2)
    g2x,g2y,h2xx,h2xy,h2yy=_jet_fast(x2,y2,t,p.A,p.B,p.nu,p.beta,phase1,phase2)
    v1=math.hypot(vx1,vy1); v2=math.hypot(vx2,vy2)
    re1=p.oseen_re_per_speed[0]*v1; re2=p.oseen_re_per_speed[1]*v2
    dX=x1-x2; dY=y1-y2; d2=dX*dX+dY*dY
    if d2<1e-20: raise FloatingPointError("pair collision singularity")
    fac=p.kappa0*alpha1*alpha2/(d2*math.sqrt(d2))
    axc=fac*p.mass2*dX; ayc=fac*p.mass2*dY
    bxc=-fac*dX; byc=-fac*dY
    ax1=-alpha1*g1x-p.lam1*(1+3*re1/16)*vx1+axc
    ay1=-alpha1*g1y-p.lam1*(1+3*re1/16)*vy1+ayc
    ax2=-alpha2*g2x-p.lam2*(1+3*re2/16)*vx2+bxc
    ay2=-alpha2*g2y-p.lam2*(1+3*re2/16)*vy2+byc
    return vx1,vy1,ax1,ay1,vx2,vy2,ax2,ay2


def rk4_pair_fast(s, t, dt, args):
    k1=_pair_fast_rhs(s,t,*args)
    q=tuple(s[i]+0.5*dt*k1[i] for i in range(8))
    k2=_pair_fast_rhs(q,t+0.5*dt,*args)
    q=tuple(s[i]+0.5*dt*k2[i] for i in range(8))
    k3=_pair_fast_rhs(q,t+0.5*dt,*args)
    q=tuple(s[i]+dt*k3[i] for i in range(8))
    k4=_pair_fast_rhs(q,t+dt,*args)
    return tuple(s[i]+dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0 for i in range(8))

