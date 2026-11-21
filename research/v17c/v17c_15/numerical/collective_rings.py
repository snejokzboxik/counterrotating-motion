"""Analytic and direct-sum observables for regular Coulomb rings."""
from __future__ import annotations
import numpy as np


def regular_ring_positions(n, radius, height=0.0, phase=0.0):
    n=int(n)
    if n<2 or radius<=0:
        raise ValueError("a ring needs N>=2 and a positive radius")
    theta=phase+2*np.pi*np.arange(n)/n
    return np.column_stack((radius*np.cos(theta),radius*np.sin(theta),
                            np.full(n,float(height))))


def intraring_energy_direct(n, radius, charge, kappa=1.0):
    x=regular_ring_positions(n,radius)
    total=0.0
    for i in range(n):
        for j in range(i+1,n):
            total+=kappa*charge**2/np.linalg.norm(x[i]-x[j])
    return float(total)


def intraring_energy_formula(n, radius, charge, kappa=1.0):
    n=int(n)
    S=sum(1/np.sin(np.pi*k/n) for k in range(1,n))
    return float(kappa*charge**2*n*S/(4*radius))


def radial_coulomb_force_direct(n, radius, charge, kappa=1.0):
    x=regular_ring_positions(n,radius)
    force=np.zeros_like(x)
    for i in range(n):
        for j in range(i+1,n):
            d=x[i]-x[j]
            fij=kappa*charge**2*d/np.linalg.norm(d)**3
            force[i]+=fij
            force[j]-=fij
    return float(force[0,0])


def radial_coulomb_force_formula(n, radius, charge, kappa=1.0):
    S=sum(1/np.sin(np.pi*k/n) for k in range(1,int(n)))
    return float(kappa*charge**2*S/(4*radius**2))


def vertical_laplacian(n, radius):
    x=regular_ring_positions(n,radius)
    L=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            w=1/np.linalg.norm(x[i]-x[j])**3
            L[i,i]+=w; L[j,j]+=w
            L[i,j]-=w; L[j,i]-=w
    return L


def vertical_laplacian_spectrum_formula(n, radius):
    n=int(n)
    return np.array([
        sum((1-np.cos(2*np.pi*m*k/n))/
            (2*radius*np.sin(np.pi*k/n))**3
            for k in range(1,n))
        for m in range(n)],dtype=float)


def interring_energy_direct(n1,n2,R1,R2,z1,z2,Q1,Q2,kappa=1.0,
                            phase=0.0):
    a=regular_ring_positions(n1,R1,z1,0.0)
    b=regular_ring_positions(n2,R2,z2,phase)
    return float(kappa*Q1*Q2*sum(
        1/np.linalg.norm(a[i]-b[j])
        for i in range(n1) for j in range(n2)))


def phase_metrics(angles):
    theta=np.unwrap(np.asarray(angles,dtype=float))
    n=len(theta)
    wrapped=np.sort(np.mod(theta,2*np.pi))
    gaps=np.diff(np.r_[wrapped,wrapped[0]+2*np.pi])
    psi=abs(np.mean(np.exp(1j*n*theta)))
    return {"psi_N":float(psi),"gaps":gaps,
            "gap_cv":float(np.std(gaps)/np.mean(gaps)),
            "gap_min":float(np.min(gaps)),"gap_max":float(np.max(gaps))}
