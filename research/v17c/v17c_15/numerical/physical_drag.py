"""Dimensionless physical-drag maps with explicit Cunningham convention."""
from __future__ import annotations
import math


def stokes_gamma(radius_m: float, viscosity_pa_s: float, cunningham: float = 1.0) -> float:
    """Correct slip-adjusted Stokes coefficient 6*pi*eta*a/Cc in kg/s."""
    a=float(radius_m); eta=float(viscosity_pa_s); cc=float(cunningham)
    if not math.isfinite(a) or not math.isfinite(eta) or not math.isfinite(cc):
        raise ValueError("radius, viscosity, and Cc must be finite")
    if a<=0 or eta<=0 or cc<1:
        raise ValueError("require radius>0, viscosity>0, and Cunningham Cc>=1")
    return 6.0*math.pi*eta*a/cc


def legacy_gamma_with_reversed_slip(radius_m: float, viscosity_pa_s: float,
                                   cunningham: float) -> float:
    """Reproduce the archived V17C.12B coefficient for paired audits only."""
    a=float(radius_m); eta=float(viscosity_pa_s); cc=float(cunningham)
    if a<=0 or eta<=0 or cc<1 or not all(map(math.isfinite,(a,eta,cc))):
        raise ValueError("require finite radius>0, viscosity>0, and Cunningham Cc>=1")
    return 6.0*math.pi*eta*a*cc


def dimensionless_lambda(radius_m: float, viscosity_pa_s: float,
                         mass_kg: float, omega_ref: float,
                         cunningham: float = 1.0) -> float:
    """Return gamma/(m*Omega_ref) using the corrected Cunningham convention."""
    m=float(mass_kg); omega=float(omega_ref)
    if not math.isfinite(m) or not math.isfinite(omega) or m<=0 or omega<=0:
        raise ValueError("mass and reference frequency must be finite and positive")
    return stokes_gamma(radius_m,viscosity_pa_s,cunningham)/(m*omega)
