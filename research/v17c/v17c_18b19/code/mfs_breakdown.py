"""MFS surface-field maxima for the 24-rod V17C RF electrode patterns.

The two RF basis solutions are solved independently. Since the two RF phases
are independent, the maximum is taken over the full two-phase torus. A support
function reduces this to a one-dimensional maximization in field direction;
direct two-phase sampling is also used for an independent numerical check.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

N = 24
A_D = 0.05
B_D = 1.10
R_D = 1.0 - A_D
GEOMETRY = "V17C.5 archived 24-electrode mixed6 pattern; 2D infinite rods + grounded circular shield"


def boundary_system(p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.arange(N, dtype=np.float64) * 2.0*np.pi/N
    centers = np.exp(1j*theta)
    alpha = np.arange(p, dtype=np.float64) * 2.0*np.pi/p
    collocation = (centers[:, None] + A_D*np.exp(1j*alpha)[None, :]).ravel()
    sources = (centers[:, None] + 0.65*A_D*np.exp(1j*alpha)[None, :]).ravel()
    z = collocation[:, None]
    s = sources[None, :]
    G = np.log(np.abs(B_D*B_D - z*np.conj(s)) / (B_D*np.abs(z-s)))
    v1 = np.exp(2j*theta)
    v2 = np.exp(-4j*theta)
    V = np.column_stack((np.repeat(v1, p), np.repeat(v2, p)))
    return G, sources, theta, V


def gradient_matrix(target: np.ndarray, sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = target[:, None]
    s = sources[None, :]
    u = B_D*B_D - z*np.conj(s)
    w = z - s
    gx = np.real(-np.conj(s)/u - 1.0/w)
    gy = np.real(-1j*np.conj(s)/u - 1j/w)
    return gx, gy


def hessian_matrix(target: np.ndarray, sources: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic 2D Hessian of each logarithmic MFS basis potential."""
    z = target[:, None]
    s = sources[None, :]
    u = B_D*B_D-z*np.conjugate(s)
    w = z-s
    hxx = np.real(-np.conjugate(s)**2/u**2+1.0/w**2)
    hxy = np.real(-1j*np.conjugate(s)**2/u**2+1j/w**2)
    hyy = np.real(np.conjugate(s)**2/u**2-1.0/w**2)
    return hxx, hxy, hyy


def surface_targets(n_angle: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arange(N, dtype=np.float64)*2.0*np.pi/N
    centers = np.exp(1j*theta)
    alpha = (np.arange(n_angle, dtype=np.float64)+0.5)*2.0*np.pi/n_angle
    rods = (centers[:, None] + A_D*np.exp(1j*alpha)[None, :]).ravel()
    shield = B_D*np.exp(1j*np.arange(max(720, n_angle), dtype=np.float64)*2.0*np.pi/max(720, n_angle))
    return rods, shield


def support_torus_max(ex1x, ex1y, ex2x, ex2y, edcx, edcy, n_direction: int = 16384):
    angle = np.arange(n_direction, dtype=np.float64)*2.0*np.pi/n_direction
    nx, ny = np.cos(angle), np.sin(angle)
    out = np.empty(len(ex1x), dtype=np.float64)
    for lo in range(0, len(ex1x), 64):
        hi = min(len(ex1x), lo+64)
        p1 = np.abs(ex1x[lo:hi, None]*nx[None, :] + ex1y[lo:hi, None]*ny[None, :])
        p2 = np.abs(ex2x[lo:hi, None]*nx[None, :] + ex2y[lo:hi, None]*ny[None, :])
        pdc = edcx[lo:hi, None]*nx[None, :] + edcy[lo:hi, None]*ny[None, :]
        out[lo:hi] = np.max(p1+p2+pdc, axis=1)
    return out


def direct_torus_max(ex1x, ex1y, ex2x, ex2y, edcx, edcy, n_phase: int = 256) -> np.ndarray:
    phase = np.arange(n_phase, dtype=np.float64)*2.0*np.pi/n_phase
    ep = np.exp(-1j*phase)
    out = np.zeros(len(ex1x), dtype=np.float64)
    for i in range(len(ex1x)):
        ex = (np.real(ex1x[i]*ep[:, None])[:, :, None]
              + np.real(ex2x[i]*ep[None, :])[None, :, None]
              + edcx[i])
        ey = (np.real(ex1y[i]*ep[:, None])[:, :, None]
              + np.real(ex2y[i]*ep[None, :])[None, :, None]
              + edcy[i])
        out[i] = float(np.max(np.sqrt(ex*ex + ey*ey)))
    return out


def sample_fields(p: int, n_angle: int, voltage: dict) -> dict:
    G, sources, theta, V = boundary_system(p)
    strengths = solve(G, V, assume_a="gen", check_finite=False)
    bnd_res = float(np.max(np.abs(G @ strengths - V)))
    rods, shield = surface_targets(n_angle)
    target = np.concatenate((rods, shield))
    gx, gy = gradient_matrix(target, sources)
    # Electric-field phasors per unit electrode amplitude; field is per D.
    e1x = -(gx @ strengths[:, 0]) * voltage["u1_electrode_peak_V"] / (voltage["D_m"])
    e1y = -(gy @ strengths[:, 0]) * voltage["u1_electrode_peak_V"] / (voltage["D_m"])
    e2x = -(gx @ strengths[:, 1]) * voltage["u2_electrode_peak_V"] / (voltage["D_m"])
    e2y = -(gy @ strengths[:, 1]) * voltage["u2_electrode_peak_V"] / (voltage["D_m"])
    # Ideal quadrupole field used by the dynamics. The DC coefficient is not
    # the endcap voltage; this term is a near-center analytic estimate only.
    pos = target
    r_m = np.abs(pos)*voltage["D_m"]
    er = voltage["v2_dc_coefficient_V"]*r_m/(voltage["R_m"]**2)
    edcx = er*np.cos(np.angle(pos))
    edcy = er*np.sin(np.angle(pos))
    emax = support_torus_max(e1x, e1y, e2x, e2y, edcx, edcy)
    zeros = np.zeros_like(edcx)
    emax_rf = support_torus_max(e1x, e1y, e2x, e2y, zeros, zeros)
    rods_n = len(rods)
    rod_max_idx = int(np.argmax(emax[:rods_n]))
    shield_max_idx = int(np.argmax(emax[rods_n:]))
    direct_idx = np.unique(np.linspace(0, len(target)-1, min(16, len(target)), dtype=int))
    direct = direct_torus_max(e1x[direct_idx], e1y[direct_idx], e2x[direct_idx], e2y[direct_idx],
                              edcx[direct_idx], edcy[direct_idx], 256)
    return {
        "p_collocation_per_rod": p,
        "n_rods": N,
        "n_surface_samples_per_rod": n_angle,
        "n_shield_samples": len(shield),
        "max_boundary_potential_residual_V": bnd_res,
        "rf_only_max_surface_field_V_m_all": float(np.max(emax_rf)),
        "rf_only_rod_max_surface_field_V_m": float(np.max(emax_rf[:rods_n])),
        "rf_only_shield_max_surface_field_V_m": float(np.max(emax_rf[rods_n:])),
        "max_surface_field_ideal_DC_analytically_added_V_m_not_a_conductor_BVP": float(np.max(emax)),
        "rod_max_surface_field_ideal_DC_added_V_m": float(np.max(emax[:rods_n])),
        "rod_max_point_index": rod_max_idx,
        "rod_max_electrode_index": int(rod_max_idx//n_angle),
        "rod_max_angle_within_rod_deg": float((rod_max_idx % n_angle + 0.5)*360.0/n_angle),
        "shield_max_surface_field_ideal_DC_added_V_m": float(np.max(emax[rods_n:])),
        "maximum_ideal_DC_field_evaluated_at_boundary_V_m_not_physical_conductor_solution": float(np.max(np.hypot(edcx, edcy))),
        "shield_max_angle_deg": float(shield_max_idx*360.0/len(shield)),
        "support_vs_direct_torus_max_abs_diff_V_m": float(np.max(np.abs(direct-emax[direct_idx]))),
        "support_vs_direct_torus_max_rel_diff": float(np.max(np.abs(direct-emax[direct_idx]) / np.maximum(emax[direct_idx], 1.0))),
        "rod_to_shield_voltage_bound_V": voltage["u1_electrode_peak_V"]+voltage["u2_electrode_peak_V"],
        "adjacent_rod_voltage_bound_V": 2.0*voltage["u1_electrode_peak_V"]*math.sin(2.0*math.pi/24.0)+voltage["u2_electrode_peak_V"],
        "assumptions": [
            "Two independent RF electrode bases solved separately by MFS and combined over independent phase torus.",
            "Infinite rods and 2D grounded circular shield; no rod ends or endcaps.",
            "RF-only surface field is the conductor-boundary result. The separate ideal DC addition is not a physical conductor-boundary solution.",
            "Surface field is computed from MFS gradients and sampled densely along each conductor boundary.",
        ],
    }


def main() -> None:
    seed = json.loads((RESULTS/"physical_seed.json").read_text(encoding="utf-8"))
    tr = seed["trap"]
    drive = seed["electrode_drive"]
    voltage = {
        "u1_electrode_peak_V": drive["U1_peak_V_electrode_not_RMS_or_peak_to_peak"],
        "u2_electrode_peak_V": drive["U2_peak_V_electrode_not_RMS_or_peak_to_peak"],
        "v2_dc_coefficient_V": drive["ideal_DC_quadrupole_coefficient_V_not_an_endcap_voltage"],
        "D_m": tr["electrode_center_radius_D_m"], "R_m": tr["free_aperture_radius_R_m"],
    }
    results = [sample_fields(p, 256, voltage) for p in (16, 24, 32, 40)]
    out = {
        "study": "24-electrode MFS RF surface fields over the full two-phase torus",
        "geometry": GEOMETRY,
        "seed": seed["label"],
        "electrode_peak_amplitudes_V": {"U1": voltage["u1_electrode_peak_V"], "U2": voltage["u2_electrode_peak_V"]},
        "convergence": results,
        "limitations": [
            "No finite axial length, endcaps, feedthroughs, guards, or real DC segmentation.",
            "No FEM cross-check yet; use these values as an MFS screening result only.",
            "Paschen, corona, and RF discharge thresholds are not inferred from Emax alone.",
            "Surface roughness and field-enhancement factor are not included.",
        ],
    }
    target = RESULTS/"mfs_breakdown_screen.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
