"""Reproducible unit quadrupole FEM on saved 30-degree sector meshes.

The m=2 boundary condition uses its own Bloch factor exp(i*pi/3), not the
legacy m=4 RF factor. Real part gives cos(2 theta) electrode voltages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from skfem import Basis, ElementTriP2, ElementTriP4, asm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap.direct_fem import (SECTOR_ANGLE, canonical_nodal_values,
    exponents, sector_mesh, vandermonde)
from three_phase_trap.fem import laplace_form

OUT = Path(__file__).resolve().parent
R = 0.008


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_quadrupole(path, degree):
    mesh, groups = sector_mesh(path)
    element = ElementTriP4() if degree == 4 else ElementTriP2()
    basis = Basis(mesh, element, intorder=2 * degree)
    minus = basis.get_dofs(facets=groups["minus"]).all()
    plus = basis.get_dofs(facets=groups["plus"]).all()
    rot = np.array([[math.cos(SECTOR_ANGLE), -math.sin(SECTOR_ANGLE)],
                    [math.sin(SECTOR_ANGLE), math.cos(SECTOR_ANGLE)]])
    mismatch, partner = cKDTree((rot @ basis.doflocs[:, minus]).T).query(basis.doflocs[:, plus].T)
    if mismatch.max() > 1e-12 or len(set(partner)) != len(minus):
        raise ValueError("Sector seam DOFs do not match")
    origin = np.flatnonzero(np.linalg.norm(basis.doflocs, axis=0) < 1e-12)
    if len(origin) != 1:
        raise ValueError("Expected one origin DOF")
    factor = np.exp(1j * math.pi / 3)  # H(theta+30 deg) = exp(i*2*30 deg) H(theta)
    parent, weight = np.arange(basis.N), np.ones(basis.N, complex)
    active = plus != origin[0]
    parent[plus[active]] = minus[partner[active]]
    weight[plus[active]] = factor
    representatives, inverse = np.unique(parent, return_inverse=True)
    transform = csr_matrix((weight, (np.arange(basis.N), inverse)), shape=(basis.N, len(representatives)))
    outer = basis.get_dofs(facets=groups["outer"]).all()
    electrode = basis.get_dofs(facets=groups["electrode"]).all()
    fixed = np.unique(inverse[np.r_[outer, electrode, origin]])
    values = np.zeros(len(representatives), complex)
    values[inverse[electrode]] = 1.0
    stiffness = asm(laplace_form, basis)
    reduced = (transform.conj().T @ stiffness @ transform).tocsc()
    interior = np.setdiff1d(np.arange(len(values)), fixed)
    matrix = reduced[interior][:, interior].tocsc()
    values[interior] = splu(matrix).solve(-(reduced @ values)[interior])
    residual = (reduced @ values)[interior]
    error = float(np.max(abs(residual)) / np.asarray(abs(reduced).sum(axis=1)).max())
    if error > 1e-10:
        raise ValueError(f"FEM residual {error:g}")
    nodal = canonical_nodal_values(basis, transform @ values)
    coefficients = np.linalg.solve(vandermonde(element.doflocs, degree), nodal.T).T
    return mesh, coefficients, {"degree": degree, "sector_dofs": int(basis.N),
                                "sector_triangles": int(mesh.nelements),
                                "max_seam_dof_mismatch_m": float(mismatch.max()),
                                "scaled_discrete_residual_max": error,
                                "bloch_factor": [float(factor.real), float(factor.imag)]}


def sample(mesh, coefficients, degree, points):
    angle = np.arctan2(points[:, 1], points[:, 0])
    k = np.floor((angle + math.pi / 12) / (math.pi / 6)).astype(int)
    local_angle = -k * math.pi / 6
    c, s = np.cos(local_angle), np.sin(local_angle)
    local = np.stack([c * points[:, 0] - s * points[:, 1],
                      s * points[:, 0] + c * points[:, 1]], axis=1)
    tri = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    ids = tri.get_trifinder()(local[:, 0], local[:, 1])
    if np.any(ids < 0):
        raise ValueError("Ring sampling left sector mesh")
    vertices = mesh.p.T[mesh.t.T[ids]]
    uv = np.linalg.solve(np.swapaxes(vertices[:, 1:] - vertices[:, [0]], 1, 2),
                         (local - vertices[:, 0])[:, :, None])[:, :, 0]
    local_phi = np.sum(coefficients[ids] * vandermonde(uv, degree), axis=1)
    return local_phi * np.exp(1j * k * math.pi / 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=int, choices=(35, 50, 70), required=True)
    parser.add_argument("--degree", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    path = ROOT / f"outputs/meshes/gmsh_{args.mesh}um/sector.msh"
    mesh, coefficients, diagnostics = solve_quadrupole(path, args.degree)
    diagnostics.update({"mesh_um": args.mesh, "sector_mesh_sha256": sha(path),
                        "radius_normalization_m": R, "samples_per_ring": 2048,
                        "electrode_basis": "cos(2*pi*j/6), j=0..11",
                        "grounded_outer_boundary": True})
    # Avoid exact seam points, where the triangulation finder is roundoff sensitive.
    theta = 2 * np.pi * (np.arange(2048) + 0.123) / 2048
    rows = []
    for radius_mm in (0.5, 1, 2, 4, 6, 7, 8):
        radius = radius_mm * 0.001
        pts = radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        h = sample(mesh, coefficients, args.degree, pts)
        phi = h.real
        for order in (0, 2, 4, 6, 10):
            a = (2 if order else 1) * np.mean(phi * np.cos(order * theta))
            b = 0 if order == 0 else 2 * np.mean(phi * np.sin(order * theta))
            rows.append({"mesh_um": args.mesh, "degree": args.degree,
                         "radius_mm": radius_mm, "order_m": order,
                         "cos_coefficient": float(a), "sin_coefficient": float(b),
                         "normalized_cos_coefficient": float(a / (radius / R)**order),
                         "normalized_sin_coefficient": float(b / (radius / R)**order)})
    stem = f"quadrupole_h{args.mesh}_P{args.degree}"
    with (OUT / f"{stem}_harmonics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (OUT / f"{stem}_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics))


if __name__ == "__main__":
    main()
