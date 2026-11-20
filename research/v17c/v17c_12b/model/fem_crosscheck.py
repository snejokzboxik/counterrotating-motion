"""Independent 2-D P1-FEM cross-check of the two RF electrode bases.

The geometry is the V17C.5/12B 24-round-rod cross-section inside a grounded
circular shield. MFS and FEM solve separate complex Dirichlet bases for k=+2
and k=-4. The two RF phases are maximized on the full phase torus by a support
function. The ideal DC quadrupole is deliberately excluded at metal surfaces:
it is not itself a conductor-boundary solution.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import gmsh
import numpy as np
from scipy.linalg import solve as dense_solve
from skfem import Basis, ElementTriP1, MeshTri, asm, condense, solve
from skfem.models.poisson import laplace

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
NROD = 24
ROD_A = 0.05
SHIELD_B = 1.10
D_OVER_R = 1.0 / 0.95


def build_mesh(hmin: float, hmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", hmin)
    gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
    gmsh.model.add("v17c12b_cross_section")

    def circle_loop(cx: float, cy: float, radius: float, reverse: bool = False) -> tuple[int, list[int]]:
        center = gmsh.model.geo.addPoint(cx, cy, 0.0)
        points = [gmsh.model.geo.addPoint(cx+radius*math.cos(k*math.pi/2),
                    cy+radius*math.sin(k*math.pi/2), 0.0) for k in range(4)]
        # Gmsh signature is (start point, center point, end point).
        arcs = [gmsh.model.geo.addCircleArc(points[k], center, points[(k+1) % 4])
                for k in range(4)]
        signed = [-arcs[3], -arcs[2], -arcs[1], -arcs[0]] if reverse else arcs
        return gmsh.model.geo.addCurveLoop(signed), arcs

    outer_loop, outer_curves = circle_loop(0.0, 0.0, SHIELD_B)
    loops = [outer_loop]
    rod_curves = []
    centers = np.exp(2j*np.pi*np.arange(NROD)/NROD)
    for z in centers:
        loop, curves = circle_loop(float(z.real), float(z.imag), ROD_A, reverse=True)
        rod_curves.extend(curves)
        loops.append(loop)
    surface = gmsh.model.geo.addPlaneSurface(loops)
    gmsh.model.geo.synchronize()
    # Refine only near the conductor surfaces; the outer bulk remains coarse.
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "CurvesList", rod_curves+outer_curves)
    gmsh.model.mesh.field.setNumber(dist, "Sampling", 100)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", dist)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", hmin)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", hmax)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.12)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    gmsh.model.mesh.generate(2)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    xy = node_coords.reshape(-1, 3)[:, :2].T
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
    etypes, _, enodes = gmsh.model.mesh.getElements(2, surface)
    tri_blocks = []
    for etype, nodes in zip(etypes, enodes):
        if int(etype) == 2:  # first-order three-node triangle
            raw = nodes.reshape(-1, 3)
            tri_blocks.append(np.vectorize(tag_to_idx.__getitem__)(raw).T)
    if not tri_blocks:
        gmsh.finalize()
        raise RuntimeError("gmsh returned no linear triangle elements")
    tri = np.concatenate(tri_blocks, axis=1).astype(np.int32)
    gmsh.finalize()

    # Circle centers used as geometric control points are not mesh vertices.
    # Drop any such unused nodes before assembling the FEM stiffness matrix.
    used = np.unique(tri)
    remap = np.full(xy.shape[1], -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    xy = xy[:, used]
    tri = remap[tri]

    # Boundary vertices are assigned by their exact geometric circles.
    radii = np.abs(xy[0]+1j*xy[1])
    shield_nodes = np.flatnonzero(np.abs(radii-SHIELD_B) < 2e-7)
    centers_xy = np.column_stack((centers.real, centers.imag))
    delta = xy.T[:, None, :] - centers_xy[None, :, :]
    dist_to_rods = np.linalg.norm(delta, axis=2)
    nearest = np.argmin(dist_to_rods, axis=1)
    nearest_dist = dist_to_rods[np.arange(xy.shape[1]), nearest]
    rod_nodes = np.flatnonzero(np.abs(nearest_dist-ROD_A) < 2e-7)
    boundary_dofs = np.unique(np.concatenate((shield_nodes, rod_nodes)))
    rod_id = nearest[rod_nodes]
    if len(boundary_dofs) == 0 or len(shield_nodes) == 0 or len(rod_nodes) == 0:
        raise RuntimeError("failed to identify conductor boundary nodes")
    return xy, tri, boundary_dofs, np.column_stack((rod_nodes, rod_id))


def solve_basis(mesh: MeshTri, p: np.ndarray, t: np.ndarray,
                boundary_dofs: np.ndarray, rod_node_ids: np.ndarray,
                mode: int) -> tuple[np.ndarray, np.ndarray]:
    basis = Basis(mesh, ElementTriP1())
    K = asm(laplace, basis)
    n = mesh.p.shape[1]
    values = np.zeros(n, dtype=np.complex128)
    rod_nodes = rod_node_ids[:, 0].astype(int)
    rod_index = rod_node_ids[:, 1].astype(int)
    phasor = np.exp(2j*np.pi*np.arange(NROD)/NROD*mode)
    values[rod_nodes] = phasor[rod_index]
    zeros = np.zeros(n, dtype=np.float64)
    ur = solve(*condense(K, zeros.copy(), D=boundary_dofs, x=values.real))
    ui = solve(*condense(K, zeros.copy(), D=boundary_dofs, x=values.imag))
    u = ur+1j*ui

    x, y = mesh.p
    x0, x1, x2 = x[t[0]], x[t[1]], x[t[2]]
    y0, y1, y2 = y[t[0]], y[t[1]], y[t[2]]
    det = (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
    g0x, g0y = (y1-y2)/det, (x2-x1)/det
    g1x, g1y = (y2-y0)/det, (x0-x2)/det
    g2x, g2y = (y0-y1)/det, (x1-x0)/det
    ex = -(u[t[0]]*g0x+u[t[1]]*g1x+u[t[2]]*g2x)
    ey = -(u[t[0]]*g0y+u[t[1]]*g1y+u[t[2]]*g2y)
    return ex, ey


def boundary_adjacent_triangles(xy: np.ndarray, tri: np.ndarray, hmin: float) -> np.ndarray:
    ctr = xy[:, tri].mean(axis=1)
    rc = np.abs(ctr[0]+1j*ctr[1])
    centers = np.exp(2j*np.pi*np.arange(NROD)/NROD)
    d = np.min(np.abs((ctr[0]+1j*ctr[1])[:, None]-centers[None, :]), axis=1)
    # Evaluate only the first couple of FEM element layers. A broad band adds
    # no useful surface information and multiplies the phase-support workload.
    band = 2.0*hmin
    rod_adj = (d >= ROD_A) & (d-ROD_A <= band)
    shield_adj = (rc <= SHIELD_B) & (SHIELD_B-rc <= band)
    return np.flatnonzero(rod_adj | shield_adj)


def support_max(ex1x: np.ndarray, ex1y: np.ndarray, ex2x: np.ndarray, ex2y: np.ndarray,
                u1: float, u2: float, ndir: int = 4096) -> float:
    angle = np.arange(ndir)*2.0*np.pi/ndir
    nx, ny = np.cos(angle), np.sin(angle)
    maximum = 0.0
    for lo in range(0, len(ex1x), 256):
        sl = slice(lo, min(lo+256, len(ex1x)))
        p1 = np.abs(ex1x[sl, None]*nx+ex1y[sl, None]*ny)*u1
        p2 = np.abs(ex2x[sl, None]*nx+ex2y[sl, None]*ny)*u2
        maximum = max(maximum, float(np.max(p1+p2)))
    return maximum


def main() -> None:
    seed = json.loads((RESULTS/"physical_seed.json").read_text(encoding="utf-8"))
    mfs = json.loads((RESULTS/"mfs_surface_refinement.json").read_text(encoding="utf-8"))
    # Compare with the RF-only conductor-boundary value. The same JSON also
    # reports an analytically added ideal-DC field, but that field does not
    # satisfy the conductor boundary-value problem and is not a valid FEM
    # reference for this cross-check.
    mfs_reference = next(row["rf_only_max_surface_field_V_m_all"]
        for row in mfs["rows"] if row["p_collocation_per_rod"] == 40
        and row["n_surface_samples_per_rod"] == 1024)
    drive = seed["electrode_drive"]
    u1 = drive["U1_peak_V_electrode_not_RMS_or_peak_to_peak"]
    u2 = drive["U2_peak_V_electrode_not_RMS_or_peak_to_peak"]
    rows = []
    for hmin in (0.010, 0.005, 0.0025):
        xy, tri, D, rod_ids = build_mesh(hmin, 0.08)
        mesh = MeshTri(xy, tri)
        f1x, f1y = solve_basis(mesh, xy, tri, D, rod_ids, 2)
        f2x, f2y = solve_basis(mesh, xy, tri, D, rod_ids, -4)
        adj = boundary_adjacent_triangles(xy, tri, hmin)
        # Divide gradients in dimensionless coordinates by D to obtain V/m.
        scale = 1.0/(seed["trap"]["electrode_center_radius_D_m"])
        peak = support_max(f1x[adj]*scale, f1y[adj]*scale,
                           f2x[adj]*scale, f2y[adj]*scale, u1, u2)
        # Compare both bases against the exact archived transfer coefficients
        # only as a global far-field check; the surface maximum is independent.
        rows.append({"hmin_over_D": hmin, "nodes": int(xy.shape[1]),
            "triangles": int(tri.shape[1]), "surface_adjacent_triangles": int(len(adj)),
            "boundary_dofs": int(len(D)), "rf_only_max_surface_field_V_m": peak,
            "rods_vs_shield_gap_elements_min_approx": 0.05/hmin,
            "mfs_reference_rf_only_V_m_p40_n1024": mfs_reference,
            "relative_to_mfs_reference": peak/mfs_reference-1.0})
    out = {"study": "independent linear FEM cross-check of 2-D RF surface fields",
        "geometry": "24 circular rods a/D=0.05, center radius D, grounded shield radius 1.10D",
        "rf_bases": ["complex electrode pattern k=+2", "complex electrode pattern k=-4"],
        "phase_maximum": "support-function maximization over independent RF phases",
        "mfs_reference_rf_only_V_m_p40_n1024": mfs_reference,
        "seed": seed["label"], "electrode_peak_V": {"U1": u1, "U2": u2},
        "refinement": rows, "limitations": [
            "2-D infinite-rod electrostatics only; no axial ends, feedthroughs, or DC electrode solution.",
            "P1 FEM gradients are elementwise constant; refinement study does not replace a 3-D FEM model.",
            "No gas discharge/corona criterion is computed from the field maximum."
        ]}
    (RESULTS/"fem_crosscheck.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
