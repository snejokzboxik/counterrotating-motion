"""Finite 3-D FEM for continuous RF rods with independent DC end-cap disks.

Solves real/imaginary basis fields for k=+2 and k=-4, plus one-volt DC
bases on each end-cap. The mesh is checkpointed after every resolution so
the end-cap field and RF transfer convergence can be audited before any
particle trajectory is integrated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gmsh
import meshio
import numpy as np
import pyamg
from scipy.signal import savgol_filter
from scipy.sparse.linalg import cg
from skfem import Basis, MeshTet, asm
from skfem.element import ElementTetP1
from skfem.models.poisson import laplace

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
BENCH = json.loads((DATA / "source_benchmarks.json").read_text(encoding="utf-8"))
PS = BENCH["v17c16_polystyrene_pair"]

R = 5e-3
D = R / .95
ROD_R = .05 * D
SHIELD_R = 1.10 * D
L = 50e-3
ROD_L = 44e-3
CAP_R = R
CAP_L = .5e-3
CAP_CENTERS = (-23e-3, 23e-3)
G = 9.80665
H_LEVELS_MM = (.60, .45, .34, .27, .24)
RF_MODES = (2, -4)


def build_mesh(hmax: float, mesh_path: Path):
    centers = np.asarray([(D * math.cos(2 * math.pi * j / 24),
                           D * math.sin(2 * math.pi * j / 24),
                           2 * math.pi * j / 24) for j in range(24)])
    if mesh_path.exists():
        loaded = meshio.read(mesh_path)
        if "tetra" not in loaded.cells_dict:
            raise RuntimeError(f"no linear tetra cell block in {mesh_path}")
        return np.asarray(loaded.points[:, :3]), np.asarray(loaded.cells_dict["tetra"], dtype=np.int32), centers, np.arange(2)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmax / 7)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.model.add("continuous_24rod_RF_with_DC_endcaps")
        occ = gmsh.model.occ
        domain = occ.addCylinder(0, 0, -L / 2, 0, 0, L, SHIELD_R)
        cutters = []
        centers = []
        for j in range(24):
            theta = 2 * math.pi * j / 24
            x, y = D * math.cos(theta), D * math.sin(theta)
            centers.append((x, y, theta))
            tag = occ.addCylinder(x, y, -ROD_L / 2, 0, 0, ROD_L, ROD_R)
            cutters.append((3, tag))
        cap_tags = []
        for zc in CAP_CENTERS:
            tag = occ.addCylinder(0, 0, zc - CAP_L / 2, 0, 0, CAP_L, CAP_R)
            cap_tags.append(tag)
            cutters.append((3, tag))
        occ.cut([(3, domain)], cutters, removeObject=True, removeTool=True)
        occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(str(mesh_path))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        etypes, _, enodes = gmsh.model.mesh.getElements(3)
        tet_nodes = None
        for typ, conn in zip(etypes, enodes):
            name, _, order, nnode, *_ = gmsh.model.mesh.getElementProperties(typ)
            if name.startswith("Tetrahedron") and order == 1 and nnode == 4:
                tet_nodes = conn.reshape(-1, 4)
                break
        if tet_nodes is None:
            raise RuntimeError("first-order tetrahedra were not generated")
        ids = np.asarray(node_tags, dtype=np.int64)
        points = np.asarray(coords, dtype=float).reshape(-1, 3)
        lut = {int(tag): i for i, tag in enumerate(ids)}
        tets = np.vectorize(lut.__getitem__)(tet_nodes).astype(np.int32)
        return points, tets, np.asarray(centers), np.asarray(cap_tags)
    finally:
        gmsh.finalize()


def classify_boundary(basis: Basis, centers: np.ndarray, hmax: float):
    xyz = basis.mesh.p.T
    boundary = np.unique(basis.mesh.facets[:, basis.mesh.boundary_facets()])
    tol = max(1.5e-7, hmax * .012)
    rho = np.hypot(xyz[:, 0], xyz[:, 1])
    grounded = (np.abs(rho[boundary] - SHIELD_R) < tol) | \
               (np.abs(np.abs(xyz[boundary, 2]) - L / 2) < tol)
    rod_id = np.full(len(xyz), -1, dtype=np.int16)
    for j, (cx, cy, _) in enumerate(centers):
        dr = np.hypot(xyz[:, 0] - cx, xyz[:, 1] - cy)
        side = (np.abs(dr - ROD_R) < tol) & (np.abs(xyz[:, 2]) <= ROD_L / 2 + tol)
        cap = (np.abs(np.abs(xyz[:, 2]) - ROD_L / 2) < tol) & (dr <= ROD_R + tol)
        rod_id[boundary[side[boundary] | cap[boundary]]] = j
    dc_id = np.full(len(xyz), -1, dtype=np.int8)
    for side, zc in enumerate(CAP_CENTERS):
        dr = rho
        wall = (np.abs(dr - CAP_R) < tol) & (np.abs(xyz[:, 2] - zc) <= CAP_L / 2 + tol)
        face = (np.abs(np.abs(xyz[:, 2] - zc) - CAP_L / 2) < tol) & (dr <= CAP_R + tol)
        dc_id[boundary[wall[boundary] | face[boundary]]] = side
    covered = grounded | (rod_id[boundary] >= 0) | (dc_id[boundary] >= 0)
    unclassified = int(np.count_nonzero(~covered))
    if unclassified:
        raise RuntimeError(f"{unclassified} boundary nodes are not assigned to a conductor")
    dofs = boundary
    free = np.setdiff1d(np.arange(basis.N), dofs, assume_unique=True)
    return xyz, boundary, rod_id, dc_id, grounded, dofs, free, tol


def solve_dirichlet(A, dofs, free, boundary, rod_id, dc_id, centers):
    Aff = A[free][:, free].tocsr()
    ml = pyamg.smoothed_aggregation_solver(Aff, symmetry="symmetric")
    preconditioner = ml.aspreconditioner(cycle="V")
    vd = np.zeros((len(dofs), 6), dtype=np.float64)
    pos = {int(node): i for i, node in enumerate(dofs)}
    for node in boundary:
        i = pos[int(node)]
        j = rod_id[node]
        if j >= 0:
            theta = centers[j, 2]
            for col, m in enumerate(RF_MODES):
                val = np.exp(1j * m * theta)
                vd[i, 2 * col] = val.real
                vd[i, 2 * col + 1] = val.imag
        dc = dc_id[node]
        if dc >= 0:
            vd[i, 4 + dc] = 1.0
    rhs = -A[free][:, dofs] @ vd
    U = np.zeros((A.shape[0], 6), dtype=np.float64)
    U[dofs] = vd
    diagnostics = []
    for col in range(rhs.shape[1]):
        iterations = [0]
        def count_iteration(_):
            iterations[0] += 1
        solution, info = cg(Aff, np.asarray(rhs[:, col]).ravel(), M=preconditioner,
                            rtol=1e-9, atol=0.0, maxiter=1000, callback=count_iteration)
        relres = float(np.linalg.norm(Aff @ solution - rhs[:, col]) /
                       max(np.linalg.norm(rhs[:, col]), 1e-300))
        if info != 0 or relres > 2e-9:
            raise RuntimeError(f"AMG-CG failed for basis {col}: info={info}, relative residual={relres:.3e}")
        U[free, col] = solution
        diagnostics.append({"basis": col, "iterations": iterations[0], "relative_residual": relres})
    del ml, preconditioner, Aff
    return U, diagnostics


def evaluate(basis: Basis, U: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # scikit-fem's tetra locator forms an inverse map for every cell and probe
    # at once; for ~1e6 cells and hundreds of probes that temporary can exceed
    # available RAM. Small batches bound peak memory without changing results.
    parts = []
    for start in range(0, pts.shape[1], 8):
        p = basis.probes(pts[:, start:start + 8])
        parts.append(np.asarray(p @ U))
    return np.vstack(parts)


def axial_dc_profile(basis: Basis, U: np.ndarray):
    z = np.linspace(-15e-3, 15e-3, 601)
    base = np.vstack((np.zeros_like(z), np.zeros_like(z), z))
    phi = evaluate(basis, U[:, 4:6], base).sum(axis=1)
    # Smooth the P1 potential over a fixed 1-mm axial window before taking
    # derivatives. Pointwise tiny differences resolve tetra-face jumps, not
    # the continuum field gradient.
    dz = z[1] - z[0]
    ez = -savgol_filter(phi, window_length=21, polyorder=3, deriv=1, delta=dz)
    dez_dz = -savgol_filter(phi, window_length=21, polyorder=3, deriv=2, delta=dz)
    return z, ez, dez_dz, phi


def linear_interp_root(x: np.ndarray, y: np.ndarray, target: float,
                       reference_z: float, side: str = "any"):
    s = y - target
    crossings = np.flatnonzero(s[:-1] * s[1:] <= 0)
    if not len(crossings):
        return None
    roots = []
    for k in crossings:
        if y[k + 1] == y[k]:
            roots.append(float(np.clip(reference_z, x[k], x[k + 1])))
        else:
            roots.append(float(x[k] + (target - y[k]) * (x[k + 1] - x[k]) /
                               (y[k + 1] - y[k])))
    if side == "below":
        allowed = [root for root in roots if root < reference_z]
        if allowed:
            roots = allowed
    elif side == "above":
        allowed = [root for root in roots if root > reference_z]
        if allowed:
            roots = allowed
    return min(roots, key=lambda root: abs(root - reference_z))


def off_axis_dc_sample(basis: Basis, U: np.ndarray, z_positions: list[float]):
    h = 50e-6
    r = .55 * R
    theta = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    result = []
    for z in z_positions:
        x, y = r * np.cos(theta), r * np.sin(theta)
        pts = np.vstack((x, y, np.full_like(x, z)))
        ex_p = pts.copy(); ex_p[0] += h
        ex_m = pts.copy(); ex_m[0] -= h
        ey_p = pts.copy(); ey_p[1] += h
        ey_m = pts.copy(); ey_m[1] -= h
        ez_p = pts.copy(); ez_p[2] += h
        ez_m = pts.copy(); ez_m[2] -= h
        up = evaluate(basis, U[:, 4:6], ex_p).sum(axis=1)
        um = evaluate(basis, U[:, 4:6], ex_m).sum(axis=1)
        ep = -(up - um) / (2 * h)
        up = evaluate(basis, U[:, 4:6], ey_p).sum(axis=1)
        um = evaluate(basis, U[:, 4:6], ey_m).sum(axis=1)
        ey = -(up - um) / (2 * h)
        up = evaluate(basis, U[:, 4:6], ez_p).sum(axis=1)
        um = evaluate(basis, U[:, 4:6], ez_m).sum(axis=1)
        ez = -(up - um) / (2 * h)
        er = ep * np.cos(theta) + ey * np.sin(theta)
        result.append({"z_mm": z * 1e3, "radius_over_R": .55,
                       "Er_per_Vcap_V_m": er.tolist(), "Ez_per_Vcap_V_m": ez.tolist(),
                       "Er_min_max_per_Vcap_V_m": [float(er.min()), float(er.max())],
                       "Ez_min_max_per_Vcap_V_m": [float(ez.min()), float(ez.max())]})
    return result


def solve_mesh(hmax: float):
    stem = f"continuous_rods_endcaps_h{hmax * 1e3:.2f}mm"
    mesh_path = RESULTS / f"{stem}.msh"
    points, tets, centers, _ = build_mesh(hmax, mesh_path)
    mesh = MeshTet(points.T, tets.T, sort_t=False)
    basis = Basis(mesh, ElementTetP1())
    A = asm(laplace, basis).tocsc()
    xyz, boundary, rod_id, dc_id, grounded, dofs, free, tol = classify_boundary(basis, centers, hmax)
    U, solver_diagnostics = solve_dirichlet(A, dofs, free, boundary, rod_id, dc_id, centers)
    del A

    r_sample = .90 * R
    theta = np.linspace(0, 2 * math.pi, 192, endpoint=False)
    sample = np.vstack((r_sample * np.cos(theta), r_sample * np.sin(theta), np.zeros_like(theta)))
    probe_values = evaluate(basis, U, sample)
    rf = []
    for family, m in enumerate(RF_MODES):
        v = probe_values[:, 2 * family] + 1j * probe_values[:, 2 * family + 1]
        coeff = np.mean(v * np.exp(-1j * m * theta))
        mfs_ref = {2: .8703168911, -4: .7982172624}[m] * (.90 ** abs(m))
        rf.append({"m": m, "complex_transfer_V_per_Vrod": [float(coeff.real), float(coeff.imag)],
                   "transfer_magnitude_at_0p9R": float(abs(coeff)),
                   "infinite_2d_MFS_at_0p9R": mfs_ref,
                   "relative_difference_from_2d_MFS": float(abs(abs(coeff) - mfs_ref) / mfs_ref)})

    z, ez, dez_dz, phi = axial_dc_profile(basis, U)
    # Pick the equal-endcap voltage that balances particle 1 at z=-2.5 mm.
    target_z1 = -2.5e-3
    ez_at_z1 = float(np.interp(target_z1, z, ez))
    particles = PS["particles"]
    eq = []
    if ez_at_z1 <= 0:
        vcap = None
    else:
        g1 = G * (1 - 1.2 / 1050.0)
        target_e1 = g1 / float(particles[0]["q_over_m_C_kg"])
        vcap = target_e1 / ez_at_z1
        for particle_index, p in enumerate(particles):
            target_e = G * (1 - 1.2 / 1050.0) / float(p["q_over_m_C_kg"])
            if particle_index == 0:
                z_eq = linear_interp_root(z, vcap * ez, target_e, reference_z=target_z1)
            else:
                # For the lower q/m sphere the gravity-balance field is larger;
                # select the nearest lower-z crossing relative to particle 1.
                z_eq = linear_interp_root(z, vcap * ez, target_e,
                                          reference_z=target_z1, side="below")
            if z_eq is None:
                eq.append({"q_over_m_C_kg": p["q_over_m_C_kg"], "target_Ez_V_m": target_e,
                           "root_found": False})
                continue
            slope = float(np.interp(z_eq, z, dez_dz) * vcap)
            eq.append({"q_over_m_C_kg": p["q_over_m_C_kg"], "target_Ez_V_m": target_e,
                       "z_eq_mm": z_eq * 1e3, "dEz_dz_V_m2": slope,
                       "axially_restoring_for_positive_charge": bool(slope < 0), "root_found": True})
    zpos = [x["z_eq_mm"] * 1e-3 for x in eq if x.get("root_found")]
    offaxis = off_axis_dc_sample(basis, U, zpos) if zpos else []

    # Store a compact field package for later interpolation only after a
    # converged mesh is selected; this routine retains basic mesh metadata.
    row = {"mesh_hmax_mm": hmax * 1e3, "mesh_file": mesh_path.name,
           "vertices": int(basis.N), "tetrahedra": int(mesh.nelements),
           "boundary_nodes": int(len(boundary)), "rod_boundary_nodes": int(np.count_nonzero(rod_id[boundary] >= 0)),
           "dc_endcap_boundary_nodes": int(np.count_nonzero(dc_id[boundary] >= 0)),
           "unclassified_boundary_nodes": 0, "boundary_classification_tolerance_mm": tol * 1e3,
           "linear_solver": {"method": "smoothed-aggregation AMG-preconditioned conjugate gradients",
                             "relative_tolerance": 1e-9, "basis_solves": solver_diagnostics},
           "geometry": {"R_mm": R * 1e3, "rod_center_radius_mm": D * 1e3,
             "rod_radius_mm": ROD_R * 1e3, "shield_radius_mm": SHIELD_R * 1e3,
             "shield_length_mm": L * 1e3, "rod_active_length_mm": ROD_L * 1e3,
             "end_DC_disks": [{"center_z_mm": zc * 1e3, "radius_mm": CAP_R * 1e3,
                                "thickness_mm": CAP_L * 1e3} for zc in CAP_CENTERS]},
           "RF_transfer": rf,
           "DC_endcap_screen": {"basis": "both DC disk endcaps at +1 V; all 24 RF rods, shield and outer end plates at 0 V",
             "particle1_target_z_mm": target_z1 * 1e3,
             "endcap_voltage_for_particle1_balance_V": vcap,
             "particle_force_balance": eq,
             "predicted_separation_mm": abs(eq[0]["z_eq_mm"] - eq[1]["z_eq_mm"]) if len(eq) == 2 and all(x.get("root_found") for x in eq) else None,
             "axial_field_derivative_method": "Savitzky-Golay cubic derivative on the FEM axial potential; 21 samples (1 mm window)",
             "axial_profile_z_mm": (z * 1e3).tolist(), "Phi_per_Vcap_V": phi.tolist(),
             "Ez_per_Vcap_V_m": ez.tolist(),
             "off_axis_r0p55R_field_samples": offaxis},
           "limitations": ["P1 tetra FEM; sharp conductor edges are not surface-field converged",
             "ideal end-disk supports/wiring and dielectric feedthroughs are omitted",
             "field solution only; no trajectories or RF breakdown qualification in this run"]}
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels-mm", nargs="+", type=float, default=list(H_LEVELS_MM))
    parser.add_argument("--force", action="store_true",
                        help="recompute requested mesh levels and replace their result rows")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    outpath = RESULTS / "continuous_rods_endcap_fem.json"
    progress = {"geometry_definition": "24 unsegmented RF rods; two separate internal DC endcap disks; grounded cylindrical shield and outer end plates",
                "model": "static Laplace conductor basis solutions; not yet particle integration",
                "incomplete_mesh_attempts": [], "mesh_levels": []}
    if outpath.exists():
        try:
            progress = json.loads(outpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    done = {round(float(x["mesh_hmax_mm"]), 5) for x in progress["mesh_levels"]}
    for h_mm in args.levels_mm:
        key = round(h_mm, 5)
        if key in done and not args.force:
            continue
        if args.force:
            progress["mesh_levels"] = [x for x in progress["mesh_levels"]
                                        if round(float(x["mesh_hmax_mm"]), 5) != key]
        row = solve_mesh(h_mm * 1e-3)
        progress["mesh_levels"].append(row)
        progress["mesh_levels"].sort(key=lambda x: float(x["mesh_hmax_mm"]), reverse=True)
        changes = []
        for coarse, fine in zip(progress["mesh_levels"][:-1], progress["mesh_levels"][1:]):
            for key in (2, -4):
                v0 = next(x["transfer_magnitude_at_0p9R"] for x in coarse["RF_transfer"] if x["m"] == key)
                v1 = next(x["transfer_magnitude_at_0p9R"] for x in fine["RF_transfer"] if x["m"] == key)
                changes.append({"metric": f"RF transfer m={key}", "coarse_hmax_mm": coarse["mesh_hmax_mm"],
                                "fine_hmax_mm": fine["mesh_hmax_mm"], "relative_change": abs(v1 - v0) / max(abs(v1), 1e-30)})
            za = coarse["DC_endcap_screen"]["particle_force_balance"]
            zb = fine["DC_endcap_screen"]["particle_force_balance"]
            if all(x.get("root_found") for x in za + zb):
                for i in range(2):
                    changes.append({"metric": f"DC equilibrium z particle {i+1}",
                      "coarse_hmax_mm": coarse["mesh_hmax_mm"], "fine_hmax_mm": fine["mesh_hmax_mm"],
                      "absolute_change_mm": abs(zb[i]["z_eq_mm"] - za[i]["z_eq_mm"]),
                      "relative_change": abs(zb[i]["z_eq_mm"] - za[i]["z_eq_mm"]) / max(abs(zb[i]["z_eq_mm"]), 1e-12)})
        progress["mesh_refinement_changes"] = changes
        # Require the last two adjacent refinements to pass.  RF transfers use
        # a relative tolerance; equilibrium positions use the specified
        # absolute tolerance (their relative error is ill-conditioned near z=0).
        last_two = changes[-8:]
        rf_ok = [x for x in last_two if x["metric"].startswith("RF transfer")]
        dc_ok = [x for x in last_two if x["metric"].startswith("DC equilibrium")]
        gate_passed = (len(last_two) == 8 and len(rf_ok) == 4 and len(dc_ok) == 4
                       and all(x["relative_change"] < .02 for x in rf_ok)
                       and all(x["absolute_change_mm"] < .1 for x in dc_ok))
        progress["convergence_gate"] = {
          "criterion": "last two consecutive refinements: <2% change in each of two RF transfers and <0.1 mm change in each of two force-balance positions",
          "evaluated_refinements": [[x["coarse_hmax_mm"], x["fine_hmax_mm"]] for x in last_two[::4]],
          "passed": bool(gate_passed)}
        outpath.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"mesh_hmax_mm": h_mm, "vertices": row["vertices"], "tetrahedra": row["tetrahedra"],
                          "RF_transfer": row["RF_transfer"], "DC": row["DC_endcap_screen"]["particle_force_balance"],
                          "convergence_gate": progress["convergence_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
