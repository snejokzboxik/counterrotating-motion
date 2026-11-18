"""Independent P1 FEM validation of the 12-rod six-phase RF field.

The finite-element solve is two-dimensional (infinite rods), with one complex
Dirichlet solve for each phase assignment. It recovers the full signed angular
spectrum and compares field-weighted coefficients and surface fields to the
archived MFS calculation.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.tri as mtri
import numpy as np
import triangle as tr
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v17c_3"))
from rf_dc_core import pseudopotential
ARCHIVE_SCAN = HERE / "input" / "archive_contents" / "v17c_six_phase" / "scan.py"
RESULTS = HERE / "results" / "fem_sixphase"
RESULTS.mkdir(parents=True, exist_ok=True)

PHASES = {
    "ADCFEBADCFEB": [0, 3, 2, 5, 4, 1] * 2,
    "ABCDEFABCDEF": [0, 1, 2, 3, 4, 5] * 2,
    "ABCABCABCABC": [0, 2, 4] * 4,
    "G1_ABCABC": [0, 2, 4] * 2,
    "G2_ABCABC": [0, 2, 4] * 2,
}
GEOMETRIES = {
    "ADCFEBADCFEB": {"N": 12, "a_D": .05, "gap_D": .05, "R_D": .95},
    "ABCDEFABCDEF": {"N": 12, "a_D": .05, "gap_D": .05, "R_D": .95},
    "ABCABCABCABC": {"N": 12, "a_D": .05, "gap_D": .05, "R_D": .95},
    "G1_ABCABC": {"N": 6, "a_D": .05, "gap_D": .05, "R_D": .95},
    "G2_ABCABC": {"N": 6, "a_D": .01, "gap_D": .02, "R_D": .99},
}
MESH_LEVELS = ((.032, 64), (.020, 112), (.012, 160))
REFINEMENT_LEVELS = {
    "ADCFEBADCFEB": ((.008, 240),),
    "G2_ABCABC": ((.008, 240), (.006, 320)),
}
SPECTRUM_K = 72


def load_archive_scan():
    spec = importlib.util.spec_from_file_location("v17c4_mfs_scan", ARCHIVE_SCAN)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def build_mesh(N: int, a: float, gap: float, hmax: float, nbound: int):
    B = 1.0 + a + gap
    vertices, segments, markers, holes = [], [], [], []
    outerN = max(200, int(math.ceil(2 * math.pi * B / hmax)))
    theta = np.arange(outerN) * 2 * np.pi / outerN
    vertices.extend(np.column_stack((B * np.cos(theta), B * np.sin(theta))).tolist())
    for j in range(outerN):
        segments.append([j, (j + 1) % outerN])
        markers.append(N + 1)
    for j in range(N):
        th = 2 * np.pi * j / N
        center = np.array([np.cos(th), np.sin(th)])
        phi = np.arange(nbound) * 2 * np.pi / nbound
        offset = len(vertices)
        vertices.extend((center + a * np.column_stack((np.cos(phi), np.sin(phi)))).tolist())
        for k in range(nbound):
            segments.append([offset + k, offset + (k + 1) % nbound])
            markers.append(j + 1)
        holes.append(center.tolist())
    pslg = {"vertices": np.asarray(vertices, float),
            "segments": np.asarray(segments, np.int32),
            "segment_markers": np.asarray(markers, np.int32)[:, None],
            "holes": np.asarray(holes, float)}
    maxarea = .5 * hmax * hmax
    mesh = tr.triangulate(pslg, f"pq28a{maxarea:.15f}Q")
    return mesh, B


def fem_geometry(name: str, levels=MESH_LEVELS):
    meta = GEOMETRIES[name]
    N, a, gap, R = int(meta["N"]), meta["a_D"], meta["gap_D"], meta["R_D"]
    phase_idx = np.asarray(PHASES[name], int)
    # e^(+i phi) is the phasor used with exp(-i Omega*t), matching
    # V_j(t)=cos(Omega*t-phi_j).
    electrode_voltage = np.exp(1j * phase_idx * np.pi / 3.0)
    surface_angles = np.arange(384) * 2 * np.pi / 384
    mfs_surface, mfs_boundary_error, mfs_coeff = mfs_reference(
        N, a, 1.0 + a + gap, phase_idx, SPECTRUM_K, p=64,
        surface_angles=surface_angles)
    all_levels = []
    for hmax, nbound in levels:
        mesh, B = build_mesh(N, a, gap, hmax, nbound)
        points, tri = mesh["vertices"], mesh["triangles"]
        seg, markers = mesh["segments"], mesh["segment_markers"].ravel()
        p = points[tri]
        xx, yy = p[:, :, 0], p[:, :, 1]
        det = (xx[:, 1] - xx[:, 0]) * (yy[:, 2] - yy[:, 0]) - (xx[:, 2] - xx[:, 0]) * (yy[:, 1] - yy[:, 0])
        area = np.abs(det) * .5
        bx = np.column_stack((yy[:, 1] - yy[:, 2], yy[:, 2] - yy[:, 0], yy[:, 0] - yy[:, 1])) / (2 * area[:, None])
        by = np.column_stack((xx[:, 2] - xx[:, 1], xx[:, 0] - xx[:, 2], xx[:, 1] - xx[:, 0])) / (2 * area[:, None])
        local = area[:, None, None] * (bx[:, :, None] * bx[:, None, :] + by[:, :, None] * by[:, None, :])
        rows = np.repeat(tri, 3, axis=1).ravel()
        cols = np.tile(tri, (1, 3)).ravel()
        stiffness = coo_matrix((local.ravel(), (rows, cols)), shape=(len(points), len(points))).tocsr()
        boundary_nodes = np.unique(seg.ravel())
        free = np.setdiff1d(np.arange(len(points)), boundary_nodes)
        boundary_value = np.zeros(len(points), np.complex128)
        electrode_nodes = []
        for j in range(N):
            nodes = np.unique(seg[markers == j + 1].ravel())
            electrode_nodes.append(nodes)
            boundary_value[nodes] = electrode_voltage[j]
        # Outer shield remains grounded (zero).
        lu = splu(stiffness[free][:, free].tocsc())
        rhs = -stiffness[free][:, boundary_nodes] @ boundary_value[boundary_nodes]
        solution = boundary_value.copy()
        solution[free] = lu.solve(rhs.real) + 1j * lu.solve(rhs.imag)

        gradx = np.sum(solution[tri] * bx, axis=1)
        grady = np.sum(solution[tri] * by, axis=1)
        field = np.column_stack((-gradx, -grady))
        # Associate each actual electrode boundary segment with its adjacent
        # P1 element and report the normal flux at that edge. Selecting every
        # triangle that merely touches a boundary node overestimates the
        # surface maximum by including interior fan elements.
        edge_to_triangle = {}
        for ti, nodes in enumerate(tri):
            for u, v in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
                edge_to_triangle[tuple(sorted((int(u), int(v))))] = ti
        surface_samples = [[] for _ in range(N)]
        for edge, mark in zip(seg, markers):
            if mark < 1 or mark > N:
                continue
            ti = edge_to_triangle.get(tuple(sorted((int(edge[0]), int(edge[1])))))
            if ti is None:
                raise RuntimeError("Could not locate a boundary-adjacent FEM triangle")
            midpoint = .5 * (points[edge[0]] + points[edge[1]])
            center = np.asarray([np.cos(2*np.pi*(mark-1)/N), np.sin(2*np.pi*(mark-1)/N)])
            normal = midpoint - center
            normal /= np.linalg.norm(normal)
            enormal = field[ti, 0] * normal[0] + field[ti, 1] * normal[1]
            surface_samples[mark-1].append(abs(enormal))
        surface_by_electrode = [float(np.max(values)) for values in surface_samples]

        triang = mtri.Triangulation(points[:, 0], points[:, 1], tri)
        interp_re = mtri.LinearTriInterpolator(triang, solution.real)
        interp_im = mtri.LinearTriInterpolator(triang, solution.imag)
        theta = np.arange(2048) * 2 * np.pi / 2048
        radii = np.asarray([.20, .30, .40, .45, .50, .60, .70, .80, .88, .92])
        modes = np.asarray([*range(-SPECTRUM_K, 0), *range(1, SPECTRUM_K + 1)], int)
        projections = {int(k): [] for k in modes}
        for s in radii:
            r = s * R
            vals = (np.asarray(interp_re(r * np.cos(theta), r * np.sin(theta)))
                    + 1j * np.asarray(interp_im(r * np.cos(theta), r * np.sin(theta))))
            if np.any(np.ma.getmaskarray(vals)):
                raise RuntimeError(f"FEM interpolation left the domain at s={s}")
            vals = np.asarray(vals)
            for k in modes:
                projections[int(k)].append(np.mean(vals * np.exp(-1j * k * theta)))
        coeff = {}
        for k in modes:
            n = abs(int(k))
            rr = radii**n
            proj = np.asarray(projections[int(k)])
            coeff[int(k)] = complex(np.vdot(rr, proj) / np.dot(rr, rr))

        # Dense angular samples on each exact circular electrode boundary.
        finite_boundary_error = float(max(abs(solution[nodes] - electrode_voltage[j]).max()
                                          for j, nodes in enumerate(electrode_nodes)))
        level = {
            "mesh_h_D": hmax, "boundary_points_per_electrode": nbound,
            "vertices": int(len(points)), "triangles": int(len(tri)),
            "circle_sagitta_D": float(a * (1 - np.cos(np.pi / nbound))),
            "boundary_dirichlet_max_residual": finite_boundary_error,
            "fem_surface_field_max_by_electrode_U0_over_D": surface_by_electrode,
            "fem_surface_field_global_max_U0_over_D": max(surface_by_electrode),
            "mfs_surface_field_max_by_electrode_U0_over_D": mfs_surface,
            "mfs_surface_field_global_max_U0_over_D": max(mfs_surface),
            "mfs_boundary_residual": mfs_boundary_error,
            "coefficients": coeff,
            "mfs_coefficients": mfs_coeff,
            "field_weighted_FEM_vs_MFS": field_weighted_errors(coeff, mfs_coeff, R),
            "pseudopotential_torque_roots_FEM": torque_roots(enforce_pi_symmetry(coeff)),
            "pseudopotential_torque_roots_MFS": torque_roots(enforce_pi_symmetry(mfs_coeff)),
        }
        all_levels.append(level)
        print(name, "h", hmax, "tri", len(tri),
              "Emax/FEM", round(level["fem_surface_field_global_max_U0_over_D"], 6),
              "Emax/MFS", round(max(mfs_surface), 6),
              "A_k", round(abs(coeff.get({"ADCFEBADCFEB": -4, "ABCDEFABCDEF": -10,
                                         "ABCABCABCABC": -8, "G1_ABCABC": -4,
                                         "G2_ABCABC": -4}[name], 0)
                             / coeff.get({"ADCFEBADCFEB": 2, "ABCDEFABCDEF": 2,
                                          "ABCABCABCABC": 4, "G1_ABCABC": 2,
                                          "G2_ABCABC": 2}[name], 1)), 9), flush=True)
    return all_levels


def mfs_reference(N, a, B, phase_idx, K, p=64, surface_angles=None):
    theta = np.arange(N) * 2 * np.pi / N
    centers = np.exp(1j * theta)
    angles = theta[:, None] + np.arange(p) * 2 * np.pi / p
    z = (centers[:, None] + a * np.exp(1j * angles)).ravel()
    src = (centers[:, None] + .65 * a * np.exp(1j * angles)).ravel()
    zph = np.exp(1j * np.asarray(phase_idx) * np.pi / 3)
    v = np.repeat(zph, p)

    def green(target):
        return np.log(np.abs(B * B - target[:, None] * src[None, :].conj()) /
                      (B * np.abs(target[:, None] - src[None, :])))

    charges = np.linalg.solve(green(z), v)
    # Residual on a staggered set of boundary collocation points.
    stagger = (centers[:, None] + a * np.exp(1j * (angles + np.pi / p))).ravel()
    boundary_error = float(np.max(np.abs(green(stagger) @ charges - np.repeat(zph, p))))
    k = np.asarray([*range(-K, 0), *range(1, K + 1)], int)
    s = src
    transfer = np.asarray([(s ** (-abs(int(i))) - (s.conj() / B**2) ** abs(int(i)))
                           * (1 - a) ** abs(int(i)) / (2 * abs(int(i))) for i in k])
    transfer[k < 0] = transfer[k < 0].conj()
    coeff = {int(i): complex(vv) for i, vv in zip(k, transfer @ charges)}
    if surface_angles is None:
        surface_angles = np.arange(384) * 2 * np.pi / 384
    surface_max = []
    for j, th in enumerate(theta):
        target = np.exp(1j * (th + surface_angles)) * a + centers[j]
        sbar = src.conj()[None, :]
        den1 = B * B - target[:, None] * sbar
        diff = target[:, None] - src[None, :]
        gx = np.real(-sbar / den1 - 1.0 / diff)
        gy = np.real(-1j * sbar / den1 - 1j / diff)
        ex = -(gx @ charges)
        ey = -(gy @ charges)
        surface_max.append(float(np.max(np.sqrt(np.abs(ex)**2 + np.abs(ey)**2))))
    return surface_max, boundary_error, coeff


def field_weighted_errors(fem: dict[int, complex], mfs: dict[int, complex], R: float):
    rows = []
    for s in (.20, .30, .40, .45, .50, .60, .70, .80, .88, .92):
        keys = set(fem) & set(mfs)
        denom = sum(abs(k * mfs[k]) * s**(abs(k) - 1) for k in keys)
        delta = sum(abs(k * (fem[k] - mfs[k])) * s**(abs(k) - 1) for k in keys)
        rows.append({"r_over_R": s, "relative_spectrum_field_L1_error": float(delta / max(denom, 1e-30))})
    return rows


def torque_value(coeff: dict[int, complex], r: float) -> float:
    return float(sum(np.sign(k) * abs(k)**2 * (abs(k) - 1) * abs(c)**2 * r**(2*abs(k)-2)
                     for k, c in coeff.items()))


def enforce_pi_symmetry(coeff: dict[int, complex]) -> dict[int, complex]:
    # Every voltage pattern here repeats after six rods (12-rod cases) or
    # after three rods (6-rod cases), so the exact phasor is pi-periodic.
    return {k: (c if abs(k) % 2 == 0 else 0j) for k, c in coeff.items()}


def torque_roots(coeff: dict[int, complex]):
    grid = np.linspace(.03, .98, 2000)
    values = [torque_value(coeff, x) for x in grid]
    return [float(brentq(lambda r: torque_value(coeff, r), lo, hi))
            for lo, hi, flo, fhi in zip(grid[:-1], grid[1:], values[:-1], values[1:])
            if flo * fhi < 0]


def main():
    results = {"scope": "2D infinite rods, grounded circular screen; complex phasor exp(-i Omega t)",
               "phase_phasor_convention": "Vhat_j=exp(+i phi_j) for V_j(t)=cos(Omega*t-phi_j)",
               "field_units": "U0/D; coefficients are Phi_hat/U0 in powers of r/R",
               "mesh_levels": list(MESH_LEVELS), "geometries": {}}
    for name in PHASES:
        levels = MESH_LEVELS + REFINEMENT_LEVELS.get(name, ())
        results["geometries"][name] = {
            "metadata": GEOMETRIES[name], "phase_indices_in_60deg_units": PHASES[name],
            "mesh_levels": fem_geometry(name, levels),
        }
    # The leading and counter-rotating coefficients and phases are explicitly
    # retained for easy comparison with the MFS files.
    for name, entry in results["geometries"].items():
        fine = entry["mesh_levels"][-1]
        cf = fine["coefficients"]
        lead_counter = {
            "ADCFEBADCFEB": (2, -4), "ABCDEFABCDEF": (2, -10),
            "ABCABCABCABC": (4, -8), "G1_ABCABC": (2, -4), "G2_ABCABC": (2, -4),
        }[name]
        lead_k, counter_k = lead_counter
        entry["leading_mode"] = lead_k
        entry["counter_rotating_mode"] = counter_k
        entry["fine_mesh_counter_to_leading_amplitude_ratio"] = float(abs(cf.get(counter_k, 0) / cf.get(lead_k, 1)))
        entry["fine_mesh_leading_and_counter_coefficients"] = {
            str(k): [float(cf[k].real), float(cf[k].imag)] for k in (2, -4, 8, -10, -16, 14, 4, -8) if k in cf
        }
        entry["adjacent_voltage_max_U0"] = float(max(
            abs(np.exp(1j * np.pi * (PHASES[name][(j+1) % len(PHASES[name])] - PHASES[name][j]) / 3) - 1)
            for j in range(len(PHASES[name]))))
        for level in entry["mesh_levels"]:
            level["coefficients"] = [[int(k), float(v.real), float(v.imag)]
                                      for k, v in sorted(level["coefficients"].items())]
            level["mfs_coefficients"] = [[int(k), float(v.real), float(v.imag)]
                                          for k, v in sorted(level["mfs_coefficients"].items())]
    out = RESULTS / "G1_G2_12electrode_6phase_P1_FEM.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
