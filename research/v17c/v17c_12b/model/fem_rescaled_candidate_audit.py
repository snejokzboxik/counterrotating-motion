"""Independent 2-D FEM RF-surface cross-check for frequency-rescaled candidates."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "rescaled_legacy_candidate_validation"
FEM_PATH = ROOT / "model" / "fem_crosscheck.py"
spec = importlib.util.spec_from_file_location("v17c_fem_crosscheck", FEM_PATH)
fem = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fem)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    meta = json.loads((OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    mfs = json.loads((OUT / "mfs_field_audit.json").read_text(encoding="utf-8"))
    candidates = {c["id"]: c["physical"] for c in meta["candidates"]}
    mfs_max = {cid: max(x["rf_only_max_surface_field_V_m_all"] for x in data["mfs_convergence"])
               for cid, data in mfs["results"].items()}
    rows = []
    # The smallest mesh repeats the earlier FEM convergence ladder; all candidates
    # use the same mesh and the same two independently solved complex RF bases.
    for hmin in (0.010, 0.005, 0.0025):
        xy, tri, D, rod_ids = fem.build_mesh(hmin, 0.08)
        mesh = fem.MeshTri(xy, tri)
        f1x, f1y = fem.solve_basis(mesh, xy, tri, D, rod_ids, 2)
        f2x, f2y = fem.solve_basis(mesh, xy, tri, D, rod_ids, -4)
        adj = fem.boundary_adjacent_triangles(xy, tri, hmin)
        candidate_values = {}
        for cid, p in candidates.items():
            scale = 1.0 / (p["trap_radius_m"] / 0.95)
            peak = fem.support_max(f1x[adj] * scale, f1y[adj] * scale,
                                   f2x[adj] * scale, f2y[adj] * scale,
                                   p["U1_peak_electrode_V"], p["U2_peak_electrode_V"])
            candidate_values[cid] = {"rf_only_surface_max_V_m": peak,
                                     "mfs_rf_only_surface_max_V_m": mfs_max[cid],
                                     "relative_to_candidate_mfs": peak / mfs_max[cid] - 1.0}
        rows.append({"hmin_over_D": hmin, "nodes": int(xy.shape[1]), "triangles": int(tri.shape[1]),
                     "surface_adjacent_triangles": int(len(adj)), "boundary_dofs": int(len(D)),
                     "rods_vs_shield_gap_elements_min_approx": 0.05 / hmin,
                     "candidate_fields": candidate_values})
        del mesh, xy, tri, D, rod_ids, f1x, f1y, f2x, f2y, adj
    output = {
        "study": "independent linear P1 FEM RF surface-field cross-check for three rescaled candidates",
        "geometry": "24 circular rods a/D=0.05, center radius D, grounded circular shield 1.10D; infinite 2-D rods",
        "phase_maximum": "support-function maximum over the two independent RF phases",
        "dc": "Excluded; ideal quadrupole coefficient is not a solved electrode-boundary potential.",
        "source_hashes": {"fem": sha(FEM_PATH), "field_model": sha(ROOT / "model" / "field_model.json"),
                          "campaign_meta": sha(OUT / "campaign_meta.json")},
        "refinement": rows,
        "limitations": [
            "P1 FEM gradients are piecewise constant and tend to under-resolve conductor-surface maxima.",
            "No finite axial ends, DC segmentation, feedthroughs, roughness, corona, or gas-breakdown model.",
            "Field maxima are screening results, not a certified electrical-strength margin.",
        ],
    }
    path = OUT / "fem_field_audit.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    compact = [{"hmin_over_D": row["hmin_over_D"], "nodes": row["nodes"], "triangles": row["triangles"],
                "fields": {cid: round(x["rf_only_surface_max_V_m"] / 1e6, 6) for cid, x in row["candidate_fields"].items()}}
               for row in rows]
    print(json.dumps({"saved": str(path), "mesh_results": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
