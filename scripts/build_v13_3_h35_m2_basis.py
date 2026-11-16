"""Solve the missing h35 m=2 P4 basis on the saved V13 geometry mesh."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas/fem"
MESH=ROOT/"outputs/meshes/gmsh_35um/sector.msh"
M4=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h35_P4/sector_solution.npz"
sys.path.insert(0,str(ROOT/"src"))
from three_phase_trap.direct_fem import SectorSolution

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/"quadrupole_h35_P4_sector_solution.npz"
    module_path=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/run_quadrupole_fem.py"
    spec=importlib.util.spec_from_file_location("v13_quadrupole_fem_h35",module_path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    m4=SectorSolution.load(M4)
    if cache.exists():
        m2=SectorSolution.load(cache)
        diag=m2.diagnostics
    else:
        mesh,coeff,diag=mod.solve_quadrupole(MESH,4)
        m2=SectorSolution(mesh.p.T,mesh.t.T,coeff,4,{**diag,"source_mesh_sha256":sha(MESH)})
    if not (m2.points.shape==m4.points.shape and m2.triangles.shape==m4.triangles.shape and
            (m2.points==m4.points).all() and (m2.triangles==m4.triangles).all()):
        raise RuntimeError("h35 m2 and m4 P4 bases are not on identical sector mesh arrays")
    m2.save(cache)
    record={"h35_m2_solution":str(cache.relative_to(ROOT)),"h35_m4_solution":str(M4.relative_to(ROOT)),
            "sector_mesh":str(MESH.relative_to(ROOT)),"degree":4,
            "same_mesh_arrays_as_m4":True,"mesh_points":int(len(m2.points)),"sector_triangles":int(len(m2.triangles)),
            "sector_mesh_sha256":sha(MESH),"m2_solution_sha256":sha(cache),"m4_solution_sha256":sha(M4),
            "quadrupole_bloch_factor":[.5,3**.5/2],"boundary":"Re H2=cos(2 theta_j), Im H2=sin(2 theta_j); grounded outer conductor",
            "solver_diagnostics":diag}
    (OUT/"h35_m2_basis_manifest.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(record,indent=2))

if __name__=="__main__":main()
