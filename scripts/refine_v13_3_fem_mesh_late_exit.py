"""Repeat the mesh-sensitive 300-cycle Q trajectory at tighter FEM step size."""
from __future__ import annotations
import csv,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
import run_v13_3_fem_controls as fem

def main():
    with (BASE/"fem_control_points.csv").open(newline="",encoding="utf-8") as f:
        point=next(r for r in csv.DictReader(f) if r["candidate_id"]=="ro_gain_positive")
    point={**point,"alpha":float(point["alpha"]),"beta":float(point["beta"]),"kappa":float(point["kappa"])}
    starts=__import__("numpy").genfromtxt(ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1/ideal_pilot/initial_conditions.csv",delimiter=",",names=True)
    initial=[float(starts[k][5]) for k in ("x0_rho","y0_rho","vx0_rho_per_tau","vy0_rho_per_tau")]
    outputs=[];basis={}
    for mesh_um in (35,50):
        fem.initialize(mesh_um=mesh_um);fem.RTOL,fem.ATOL=1e-10,1e-12;fem.MAX_STEP=2*math.pi/128
        row=fem.simulate_segment(point,"Q",1.,0.,0.,5,initial,initial,0,300,0.)
        row["mesh_um"]=mesh_um;row["integrator_refined"]=True;outputs.append(row)
        mesh_path=ROOT/f"outputs/meshes/gmsh_{mesh_um}um/trap.msh"
        f4=ROOT/f"outputs/abc_validation/direct_fem_validation/v2/h{mesh_um}_P4/sector_solution.npz"
        f2=(ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1/fem/quadrupole_h50_P4_sector_solution.npz" if mesh_um==50
            else BASE/"fem/quadrupole_h35_P4_sector_solution.npz")
        for p in (mesh_path,f4,f2):basis[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    out=BASE/"fem_mesh_late_exit_refinement.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(outputs[0]));w.writeheader();w.writerows(outputs)
    metadata={"candidate_id":"ro_gain_positive","model":"Q","tau0_deg":0,"state_i":5,"horizon_RF":300,
              "common_initial_state_rho_tau":initial,"aperture_rho":1.0,"mesh_um":[35,50],"field_sha256":basis,
              "integrator":{"method":"DOP853","rtol":1e-10,"atol":1e-12,"max_step_tau":2*math.pi/128},
              "purpose":"resolve the sole h35/h50 status difference at 300 RF cycles; any persistent mesh-dependent status remains unresolved in ensemble bounds"}
    (BASE/"fem_mesh_late_exit_refinement.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"results":[{k:r.get(k) for k in ("mesh_um","status","exit_periods","rho_max_cumulative","nfev")} for r in outputs],"metadata":metadata},indent=2))
if __name__=="__main__":main()
