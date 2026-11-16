"""Half-step convergence audit for ideal/FEM status mismatches on selected mass rays."""
from __future__ import annotations
import csv,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
RUN=BASE/"fem_nonlinear_gain_h50"


def read(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def write(path,rows):
    if not rows:return
    fields=[]
    for row in rows:fields.extend(k for k in row if k not in fields)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    import sys
    sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
    import run_v13_3_fem_controls as fem
    fem.RTOL,fem.ATOL,fem.MAX_STEP=1e-10,1e-12,2*math.pi/128
    fem.initialize(50)
    points={r["candidate_id"]:{**r,"alpha":float(r["alpha"]),"beta":float(r["beta"]),"kappa":float(r["kappa"])}
            for r in read(BASE/"fem_nonlinear_gain_points.csv")}
    pairs=read(RUN/"ideal_fem_paired_trajectories.csv")
    mismatches=[r for r in pairs if r["ideal_status"]!=r["FEM_h50_status"]]
    trajectories=read(RUN/"trajectory_results.csv")
    start_index={(r["candidate_id"],r["model"],float(r["tau0_deg"]),int(r["state_i"])):r
                 for r in trajectories if r["field"]=="ideal"}
    refined=[]
    for pair in mismatches:
        key=(pair["candidate_id"],pair["model"],float(pair["phase_deg"]),int(pair["state_i"]))
        start=start_index[key];point=points[pair["candidate_id"]]
        initial=[float(start[k]) for k in ("x0_rho","y0_rho","vx0_rho_tau","vy0_rho_tau")]
        result=fem.simulate_segment(point,pair["model"],float(pair["mu"]),
                                    math.radians(float(pair["phase_deg"])),0.,int(pair["state_i"]),
                                    initial,initial,0,100,0.)
        refined.append({"candidate_id":pair["candidate_id"],"model":pair["model"],
                        "mu":float(pair["mu"]),"phase_deg":float(pair["phase_deg"]),
                        "state_i":int(pair["state_i"]),"ideal_status":pair["ideal_status"],
                        "original_FEM_h50_status":pair["FEM_h50_status"],
                        "refined_FEM_h50_status":result["status"],
                        "original_FEM_exit_RF":pair["FEM_exit_RF"],
                        "refined_FEM_exit_RF":result["exit_periods"],
                        "original_FEM_rho_max":pair["FEM_rho_max"],
                        "refined_FEM_rho_max":result["rho_max_cumulative"],
                        "nfev_refined":result["nfev"],"rtol":fem.RTOL,"atol":fem.ATOL,
                        "max_step_tau":fem.MAX_STEP,
                        "status_reproduced":result["status"]==pair["FEM_h50_status"]})
    output=BASE/"fem_nonlinear_gain_h50_step_refinement.csv"
    write(output,refined)
    metadata={"source_run":"fem_nonlinear_gain_h50","source_mismatches":len(mismatches),
              "refined_rows":len(refined),"mesh_um":50,"field":"full P4 FEM superposition",
              "integrator":"DOP853","rtol":fem.RTOL,"atol":fem.ATOL,
              "max_step_tau":fem.MAX_STEP,"from_initial_state":True,
              "mismatch_reproduced":sum(r["status_reproduced"] for r in refined),
              "purpose":"separate solver-step sensitivity from ideal-vs-FEM field differences"}
    (BASE/"fem_nonlinear_gain_h50_step_refinement.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":main()
