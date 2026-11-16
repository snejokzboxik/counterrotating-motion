"""Compare selected V13.3 ideal trajectories with exact V13.2 grid nodes."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from three_phase_trap.mass_filter_v13_3 import acceleration

BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13"
V13_2 = BASE / "v13_2_nonlinear_stability"
OUT = BASE / "v13_3_three_parameter_atlas"
CASES = (
    (0.018, 0.145),
    (0.026571428571428572, 0.1572222222222222),
    (0.026571428571428572, 0.17555555555555555),
)
MODELS = (("Q", 0.0, 0.0), ("RO", -0.05, -0.05))
PHASES = (0.0, math.pi/2)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def integrate(row: dict) -> dict:
    alpha, beta = float(row["alpha"]), float(row["beta"])
    eta, phase = float(row["eta"]), float(row["phase_rad"])
    kappa = eta*beta
    state = np.array([float(row[k]) for k in ("x0", "y0", "vx0", "vy0")])

    def rhs(t, y):
        ax, ay = acceleration(y[0], y[1], t, kappa, beta, alpha, delta=0.0, tau0=phase)
        return np.array([y[2], y[3], ax, ay])

    def event(t, y):
        return math.hypot(y[0], y[1])-1.0

    event.terminal = True
    event.direction = 1.0
    sol = solve_ivp(rhs, (0, 2*math.pi*100), state, method="DOP853",
                    rtol=2e-10, atol=2e-12, max_step=2*math.pi/64,
                    events=event, dense_output=True)
    if not sol.success:
        status, exitp = "unresolved", math.nan
    elif len(sol.t_events[0]):
        status, exitp = "escaped", float(sol.t_events[0][0]/(2*math.pi))
    else:
        status, exitp = "retained", math.nan
    rho_max = float(np.max(np.hypot(sol.y[0], sol.y[1])))
    if len(sol.t)>1:
        mid=sol.sol((sol.t[:-1]+sol.t[1:])/2)
        rho_max=max(rho_max,float(np.max(np.hypot(mid[0],mid[1]))))
    if status=="escaped": rho_max=max(1.0,rho_max)
    return {"alpha":alpha,"beta":beta,"kappa":kappa,"eta":eta,
            "phase_rad":phase,"phase_deg":math.degrees(phase),
            "model":row["model"],"index":int(row["index"]),
            "status_new":status,"status_old":row["status_100"],
            "exit_periods_new":exitp,
            "exit_periods_old":float(row["exit_periods_100"]) if row["exit_periods_100"] else math.nan,
            "rho_max_new":rho_max,"rho_max_old":float(row["max_rho_100"]),
            "status_match":status==row["status_100"],
            "exit_period_difference":(exitp-float(row["exit_periods_100"]))
                if status=="escaped" and row["exit_periods_100"] else math.nan,
            "rho_max_difference":rho_max-float(row["max_rho_100"]),
            "nfev_new":int(sol.nfev)}


def main() -> None:
    old = read_csv(V13_2/"stage_b_trajectory_results.csv")
    selected=[]
    for alpha,beta in CASES:
        for model,eta,eta_match in MODELS:
            for phase in PHASES:
                for index in range(8):
                    match=[r for r in old if abs(float(r["alpha"])-alpha)<1e-12
                           and abs(float(r["beta"])-beta)<1e-12
                           and r["model"]==model
                           and abs(float(r["eta"])-eta_match)<1e-12
                           and abs(float(r["phase_rad"])-phase)<1e-12
                           and int(r["index"])==index]
                    if len(match)!=1:
                        raise RuntimeError(f"Expected one V13.2 control, found {len(match)} at {alpha=},{beta=},{model=},{phase=},{index=}")
                    selected.append({**match[0],"eta":eta})
    results=[integrate(row) for row in selected]
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"v13_2_exact_node_controls.csv"
    fields=list(results[0])
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(results)
    summary={"n_controls":len(results),"status_mismatches":sum(not r["status_match"] for r in results),
             "max_abs_exit_period_difference":max((abs(r["exit_period_difference"]) for r in results
                  if np.isfinite(r["exit_period_difference"])),default=0.0),
             "max_abs_rho_max_difference":max(abs(r["rho_max_difference"]) for r in results),
             "nodes":[{"alpha":a,"beta":b} for a,b in CASES],
             "models":["Q","Q+RO eta=-0.05"],"phases_deg":[0,90],
             "states_per_group":8,"N_RF":100,
             "meaning":"Exact V13.2 parameter nodes; they are external controls, not literal overlaps with the new V13.3 tensor grid."}
    (OUT/"v13_2_exact_node_controls_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
