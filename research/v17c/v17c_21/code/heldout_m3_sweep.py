"""Independent M3 predictions for the three remaining predeclared RK4 hold-outs."""
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
import analytical_model as model  # noqa: E402


def solve_case(spec):
    key, factor = spec
    data = model.load_inputs()
    if key in ("A", "B"):
        family = 0 if key == "A" else 1
        data["modes"] = [
            replace(m, amplitude=m.amplitude*factor) if m.family == family else m
            for m in data["modes"]
        ]
    elif key == "nu":
        data["modes"] = [
            replace(m, frequency=m.frequency*factor) if m.family == 1 else m
            for m in data["modes"]
        ]
    elif key == "beta":
        data["beta"] *= factor
    else:
        raise ValueError(key)
    seed = model.solve_m12(data, list(range(len(data["modes"]))))
    pred = model.solve_m3(data, seed, (96, 12))
    return key, factor, pred


def main():
    specs = [("B", 0.95), ("nu", 1.02), ("beta", 1.05)]
    # Each worker uses modest Fourier grids; three independent solves fit well
    # within the current workstation's memory budget.
    with ProcessPoolExecutor(max_workers=3) as pool:
        solved = list(pool.map(solve_case, specs))
    rk = json.loads((BASE/"results"/"heldout_validation.json").read_text(encoding="utf-8"))
    rows = []
    for key, factor, pred in solved:
        ref = next(row for row in rk["rows"]
                   if row["parameter"] == key and row["factor"] == factor)
        rows.append({
            "heldout_parameter": key, "factor": factor,
            "M3_radii_R": pred["radii_R"], "M3_mean_z_R": pred["mean_z_R"],
            "M3_frequency_Hz": pred["orbital_frequency_Hz"],
            "RK4_radii_R": ref["mean_radius_R"], "RK4_frequency_Hz": ref["frequency_Hz"],
            "radius_relative_error_percent": [
                100*(a-b)/b for a,b in zip(pred["radii_R"],ref["mean_radius_R"])],
            "frequency_relative_error_percent": [
                100*(a-b)/abs(b) for a,b in zip(pred["orbital_frequency_Hz"],ref["frequency_Hz"])],
            "solver_residual_scaled": pred["solver_residual_scaled"],
            "phase_grid": pred["phase_grid"],
            "HB_status": [
                {"converged":p["converged"],"relative_fixed_point_residual":p["relative_fixed_point_residual"]}
                for p in pred["inner_harmonic_balance"]["particles"]],
            "qualification":"Local finite-horizon validation point; not a global accuracy guarantee.",
        })
    a_case = json.loads((BASE/"results"/"heldout_M3_A105.json").read_text(encoding="utf-8"))
    a_pred = a_case["analytical_M3"]
    a_ref = a_case["full_RK4_reference"]
    rows.insert(0, {
        "heldout_parameter":"A", "factor":1.05,
        "M3_radii_R":a_pred["radii_R"], "M3_mean_z_R":a_pred["mean_z_R"],
        "M3_frequency_Hz":a_pred["frequency_Hz"],
        "RK4_radii_R":a_ref["mean_radius_R"], "RK4_frequency_Hz":a_ref["frequency_Hz"],
        "radius_relative_error_percent":a_case["radius_relative_error_percent"],
        "frequency_relative_error_percent":a_case["frequency_relative_error_percent"],
        "solver_residual_scaled":a_pred["solver_residual_scaled"],
        "phase_grid":a_pred["torus_grid"], "HB_status":a_pred["particle_hb"],
        "qualification":a_case["qualification"],
    })
    out = {
        "study":"V17C.21 M3 held-out checks",
        "method":"Predeclared full-RHS RK4 holds; M3 roots evaluated without fitted coefficients.",
        "workers":3,
        "rows":rows,
    }
    (BASE/"results"/"heldout_M3_all.json").write_text(
        json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
