"""No-fit averaged-vs-exact error map on the archived Local15 RF scan.

The exact points are reused from the fixed 3,000-period panel.  Only the
reduced equilibrium equations are reevaluated at sampled frequency ratios;
this script does not launch RF trajectories or optimize trap parameters.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "results/rotation_sign_scan"
INPUT = SCAN / "local15_sign_screen.json"
OUTPUT_JSON = ROOT / "results/moving_orbit_error_map_local15.json"
OUTPUT_CSV = ROOT / "results/figure_data/moving_orbit_error_map_local15.csv"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import moving_orbit_correction_local15 as moving  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_exact(rows: list[dict]) -> dict:
    if len(rows) != 24:
        raise ValueError(f"expected the fixed 24-start panel, got {len(rows)}")
    retained = [row for row in rows if not row["escaped"]]
    radii = np.array([float(row["rho_mean_last_500"]) for row in retained])
    omegas = np.array([float(row["omega_last_500"]) for row in retained])
    return {
        "n_starts": len(rows), "n_retained": len(retained),
        "n_escaped": len(rows) - len(retained),
        "mean_rho": float(np.mean(radii)), "sd_rho_across_retained": float(np.std(radii, ddof=1)),
        "median_rho": float(np.median(radii)),
        "mean_omega": float(np.mean(omegas)), "sd_omega_across_retained": float(np.std(omegas, ddof=1)),
        "median_omega": float(np.median(omegas)),
        "sign_counts": {"negative": int(np.sum(omegas < 0)),
                        "positive": int(np.sum(omegas > 0)),
                        "zero": int(np.sum(omegas == 0))},
    }


def epsilon_metrics(radius: float, omega: float, species: dict, nu: float) -> dict:
    alpha, lam = float(species["alpha"]), float(species["lambda"])
    modes = []
    for k, carrier, amplitude in ((2, 1.0, moving.PARAMS["A"]),
                                  (-4, nu, moving.PARAMS["B"])):
        _, jac = moving.planar_rf_force_jet(radius, alpha, k, amplitude)
        eps = float(np.linalg.norm(jac, ord=2) /
                    (carrier * np.sqrt(carrier**2 + lam**2)))
        modes.append({"k": k, "carrier": carrier, "epsilon_mu": eps,
                      "orbital_frequency_shift_ratio": float(abs(k * omega) / carrier)})
    return {"modes": modes,
            "epsilon_mu_max": max(item["epsilon_mu"] for item in modes),
            "orbital_frequency_shift_ratio_max": max(
                item["orbital_frequency_shift_ratio"] for item in modes)}


def model_predictions(result: dict, nu: float, exact: dict) -> list[dict]:
    stable_roots = [row for row in result["leading_roots"] if row["stable_reduced"]]
    if len(stable_roots) != 1:
        raise ValueError(f"expected one stable leading root, got {len(stable_roots)}")
    species_cfg = moving.PARAMS[result["species"]].copy()
    # Continuation roots are unique and stable throughout the sampled Local15 slice.
    root_row = min(stable_roots, key=lambda row: row["rho"])
    species_cfg["leading_rho"] = float(root_row["rho"])
    species_cfg["leading_omega"] = float(root_row["omega"])
    leading = {"rho": species_cfg["leading_rho"], "omega": species_cfg["leading_omega"]}
    moving.PARAMS["nu"] = float(nu)
    rho_kin, omega_kin, _, kin_success, kin_message = moving.solve_variant(
        species_cfg, np.asarray([leading["rho"], leading["omega"]]), False)
    rho_full, omega_full, _, full_success, full_message = moving.solve_variant(
        species_cfg, np.asarray([leading["rho"], leading["omega"]]), True)
    variants = [
        ("leading", leading, True, "analytic leading-order root"),
        ("kinematic_only", {"rho": rho_kin, "omega": omega_kin},
         kin_success, kin_message),
        ("full_local_response", {"rho": rho_full, "omega": omega_full},
         full_success, full_message),
    ]
    eps = epsilon_metrics(float(leading["rho"]), float(leading["omega"]), species_cfg, nu)
    predictions = []
    for name, prediction, solver_success, solver_message in variants:
        rho, omega = float(prediction["rho"]), float(prediction["omega"])
        predictions.append({
            "model": name, "rho_prediction": rho, "omega_prediction": omega,
            "solver_success": bool(solver_success), "solver_message": solver_message,
            "balance_residual_norm": (0.0 if name == "leading" else float(np.linalg.norm(
                moving.corrected_balance(np.asarray([rho, omega]), species_cfg,
                                         name == "full_local_response")))),
            "rho_error_percent_prediction_over_exact_mean": 100 * (rho / exact["mean_rho"] - 1),
            "omega_error_absolute_prediction_minus_exact_mean": omega - exact["mean_omega"],
            "omega_error_percent_abs_prediction_over_exact_mean": (
                None if abs(exact["mean_omega"]) < 5e-5 else
                100 * (abs(omega) / abs(exact["mean_omega"]) - 1)),
            "rotation_sign_agreement_with_exact_panel_mean": (
                int(np.sign(omega)) == int(np.sign(exact["mean_omega"]))),
            "rho_panel_spread_sd": exact["sd_rho_across_retained"],
            "omega_panel_spread_sd": exact["sd_omega_across_retained"],
        })
    return [{**eps, **row} for row in predictions]


def main() -> None:
    doc = json.loads(INPUT.read_text(encoding="utf-8"))
    expected_parameters = doc["parameters"]
    for field in ("A", "B", "beta"):
        if not np.isclose(float(expected_parameters[field]), moving.PARAMS[field],
                          rtol=0, atol=1e-15):
            raise ValueError(f"field parameter mismatch for {field}")
    for name in ("species1", "species2"):
        for field in ("alpha", "lambda"):
            if not np.isclose(float(expected_parameters["species"][name][field]),
                              moving.PARAMS[name][field], rtol=0, atol=1e-15):
                raise ValueError(f"species parameter mismatch for {name}.{field}")
    if doc["periods"] != 3000 or doc["measurement_window_periods"] != 500:
        raise ValueError("this map requires the common 3000-period / 500-period-window screen")
    if doc["panel_is_iid_probability_sample"] is not False:
        raise ValueError("the fixed comparison panel must not be described as an iid sample")
    grouped = {}
    for result in doc["results"]:
        key = (result["species"], float(result["nu"]))
        grouped[key] = result
    if len(grouped) != 10:
        raise ValueError(f"expected ten species-frequency nodes, got {len(grouped)}")
    nodes = []
    rows = []
    summary = {}
    for (species, nu), result in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        exact = summarize_exact(result["rows"])
        predictions = model_predictions(result, nu, exact)
        node = {"species": species, "nu": nu, "horizon_periods": doc["periods"],
                "measurement_window_periods": doc["measurement_window_periods"],
                "panel_sha256": doc["panel_sha256"], "panel_is_iid_probability_sample": False,
                "leading_stable_roots": [r for r in result["leading_roots"] if r["stable_reduced"]],
                "exact_panel": exact, "models": predictions}
        nodes.append(node)
        for prediction in predictions:
            rows.append({"species": species, "nu": nu,
                         "horizon_periods": doc["periods"],
                         "panel_n_starts": exact["n_starts"],
                         "panel_n_retained": exact["n_retained"],
                         "panel_n_escaped": exact["n_escaped"],
                         "exact_mean_rho": exact["mean_rho"],
                         "exact_sd_rho_across_retained": exact["sd_rho_across_retained"],
                         "exact_mean_omega": exact["mean_omega"],
                         "exact_sd_omega_across_retained": exact["sd_omega_across_retained"],
                         **prediction})

    for species in ("species1", "species2"):
        species_rows = [r for r in rows if r["species"] == species]
        summary[species] = {}
        for model in ("leading", "kinematic_only", "full_local_response"):
            selected = [r for r in species_rows if r["model"] == model]
            leading_by_nu = {r["nu"]: r for r in species_rows if r["model"] == "leading"}
            summary[species][model] = {
                "sampled_nodes": len(selected),
                "mean_abs_radius_error_percent": float(np.mean([
                    abs(r["rho_error_percent_prediction_over_exact_mean"]) for r in selected])),
                "mean_abs_frequency_error": float(np.mean([
                    abs(r["omega_error_absolute_prediction_minus_exact_mean"]) for r in selected])),
                "radius_error_improved_vs_leading_nodes": sum(
                    abs(r["rho_error_percent_prediction_over_exact_mean"]) <
                    abs(leading_by_nu[r["nu"]]["rho_error_percent_prediction_over_exact_mean"])
                    for r in selected),
                "frequency_error_improved_vs_leading_nodes": sum(
                    abs(r["omega_error_absolute_prediction_minus_exact_mean"]) <
                    abs(leading_by_nu[r["nu"]]["omega_error_absolute_prediction_minus_exact_mean"])
                    for r in selected),
                "rotation_sign_agreement_nodes": sum(
                    r["rotation_sign_agreement_with_exact_panel_mean"] for r in selected),
            }

    output = {
        "status": "completed_reduced_model_parameter_slice",
        "model_scope": "ideal k=+2,-4 Local15 RF; C=0; exact comparator reused from frozen 3000-period screen; no new RF integrations",
        "question": "How do leading and local moving-orbit reduced-model errors vary along the sampled frequency-ratio slice and its derived epsilon_mu values?",
        "varying_parameter": "nu=omega2/Omega1",
        "fixed_parameters": {k: moving.PARAMS[k] for k in ("A", "B", "beta")},
        "species_parameters": {key: moving.PARAMS[key]
                                for key in ("species1", "species2")},
        "exact_data_source": INPUT.relative_to(ROOT).as_posix(),
        "exact_data_sha256": sha256(INPUT),
        "panel_seed": doc["panel_seed"], "panel_sha256": doc["panel_sha256"],
        "panel_is_iid_probability_sample": False,
        "exact_horizon_periods": doc["periods"],
        "measurement_window_periods": doc["measurement_window_periods"],
        "epsilon_mu_definition": "||D f_k||_2 / (nu_k sqrt(nu_k^2+lambda^2)) at the leading reduced root",
        "epsilon_mu_is_derived_coordinate_not_independent_sweep": True,
        "nodes": nodes, "summary": summary,
        "limitations": [
            "frequency ratios are discrete archived nodes, not a continuous or full multidimensional error map",
            "all other field coefficients and species drag parameters are fixed; epsilon_mu is derived and not independently varied",
            "exact reference is the mean over a deterministic fixed panel of retained starts; reported SD is panel spread, not a confidence interval",
            "near-zero measured rotation is compared by signed absolute frequency error; relative frequency errors are suppressed below 5e-5",
            "ideal two-mode point-particle dynamics only: no Coulomb, MFS higher harmonics, or electrode geometry",
            "the full local-response equilibrium is a diagnostic reduced model, not a systematic next-order asymptotic formula",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"nodes": len(nodes), "rows": len(rows), "summary": summary},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
