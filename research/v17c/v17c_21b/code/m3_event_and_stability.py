"""Select candidate 1-D transitions and compute M3-local stability data."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "v17c" / "v17c_21a" / "code"))
sys.path.insert(0, str(ROOT / "research" / "v17c" / "v17c_21" / "code"))

import analytical_model as legacy  # noqa: E402
import m3_continuation as one_d  # noqa: E402
from m3_adiabatic_slow_flow import slow_flow_jacobian  # noqa: E402
from m3_harmonic_balance import HBConfig  # noqa: E402
from m3_system import M3PairSystem  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _root_vector(row: dict) -> np.ndarray:
    root = row["root"]
    return np.array([root["radii_R"][0], root["omega_over_Omega1"][0],
                     root["radii_R"][1], root["omega_over_Omega1"][1],
                     root["mean_z_R"][0], root["mean_z_R"][1]], dtype=float)


def _slow_state(row: dict) -> np.ndarray:
    root = row["root"]
    return np.array([root["radii_R"][0], 0.0, root["omega_over_Omega1"][0],
                     root["mean_z_R"][0], 0.0, root["radii_R"][1], 0.0,
                     root["omega_over_Omega1"][1], root["mean_z_R"][1], 0.0], dtype=float)


def _continuation_rows(data: dict) -> list[tuple[str, dict]]:
    rows = [("nominal", data["baseline"])]
    for direction, sequence in data.get("directions", {}).items():
        rows.extend((direction, row) for row in sequence if row.get("status") == "VERIFIED_ROOT")
    return rows


def _nearest_event_candidates(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    all_rows = _continuation_rows(payload)
    select: dict[tuple[str, float], dict] = {}
    # Analyze only the nominal root and the nearest computed roots bracketing
    # each observed spin-sign change.  For a parameter without a sign change,
    # retain only the two outer range endpoints.
    found_any = False
    for direction in ("lower", "upper"):
        seq = sorted([(row["factor_from_nominal"], row) for dr, row in all_rows
                      if dr == direction], key=lambda pair: pair[0])
        for (_, left), (_, right) in zip(seq[:-1], seq[1:]):
            wl, wr = left["root"]["omega_over_Omega1"], right["root"]["omega_over_Omega1"]
            if any(a == 0.0 or b == 0.0 or np.sign(a) != np.sign(b) for a, b in zip(wl, wr)):
                for row in (left, right):
                    key = (payload["parameter"], round(float(row["value"]), 14))
                    select[key] = row
                found_any = True
                break
    if not found_any:
        branch_rows = [(row["factor_from_nominal"], row) for _, row in all_rows if row is not payload["baseline"]]
        if branch_rows:
            for _, row in (min(branch_rows, key=lambda x: x[0]), max(branch_rows, key=lambda x: x[0])):
                key = (payload["parameter"], round(float(row["value"]), 14))
                select[key] = row
    key = ("nominal", 1.0)
    select[key] = payload["baseline"]
    return [{"parameter": name, "row": row} for (name, _), row in select.items()]


def _stationary_jacobian(data: dict, x: np.ndarray, cfg: HBConfig,
                         root_seed_xi: list[np.ndarray], hfactor: float) -> dict:
    system = M3PairSystem(data, cfg, xi_seed=root_seed_xi)
    steps0 = np.array([2e-4, 2e-6, 2e-4, 2e-6, 2e-4, 2e-4]) * hfactor
    J = np.empty((6, 6), dtype=float)
    f0, _ = system.balance(x)
    for j, h in enumerate(steps0):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        fp, _ = system.balance(xp)
        fm, _ = system.balance(xm)
        J[:, j] = (fp - fm) / (2 * h) * one_d.X_SCALE[j]
    s = np.linalg.svd(J, compute_uv=False)
    return {"finite_difference_factor": hfactor,
            "finite_difference_steps": steps0.tolist(),
            "scaled_state_scales": one_d.X_SCALE.tolist(),
            "scaled_stationary_balance_jacobian": J.tolist(),
            "singular_values": s.tolist(),
            "minimum_singular_value": float(s[-1]),
            "maximum_singular_value": float(s[0]),
            "condition_number": float(s[0] / s[-1]),
            "balance_evaluations": system.evaluations,
            "balance_at_root_inf_norm": float(np.max(np.abs(f0)))}


def main() -> None:
    data_files = [BASE / "results" / f"continuation_{name}.json"
                  for name in ("A", "B", "nu", "beta")]
    available = [p for p in data_files if p.exists()]
    if not available:
        raise RuntimeError("Run one-dimensional M3 continuation first.")
    selected: list[dict] = []
    for p in available:
        selected.extend(_nearest_event_candidates(p))
    # Deduplicate the common nominal root, retaining the parameter label that
    # motivated selection when it is a true non-nominal state.
    unique: dict[tuple, dict] = {}
    for item in selected:
        row = item["row"]
        if item["parameter"] == "nominal":
            key = ("nominal", 1.0)
        else:
            key = (item["parameter"], round(float(row["factor_from_nominal"]), 8))
        unique[key] = item

    base = legacy.load_inputs()
    cfg = HBConfig(96, 12, cutoff=0.05, oversample=1, method="newton_krylov",
                   tolerance=1e-7, equation_tolerance=5e-5, max_iter=30,
                   relaxation=0.5, history=6)
    with np.load(ROOT / "research" / "v17c" / "v17c_21a" / "results" / "xi_seed_192_16.npz") as f:
        xi_ref = [f["xi1"].copy(), f["xi2"].copy()]
    outpath = BASE / "results" / "selected_m3_stability.json"
    output = {
        "study": "V17C.21.2 M3 branch-local reduced-flow stability",
        "source_commit": "94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8",
        "grid": [96, 12, 12], "cutoff": 0.05,
        "stability_object": "10D adiabatic quasi-steady M3 slow-flow eigenvalues; not full RF Lyapunov/Floquet exponents",
        "stationary_balance_jacobian": "finite-difference scaled 6x6 Jacobian with inner HB re-solved at each perturbation",
        "selected_points": [], "status": "IN_PROGRESS",
        "finite_difference_policy": "one central-difference stationary Jacobian per selected root; extra h/2 and 2h checks at nominal only",
    }
    _write(outpath, output)
    for idx, item in enumerate(unique.values()):
        label, row = item["parameter"], item["row"]
        if label != "nominal":
            name = label
            data_i = one_d.change_field_parameter(base, name, one_d._parameter_value(base, name) *
                                                  float(row["factor_from_nominal"]))
        else:
            data_i = base
        root_x = _root_vector(row)
        state = _slow_state(row)
        started = time.perf_counter()
        result = {
            "parameter_label": label,
            "factor_from_nominal": 1.0 if label == "nominal" else float(row["factor_from_nominal"]),
            "status": "IN_PROGRESS", "classification": row.get("classification"),
            "root": row["root"], "stationary_jacobian_variants": [],
            "reduced_slowflow": None,
        }
        output["selected_points"].append(result)
        _write(outpath, output)
        try:
            hvariants = (1.0, 0.5, 2.0) if label == "nominal" else (1.0,)
            for hf in hvariants:
                result["stationary_jacobian_variants"].append(
                    _stationary_jacobian(data_i, root_x, cfg, xi_ref, hf))
                _write(outpath, output)
            steps = np.array([1e-4, 1e-4, 2e-6, 1e-4, 1e-4] * 2)
            reduced = slow_flow_jacobian(data_i, state, cfg, steps, steps)
            result["reduced_slowflow"] = {
                "eigenvalues_real_imag": reduced["eigenvalues_real_imag"],
                "max_real_eigenvalue": reduced["max_real_eigenvalue"],
                "all_modes_stable_in_reduced_flow": reduced["all_modes_stable_in_reduced_flow"],
                "equilibrium_acceleration_torque_vector": reduced["equilibrium_acceleration_torque_vector"],
                "dL_domega": reduced["dL_domega"],
                "inner_residuals": reduced["observables_inner_residual"],
                "approximation": reduced["approximation"],
            }
            result["status"] = "COMPLETE"
        except Exception as exc:
            result["status"] = "NUMERICALLY_UNRESOLVED"
            result["failure_type"] = type(exc).__name__
            result["message"] = str(exc)
        result["wall_time_s"] = time.perf_counter() - started
        _write(outpath, output)
        print(f"M3 stability selected {label} x{result['factor_from_nominal']:.4g} "
              f"status={result['status']}", flush=True)
    output["status"] = "COMPLETE_WITH_UNRESOLVED" if any(
        row["status"] != "COMPLETE" for row in output["selected_points"]
    ) else "COMPLETE"
    _write(outpath, output)


if __name__ == "__main__":
    main()
