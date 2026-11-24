"""Auditable leading-order radius-eliminated counterrotation criterion.

This is M0 of the V2 hierarchy, not the historical V21 M3 closure.
It also maps the published V21B one-parameter M3 continuation onto the
criterion for a sparse, non-independent validation (not a parameter sweep).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def radial_root(A: float, B: float, nu: float, beta: float, alpha: float,
                lam: float, c2: complex = 1.0,
                cm4: complex = 1.0) -> float | None:
    """M1 zero-orbit-speed radial root, including finite lambda/DC terms."""
    k1, k2 = 1 + alpha * beta, nu**2 + alpha * beta
    d1, d2 = k1**2 + lam**2, k2**2 + (lam * nu)**2
    numerator = alpha * beta - 4 * alpha**2 * A**2 * abs(c2)**2 * k1 / d1
    denominator = 48 * alpha**2 * B**2 * abs(cm4)**2 * k2 / d2
    if denominator <= 0 or numerator <= 0:
        return None
    return (numerator / denominator) ** 0.25


def critical_radius(A: float, B: float, nu: float, alpha: float,
                    lam: float, beta: float, c2: complex = 1.0,
                    cm4: complex = 1.0) -> float:
    """M1 zero slow RF torque radius at zero mean orbital speed."""
    d1 = (1 + alpha * beta) ** 2 + lam**2
    d2 = (nu**2 + alpha * beta) ** 2 + (lam * nu) ** 2
    r4 = A**2 * abs(c2)**2 * d2 / (12 * B**2 * abs(cm4)**2 * nu * d1)
    return r4**0.25


def exact_ratio_s(A: float, B: float, nu: float, beta: float,
                  alpha: float, lam: float, c2: complex = 1.0,
                  cm4: complex = 1.0) -> float | None:
    r = radial_root(A, B, nu, beta, alpha, lam, c2, cm4)
    if r is None:
        return None
    return (r / critical_radius(A, B, nu, alpha, lam, beta, c2, cm4))**4


def simplified_s(Pi: float, nu: float, alpha: float) -> float:
    return (Pi / alpha - 1) / nu


def leading_cr_condition(Pi: float, nu: float, alpha2: float) -> bool:
    """M0 CR(+,-) sign test plus positive-root constraints, under stated assumptions."""
    return (nu > 0 and 0 < alpha2 < 1 and
            max(1.0, alpha2 * (1 + nu)) < Pi < 1 + nu)


def _sign(x: float, tol: float = 1e-12) -> str:
    return "+" if x > tol else "-" if x < -tol else "0/near-zero"


def evaluate() -> dict:
    pars = json.loads((ROOT / "data/m3_nominal_torque_decomposition.json").read_text())
    A, B, nu, beta, mass_ratio, C, lam1, lam2 = pars["parameters"]
    alpha = [p["alpha"] for p in pars["particles"]]
    Pi = beta / (4 * A**2)
    lo, hi = max(1.0, alpha[1] * (1 + nu)), 1 + nu
    exact_rows = []
    for i in range(2):
        s = exact_ratio_s(A, B, nu, beta, alpha[i], (lam1, lam2)[i])
        s0 = simplified_s(Pi, nu, alpha[i])
        exact_rows.append({"particle": i + 1, "alpha": alpha[i],
                           "S_M1_finite_lambda_DC": s,
                           "S_M0_weak_damping": s0,
                           "M1_positive_radius": radial_root(
                               A, B, nu, beta, alpha[i], (lam1, lam2)[i]) is not None,
                           "M1_zero_speed_slow_torque_sign": _sign(1 - s),
                           "M1_radius_R": radial_root(
                               A, B, nu, beta, alpha[i], (lam1, lam2)[i])})

    rows = []
    path = ROOT / "research/v17c/v17c_21b/tables/m3_1d_continuation.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            par, value = row["parameter"], float(row["value"])
            aa, bb, nn, be = A, B, nu, beta
            if par == "A": aa = value
            elif par == "B": bb = value
            elif par == "nu": nn = value
            elif par == "beta": be = value
            pi = be / (4 * aa**2)
            s0 = [simplified_s(pi, nn, a) for a in alpha]
            pred = ("CR_PLUS_MINUS" if 0 < s0[0] < 1 and s0[1] > 1
                    else "M0_OTHER_OR_NO_RADIAL_ROOT")
            rows.append({"varied_parameter": par, "value": value,
                         "Pi": pi, "nu": nn, "alpha2": alpha[1],
                         "S1_M0": s0[0], "S2_M0": s0[1],
                         "M0_CR_plus_minus": pred == "CR_PLUS_MINUS",
                         "M3_classification": row["classification"],
                         "M3_status": row["status"],
                         "M3_omega_over_Omega1": [float(row["omega1_over_Omega1"]),
                                                   float(row["omega2_over_Omega1"])],
                         "M0_torque_signs": [_sign(1 - s0[0]), _sign(1 - s0[1])],
                         "M0_sign_match_M3": (
                             _sign(1 - s0[0]) == _sign(float(row["omega1_over_Omega1"]), 1e-5) and
                             _sign(1 - s0[1]) == _sign(float(row["omega2_over_Omega1"]), 1e-5)
                         )})
    matched = sum(r["M0_sign_match_M3"] for r in rows)
    return {
        "definition": "Pi=beta/(4 A^2); alpha1=1; alpha2=(q2/m2)/(q1/m1)",
        "source_data": "data/m3_nominal_torque_decomposition.json; V21B m3_1d_continuation.csv",
        "nominal": {"A": A, "B": B, "nu": nu, "beta": beta,
                    "alpha2": alpha[1], "Pi": Pi,
                    "M0_CR_interval_open": [lo, hi],
                    "M0_predicts_CR_plus_minus": leading_cr_condition(Pi, nu, alpha[1]),
                    "particles": exact_rows},
        "sparse_M3_continuation_comparison": {
            "rows": rows, "total": len(rows), "sign_matches_omega_near_zero_tolerance_1e-5": matched,
            "warning": "This reuses published branch points and is not an independent M3 scan. M0 predicts signs from zero-speed torque and assumes positive effective angular damping; a point near S=1 is intrinsically sensitive."},
        "assumptions": ["c_(+2)=c_(-4)=1 normalized leading MFS coefficients",
                        "A,B,beta>0; nu>0; alpha_i>0; lambda_i small",
                        "zero slow orbit speed for the torque threshold",
                        "Coulomb, radial drag, higher harmonics, finite orbit speed omitted",
                        "positive effective angular damping needed to transfer torque sign to stationary omega sign"]}


if __name__ == "__main__":
    out = ROOT / "data/three_parameter_criterion_validation.json"
    out.write_text(json.dumps(evaluate(), indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
