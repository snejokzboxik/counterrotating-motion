"""Leading two-mode radial localization and archived M3 balance audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def leading_radial_force(r: float, alpha: float, lam: float, beta: float,
                         amplitude: float, frequency: float, n: int,
                         coefficient: complex = 1.0) -> float:
    """Zero-orbit-frequency M2 mean radial RF force for one traveling mode."""
    stiffness = frequency**2 + alpha * beta
    denom = stiffness**2 + (lam * frequency)**2
    return (-alpha**2 * n**2 * (n-1) * abs(amplitude * coefficient)**2
            * r**(2*n-3) * stiffness / denom)


def leading_radius(A: float, B: float, nu: float, beta: float,
                   alpha: float, lam: float,
                   c2: complex = 1.0, cm4: complex = 1.0) -> float | None:
    """Positive root of alpha*beta*r+F_RF,+2+F_RF,-4=0.

    Coulomb, drag radial mean, centrifugal acceleration, orbital motion and
    higher spatial harmonics are omitted; this is not a stability criterion.
    """
    k1 = 1 + alpha * beta
    k2 = nu**2 + alpha * beta
    d1 = k1**2 + lam**2
    d2 = k2**2 + (lam * nu)**2
    numerator = alpha*beta - 4*alpha**2*A*A*abs(c2)**2*k1/d1
    denominator = 48*alpha**2*B*B*abs(cm4)**2*k2/d2
    if denominator <= 0 or numerator <= 0:
        return None
    return float((numerator / denominator)**0.25)


def ring_coulomb(r1: float, r2: float, z1: float, z2: float,
                 C: float, mass_s: float, n: int = 16384) -> np.ndarray:
    angle = 2*np.pi*np.arange(n)/n
    co = np.cos(angle)
    dz = z1-z2
    d3 = (r1*r1+r2*r2-2*r1*r2*co+dz*dz)**1.5
    return np.asarray([
        C*np.mean((r1-r2*co)/d3),
        mass_s*C*np.mean((r2-r1*co)/d3),
    ])


def evaluate() -> dict:
    torques = json.loads((ROOT / "data/m3_nominal_torque_decomposition.json").read_text())
    archive = json.loads((ROOT / "research/v17c/v17c_21a/results/full_grid_campaign.json").read_text())
    held = json.loads((ROOT / "research/v17c/v17c_21a/results/heldout_m3_resolved.json").read_text())
    full_rf = json.loads((ROOT / "research/v17c/v17c_20b/results/refine12_240k_corrected_drag_summary.json").read_text())
    p = torques["parameters"]
    A, B, nu, beta, mass_s, C, _, _ = p
    root = archive["stages"][-1]["root"]
    fc = ring_coulomb(*root["radii_R"], *root["mean_z_R"], C, mass_s)
    rows = []
    for i, part in enumerate(torques["particles"]):
        r, omega, alpha, lam = (part["r_over_R"], part["omega_over_Omega1"],
                                part["alpha"], part["lambda"])
        radial = {
            "centrifugal": r*omega*omega,
            "DC_outward": alpha*beta*r,
            "RF_inward": part["m3_RF_force_rotating_xy"][0],
            "drag": part["m3_drag_force_rotating_xyz"][0],
            "Coulomb": float(fc[i]),
        }
        estimate = leading_radius(A, B, nu, beta, alpha, lam)
        rows.append({
            "particle": i+1, "M3_radius_R": r, "leading_radius_R": estimate,
            "leading_vs_M3_error_percent": 100*(estimate/r-1) if estimate else None,
            "full_RF_RK4_240k_radius_R": full_rf["late_mean_radius_R"][i],
            "leading_vs_full_RF_error_percent": 100*(estimate/full_rf["late_mean_radius_R"][i]-1) if estimate else None,
            "M3_radial_terms": radial,
            "M3_radial_sum": sum(radial.values()),
            "M3_RF_per_mode_radial": [
                {"k": x["k"], "mean_radial_force": x["mean_radial_force"]}
                for x in part["m3_modes"]],
            "leading_RF_plus2": leading_radial_force(r, alpha, lam, beta, A, 1.0, 2),
            "leading_RF_minus4": leading_radial_force(r, alpha, lam, beta, B, nu, 4),
        })
    holdouts = []
    for row in held["rows"]:
        assert row["status"] == "VERIFIED_ROOT"
        ph = list(p)
        index = {"A": 0, "B": 1, "nu": 2, "beta": 3}[row["parameter"]]
        ph[index] *= row["factor"]
        estimates = [leading_radius(ph[0], ph[1], ph[2], ph[3],
                                    torques["particles"][i]["alpha"], ph[6+i])
                     for i in range(2)]
        exact = row["root"]["radii_R"]
        holdouts.append({"parameter": row["parameter"], "factor": row["factor"],
                         "M3_grid": row["grid"], "M3_radii_R": exact,
                         "leading_radii_R": estimates,
                         "leading_vs_M3_error_percent": [100*(a/b-1) if a else None
                                                         for a, b in zip(estimates, exact)],
                         "leading_vs_full_RF_error_percent": [
                             100*(a/b-1) if a else None
                             for a, b in zip(estimates, row["RK4_reference"]["mean_radius_R"])],
                         "full_RF_RK4_reference": row["RK4_reference"]})
    return {
        "formula": "zero-orbital-frequency leading +2/-4 radial balance; no fitted coefficients",
        "nominal_M3_grid": archive["stages"][-1]["grid"],
        "nominal": rows, "published_holdouts": holdouts,
        "limits": "Coulomb, drag radial mean, centrifugal and higher harmonics omitted in leading radius; holdouts were published before this calculation. No attractor or global localization proof.",
    }


if __name__ == "__main__":
    result = evaluate()
    out = ROOT / "data/radial_localization_validation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for row in result["nominal"]:
        print(row["particle"], row["M3_radius_R"], row["leading_radius_R"],
              row["leading_vs_M3_error_percent"], row["M3_radial_sum"])
    for row in result["published_holdouts"]:
        print(row["parameter"], row["leading_vs_M3_error_percent"])
