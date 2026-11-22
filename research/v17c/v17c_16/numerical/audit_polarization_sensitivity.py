"""Audit polarization-run time-step refinement and save a compact comparison."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(tag: str) -> dict:
    path = ROOT/"results"/f"trajectory_validation_{tag}"/"summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["results"][0]


def main() -> None:
    p16 = load("mfs250_dep_eps26")
    p32 = load("mfs250_dep_eps26_s32")
    long = load("mfs1000_dep_eps26")
    rows = []
    for key in ("mean_radius_dimensionless_last_80pct",
                "mean_angular_velocity_dimensionless_last_80pct"):
        a, b = p16[key], p32[key]
        rows.append({
            "metric": key,
            "steps16": a,
            "steps32": b,
            "absolute_change_by_particle": [abs(x-y) for x, y in zip(a, b)],
            "relative_change_percent_by_particle": [
                (abs(x-y)/abs(y)*100.0 if y != 0.0 else None)
                for x, y in zip(a, b)],
        })
    result = {
        "comparison": "same nominal MFS/Oseen/Coulomb/epsilon_r=2.6 start; 250 primary RF periods",
        "step16_classification": p16["classification"],
        "step32_classification": p32["classification"],
        "step16_retained": p16["retained_contact_free_screen"],
        "step32_retained": p32["retained_contact_free_screen"],
        "step_refinement": rows,
        "long_1000_period_result": {
            "classification": long["classification"],
            "mean_radius_dimensionless_last_80pct":
                long["mean_radius_dimensionless_last_80pct"],
            "mean_angular_velocity_dimensionless_last_80pct":
                long["mean_angular_velocity_dimensionless_last_80pct"],
            "limitation": "single initial condition; constant real illustrative polarizability",
        },
    }
    out = ROOT/"results"/"polarization_sensitivity_audit.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n",
                   encoding="utf-8")
    print(out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
