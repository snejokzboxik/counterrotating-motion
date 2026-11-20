"""Check the integrated post-load energy/work diagnostic at several RK4 steps."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import nbody_dynamics as nb

HERE = Path(__file__).resolve().parent
OUT = HERE / "p1_energy_balance_validation.json"


def main() -> None:
    p, family, modes, coeff, qratio = nb.load_local15()
    protocols = nb.load_protocols()
    groups = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    radii = np.asarray([0.0002] * 3 + [0.00189] * 3, dtype=np.float64)
    initial, phase = nb.initialize_rings((0.45, 0.63), (0.0, 0.1), 0.0, groups=groups)
    periods, skip = 600, 200
    records = []
    for steps in (64, 96, 128, 192):
        result = nb.integrate(initial, phase, p, protocols["constant"], groups,
                              family, modes, coeff, qratio, radii, periods,
                              steps, skip, False, True)
        work = result[14]
        records.append({
            "steps_per_primary_period": steps,
            "completed_periods": int(result[9]),
            "energy_change": float(work[-1, 1] - work[0, 1]),
            "integrated_rf_explicit_work": float(work[-1, 2]),
            "integrated_drag_work": float(work[-1, 3]),
            "final_balance_residual": float(work[-1, 4]),
            "max_abs_balance_residual": float(np.max(np.abs(work[:, 4]))),
        })
    errors = [abs(r["final_balance_residual"]) for r in records]
    result = {
        "status": "passed",
        "case": "N=6 ordered rings from rest, constant local15, fixed field after skip",
        "initial_radii_R": [0.45, 0.63], "initial_z_centers_R": [0.0, 0.1],
        "periods": periods, "post_load_skip_periods": skip,
        "all_particles_unconstrained": True,
        "energy_definition": "sum_i 0.5*(m_i/m1)*|v_i|^2 + full RF + DC + all-pair Coulomb potential, normalized by m1*R^2*Omega1^2",
        "work_definition": "RK4 quadrature of explicit RF partial-time derivative and drag power; DC and pair potential are static after skip",
        "records": records,
        "residual_decreases_with_step_refinement": all(errors[i + 1] < errors[i] for i in range(len(errors) - 1)),
        "relative_residual_at_96_steps": float(errors[1] / max(abs(records[1]["integrated_rf_explicit_work"]), 1e-30)),
        "relative_residual_at_192_steps": float(errors[-1] / max(abs(records[-1]["integrated_rf_explicit_work"]), 1e-30)),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["residual_decreases_with_step_refinement"]:
        raise SystemExit("energy residual failed step-refinement check")


if __name__ == "__main__":
    main()
