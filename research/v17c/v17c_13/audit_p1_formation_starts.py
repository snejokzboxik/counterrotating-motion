"""Validate frozen N=6 start ensembles before launching their integrations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prepare_p1_crystal_ensembles import (
    COUNT, ENSEMBLES, OUT, PARTICLE_RADII_R, digest,
)


def main() -> None:
    results = {}
    errors = []
    for name in ENSEMBLES:
        with np.load(OUT / f"{name}_initials.npz") as data:
            initial = data["initial"].copy()
            phases = data["phases"].copy()
            ids = data["accepted_sobol_indices"].copy()
            groups = data["groups"].copy()
            radii = data["particle_radii_R"].copy()
        meta = json.loads((OUT / f"{name}_metadata.json").read_text(encoding="utf-8"))
        if initial.shape != (COUNT, 6, 6) or phases.shape != (COUNT, 2):
            errors.append(f"{name}: wrong array shape")
        if groups.tolist() != [0, 0, 0, 1, 1, 1]:
            errors.append(f"{name}: group assignment differs")
        if not np.array_equal(radii, PARTICLE_RADII_R):
            errors.append(f"{name}: finite radii differ")
        if digest(initial, phases) != meta["start_and_phase_sha256"]:
            errors.append(f"{name}: start hash mismatch")
        if len(np.unique(ids)) != COUNT:
            errors.append(f"{name}: duplicate Sobol IDs")
        if np.any(phases < 0.0) or np.any(phases >= 2.0 * np.pi):
            errors.append(f"{name}: phase outside [0,2pi)")
        minimum_start_separation = float("inf")
        for case in initial:
            for i in range(5):
                for j in range(i + 1, 6):
                    distance = float(np.linalg.norm(case[i, :3] - case[j, :3]))
                    minimum_start_separation = min(minimum_start_separation, distance)
                    if distance <= PARTICLE_RADII_R[i] + PARTICLE_RADII_R[j]:
                        errors.append(f"{name}: initial overlap")
        if name in ("ordered", "perturbed", "random_azimuth", "random_position_rest"):
            if not np.all(initial[:, :, 3:] == 0.0):
                errors.append(f"{name}: expected initial rest state")
        if name == "random_position_moving" and np.max(np.abs(initial[:, :, 3:])) == 0.0:
            errors.append(f"{name}: moving starts are zero")
        results[name] = {
            "count": int(initial.shape[0]), "seed": meta["seed"],
            "start_sha256": meta["start_and_phase_sha256"],
            "accepted_sobol_indices": ids.tolist(),
            "minimum_center_separation_R": minimum_start_separation,
            "initial_velocity_component_rms": float(np.sqrt(np.mean(initial[:, :, 3:] ** 2))),
            "phase_range_rad": [[float(phases[:, k].min()), float(phases[:, k].max())] for k in range(2)],
            "definition": meta["definition"],
        }
    output = {"status": "passed" if not errors else "failed",
              "n_per_ensemble": COUNT, "protocol": "P2_505",
              "overlap_filter_applied_before_dynamics": True,
              "failures_retained_in_later_outcome_denominator": True,
              "errors": errors, "ensembles": results}
    (OUT / "formation_start_audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "errors": errors,
                      "start_sha256": {k: v["start_sha256"] for k, v in results.items()}}, indent=2))
    if errors:
        raise SystemExit("start integrity audit failed")


if __name__ == "__main__":
    main()
