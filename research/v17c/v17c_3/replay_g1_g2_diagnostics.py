"""Recreate and retain full trajectories for the longest diagnostic runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from rf_dc_core import RESULTS, load_geometry, run_rk4

ROOT = RESULTS / "g1_g2_joint"
OUT = ROOT / "trajectories"
OUT.mkdir(parents=True, exist_ok=True)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def min_turns(row: dict[str, str]) -> float:
    return min(abs(float(row["turns1"])), abs(float(row["turns2"])))


def params(row: dict[str, str]) -> dict:
    alpha1 = float(row["rf_alpha1"])
    qom = float(row["q_over_m_ratio"])
    return {
        "alpha": [alpha1, alpha1*qom],
        "beta": [float(row["beta1"]), float(row["beta1"])*qom],
        "lambda": [float(row["lambda1"]), float(row["lambda2"])],
        "mass": [1.0, float(row["mass2_over_mass1"])],
        "charge": [1.0, float(row["charge2_over_charge1"])],
        "chi": float(row["chi"]),
    }


def replay(row: dict[str, str], dataset: str, steps: int) -> dict:
    name = row["geometry"]
    meta, modes, coeff = load_geometry(name)
    geom = {**meta, "modes": modes, "coeff": coeff}
    p = params(row)
    duration = float(row["periods"])
    # Re-run the same requested horizon as the source ensemble. Ending the
    # replay at its rounded reported duration can omit the boundary-crossing
    # step that set the original escape flag.
    horizon = 320 if dataset == "broad" else 600
    out = run_rk4(
        geom, [float(row["r1_0"]), float(row["r2_0"])],
        [float(row["theta1_0"]), float(row["theta2_0"])],
        [float(row["omega1_0"]), float(row["omega2_0"])],
        p["alpha"], p["beta"], p["lambda"], p["mass"], p["charge"], p["chi"],
        z0=[float(row["z1_0"]), float(row["z2_0"])],
        vz=[float(row["vz1_0"]), float(row["vz2_0"])],
        periods=horizon, steps_per_period=steps, save_per_period=1,
        phase0=float(row["phase0"]), escape_radius=.995, z_limit=1.0)
    path = OUT / f"{dataset}_{row['run_id']}.npz"
    np.savez_compressed(path, t=out["t"], state=out["state"],
                        parameters_json=np.asarray(json.dumps(p)),
                        initial_conditions_json=np.asarray(json.dumps({k: row[k] for k in row}))
                        )
    return {
        "dataset": dataset, "run_id": row["run_id"], "geometry": name,
        "source_duration_rf_periods": duration,
        "requested_horizon_rf_periods": horizon,
        "replayed_duration_rf_periods": out["duration_rf_periods"],
        "source_turns": [float(row["turns1"]), float(row["turns2"])],
        "replayed_turns": [x["turns"] for x in out["particles"]],
        "escaped": out["escaped"], "collided": out["collided"],
        "steps_per_period": steps, "trajectory_file": path.name,
    }


def main() -> None:
    broad = read(ROOT / "screen_320RF_n900_seed17032026" / "G1_G2_screening_all.csv")
    targeted = read(ROOT / "barrier_targeted_600RF_n600_seed17032027" / "G1_G2_barrier_targeted_all.csv")
    chosen: dict[tuple[str, str], tuple[dict[str, str], int]] = {}
    for row in broad:
        if row["escaped"] == "False" and row["collided"] == "False":
            chosen[("broad", row["run_id"])] = (row, 32)
    for geometry in ("G1", "G2"):
        group = [r for r in targeted if r["geometry"] == geometry]
        for row in (max(group, key=lambda r: float(r["periods"])),
                    max(group, key=min_turns)):
            chosen[("barrier", row["run_id"])] = (row, 48)
    result = [replay(row, dataset, steps) for (dataset, _), (row, steps) in sorted(chosen.items())]
    manifest = {"purpose": "retain full trajectories for all broad-screen horizon survivors and top barrier-targeted failures",
                "trajectory_count": len(result), "trajectories": result}
    (OUT / "trajectory_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
