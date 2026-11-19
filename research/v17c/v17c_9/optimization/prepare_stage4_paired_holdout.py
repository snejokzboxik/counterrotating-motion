"""Create frozen, previously unused N=128 ensembles for paired candidate52/48 holdout."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "stage4_candidate48" / "holdout_starts"
SEEDS = {"rest": 20261150, "weak": 20261151, "wide": 20261152}


def main():
    import sys
    sys.path.insert(0, str(HERE.parent / "baseline"))
    from basin_scan import make_initials

    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for group, seed in SEEDS.items():
        path = OUT / f"{group}_N128_seed{seed}.npz"
        initial, phases = make_initials(group, 128, seed)
        np.savez_compressed(path, initial=initial, phases=phases, seed=seed,
                            ensemble=group, generation="scrambled Sobol fresh holdout",
                            radial_distribution="uniform in transverse area, r in [0.03,0.90]R",
                            phase_distribution="independent uniform [0,2pi)")
        records.append({"ensemble": group, "seed": seed, "count": 128,
                        "file": path.name, "initial_velocity_rule":
                        "zero for rest; Gaussian dimensionless sigma 0.01 weak / 0.05 wide"})
    meta = {"purpose": "paired holdout for stage4 config52 vs stage2_speed candidate48",
            "seeds": records, "shared_starts_and_RF_phases": True,
            "validation_horizon_periods": 15000, "steps_per_period": 64,
            "finite_spheres_a_over_R": [0.0002, 0.00189]}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
