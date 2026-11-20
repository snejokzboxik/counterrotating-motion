"""Create frozen, reproducible small N=6 formation ensembles before any runs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
OUT = HERE / "p1_formation_ensembles"
GROUPS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
PARTICLE_RADII_R = np.asarray([0.0002] * 3 + [0.00189] * 3, dtype=np.float64)
COUNT = 16
SEED_BASE = 202613130
Z_CHOICES = np.asarray([0.0, 0.1, -0.1, 0.25, -0.25])
ENSEMBLES = ("ordered", "perturbed", "random_azimuth",
             "random_position_rest", "random_position_moving")


def digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode("ascii"))
        h.update(a.view(np.uint8))
    return h.hexdigest()


def choose_z(u: float) -> float:
    return float(Z_CHOICES[min(int(u * len(Z_CHOICES)), len(Z_CHOICES) - 1)])


def make_candidate(name: str, u: np.ndarray) -> np.ndarray:
    y = np.zeros((6, 6), dtype=np.float64)
    global_angle = 2.0 * math.pi * u[0]
    relative_angle = 2.0 * math.pi * u[1]
    z_centers = (choose_z(u[2]), choose_z(u[3]))

    if name in ("ordered", "perturbed", "random_azimuth"):
        for g, (radius, zc) in enumerate(zip((0.45, 0.63), z_centers)):
            base = global_angle + (relative_angle if g == 1 else 0.0)
            for j in range(3):
                i = 3 * g + j
                theta = base + 2.0 * math.pi * j / 3.0
                r = radius
                z = zc
                if name == "perturbed":
                    r += 0.05 * (u[4 + 6 * i] - 0.5)  # independent ±0.025 R
                    theta += math.radians(5.0) * ndtri(np.clip(u[4 + 6 * i + 1], 1e-9, 1.0 - 1e-9))
                    z += 0.05 * (u[4 + 6 * i + 2] - 0.5)  # independent ±0.025 R
                elif name == "random_azimuth":
                    theta = 2.0 * math.pi * u[4 + 6 * i]
                    z = zc
                y[i, 0] = r * math.cos(theta)
                y[i, 1] = r * math.sin(theta)
                y[i, 2] = z
    else:
        for i in range(6):
            ur, ut, uz = u[4 + 6 * i:4 + 6 * i + 3]
            r = math.sqrt(0.03**2 + ur * (0.90**2 - 0.03**2))
            theta = 2.0 * math.pi * ut
            y[i, 0] = r * math.cos(theta)
            y[i, 1] = r * math.sin(theta)
            y[i, 2] = (2.0 * uz - 1.0) * 0.5
            if name == "random_position_moving":
                vv = np.clip(u[4 + 6 * i + 3:4 + 6 * i + 6], 1e-9, 1.0 - 1e-9)
                y[i, 3:6] = 0.05 * ndtri(vv)  # same component sigma as V17C.9 wide
    return y


def is_initially_nonoverlapping(y: np.ndarray) -> bool:
    for i in range(5):
        for j in range(i + 1, 6):
            if np.linalg.norm(y[i, :3] - y[j, :3]) <= PARTICLE_RADII_R[i] + PARTICLE_RADII_R[j]:
                return False
    return True


def generate(name: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    for power in range(9, 15):
        points = qmc.Sobol(d=42, scramble=True, seed=seed).random_base2(power)
        starts, phases, accepted = [], [], []
        rejected = 0
        for index, u in enumerate(points):
            y = make_candidate(name, u)
            if not is_initially_nonoverlapping(y):
                rejected += 1
                continue
            starts.append(y)
            phases.append(2.0 * math.pi * u[40:42])
            accepted.append(index)
            if len(starts) == COUNT:
                return np.asarray(starts), np.asarray(phases), np.asarray(accepted, dtype=np.int64), rejected, power
    raise RuntimeError(f"could not prepare {COUNT} nonoverlapping starts for {name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"status": "prepared", "count_per_ensemble": COUNT,
               "protocol_for_all_runs": "P2_505", "seeds": {}, "start_sha256": {},
               "ensembles": {}}
    definitions = {
        "ordered": "Exact regular triangles at r/R=0.45 and 0.63, zero velocities; independent global/relative orientation, z-centers sampled from {0, ±0.1, ±0.25}R.",
        "perturbed": "Same triangle scaffold with independent radial offsets ±0.025R, Gaussian angular offsets σ=5 degrees, z offsets ±0.025R, zero velocities.",
        "random_azimuth": "Three particles per group on r/R=0.45 and 0.63 with independent uniform azimuths, z-centers from {0, ±0.1, ±0.25}R, zero velocities.",
        "random_position_rest": "All six independently uniform in area over 0.03≤r/R≤0.90 and uniform in −0.5≤z/R≤0.5; zero velocities.",
        "random_position_moving": "Same positions as random_position_rest; independent Gaussian Cartesian velocity components σ=0.05 in dimensionless units, matching V17C.9 wide.",
    }
    for idx, name in enumerate(ENSEMBLES):
        seed = SEED_BASE + idx
        path = OUT / f"{name}_initials.npz"
        meta_path = OUT / f"{name}_metadata.json"
        if path.exists() and meta_path.exists():
            with np.load(path) as data:
                initial = data["initial"].copy()
                phases = data["phases"].copy()
                accepted_indices = data["accepted_sobol_indices"].copy()
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if initial.shape != (COUNT, 6, 6) or phases.shape != (COUNT, 2):
                raise ValueError(f"bad frozen start shape for {name}")
            if digest(initial, phases) != metadata["start_and_phase_sha256"]:
                raise ValueError(f"checksum mismatch for frozen starts {name}")
            summary["ensembles"][name] = metadata
        else:
            initial, phases, accepted_indices, rejected, power = generate(name, seed)
            sha = digest(initial, phases)
            metadata = {
                "ensemble": name, "count": COUNT, "seed": seed,
                "generator": "SciPy scrambled Sobol, d=42, first 16 valid non-overlapping starts",
                "sobol_base2_power": power, "overlap_rejections_before_16_accepts": rejected,
                "accepted_sobol_candidate_indices": accepted_indices.tolist(),
                "definition": definitions[name],
                "phase_sampling": "two independent uniform phases in [0,2π), fixed per run",
                "physical_overlap_filter": "reject only if any center distance <= sum of the prescribed finite radii; rejected candidates are counted in metadata before dynamics",
                "protocol": "P2_505, identical for every initial condition; continuous carriers",
                "steps_per_period": 96, "post_load_periods": 15000,
                "start_and_phase_sha256": sha,
            }
            np.savez_compressed(path, initial=initial, phases=phases,
                                accepted_sobol_indices=accepted_indices,
                                groups=GROUPS, particle_radii_R=PARTICLE_RADII_R)
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            summary["ensembles"][name] = metadata
        summary["seeds"][name] = seed
        summary["start_sha256"][name] = summary["ensembles"][name]["start_and_phase_sha256"]
        print(name, "n=16", "seed", seed, "sha256", summary["start_sha256"][name])
    (OUT / "ensemble_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
