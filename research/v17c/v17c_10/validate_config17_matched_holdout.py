"""Audit config17's strict matched holdout and compare it pairwise to config52."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V17C9 = HERE.parent / "v17c_9"
STAGE4 = V17C9 / "optimization" / "stage4_candidate48"
RUN52 = STAGE4 / "holdout_runs"
RUN17 = HERE / "candidate17_holdout"
STARTS = STAGE4 / "holdout_starts"
GROUPS = ("rest", "weak", "wide")
RUN_TAG = "pairedholdout20261150"
N = 128
PERIODS = 15000
STEPS = 64
TRANSIENT = 200
RMIN_CLASS = 0.02
CONFIG52_TAG = "cfg52_{group}_matched_holdout_none_chi0-0_a0.0002-0.00189_pairedholdout20261150_128_15000p_64s"
CONFIG17_TAG = "cfg17_{group}_matched_holdout_none_chi0-0_a0.0002-0.00189_pairedholdout20261150_128_15000p_64s"
PARAMETER_NAMES = ("A", "B", "nu", "beta", "s_m1_over_m2", "C", "lambda1", "lambda2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keyed = {int(row["index"]): row for row in rows}
    if len(rows) != N or len(keyed) != N or set(keyed) != set(range(N)):
        raise ValueError(f"expected 128 unique indexed rows in {path}; got {len(rows)}/{len(keyed)}")
    return keyed


def strict_cr_reproduced(row):
    windows = np.asarray(row["window_turns"], dtype=float)
    turns = np.asarray(row["turns"], dtype=float)
    rmin = np.asarray(row["r_min"], dtype=float)
    signs = np.where(np.abs(windows) < 0.05, 0.0, np.sign(windows))
    stable_sign = bool(np.all(signs != 0.0))
    opposite = bool(np.all(signs[:, 0] * signs[:, 1] < 0.0))
    enough_turns = bool(np.all(np.abs(turns) >= 10.0))
    return bool(
        row["retained"]
        and not row["escaped"]
        and not row["singular_unresolved"]
        and int(row["completed_periods"]) >= PERIODS
        and np.all(rmin >= RMIN_CLASS)
        and enough_turns
        and stable_sign
        and opposite
    )


def main():
    p17_doc = json.loads((HERE / "sobol_leader_config17.json").read_text(encoding="utf-8"))
    vals17 = p17_doc["parameters"]
    p17 = [float(vals17[key]) for key in PARAMETER_NAMES]
    q17 = float(vals17["q2_over_q1"])
    holdout_manifest = json.loads((STAGE4 / "PAIRED_HOLDOUT_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_hashes = {item["path"]: item["sha256"] for item in holdout_manifest["files"]}
    validation = {
        "source_start_hashes_match_manifest": True,
        "candidate17_rows_match_frozen_starts": True,
        "candidate52_rows_match_frozen_starts": True,
        "candidate17_strict_labels_reproduce": True,
        "candidate52_strict_labels_reproduce": True,
        "candidate17_metadata_matches_training_leader": True,
        "candidate52_metadata_matches_original_holdout": True,
    }
    by_group = {}

    for group_idx, group in enumerate(GROUPS):
        start_path = STARTS / f"{group}_N128_seed{20261150 + group_idx}.npz"
        start_rel = f"optimization/stage4_candidate48/holdout_starts/{start_path.name}"
        if manifest_hashes.get(start_rel) != sha256(start_path):
            validation["source_start_hashes_match_manifest"] = False
        with np.load(start_path, allow_pickle=False) as frozen:
            initial = frozen["initial"].astype(np.float64)
            phases = frozen["phases"].astype(np.float64)
            seed = int(frozen["seed"])
        if seed != 20261150 + group_idx or initial.shape != (N, 12) or phases.shape != (N, 2):
            validation["source_start_hashes_match_manifest"] = False

        tag52 = CONFIG52_TAG.format(group=group)
        tag17 = CONFIG17_TAG.format(group=group)
        meta52 = json.loads((RUN52 / f"{tag52}.meta.json").read_text(encoding="utf-8"))
        meta17_path = RUN17 / f"{tag17}.meta.json"
        meta17 = json.loads(meta17_path.read_text(encoding="utf-8"))
        rows52 = load_rows(RUN52 / f"{tag52}.jsonl")
        rows17 = load_rows(RUN17 / f"{tag17}.jsonl")

        validation["candidate17_metadata_matches_training_leader"] &= bool(
            meta17["config_id"] == 17
            and np.array_equal(np.asarray(meta17["parameters"], dtype=float), np.asarray(p17))
            and meta17["qratio"] == q17
            and meta17["ensemble"] == group
            and meta17["n"] == N
            and meta17["periods"] == PERIODS
            and meta17["steps"] == STEPS
            and meta17["transient"] == TRANSIENT
            and meta17["hydro_mode"] == "none"
            and meta17["a1_over_R"] == 0.0002
            and meta17["a2_over_R"] == 0.00189
            and meta17["frozen_start_sha256"] == sha256(start_path)
            and meta17["runner_sha256"] == sha256(HERE / "run_config17_matched_holdout.py")
        )
        validation["candidate52_metadata_matches_original_holdout"] &= bool(
            meta52["config_id"] == 52
            and meta52["ensemble"] == group
            and meta52["n"] == N
            and meta52["periods"] == PERIODS
            and meta52["steps"] == STEPS
            and meta52["transient"] == TRANSIENT
            and meta52["hydro_mode"] == "none"
            and meta52["a1_over_R"] == 0.0002
            and meta52["a2_over_R"] == 0.00189
        )

        cats17 = Counter(row["category"] for row in rows17.values())
        cats52 = Counter(row["category"] for row in rows52.values())
        transitions = Counter()
        timing_by_category = {}
        for i in range(N):
            r17, r52 = rows17[i], rows52[i]
            match17 = (
                np.array_equal(np.asarray(r17["initial"], dtype=float), initial[i])
                and np.array_equal(np.asarray(r17["phase"], dtype=float), phases[i])
            )
            match52 = (
                np.array_equal(np.asarray(r52["initial"], dtype=float), initial[i])
                and np.array_equal(np.asarray(r52["phase"], dtype=float), phases[i])
            )
            validation["candidate17_rows_match_frozen_starts"] &= bool(match17)
            validation["candidate52_rows_match_frozen_starts"] &= bool(match52)

            strict17 = strict_cr_reproduced(r17)
            strict52 = strict_cr_reproduced(r52)
            validation["candidate17_strict_labels_reproduce"] &= bool(strict17 == (r17["category"] == "CR_screen"))
            validation["candidate52_strict_labels_reproduce"] &= bool(strict52 == (r52["category"] == "CR_screen"))
            transitions[f"{r52['category']} -> {r17['category']}"] += 1
            if not strict17:
                timing_by_category.setdefault(r17["category"], []).append(int(r17["completed_periods"]))

        s52 = int(cats52.get("CR_screen", 0))
        s17 = int(cats17.get("CR_screen", 0))
        both = sum(rows52[i]["category"] == "CR_screen" and rows17[i]["category"] == "CR_screen" for i in range(N))
        only52 = sum(rows52[i]["category"] == "CR_screen" and rows17[i]["category"] != "CR_screen" for i in range(N))
        only17 = sum(rows52[i]["category"] != "CR_screen" and rows17[i]["category"] == "CR_screen" for i in range(N))
        neither = N - both - only52 - only17
        by_group[group] = {
            "n": N,
            "candidate52_strict_cr": s52,
            "candidate52_fraction": s52/N,
            "config17_strict_cr": s17,
            "config17_fraction": s17/N,
            "difference_config17_minus_candidate52_percentage_points": 100.0*(s17-s52)/N,
            "category_counts_candidate52": dict(cats52),
            "category_counts_config17": dict(cats17),
            "paired_success_transitions": {
                "both_success": int(both), "candidate52_only": int(only52),
                "config17_only": int(only17), "neither": int(neither),
            },
            "category_transition_counts": dict(transitions),
            "config17_completed_periods_by_failure_category": {
                category: {
                    "n": len(values), "median": float(np.median(values)),
                    "q90": float(np.quantile(values, .9)), "max": int(np.max(values)),
                }
                for category, values in timing_by_category.items()
            },
        }

    result = {
        "study": "V17C.10 P2 strict matched holdout: training config17 vs candidate52",
        "status": "complete; descriptive paired finite-horizon comparison",
        "holdout": {
            "ensembles": {group: N for group in GROUPS},
            "periods": PERIODS, "steps_per_period": STEPS,
            "transient_periods": TRANSIENT,
            "finite_radii_over_R": [0.0002, 0.00189],
            "hydrodynamics": "none",
            "same_frozen_starts_and_phases": True,
            "candidate17_parameters": p17, "candidate17_qratio": q17,
        },
        "all_checks_pass": all(validation.values()),
        "checks": validation,
        "by_ensemble": by_group,
        "interpretation_note": "Paired counts are descriptive for these randomized Sobol holdout starts. Ensembles have different input distributions; do not collapse their rates into one universal basin probability or treat the training-selected config17 comparison as an untouched confirmatory test.",
    }
    out_path = RUN17 / "paired_validation_summary.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": result["all_checks_pass"],
        "checks": validation,
        "by_ensemble": {
            g: {
                "candidate52_strict_cr": by_group[g]["candidate52_strict_cr"],
                "config17_strict_cr": by_group[g]["config17_strict_cr"],
                "difference_percentage_points": by_group[g]["difference_config17_minus_candidate52_percentage_points"],
                "categories_17": by_group[g]["category_counts_config17"],
                "paired": by_group[g]["paired_success_transitions"],
            }
            for g in GROUPS
        },
        "summary_file": str(out_path),
    }, indent=2, ensure_ascii=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
