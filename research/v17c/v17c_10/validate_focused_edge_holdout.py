"""Audit the fresh matched strict holdout from the focused edge screen."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import run_focused_edge_holdout as setup

HERE = Path(__file__).resolve().parent
OUT = HERE / "focused_edge_validation"
GROUPS = setup.GROUPS
SELECTED = setup.SELECTED
FILE_KEYS = setup.FILE_KEYS
N = setup.N
PERIODS = setup.PERIODS
STEPS = setup.STEPS
TRANSIENT = setup.TRANSIENT
RMIN_CLASS = 0.02


def strict_cr_reproduced(row):
    windows = np.asarray(row["window_turns"], dtype=float)
    turns = np.asarray(row["turns"], dtype=float)
    rmin = np.asarray(row["r_min"], dtype=float)
    signs = np.where(np.abs(windows) < 0.05, 0.0, np.sign(windows))
    stable_sign = bool(np.all(signs != 0.0))
    opposite = bool(np.all(signs[:, 0] * signs[:, 1] < 0.0))
    enough_turns = bool(np.all(np.abs(turns) >= 10.0))
    return bool(
        row["retained"] and not row["escaped"] and not row["singular_unresolved"]
        and int(row["completed_periods"]) >= PERIODS
        and np.all(rmin >= RMIN_CLASS) and enough_turns and stable_sign and opposite
    )


def load_rows(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = int(row["index"])
        if i in rows or not 0 <= i < N:
            raise ValueError(f"duplicate or out-of-range index {i} in {path}")
        rows[i] = row
    if len(rows) != N or set(rows) != set(range(N)):
        raise ValueError(f"expected {N} unique rows in {path}; got {len(rows)}")
    return rows


def run_path(candidate_id, group):
    key = FILE_KEYS[candidate_id]
    stem = f"{key}_{group}_matched_strict_none_a0.0002-0.00189_{setup.RUN_TAG}_{N}_{PERIODS}p_{STEPS}s"
    return OUT / f"{stem}.jsonl", OUT / f"{stem}.meta.json"


def main():
    candidates = setup.load_candidates()
    starts, phases = setup.make_inputs()
    start_paths = setup.prepare_starts(starts, phases)
    validation = {
        "all_20_candidate_group_files_complete": True,
        "fresh_frozen_starts_reproduce_from_seeds": True,
        "frozen_start_hashes_match_metadata": True,
        "candidate_parameters_match_saved_screen": True,
        "holdout_metadata_matches_protocol": True,
        "all_rows_match_frozen_starts_and_phases": True,
        "all_strict_cr_labels_reproduce": True,
        "all_2560_candidate_group_index_keys_unique": False,
    }
    by_candidate = {}
    by_group = {}
    keyed_rows = {}
    all_categories = Counter()
    total_rows = 0

    for group in GROUPS:
        start_path = start_paths[group]
        with np.load(start_path, allow_pickle=False) as saved:
            validation["fresh_frozen_starts_reproduce_from_seeds"] &= bool(
                int(saved["seed"]) == setup.START_SEEDS[group]
                and np.array_equal(saved["initial"], starts[group])
                and np.array_equal(saved["phases"], phases[group])
            )
        group_rows = {}
        for candidate_id in SELECTED:
            candidate = candidates[candidate_id]
            run_file, meta_file = run_path(candidate_id, group)
            rows = load_rows(run_file)
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            key = FILE_KEYS[candidate_id]
            p = np.asarray(candidate["p"], dtype=float)
            validation["frozen_start_hashes_match_metadata"] &= bool(
                meta["frozen_start_sha256"] == setup.sha256(start_path)
            )
            validation["candidate_parameters_match_saved_screen"] &= bool(
                np.array_equal(np.asarray(meta["parameters"], dtype=float), p)
                and float(meta["qratio"]) == float(candidate["qratio"])
                and int(meta["screen_config_id"]) == candidate_id
            )
            validation["holdout_metadata_matches_protocol"] &= bool(
                meta["ensemble"] == group and int(meta["n"]) == N
                and int(meta["seed"]) == setup.START_SEEDS[group]
                and int(meta["periods"]) == PERIODS and int(meta["steps"]) == STEPS
                and int(meta["transient"]) == TRANSIENT
                and meta["hydro_mode"] == "none"
                and float(meta["a1_over_R"]) == setup.SIZES[0]
                and float(meta["a2_over_R"]) == setup.SIZES[1]
                and meta["run_tag"] == setup.RUN_TAG
                and meta["fixed_field_config_sha256"] == setup.sha256(HERE.parent / "v17c_5" / "input" / "config.json")
                and meta["runner_sha256"] == setup.sha256(HERE / "run_focused_edge_holdout.py")
            )
            cats = Counter()
            strict_count = 0
            for i, row in rows.items():
                validation["all_rows_match_frozen_starts_and_phases"] &= bool(
                    np.array_equal(np.asarray(row["initial"], dtype=float), starts[group][i])
                    and np.array_equal(np.asarray(row["phase"], dtype=float), phases[group][i])
                    and row["hydro_model"] == "none"
                    and float(row["a1_over_R"]) == setup.SIZES[0]
                    and float(row["a2_over_R"]) == setup.SIZES[1]
                )
                reproduced = strict_cr_reproduced(row)
                validation["all_strict_cr_labels_reproduce"] &= bool(
                    reproduced == (row["category"] == "CR_screen")
                )
                cats[row["category"]] += 1
                all_categories[row["category"]] += 1
                strict_count += reproduced
                group_rows[key] = rows
                keyed_rows[(candidate_id, group, i)] = row
            by_candidate.setdefault(key, {})[group] = {
                "n": N, "strict_cr": strict_count, "strict_fraction": strict_count / N,
                "retained": sum(bool(r["retained"]) for r in rows.values()),
                "category_counts": dict(cats),
            }
            total_rows += len(rows)

        # Pair each candidate to the anchor on identical starts within this group.
        anchor_id = 36
        anchor_rows = group_rows[FILE_KEYS[anchor_id]]
        for candidate_id in SELECTED:
            if candidate_id == anchor_id:
                continue
            candidate_rows = group_rows[FILE_KEYS[candidate_id]]
            both = only_candidate = only_anchor = neither = 0
            transitions = Counter()
            for i in range(N):
                rc, ra = candidate_rows[i], anchor_rows[i]
                sc = rc["category"] == "CR_screen"
                sa = ra["category"] == "CR_screen"
                both += sc and sa
                only_candidate += sc and not sa
                only_anchor += sa and not sc
                neither += not sc and not sa
                transitions[f"{ra['category']} -> {rc['category']}"] += 1
            c_count = by_candidate[FILE_KEYS[candidate_id]][group]["strict_cr"]
            a_count = by_candidate[FILE_KEYS[anchor_id]][group]["strict_cr"]
            by_group.setdefault(group, {})[FILE_KEYS[candidate_id]] = {
                "candidate_strict_cr": c_count,
                "anchor_candidate52_strict_cr": a_count,
                "difference_vs_candidate52": c_count - a_count,
                "difference_percentage_points": 100.0 * (c_count - a_count) / N,
                "paired_success_transitions": {
                    "both_success": int(both), "candidate_only": int(only_candidate),
                    "candidate52_only": int(only_anchor), "neither": int(neither),
                },
                "category_transition_counts": dict(transitions),
            }

    validation["all_20_candidate_group_files_complete"] = total_rows == len(SELECTED) * len(GROUPS) * N
    validation["all_2560_candidate_group_index_keys_unique"] = (
        total_rows == 2560 and len(keyed_rows) == 2560
    )
    result = {
        "study": "V17C.10 P3 fresh strict matched holdout",
        "status": "complete; descriptive paired finite-horizon comparison",
        "protocol": {
            "candidates": {FILE_KEYS[i]: i for i in SELECTED},
            "ensembles": {g: N for g in GROUPS},
            "periods": PERIODS, "steps_per_period": STEPS,
            "transient_periods": TRANSIENT,
            "finite_radii_over_R": list(setup.SIZES),
            "hydrodynamics": "none", "strict_axis_rmin_threshold_over_R": RMIN_CLASS,
            "strict_turn_threshold": 10.0, "window_sign_deadband_turns": 0.05,
            "same_frozen_starts_and_phases_for_all_candidates": True,
            "edge_ensemble_note": "Separately targeted starts with at least one particle initialized at 0.70-0.90R; not pooled with rest/weak/wide.",
        },
        "all_checks_pass": all(validation.values()),
        "checks": validation,
        "category_counts_all_candidate_runs": dict(all_categories),
        "by_candidate_and_ensemble": by_candidate,
        "paired_against_candidate52_by_ensemble": by_group,
        "interpretation_note": "Differences are descriptive for one fresh matched Sobol holdout per group. Candidates were selected from a preceding screen, so this holdout is an independent-start comparison but not a fully untouched preregistered confirmatory test. Keep rest, weak, wide, and the targeted edge ensemble separate; do not interpret a pooled rate as a universal basin probability.",
    }
    out_path = OUT / "paired_validation_summary.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": result["all_checks_pass"], "checks": validation,
        "total_rows": total_rows, "category_counts": dict(all_categories),
        "by_candidate_and_ensemble": by_candidate,
        "paired_against_candidate52_by_ensemble": by_group,
        "summary_file": str(out_path),
    }, indent=2, ensure_ascii=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
