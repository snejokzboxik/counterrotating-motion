"""Audit strict matched holdout results for the refine12-neighborhood screen."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import run_refine12_neighborhood_holdout as study
import run_local15_refinement_holdout as setup

HERE = Path(__file__).resolve().parent
OUT = study.OUT


def strict_cr_reproduced(row):
    windows = np.asarray(row["window_turns"], dtype=float)
    turns = np.asarray(row["turns"], dtype=float)
    rmin = np.asarray(row["r_min"], dtype=float)
    signs = np.where(np.abs(windows) < 0.05, 0.0, np.sign(windows))
    return bool(
        row["retained"] and not row["escaped"] and not row["singular_unresolved"]
        and int(row["completed_periods"]) >= setup.PERIODS
        and np.all(rmin >= 0.02) and np.all(np.abs(turns) >= 10.0)
        and np.all(signs != 0.0) and np.all(signs[:, 0] * signs[:, 1] < 0.0)
    )


def load_rows(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = int(row["index"])
        if i in rows or not 0 <= i < setup.N:
            raise ValueError(f"duplicate or invalid index {i} in {path}")
        rows[i] = row
    if len(rows) != setup.N or set(rows) != set(range(setup.N)):
        raise ValueError(f"expected {setup.N} unique indexed rows in {path}; got {len(rows)}")
    return rows


def paths_for(candidate_id, group):
    key = setup.FILE_KEYS[candidate_id]
    stem = (f"{key}_{group}_matched_strict_none_a0.0002-0.00189_"
            f"{setup.RUN_TAG}_{setup.N}_{setup.PERIODS}p_{setup.STEPS}s")
    return OUT / f"{stem}.jsonl", OUT / f"{stem}.meta.json"


def main():
    study.configure()
    candidates = study.load_candidates()
    starts, phases = setup.make_inputs()
    start_paths = setup.prepare_starts(starts, phases)
    expected_rows = len(setup.SELECTED) * len(setup.GROUPS) * setup.N
    checks = {
        "all_candidate_group_files_complete": True,
        "fresh_starts_reproduce_from_seed_values": True,
        "frozen_start_hashes_match_metadata": True,
        "parameters_match_neighborhood_screen": True,
        "strict_metadata_matches_protocol": True,
        "rows_match_exact_frozen_initials_and_phases": True,
        "strict_labels_reproduce": True,
        "all_candidate_group_index_keys_unique": False,
    }
    by_candidate = {}
    pairwise = {}
    rows_by_group = {}
    total_rows = 0
    categories_all = Counter()
    all_keys = set()

    for group in setup.GROUPS:
        with np.load(start_paths[group], allow_pickle=False) as saved:
            checks["fresh_starts_reproduce_from_seed_values"] &= bool(
                int(saved["seed"]) == setup.START_SEEDS[group]
                and np.array_equal(saved["initial"], starts[group])
                and np.array_equal(saved["phases"], phases[group])
            )
        rows_by_group[group] = {}
        for cid in setup.SELECTED:
            candidate = candidates[cid]
            run_path, meta_path = paths_for(cid, group)
            rows = load_rows(run_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            checks["frozen_start_hashes_match_metadata"] &= bool(
                meta["frozen_start_sha256"] == setup.sha256(start_paths[group])
            )
            checks["parameters_match_neighborhood_screen"] &= bool(
                int(meta["screen_config_id"]) == cid
                and np.array_equal(np.asarray(meta["parameters"], dtype=float), np.asarray(candidate["p"], dtype=float))
                and float(meta["qratio"]) == float(candidate["qratio"])
            )
            checks["strict_metadata_matches_protocol"] &= bool(
                meta["ensemble"] == group and int(meta["n"]) == setup.N
                and int(meta["seed"]) == setup.START_SEEDS[group]
                and int(meta["periods"]) == setup.PERIODS and int(meta["steps"]) == setup.STEPS
                and int(meta["transient"]) == setup.TRANSIENT
                and meta["run_tag"] == setup.RUN_TAG and meta["hydro_mode"] == "none"
                and float(meta["a1_over_R"]) == setup.SIZES[0]
                and float(meta["a2_over_R"]) == setup.SIZES[1]
                and meta["candidate_source"] == "V17C.10 strict holdout for refine12-neighborhood screen"
                and meta["fixed_field_config_sha256"] == setup.sha256(HERE.parent / "v17c_5" / "input" / "config.json")
                and meta["runner_sha256"] == setup.sha256(Path(study.__file__))
            )
            strict_count = 0
            cats = Counter()
            for i, row in rows.items():
                key = (cid, group, i)
                if key in all_keys:
                    raise ValueError(f"duplicate candidate/group/index key {key}")
                all_keys.add(key)
                checks["rows_match_exact_frozen_initials_and_phases"] &= bool(
                    np.array_equal(np.asarray(row["initial"], dtype=float), starts[group][i])
                    and np.array_equal(np.asarray(row["phase"], dtype=float), phases[group][i])
                    and int(row["seed"]) == setup.START_SEEDS[group]
                    and row["hydro_model"] == "none"
                    and float(row["a1_over_R"]) == setup.SIZES[0]
                    and float(row["a2_over_R"]) == setup.SIZES[1]
                )
                reproduced = strict_cr_reproduced(row)
                checks["strict_labels_reproduce"] &= bool(reproduced == (row["category"] == "CR_screen"))
                strict_count += reproduced
                cats[row["category"]] += 1
                categories_all[row["category"]] += 1
            name = setup.FILE_KEYS[cid]
            by_candidate.setdefault(name, {})[group] = {
                "n": setup.N, "strict_cr": strict_count,
                "strict_fraction": strict_count / setup.N,
                "retained": sum(bool(r["retained"]) for r in rows.values()),
                "category_counts": dict(cats),
            }
            rows_by_group[group][cid] = rows
            total_rows += len(rows)

        new_ids = (17, 10, 23)
        control_ids = (32, 34, 35, 36)
        pairs = [(a, b) for a in new_ids for b in control_ids]
        pairs += [(17, 10), (17, 23), (10, 23)]
        for candidate_a, candidate_b in pairs:
            akey, bkey = setup.FILE_KEYS[candidate_a], setup.FILE_KEYS[candidate_b]
            arows, brows = rows_by_group[group][candidate_a], rows_by_group[group][candidate_b]
            both = only_a = only_b = neither = 0
            transitions = Counter()
            for i in range(setup.N):
                aok = arows[i]["category"] == "CR_screen"
                bok = brows[i]["category"] == "CR_screen"
                both += aok and bok
                only_a += aok and not bok
                only_b += bok and not aok
                neither += not aok and not bok
                transitions[f"{brows[i]['category']} -> {arows[i]['category']}"] += 1
            ac = by_candidate[akey][group]["strict_cr"]
            bc = by_candidate[bkey][group]["strict_cr"]
            pairwise.setdefault(group, {})[f"{akey}_vs_{bkey}"] = {
                "candidate_a_strict_cr": ac, "candidate_b_strict_cr": bc,
                "difference_a_minus_b": ac - bc,
                "difference_percentage_points": 100.0 * (ac - bc) / setup.N,
                "paired_success_transitions": {
                    "both_success": int(both), "candidate_a_only": int(only_a),
                    "candidate_b_only": int(only_b), "neither": int(neither),
                },
                "category_transition_counts": dict(transitions),
            }

    checks["all_candidate_group_files_complete"] = total_rows == expected_rows
    checks["all_candidate_group_index_keys_unique"] = total_rows == expected_rows and len(all_keys) == expected_rows
    result = {
        "study": "V17C.10 refine12-neighborhood strict matched holdout",
        "status": "complete; descriptive independent-start comparison",
        "protocol": {
            "candidates": {setup.FILE_KEYS[i]: i for i in setup.SELECTED},
            "ensembles": {g: setup.N for g in setup.GROUPS},
            "periods": setup.PERIODS, "steps_per_period": setup.STEPS,
            "transient_periods": setup.TRANSIENT, "finite_radii_over_R": list(setup.SIZES),
            "hydrodynamics": "none", "strict_turn_threshold": 10.0,
            "axis_rmin_threshold_over_R": 0.02, "window_sign_deadband_turns": 0.05,
            "same_frozen_starts_and_phases_for_all_candidates": True,
        },
        "all_checks_pass": all(checks.values()), "checks": checks,
        "total_rows": total_rows, "category_counts_all_runs": dict(categories_all),
        "by_candidate_and_ensemble": by_candidate,
        "paired_comparisons_by_ensemble": pairwise,
        "interpretation_note": "The new candidates were selected from a short training screen. Results are exploratory; keep the four initial-condition distributions separate.",
    }
    out = OUT / "paired_validation_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": result["all_checks_pass"], "checks": checks,
        "total_rows": total_rows, "category_counts": dict(categories_all),
        "by_candidate_and_ensemble": by_candidate,
        "paired_comparisons_by_ensemble": pairwise,
        "summary_file": str(out),
    }, indent=2, ensure_ascii=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
