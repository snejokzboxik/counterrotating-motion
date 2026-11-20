"""Audit the fresh N=256 strict matched replication."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import run_focused_edge_replication as setup

HERE = Path(__file__).resolve().parent
OUT = HERE / "focused_edge_replication"
RMIN_CLASS = 0.02


def strict_cr_reproduced(row):
    windows = np.asarray(row["window_turns"], dtype=float)
    turns = np.asarray(row["turns"], dtype=float)
    rmin = np.asarray(row["r_min"], dtype=float)
    signs = np.where(np.abs(windows) < 0.05, 0.0, np.sign(windows))
    return bool(
        row["retained"] and not row["escaped"] and not row["singular_unresolved"]
        and int(row["completed_periods"]) >= setup.PERIODS
        and np.all(rmin >= RMIN_CLASS)
        and np.all(np.abs(turns) >= 10.0)
        and np.all(signs != 0.0)
        and np.all(signs[:, 0] * signs[:, 1] < 0.0)
    )


def load_rows(path: Path):
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
        raise ValueError(f"expected {setup.N} unique rows in {path}; got {len(rows)}")
    return rows


def path_for(candidate_id, group):
    key = setup.FILE_KEYS[candidate_id]
    stem = f"{key}_{group}_matched_strict_none_a0.0002-0.00189_{setup.RUN_TAG}_{setup.N}_{setup.PERIODS}p_{setup.STEPS}s"
    return OUT / f"{stem}.jsonl", OUT / f"{stem}.meta.json"


def main():
    candidates = setup.load_candidates()
    starts, phases = setup.make_inputs()
    start_paths = setup.prepare_starts(starts, phases)
    validation = {
        "all_12_candidate_group_files_complete": True,
        "replication_starts_reproduce_from_seeds": True,
        "start_hashes_match_metadata": True,
        "candidate_parameters_match_saved_screen": True,
        "strict_protocol_metadata_matches": True,
        "all_rows_match_frozen_starts_and_phases": True,
        "all_strict_labels_reproduce": True,
        "all_3072_candidate_group_index_keys_unique": False,
    }
    by_candidate = {}
    by_group = {}
    all_rows = {}
    all_categories = Counter()
    total_rows = 0

    for group in setup.GROUPS:
        start_path = start_paths[group]
        with np.load(start_path, allow_pickle=False) as saved:
            validation["replication_starts_reproduce_from_seeds"] &= bool(
                int(saved["seed"]) == setup.START_SEEDS[group]
                and np.array_equal(saved["initial"], starts[group])
                and np.array_equal(saved["phases"], phases[group])
            )
        group_rows = {}
        for candidate_id in setup.SELECTED:
            candidate = candidates[candidate_id]
            run_file, meta_file = path_for(candidate_id, group)
            rows = load_rows(run_file)
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            validation["start_hashes_match_metadata"] &= bool(
                meta["frozen_start_sha256"] == setup.sha256(start_path)
            )
            validation["candidate_parameters_match_saved_screen"] &= bool(
                int(meta["screen_config_id"]) == candidate_id
                and np.array_equal(np.asarray(meta["parameters"], dtype=float), np.asarray(candidate["p"], dtype=float))
                and float(meta["qratio"]) == float(candidate["qratio"])
            )
            validation["strict_protocol_metadata_matches"] &= bool(
                meta["ensemble"] == group and int(meta["n"]) == setup.N
                and int(meta["seed"]) == setup.START_SEEDS[group]
                and int(meta["periods"]) == setup.PERIODS and int(meta["steps"]) == setup.STEPS
                and int(meta["transient"]) == setup.TRANSIENT
                and meta["hydro_mode"] == "none"
                and float(meta["a1_over_R"]) == setup.SIZES[0]
                and float(meta["a2_over_R"]) == setup.SIZES[1]
                and meta["run_tag"] == setup.RUN_TAG
                and meta["fixed_field_config_sha256"] == setup.sha256(HERE.parent / "v17c_5" / "input" / "config.json")
                and meta["runner_sha256"] == setup.sha256(HERE / "run_focused_edge_replication.py")
            )
            cats = Counter()
            strict_count = 0
            for i, row in rows.items():
                validation["all_rows_match_frozen_starts_and_phases"] &= bool(
                    np.array_equal(np.asarray(row["initial"], dtype=float), starts[group][i])
                    and np.array_equal(np.asarray(row["phase"], dtype=float), phases[group][i])
                    and int(row["seed"]) == setup.START_SEEDS[group]
                    and row["hydro_model"] == "none"
                    and float(row["a1_over_R"]) == setup.SIZES[0]
                    and float(row["a2_over_R"]) == setup.SIZES[1]
                )
                reproduced = strict_cr_reproduced(row)
                validation["all_strict_labels_reproduce"] &= bool(reproduced == (row["category"] == "CR_screen"))
                strict_count += reproduced
                cats[row["category"]] += 1
                all_categories[row["category"]] += 1
                group_rows[setup.FILE_KEYS[candidate_id]] = rows
                all_rows[(candidate_id, group, i)] = row
            by_candidate.setdefault(setup.FILE_KEYS[candidate_id], {})[group] = {
                "n": setup.N, "strict_cr": strict_count, "strict_fraction": strict_count / setup.N,
                "retained": sum(bool(r["retained"]) for r in rows.values()),
                "category_counts": dict(cats),
            }
            total_rows += len(rows)

        for candidate_a, candidate_b in ((15, 36), (6, 36), (15, 6)):
            akey, bkey = setup.FILE_KEYS[candidate_a], setup.FILE_KEYS[candidate_b]
            arows, brows = group_rows[akey], group_rows[bkey]
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
            by_group.setdefault(group, {})[akey + "_vs_" + bkey] = {
                "candidate_a_strict_cr": ac, "candidate_b_strict_cr": bc,
                "difference_a_minus_b": ac - bc,
                "difference_percentage_points": 100.0 * (ac - bc) / setup.N,
                "paired_success_transitions": {
                    "both_success": int(both), "candidate_a_only": int(only_a),
                    "candidate_b_only": int(only_b), "neither": int(neither),
                },
                "category_transition_counts": dict(transitions),
            }

    validation["all_12_candidate_group_files_complete"] = total_rows == 3072
    validation["all_3072_candidate_group_index_keys_unique"] = total_rows == 3072 and len(all_rows) == 3072
    result = {
        "study": "V17C.10 independent focused edge strict replication",
        "status": "complete; descriptive matched comparison",
        "protocol": {
            "candidates": {setup.FILE_KEYS[i]: i for i in setup.SELECTED},
            "ensembles": {g: setup.N for g in setup.GROUPS},
            "periods": setup.PERIODS, "steps_per_period": setup.STEPS,
            "transient_periods": setup.TRANSIENT,
            "finite_radii_over_R": list(setup.SIZES), "hydrodynamics": "none",
            "strict_axis_rmin_threshold_over_R": RMIN_CLASS,
            "strict_turn_threshold": 10.0, "window_sign_deadband_turns": 0.05,
            "same_frozen_starts_and_phases_for_all_candidates": True,
            "edge_ensemble_note": "Targeted starts with one particle initialized in [0.70R,0.90R]; analyzed separately.",
        },
        "all_checks_pass": all(validation.values()), "checks": validation,
        "category_counts_all_candidate_runs": dict(all_categories),
        "by_candidate_and_ensemble": by_candidate,
        "paired_comparisons_by_ensemble": by_group,
        "interpretation_note": "Fresh matched-start replication; each configuration was selected from earlier screens, and local15 was highlighted after a prior N=128 holdout. Treat results as an independent replication, not a fully preregistered test. Keep all start distributions separate.",
    }
    out = OUT / "replication_validation_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": result["all_checks_pass"], "checks": validation,
        "total_rows": total_rows, "category_counts": dict(all_categories),
        "by_candidate_and_ensemble": by_candidate,
        "paired_comparisons_by_ensemble": by_group,
        "summary_file": str(out),
    }, indent=2, ensure_ascii=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
