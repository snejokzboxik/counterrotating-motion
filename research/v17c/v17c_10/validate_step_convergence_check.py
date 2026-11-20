"""Compare same-start 64-step and 96-step strict classifications."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import run_step_convergence_check as setup

HERE = Path(__file__).resolve().parent
OUT = HERE / "focused_edge_convergence"
REPLICATION = HERE / "focused_edge_replication"
REPL_TAG = "replicate20261410"
REPL_N = 256
REPL_STEPS = 64


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


def load_rows(path: Path, expected_n):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = int(row["index"])
        if i in rows or not 0 <= i < expected_n:
            raise ValueError(f"duplicate or invalid row {i} in {path}")
        rows[i] = row
    if len(rows) != expected_n or set(rows) != set(range(expected_n)):
        raise ValueError(f"expected {expected_n} rows in {path}; got {len(rows)}")
    return rows


def file_path(candidate_id, group, steps):
    key = setup.FILE_KEYS[candidate_id]
    if steps == setup.STEPS:
        stem = f"{key}_{group}_matched_strict_none_a0.0002-0.00189_{setup.RUN_TAG}_{setup.N}_{setup.PERIODS}p_{setup.STEPS}s"
        return OUT / f"{stem}.jsonl", OUT / f"{stem}.meta.json"
    stem = f"{key}_{group}_matched_strict_none_a0.0002-0.00189_{REPL_TAG}_{REPL_N}_{setup.PERIODS}p_{REPL_STEPS}s"
    return REPLICATION / f"{stem}.jsonl", REPLICATION / f"{stem}.meta.json"


def main():
    candidates = setup.load_candidates()
    starts, start_paths = setup.prepare_starts()
    validation = {
        "all_8_96_step_files_complete": True,
        "subset_starts_match_N256_replication_prefix": True,
        "metadata_and_hashes_match_protocol": True,
        "parameters_match_selected_candidates": True,
        "all_64_and_96_step_strict_labels_reproduce": True,
        "all_1024_candidate_group_index_keys_unique": False,
    }
    by_candidate = {}
    by_group = {}
    all_categories_96 = Counter()
    total_rows = 0

    for candidate_id in setup.SELECTED:
        key = setup.FILE_KEYS[candidate_id]
        for group in setup.GROUPS:
            rows96_path, meta_path = file_path(candidate_id, group, setup.STEPS)
            rows64_path, meta64_path = file_path(candidate_id, group, REPL_STEPS)
            rows96 = load_rows(rows96_path, setup.N)
            rows64_all = load_rows(rows64_path, REPL_N)
            rows64 = {i: rows64_all[i] for i in range(setup.N)}
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta64 = json.loads(meta64_path.read_text(encoding="utf-8"))
            initial, phases, seed = starts[group]
            source_file = REPLICATION / "replication_starts" / f"{group}_N256_seed{seed}.npz"
            validation["subset_starts_match_N256_replication_prefix"] &= bool(
                all(np.array_equal(np.asarray(rows96[i]["initial"], dtype=float), initial[i])
                    and np.array_equal(np.asarray(rows96[i]["phase"], dtype=float), phases[i])
                    and np.array_equal(np.asarray(rows64[i]["initial"], dtype=float), initial[i])
                    and np.array_equal(np.asarray(rows64[i]["phase"], dtype=float), phases[i])
                    for i in range(setup.N))
            )
            validation["metadata_and_hashes_match_protocol"] &= bool(
                meta["ensemble"] == group and meta["n"] == setup.N and meta["seed"] == seed
                and meta["periods"] == setup.PERIODS and meta["steps"] == setup.STEPS
                and meta["transient"] == setup.TRANSIENT and meta["hydro_mode"] == "none"
                and meta["frozen_start_sha256"] == setup.sha256(start_paths[group])
                and meta["source_start_sha256"] == setup.sha256(source_file)
                and meta["runner_sha256"] == setup.sha256(HERE / "run_step_convergence_check.py")
                and meta64["n"] == REPL_N and meta64["steps"] == REPL_STEPS
                and meta64["periods"] == setup.PERIODS and meta64["ensemble"] == group
            )
            validation["parameters_match_selected_candidates"] &= bool(
                int(meta["screen_config_id"]) == candidate_id
                and np.array_equal(np.asarray(meta["parameters"], dtype=float), np.asarray(candidates[candidate_id]["p"], dtype=float))
                and float(meta["qratio"]) == float(candidates[candidate_id]["qratio"])
            )
            c64 = Counter()
            c96 = Counter()
            both = only64 = only96 = neither = 0
            exact_category_match = 0
            strict_match = 0
            transitions = Counter()
            turns_errors = []
            rmin_errors = []
            for i in range(setup.N):
                a, b = rows64[i], rows96[i]
                ok64 = strict_cr_reproduced(a)
                ok96 = strict_cr_reproduced(b)
                validation["all_64_and_96_step_strict_labels_reproduce"] &= bool(
                    ok64 == (a["category"] == "CR_screen")
                    and ok96 == (b["category"] == "CR_screen")
                )
                c64[a["category"]] += 1
                c96[b["category"]] += 1
                all_categories_96[b["category"]] += 1
                both += ok64 and ok96
                only64 += ok64 and not ok96
                only96 += ok96 and not ok64
                neither += not ok64 and not ok96
                exact_category_match += a["category"] == b["category"]
                strict_match += ok64 == ok96
                transitions[f"{a['category']} -> {b['category']}"] += 1
                if a["completed_periods"] >= setup.PERIODS and b["completed_periods"] >= setup.PERIODS:
                    turns_errors.extend(np.abs(np.asarray(a["turns"]) - np.asarray(b["turns"])).tolist())
                    rmin_errors.extend(np.abs(np.asarray(a["r_min"]) - np.asarray(b["r_min"])).tolist())
            n = setup.N
            by_candidate.setdefault(key, {})[group] = {
                "n": n, "strict_cr_64_steps": int(sum(x["category"] == "CR_screen" for x in rows64.values())),
                "strict_cr_96_steps": int(sum(x["category"] == "CR_screen" for x in rows96.values())),
                "difference_96_minus_64": int(sum(x["category"] == "CR_screen" for x in rows96.values()) - sum(x["category"] == "CR_screen" for x in rows64.values())),
                "strict_label_agreement_n": strict_match,
                "strict_label_agreement_fraction": strict_match / n,
                "exact_category_agreement_n": exact_category_match,
                "success_transitions_64_to_96": {
                    "both_success": int(both), "64_step_only": int(only64),
                    "96_step_only": int(only96), "neither": int(neither),
                },
                "category_counts_64": dict(c64), "category_counts_96": dict(c96),
                "category_transition_counts_64_to_96": dict(transitions),
                "continuous_metric_differences_for_full_horizon_pairs": {
                    "n_particles_turn_differences": len(turns_errors),
                    "turn_abs_difference_median": float(np.median(turns_errors)) if turns_errors else None,
                    "turn_abs_difference_q95": float(np.quantile(turns_errors, 0.95)) if turns_errors else None,
                    "rmin_abs_difference_median": float(np.median(rmin_errors)) if rmin_errors else None,
                    "rmin_abs_difference_q95": float(np.quantile(rmin_errors, 0.95)) if rmin_errors else None,
                },
            }
            by_group.setdefault(group, {})[key] = by_candidate[key][group]
            total_rows += n

    validation["all_8_96_step_files_complete"] = total_rows == 1024
    validation["all_1024_candidate_group_index_keys_unique"] = total_rows == 1024
    result = {
        "study": "V17C.10 same-start 64-vs-96-step numerical sensitivity",
        "status": "complete; same-start subset comparison",
        "protocol": {
            "candidates": {setup.FILE_KEYS[i]: i for i in setup.SELECTED},
            "ensembles": {g: setup.N for g in setup.GROUPS},
            "periods": setup.PERIODS, "transient_periods": setup.TRANSIENT,
            "steps_compared": [REPL_STEPS, setup.STEPS],
            "same_start_indices": "0..127 from each independent N=256 replication group",
            "finite_radii_over_R": [0.0002, 0.00189], "hydrodynamics": "none",
        },
        "all_checks_pass": all(validation.values()), "checks": validation,
        "category_counts_96_step_runs": dict(all_categories_96),
        "by_candidate_and_ensemble": by_candidate,
        "interpretation_note": "This measures numerical sensitivity on the first 128 paired starts per group, not a new basin-probability sample. Report category transition rates by group; do not pool distinct start distributions.",
    }
    out = OUT / "step_convergence_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": result["all_checks_pass"], "checks": validation,
        "total_rows": total_rows, "category_counts_96_steps": dict(all_categories_96),
        "by_candidate_and_ensemble": by_candidate, "summary_file": str(out),
    }, indent=2, ensure_ascii=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
