"""Validate the completed V17C.9 Sobol checkpoint and summarize its ranking."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V17C = HERE.parent
V17C9 = V17C / "v17c_9"
OPT = V17C9 / "optimization"
STAGE4 = OPT / "stage4_candidate48"
STAGE5 = OPT / "stage5_candidate52"
sys.path.insert(0, str(V17C9 / "baseline"))
sys.path.insert(0, str(OPT))
import basin_scan
import refine_candidate52_screen as scan

PRIOR_ROWS = 849
PRIOR_RUNS_SHA256 = "dd2f3295915a4137e07b496f868b5cebf3856d4befd91980e91965193c0f3b3c"


def same_array(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return bool(left.shape == right.shape and np.array_equal(left, right))


def main():
    runs_path = STAGE5 / "runs.jsonl"
    raw_lines = runs_path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in raw_lines if line.strip()]
    keys = [(r["config_id"], r["ensemble"], r["start_id"]) for r in rows]
    prior_prefix_sha256 = hashlib.sha256(b"".join(raw_lines[:PRIOR_ROWS])).hexdigest()

    old_config = np.load(STAGE4 / "configurations.npz")
    base = np.asarray(old_config["p"][scan.BASE_ID], dtype=float)
    base_qratio = float(old_config["qratio"][scan.BASE_ID])
    configs = scan.make_configs(base, base_qratio)
    expected_p = np.asarray([c["p"] for c in configs])
    expected_q = np.asarray([c["qratio"] for c in configs])
    frozen = np.load(STAGE5 / "configurations.npz")
    initials, phases = {}, {}
    checks = {
        "all_3120_rows": len(rows) == 65 * 3 * 16,
        "all_task_keys_unique": len(set(keys)) == len(keys),
        "all_65_configs_present": {r["config_id"] for r in rows} == set(range(65)),
        "all_16_starts_per_group": True,
        "configuration_parameters_unchanged": same_array(frozen["p"], expected_p),
        "charge_ratios_unchanged": same_array(frozen["qratio"], expected_q),
        "original_849_rows_preserved": prior_prefix_sha256 == PRIOR_RUNS_SHA256,
    }
    groups = ("rest", "weak", "wide")
    counts = Counter((r["config_id"], r["ensemble"]) for r in rows)
    checks["all_16_starts_per_group"] = all(
        counts[(cfg, group)] == 16 for cfg in range(65) for group in groups
    )

    for group, seed in scan.START_SEEDS.items():
        initials[group], phases[group] = basin_scan.make_initials(
            group, scan.N_PER_ENSEMBLE, seed
        )
        checks[f"{group}_frozen_initials_exact"] = same_array(
            frozen[f"initial_{group}"], initials[group]
        )
        checks[f"{group}_frozen_phases_exact"] = same_array(
            frozen[f"phase_{group}"], phases[group]
        )

    checks["all_run_inputs_and_seeds_exact"] = all(
        r["seed"] == scan.START_SEEDS[r["ensemble"]]
        and same_array(r["initial"], initials[r["ensemble"]][r["start_id"]])
        and same_array(r["phase"], phases[r["ensemble"]][r["start_id"]])
        for r in rows
    )
    checks["all_proxy_labels_reproduce"] = all(
        r["cr_proxy"]
        == scan.cr_proxy(
            r["retained"],
            np.asarray(r["turns"]),
            np.asarray(r["window_turns"]),
            np.asarray(r["r_min"]),
        )
        for r in rows
    )
    meta = json.loads((STAGE5 / "meta.json").read_text(encoding="utf-8"))
    checks["horizon_step_skip_and_sphere_metadata"] = (
        meta["n_configs"] == 65
        and meta["N_per_ensemble"] == 16
        and meta["periods"] == 1800
        and meta["steps"] == 48
        and meta["transient_cycles"] == 400
        and meta["config_seed"] == 20261180
        and meta["start_seeds"] == scan.START_SEEDS
        and meta["finite_spheres_a_over_R"] == [0.0002, 0.00189]
    )

    grouped = {
        cfg: {
            group: [r for r in rows if r["config_id"] == cfg and r["ensemble"] == group]
            for group in groups
        }
        for cfg in range(65)
    }
    ranking = []
    for cfg in range(65):
        cr = {g: sum(bool(r["cr_proxy"]) for r in grouped[cfg][g]) / 16 for g in groups}
        retained = {
            g: sum(bool(r["retained"]) for r in grouped[cfg][g]) / 16 for g in groups
        }
        ranking.append(
            {
                "config_id": cfg,
                "P_CR_proxy_by_ensemble": cr,
                "P_retained_by_ensemble": retained,
                "balanced_min_CR_proxy": min(cr.values()),
                "balanced_mean_CR_proxy": float(np.mean(list(cr.values()))),
                "balanced_min_retention": min(retained.values()),
                "parameters": {
                    k: v for k, v in configs[cfg].items() if k not in ("p", "source")
                },
            }
        )
    ranking.sort(
        key=lambda r: (
            r["balanced_min_CR_proxy"],
            r["balanced_mean_CR_proxy"],
            r["balanced_min_retention"],
        ),
        reverse=True,
    )
    saved_ranking = json.loads((STAGE5 / "ranked.json").read_text(encoding="utf-8"))
    checks["saved_ranking_reproduces"] = len(saved_ranking) == 65 and all(
        saved_ranking[i]["config_id"] == ranking[i]["config_id"]
        and saved_ranking[i]["balanced_min_CR_proxy"]
        == ranking[i]["balanced_min_CR_proxy"]
        and saved_ranking[i]["balanced_mean_CR_proxy"]
        == ranking[i]["balanced_mean_CR_proxy"]
        for i in range(65)
    )

    category_counts = Counter(r["category"] for r in rows)
    by_ensemble = {
        g: {
            "n": sum(r["ensemble"] == g for r in rows),
            "cr_proxy": sum(r["ensemble"] == g and r["cr_proxy"] for r in rows),
            "retained": sum(r["ensemble"] == g and r["retained"] for r in rows),
            "categories": dict(Counter(r["category"] for r in rows if r["ensemble"] == g)),
        }
        for g in groups
    }
    outcome = {
        "study": "V17C.9 Sobol screen continuation",
        "remote_base_commit": "cc41be9082c62e9b907eb459f453ac51001d4267",
        "status": "complete; training proxy only; no independent candidate validation",
        "planned_trials": 3120,
        "saved_rows": len(rows),
        "unique_task_keys": len(set(keys)),
        "new_trials_completed": len(rows) - PRIOR_ROWS,
        "continuation_wall_seconds": 739.2,
        "continuation_throughput_trials_per_second": round((len(rows) - PRIOR_ROWS) / 739.2, 4),
        "prior_849_prefix_sha256": prior_prefix_sha256,
        "complete_runs_sha256": hashlib.sha256(runs_path.read_bytes()).hexdigest(),
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "category_counts_all_ensembles": dict(category_counts),
        "by_ensemble": by_ensemble,
        "ranked_top_10": ranking[:10],
        "anchor_candidate52": next(r for r in ranking if r["config_id"] == 64),
        "interpretation": "The 16-start short-horizon Sobol score is a training proxy, not a strict P_CR estimate or holdout result.",
    }
    (HERE / "screen_completion_summary.json").write_text(
        json.dumps(outcome, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (HERE / "screen_completion_validation.json").write_text(
        json.dumps({"checks": checks, "all_checks_pass": all(checks.values())}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "all_checks_pass": outcome["all_checks_pass"],
                "checks": checks,
                "saved_rows": len(rows),
                "category_counts": dict(category_counts),
                "by_ensemble": by_ensemble,
                "top_10": [
                    {
                        "config_id": r["config_id"],
                        "J": r["balanced_min_CR_proxy"],
                        "J_mean": r["balanced_mean_CR_proxy"],
                        "minimum_retention": r["balanced_min_retention"],
                        "P_CR_proxy_by_ensemble": r["P_CR_proxy_by_ensemble"],
                    }
                    for r in ranking[:10]
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not outcome["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
