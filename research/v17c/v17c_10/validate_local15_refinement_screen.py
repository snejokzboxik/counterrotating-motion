"""Audit the horizon-scaled local15 refinement training screen."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import run_local15_refinement_screen as setup

HERE = Path(__file__).resolve().parent
OUT = HERE / "local15_refinement_screen"


def main():
    configs = setup.local_screen_configs()
    starts, phases = setup.make_inputs()
    input_path = OUT / "configurations_and_training_starts.npz"
    with np.load(input_path, allow_pickle=False) as saved:
        expected = {"p": np.asarray([c["p"] for c in configs]),
                    "qratio": np.asarray([c["qratio"] for c in configs])}
        for group in setup.GROUPS:
            expected[f"initial_{group}"] = starts[group]
            expected[f"phase_{group}"] = phases[group]
        inputs_match = all(np.array_equal(saved[name], value) for name, value in expected.items())
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    validation = {
        "configuration_and_training_inputs_reproduce": bool(inputs_match),
        "metadata_matches_protocol_and_hashes": bool(
            meta["config_count"] == len(configs)
            and meta["sobol_config_count"] == setup.N_CONFIGS
            and meta["anchor_config_count"] == setup.N_ANCHORS
            and meta["n_per_group"] == setup.N_PER_GROUP
            and meta["periods"] == setup.PERIODS and meta["steps"] == setup.STEPS
            and meta["transient_periods"] == setup.TRANSIENT
            and meta["config_seed"] == setup.CONFIG_SEED
            and meta["horizon_scaled_min_abs_turns"] == setup.SCALED_TURN_MIN
            and meta["input_arrays_sha256"] == setup.sha256(input_path)
            and meta["runner_sha256"] == setup.sha256(HERE / "run_local15_refinement_screen.py")
            and meta["fixed_field_config_sha256"] == setup.sha256(HERE.parent / "v17c_5" / "input" / "config.json")
        ),
        "all_trial_keys_unique_and_complete": False,
        "all_trial_parameters_and_qratios_match": True,
        "all_trial_initials_phases_and_seeds_match": True,
        "all_scaled_turn_proxy_labels_reproduce": True,
        "saved_ranking_reproduces": False,
    }
    config_by_id = {int(c["config_id"]): c for c in configs}
    rows = []
    keys = set()
    with (OUT / "runs.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            cid, group, start_id = int(row["config_id"]), row["ensemble"], int(row["start_id"])
            key = (cid, group, start_id)
            if key in keys:
                raise ValueError(f"duplicate local refinement task key {key}")
            keys.add(key)
            rows.append(row)
            config = config_by_id[cid]
            validation["all_trial_parameters_and_qratios_match"] &= bool(
                np.array_equal(np.asarray(row["p"], dtype=float), np.asarray(config["p"], dtype=float))
                and float(row["qratio"]) == float(config["qratio"])
            )
            validation["all_trial_initials_phases_and_seeds_match"] &= bool(
                np.array_equal(np.asarray(row["initial"], dtype=float), starts[group][start_id])
                and np.array_equal(np.asarray(row["phase"], dtype=float), phases[group][start_id])
                and int(row["seed"]) == setup.START_SEEDS[group]
            )
            validation["all_scaled_turn_proxy_labels_reproduce"] &= bool(
                setup.proxy_label(row) == bool(row["proxy_cr_scaled_turns"])
            )
    expected_rows = len(configs) * len(setup.GROUPS) * setup.N_PER_GROUP
    validation["all_trial_keys_unique_and_complete"] = len(rows) == expected_rows and len(keys) == expected_rows

    ranked, _stats = setup.rank_rows(rows, configs)
    saved_ranked = json.loads((OUT / "ranked.json").read_text(encoding="utf-8"))
    validation["saved_ranking_reproduces"] = bool(
        len(ranked) == len(saved_ranked)
        and all(ranked[i]["config_id"] == saved_ranked[i]["config_id"]
                and np.allclose(list(ranked[i]["proxy_by_group"].values()),
                                list(saved_ranked[i]["proxy_by_group"].values()))
                and np.allclose(list(ranked[i]["retention_by_group"].values()),
                                list(saved_ranked[i]["retention_by_group"].values()))
                for i in range(len(ranked)))
    )
    category_counts = Counter(row["category"] for row in rows)
    result = {
        "study": "V17C.10 local15 refinement screen validation",
        "status": "complete; training-only proxy ranking",
        "all_checks_pass": all(validation.values()), "checks": validation,
        "config_count": len(configs), "n_per_group": setup.N_PER_GROUP,
        "total_rows": len(rows), "expected_rows": expected_rows,
        "horizon_scaled_turn_min": setup.SCALED_TURN_MIN,
        "category_counts": dict(category_counts),
        "ranked_top_10": ranked[:10],
        "interpretation_note": "Short-horizon re-ranking is exploratory. The scaled turn threshold aligns active duration linearly with the strict criterion but does not replace a new strict matched holdout.",
    }
    out = OUT / "validation_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": result["all_checks_pass"], "checks": validation,
        "total_rows": len(rows), "expected_rows": expected_rows,
        "category_counts": dict(category_counts),
        "top_10": [{"config_id": r["config_id"], "label": r["label"],
                    "proxy_by_group": r["proxy_by_group"], "objective": r["objective"]}
                   for r in ranked[:10]],
        "summary_file": str(out),
    }, indent=2, ensure_ascii=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
