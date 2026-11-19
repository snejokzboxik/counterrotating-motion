"""Audit all saved focused-screen inputs, labels, and rankings."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import run_focused_edge_screen as setup

HERE = Path(__file__).resolve().parent
OUT = HERE / "focused_edge_screen"


def main():
    stage4 = np.load(setup.OPT / "stage4_candidate48" / "configurations.npz")
    p0 = np.load(setup.OPT / "stage5_candidate52" / "configurations.npz")
    base = np.asarray(stage4["p"][52], dtype=np.float64)
    configs = setup.make_search_configs(base, float(stage4["qratio"][52]), p0)
    starts, phases = setup.make_inputs()
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    saved_inputs = np.load(OUT / "configurations_and_training_starts.npz", allow_pickle=False)
    checks = {
        "37_configurations_and_four_groups": meta["config_count"] == 37 and all(saved_inputs[f"initial_{g}"].shape == (32, 12) for g in setup.GROUPS),
        "configurations_match_saved_training_npz": True,
        "sobol_and_anchor_definitions_reproduce": True,
        "fresh_start_arrays_reproduce_from_seeds": True,
        "edge_group_all_rmax_ge_0_70": True,
        "all_4736_trial_keys_unique": False,
        "all_trial_parameters_and_qratios_match": True,
        "all_trial_inputs_and_phases_match": True,
        "all_proxy_labels_reproduce": True,
        "saved_ranking_reproduces": False,
    }

    expected_arrays = {"p": np.asarray([c["p"] for c in configs]), "qratio": np.asarray([c["qratio"] for c in configs])}
    for group in setup.GROUPS:
        expected_arrays[f"initial_{group}"] = starts[group]
        expected_arrays[f"phase_{group}"] = phases[group]
    for name, expected in expected_arrays.items():
        checks["configurations_match_saved_training_npz"] &= bool(np.array_equal(saved_inputs[name], expected))
    checks["sobol_and_anchor_definitions_reproduce"] = (
        len(configs) == 37
        and [c["config_id"] for c in configs] == list(range(37))
        and [c.get("P0_config_id") for c in configs[32:]] == list(setup.P0_TOP_IDS)
    )
    for group in setup.GROUPS:
        checks["fresh_start_arrays_reproduce_from_seeds"] &= bool(
            np.array_equal(saved_inputs[f"initial_{group}"], starts[group])
            and np.array_equal(saved_inputs[f"phase_{group}"], phases[group])
        )
    edge_rmax = np.maximum(np.hypot(starts["edge"][:, 0], starts["edge"][:, 1]), np.hypot(starts["edge"][:, 6], starts["edge"][:, 7]))
    checks["edge_group_all_rmax_ge_0_70"] = bool(np.all(edge_rmax >= setup.EDGE_RMIN))

    expected_config = {c["config_id"]: c for c in configs}
    rows = []
    keys = set()
    with (OUT / "runs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["config_id"]), row["ensemble"], int(row["start_id"]))
            if key in keys:
                raise ValueError(f"duplicate task key: {key}")
            keys.add(key)
            rows.append(row)
            c = expected_config[key[0]]
            g, i = key[1], key[2]
            checks["all_trial_parameters_and_qratios_match"] &= bool(
                np.array_equal(np.asarray(row["p"], dtype=float), np.asarray(c["p"], dtype=float))
                and float(row["qratio"]) == float(c["qratio"])
            )
            checks["all_trial_inputs_and_phases_match"] &= bool(
                np.array_equal(np.asarray(row["initial"], dtype=float), starts[g][i])
                and np.array_equal(np.asarray(row["phase"], dtype=float), phases[g][i])
                and int(row["seed"]) == int(setup.START_SEEDS[g])
            )
            expected_proxy = setup.cr_proxy(
                bool(row["retained"]), np.asarray(row["turns"], dtype=float),
                np.asarray(row["window_turns"], dtype=float), np.asarray(row["r_min"], dtype=float),
            )
            checks["all_proxy_labels_reproduce"] &= bool(expected_proxy == row["cr_proxy"])
    checks["all_4736_trial_keys_unique"] = len(rows) == 4736 and len(keys) == 4736

    stats = {
        c["config_id"]: {
            g: {"n": 0, "cr": 0, "retained": 0, "categories": Counter()}
            for g in setup.GROUPS
        }
        for c in configs
    }
    category_counts = Counter()
    for row in rows:
        s = stats[int(row["config_id"])][row["ensemble"]]
        s["n"] += 1
        s["cr"] += bool(row["cr_proxy"])
        s["retained"] += bool(row["retained"])
        s["categories"][row["category"]] += 1
        category_counts[row["category"]] += 1

    ranked = []
    for c in configs:
        by_group = stats[c["config_id"]]
        rates = {g: by_group[g]["cr"]/setup.N_PER_GROUP for g in setup.GROUPS}
        retention = {g: by_group[g]["retained"]/setup.N_PER_GROUP for g in setup.GROUPS}
        objective = (
            min(rates.values()), retention["edge"],
            float(np.mean([rates[g] for g in setup.STANDARD_GROUPS])),
            min(retention.values()),
        )
        ranked.append({
            "config_id": c["config_id"], "label": c["label"],
            "proxy_by_group": rates, "retention_by_group": retention,
            "categories_by_group": {g: dict(by_group[g]["categories"]) for g in setup.GROUPS},
            "objective_tuple": list(objective),
            "p": np.asarray(c["p"], dtype=float).tolist(), "qratio": float(c["qratio"]),
        })
    ranked.sort(key=lambda r: tuple(r["objective_tuple"]), reverse=True)
    saved_ranked = json.loads((OUT / "ranked.json").read_text(encoding="utf-8"))
    checks["saved_ranking_reproduces"] = bool(
        len(saved_ranked) == len(ranked)
        and all(saved_ranked[i]["config_id"] == ranked[i]["config_id"]
                and np.allclose(list(saved_ranked[i]["P_CR_proxy_by_group"].values()), list(ranked[i]["proxy_by_group"].values()))
                and np.allclose(list(saved_ranked[i]["P_retained_by_group"].values()), list(ranked[i]["retention_by_group"].values()))
                for i in range(len(ranked)))
    )

    result = {
        "study": "V17C.10 focused edge screen validation",
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "n_configs": len(configs), "n_groups": len(setup.GROUPS),
        "n_per_config_group": setup.N_PER_GROUP,
        "total_trials": len(rows),
        "category_counts_all_trials": dict(category_counts),
        "ranked_top_10": ranked[:10],
        "anchor_results": [row for row in ranked if row["config_id"] >= setup.N_CONFIGS],
        "objective_note": "Training proxy on fresh scrambled Sobol sets. The edge group is an additional targeted distribution and is not a reweighted estimate of rest/weak/wide.",
    }
    output = OUT / "validation_summary.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": result["all_checks_pass"],
        "checks": checks,
        "total_trials": len(rows),
        "category_counts": dict(category_counts),
        "ranked_top_10": [
            {"config_id": x["config_id"], "label": x["label"], "proxy_by_group": x["proxy_by_group"],
             "retention_by_group": x["retention_by_group"], "objective_tuple": x["objective_tuple"]}
            for x in ranked[:10]
        ],
        "anchor_results": [
            {"config_id": x["config_id"], "label": x["label"], "proxy_by_group": x["proxy_by_group"],
             "retention_by_group": x["retention_by_group"], "categories_by_group": x["categories_by_group"]}
            for x in ranked if x["config_id"] >= setup.N_CONFIGS
        ],
        "summary_file": str(output),
    }, indent=2, ensure_ascii=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
