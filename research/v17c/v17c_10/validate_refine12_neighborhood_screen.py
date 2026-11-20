"""Audit the fresh Sobol training screen centered on refine12."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import run_refine12_neighborhood_screen as study

HERE = Path(__file__).resolve().parent
OUT = study.OUT


def main():
    study.configure_base()
    configs, center = study.local_configs()
    base = study.base
    starts, phases = study.make_inputs()
    input_path = OUT / "configurations_and_training_starts.npz"
    metadata = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    checks = {
        "37_configs_include_32_sobol_and_5_anchors": len(configs) == 37,
        "metadata_matches_protocol": False,
        "runner_and_field_hashes_match": False,
        "saved_arrays_reproduce_from_seeds": False,
        "all_9472_rows_unique_and_complete": False,
        "parameters_and_starts_match_configs": True,
        "scaled_proxy_labels_reproduce": True,
        "ranking_reproduces": False,
    }
    expected_parameters = {
        "study": "V17C.10 fresh Sobol neighborhood screen around refine12",
        "config_count": len(configs), "sobol_config_count": study.N_CONFIGS,
        "anchor_config_count": study.N_ANCHORS, "n_per_group": study.N_PER_GROUP,
        "groups": list(study.GROUPS), "periods": study.PERIODS,
        "steps": study.STEPS, "transient_periods": study.TRANSIENT,
        "config_seed": study.CONFIG_SEED, "start_seeds": study.START_SEEDS,
        "finite_radii_over_R": list(study.SIZES),
        "horizon_scaled_min_abs_turns": study.SCALED_TURN_MIN,
        "strict_reference": {"periods": study.STRICT_PERIODS,
                             "transient": study.STRICT_TRANSIENT,
                             "min_abs_turns": study.STRICT_TURNS},
        "parameter_ranges": study.PARAMETER_RANGES,
    }
    checks["metadata_matches_protocol"] = all(metadata.get(k) == v for k, v in expected_parameters.items())
    checks["runner_and_field_hashes_match"] = bool(
        metadata["runner_sha256"] == study.sha256(Path(study.__file__))
        and metadata["input_arrays_sha256"] == study.sha256(input_path)
        and metadata["fixed_field_config_sha256"] == study.sha256(
            study.base.BASE.parent.parent / "v17c_5" / "input" / "config.json"
        )
    )

    arrays = {"p": np.asarray([c["p"] for c in configs]),
              "qratio": np.asarray([c["qratio"] for c in configs])}
    for group in study.GROUPS:
        arrays[f"initial_{group}"] = starts[group]
        arrays[f"phase_{group}"] = phases[group]
    with np.load(input_path, allow_pickle=False) as saved:
        checks["saved_arrays_reproduce_from_seeds"] = all(
            np.array_equal(saved[name], expected) for name, expected in arrays.items()
        )

    by_id = {int(c["config_id"]): c for c in configs}
    run_path = OUT / "runs.jsonl"
    rows = []
    seen = set()
    for line in run_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid, group, i = int(row["config_id"]), row["ensemble"], int(row["start_id"])
        key = (cid, group, i)
        if key in seen or cid not in by_id or group not in study.GROUPS or not 0 <= i < study.N_PER_GROUP:
            raise ValueError(f"duplicate or invalid row key: {key}")
        seen.add(key)
        cfg = by_id[cid]
        checks["parameters_and_starts_match_configs"] &= bool(
            np.array_equal(np.asarray(row["p"], dtype=float), np.asarray(cfg["p"], dtype=float))
            and float(row["qratio"]) == float(cfg["qratio"])
            and int(row["seed"]) == study.START_SEEDS[group]
            and np.array_equal(np.asarray(row["initial"], dtype=float), starts[group][i])
            and np.array_equal(np.asarray(row["phase"], dtype=float), phases[group][i])
            and row["hydro_model"] == "none"
            and float(row["a1_over_R"]) == study.SIZES[0]
            and float(row["a2_over_R"]) == study.SIZES[1]
        )
        checks["scaled_proxy_labels_reproduce"] &= bool(
            bool(row["proxy_cr_scaled_turns"]) == bool(base.proxy_label(row))
        )
        rows.append(row)

    expected_rows = len(configs) * len(study.GROUPS) * study.N_PER_GROUP
    checks["all_9472_rows_unique_and_complete"] = len(rows) == expected_rows == 9472 and len(seen) == expected_rows
    ranked, _ = base.rank_rows(rows, configs)
    saved_ranked = json.loads((OUT / "ranked.json").read_text(encoding="utf-8"))
    checks["ranking_reproduces"] = ranked == saved_ranked
    categories = dict(Counter(row["category"] for row in rows))
    expected_scaled = 10.0 * (2400 - 200) / (15000 - 200)
    result = {
        "study": "V17C.10 refine12 neighborhood screen audit",
        "all_checks_pass": all(checks.values()), "checks": checks,
        "total_rows": len(rows), "expected_rows": expected_rows,
        "category_counts": categories, "scaled_turn_min_recomputed": expected_scaled,
        "top_10": ranked[:10],
        "note": "Exploratory short-horizon ranking; strict confirmation is required.",
    }
    out = OUT / "screen_validation_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_pass": result["all_checks_pass"], "checks": checks,
                      "total_rows": len(rows), "category_counts": categories,
                      "top_10": ranked[:10], "summary_file": str(out)},
                     indent=2, ensure_ascii=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
