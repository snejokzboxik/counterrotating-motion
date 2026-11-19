"""Re-rank P3 screen rows with a turn cutoff scaled to the strict horizon."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

import run_focused_edge_screen as setup

HERE = Path(__file__).resolve().parent
OUT = HERE / "focused_edge_screen"
STRICT_PERIODS, STRICT_TRANSIENT, STRICT_TURN_MIN = 15000, 200, 10.0
STRICT_ACTIVE = STRICT_PERIODS - STRICT_TRANSIENT
TRAIN_ACTIVE = setup.PERIODS - setup.SKIP
SCALED_TURN_MIN = STRICT_TURN_MIN * TRAIN_ACTIVE / STRICT_ACTIVE


def main():
    configs_and_starts = np.load(OUT / "configurations_and_training_starts.npz", allow_pickle=False)
    configs = {i: {"p": configs_and_starts["p"][i], "qratio": float(configs_and_starts["qratio"][i])}
               for i in range(len(configs_and_starts["p"]))}
    stats = {i: {g: {"n": 0, "cr": 0, "retained": 0, "categories": Counter(), "turn_failure": 0}
                 for g in setup.GROUPS} for i in configs}
    rows_count = 0
    with (OUT / "runs.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            cid, group = int(row["config_id"]), row["ensemble"]
            turns = np.asarray(row["turns"], dtype=float)
            windows = np.asarray(row["window_turns"], dtype=float)
            rmin = np.asarray(row["r_min"], dtype=float)
            s = stats[cid][group]
            s["n"] += 1
            s["retained"] += bool(row["retained"])
            s["categories"][row["category"]] += 1
            turns_ok = np.min(np.abs(turns)) >= SCALED_TURN_MIN
            if row["retained"] and np.min(rmin) >= 0.02 and turns_ok:
                signs = np.where(np.abs(windows) < 0.05, 0.0, np.sign(windows))
                if np.all(signs != 0.0) and np.all(signs[:, 0] * signs[:, 1] < 0.0):
                    s["cr"] += 1
            if row["cr_proxy"] and not turns_ok:
                s["turn_failure"] += 1
            rows_count += 1

    ranked = []
    for cid, groups in stats.items():
        if any(groups[g]["n"] != setup.N_PER_GROUP for g in setup.GROUPS):
            raise ValueError(f"incomplete P3 screen rows for config {cid}")
        cr = {g: groups[g]["cr"] / setup.N_PER_GROUP for g in setup.GROUPS}
        retention = {g: groups[g]["retained"] / setup.N_PER_GROUP for g in setup.GROUPS}
        ranked.append({
            "config_id": cid,
            "label": f"config_{cid:02d}",
            "scaled_proxy_by_group": cr,
            "retention_by_group": retention,
            "original_proxy_positive_but_scaled_turn_failures": {
                g: groups[g]["turn_failure"] for g in setup.GROUPS
            },
            "categories_by_group": {g: dict(groups[g]["categories"]) for g in setup.GROUPS},
            "objective": [min(cr.values()), retention["edge"],
                          float(np.mean([cr[g] for g in setup.STANDARD_GROUPS])),
                          min(retention.values())],
            "p": configs[cid]["p"].tolist(), "qratio": configs[cid]["qratio"],
        })
    ranked.sort(key=lambda r: tuple(r["objective"]), reverse=True)
    saved_original_ranked = json.loads((OUT / "ranked.json").read_text(encoding="utf-8"))
    original_by_id = {int(row["config_id"]): row for row in saved_original_ranked}
    for row in ranked:
        row["label"] = original_by_id[row["config_id"]]["label"]
    candidate31 = next(row for row in ranked if row["config_id"] == 31)
    old_candidate31 = saved_original_ranked[0]
    result = {
        "study": "V17C.10 P3 post hoc horizon-scaled turn-threshold reanalysis",
        "status": "complete; reclassification of existing training rows, no integrations rerun",
        "training_horizon_periods": setup.PERIODS,
        "training_transient_periods": setup.SKIP,
        "strict_horizon_periods": STRICT_PERIODS,
        "strict_transient_periods": STRICT_TRANSIENT,
        "strict_min_absolute_turns": STRICT_TURN_MIN,
        "scaled_training_min_absolute_turns": SCALED_TURN_MIN,
        "scaling_note": "Linear horizon scaling of the strict 10-turn cutoff by active periods (1900/14800). This is a diagnostic proxy, not an exact predictor of strict success.",
        "training_rows": rows_count,
        "original_screen_leader": {"config_id": 31, "proxy_by_group": old_candidate31["P_CR_proxy_by_group"]},
        "config31_reanalysis": candidate31,
        "top_10_after_turn_threshold_scaling": ranked[:10],
        "all_ranked": ranked,
        "interpretation_note": "The short P3 proxy used a 1.05-turn cutoff, which admitted candidates whose slower vortex did not accumulate the strict 10 turns over the long holdout. The scaled threshold improves horizon alignment but still does not replace strict validation.",
    }
    out = OUT / "horizon_scaled_reanalysis.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "training_rows": rows_count,
        "scaled_training_min_absolute_turns": SCALED_TURN_MIN,
        "top_10": [{"config_id": x["config_id"], "label": x["label"],
                    "scaled_proxy_by_group": x["scaled_proxy_by_group"],
                    "objective": x["objective"]} for x in ranked[:10]],
        "config31": {"scaled_proxy_by_group": candidate31["scaled_proxy_by_group"],
                     "turn_failures": candidate31["original_proxy_positive_but_scaled_turn_failures"]},
        "output": str(out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
