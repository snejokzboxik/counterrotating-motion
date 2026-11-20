"""Rank the completed physical screen under the updated any-direction CR objective.

This is a deterministic postprocessor. It does not change trajectory labels or
reinterpret the 3,000-period discovery screen as a long-horizon probability.
"""
from __future__ import annotations

import json
import gzip
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "physical_parameter_sobol_screen"
GROUPS = ("rest", "weak", "wide", "edge")


def trajectory_reader():
    raw = OUT / "trajectories.jsonl"
    archived = OUT / "trajectories.jsonl.gz"
    if raw.exists():
        return raw.open(encoding="utf-8")
    return gzip.open(archived, "rt", encoding="utf-8")


def main() -> None:
    meta = json.loads((OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "independent_audit.json").read_text(encoding="utf-8"))
    configs = {c["id"]: c for c in meta["configurations"]}
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    retained_counts: dict[str, Counter[str]] = defaultdict(Counter)
    with trajectory_reader() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[row["configuration"]][row["ensemble"]][row["provisional_category"]] += 1
            retained_counts[row["configuration"]][row["ensemble"]] += int(bool(row["retained"]))

    loss_data = audit["loss_mechanisms_by_configuration"]
    ranked = []
    for cid, cfg in configs.items():
        by_group = {}
        agg = Counter()
        for group in GROUPS:
            c = counts[cid][group]
            by_group[group] = {
                "n": sum(c.values()),
                "counterrotation_any_direction": c["PROVISIONAL_CR_PLUS_MINUS"] + c["PROVISIONAL_CR_MINUS_PLUS"],
                "counterrotation_plus_minus": c["PROVISIONAL_CR_PLUS_MINUS"],
                "counterrotation_minus_plus": c["PROVISIONAL_CR_MINUS_PLUS"],
                "corotation": c["PROVISIONAL_CO_PLUS_PLUS"] + c["PROVISIONAL_CO_MINUS_MINUS"],
                "retained": int(retained_counts[cid][group]),
                "wall_or_axial_escape": c["WALL_OR_AXIAL_ESCAPE"],
                "unresolved_short": c["UNRESOLVED_SHORT"],
            }
            agg.update(c)
        cr = agg["PROVISIONAL_CR_PLUS_MINUS"] + agg["PROVISIONAL_CR_MINUS_PLUS"]
        co = agg["PROVISIONAL_CO_PLUS_PLUS"] + agg["PROVISIONAL_CO_MINUS_MINUS"]
        retained = sum(retained_counts[cid][g] for g in GROUPS)
        phys = cfg["physical"]
        ranked.append({
            "configuration": cid,
            "sobol_index": cfg["sobol_index"],
            "preliminary_CR_count_of_128": cr,
            "preliminary_CR_fraction_of_128": cr / 128,
            "CR_plus_minus_count": agg["PROVISIONAL_CR_PLUS_MINUS"],
            "CR_minus_plus_count": agg["PROVISIONAL_CR_MINUS_PLUS"],
            "co_count": co,
            "co_fraction_of_128": co / 128,
            "retained_count_of_128": retained,
            "escape_count": agg["WALL_OR_AXIAL_ESCAPE"],
            "unresolved_short_count": agg["UNRESOLVED_SHORT"],
            "axis_ambiguous_count": agg["AXIS_AMBIGUOUS"],
            "nonrotating_count": agg["NONROTATING"],
            "loss_mechanisms": loss_data.get(cid, {"radial": 0, "axial": 0, "both": 0}),
            "per_ensemble": by_group,
            "dimensionless_parameters": cfg["p"],
            "q2_over_q1": cfg["qratio"],
            "physical": phys,
            "classification": "3,000-period RK4 discovery labels; not a strict long-horizon probability or absence proof",
        })
    # New objective: maximize either counter-rotating label, then suppress
    # co-rotation, wall loss, unresolved outcomes, and axis-ambiguous outcomes.
    ranked.sort(key=lambda x: (
        -x["preliminary_CR_count_of_128"],
        x["co_count"],
        x["escape_count"],
        x["unresolved_short_count"],
        x["axis_ambiguous_count"],
        x["nonrotating_count"],
        x["configuration"],
    ))
    for i, row in enumerate(ranked, 1):
        row["rank_updated_objective"] = i
        row["meets_80pct_discovery_threshold"] = row["preliminary_CR_count_of_128"] >= 103

    result = {
        "study": "V17C.12B physical-parameter Sobol screen ranking under updated adviser objective",
        "objective": "maximize P(CR+−)+P(CR−+); no branch-balance requirement; suppress co-rotation and losses",
        "source_rows": summary["rows"],
        "source_trajectory_sha256": summary["trajectory_sha256"],
        "independent_audit_integrity_pass": audit["integrity_pass"],
        "starts_per_configuration": 128,
        "ensembles": {g: 32 for g in GROUPS},
        "80pct_threshold_counts": {"of_128": 103, "of_32_per_ensemble": 26},
        "number_of_configurations_with_any_CR": sum(x["preliminary_CR_count_of_128"] > 0 for x in ranked),
        "number_of_configurations_meeting_80pct_discovery_threshold": sum(x["meets_80pct_discovery_threshold"] for x in ranked),
        "top_50": ranked[:50],
        "all_configurations": ranked,
        "interpretation": "The frozen start panels are reused across configurations and each ensemble has n=32. These are discovery counts, not IID population probabilities. A threshold hit would require fresh, longer-horizon validation.",
    }
    path = OUT / "updated_objective_ranking.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("top_50", "all_configurations")}, indent=2, ensure_ascii=True))
    print("TOP10")
    for row in ranked[:10]:
        print(json.dumps({k: row[k] for k in ("rank_updated_objective", "configuration", "preliminary_CR_count_of_128", "CR_plus_minus_count", "CR_minus_plus_count", "co_count", "retained_count_of_128", "escape_count", "axis_ambiguous_count", "nonrotating_count", "unresolved_short_count", "per_ensemble")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
