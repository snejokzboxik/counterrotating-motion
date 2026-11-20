"""Post-process the short Sobol screen without treating all-loss rows as leaders."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "identical_physical_sobol_screen_corrected"


def main() -> None:
    meta = json.loads((OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(s) for s in (OUT / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    by: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by[int(row["screen_index"])].append(row)

    ranked = []
    for c in meta["configurations"]:
        ci = int(c["screen_index"])
        group = by[ci]
        counts = Counter(r["provisional_category"] for r in group)
        n = len(group)
        cr_pm = counts.get("PROVISIONAL_CR_PLUS_MINUS", 0)
        cr_mp = counts.get("PROVISIONAL_CR_MINUS_PLUS", 0)
        co_pp = counts.get("PROVISIONAL_CO_PLUS_PLUS", 0)
        co_mm = counts.get("PROVISIONAL_CO_MINUS_MINUS", 0)
        retained = sum(bool(r["retained"]) for r in group)
        # Discovery ordering: first maximize pair retention; among ties maximize
        # balanced counterrotation, then CR count, and minimize co-rotation.
        balance = min(cr_pm, cr_mp) / (cr_pm + cr_mp) if cr_pm + cr_mp else 0.0
        ranked.append({
            "screen_index": ci, "config_id": c["id"], "p": c["p"],
            "physical": c["physical"], "n": n, "retained_pair_count": retained,
            "retained_pair_fraction_short_screen": retained / n if n else None,
            "provisional_counts": dict(sorted(counts.items())),
            "cr_plus_minus_count": cr_pm, "cr_minus_plus_count": cr_mp,
            "co_plus_plus_count": co_pp, "co_minus_minus_count": co_mm,
            "both_cr_directions_seen": cr_pm > 0 and cr_mp > 0,
            "cr_direction_balance_b_if_any": balance,
            "empirical_charge_window": c["empirical_charge_window"],
            "preferred_charge_window": c["preferred_charge_window"],
            "reported_voltage_envelope_4p4kV": c["reported_voltage_envelope_4p4kV"],
            "flat_Paschen_screen_under_unity": c["flat_Paschen_screen_under_unity"]})

    ranked.sort(key=lambda r: (
        r["retained_pair_count"], min(r["cr_plus_minus_count"], r["cr_minus_plus_count"]),
        r["cr_plus_minus_count"] + r["cr_minus_plus_count"],
        -(r["co_plus_plus_count"] + r["co_minus_minus_count"]),
        r["preferred_charge_window"], r["flat_Paschen_screen_under_unity"]), reverse=True)
    audit = json.loads((OUT / "independent_audit.json").read_text(encoding="utf-8"))
    result = {
        "method": "post-hoc screen ordering: pair retention first; balanced CR directions next; discovery labels only",
        "not_a_final_probability_or_leader_claim": True,
        "n_configs": len(ranked),
        "configs_with_both_CR_directions": sum(r["both_cr_directions_seen"] for r in ranked),
        "configs_with_at_least_6_of_8_retained": sum(r["retained_pair_count"] >= 6 for r in ranked),
        "top_50_by_retention_first": ranked[:50],
        "all_configs": ranked,
        "campaign_audit_integrity_pass": audit["integrity_pass"],
        "automated_campaign_J_discovery_warning": "The campaign's J_discovery assigns the best value to zero-retention configurations if they have no provisional CO label. Its top_config must not be interpreted as a physical leader."}
    path = OUT / "independent_retention_ranking.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    top = [{k: r[k] for k in ("config_id", "retained_pair_count", "provisional_counts",
        "physical", "preferred_charge_window", "flat_Paschen_screen_under_unity")}
        for r in ranked[:10]]
    print(json.dumps({"configs_with_both_CR_directions": result["configs_with_both_CR_directions"],
        "configs_with_at_least_6_of_8_retained": result["configs_with_at_least_6_of_8_retained"],
        "top_10_by_pair_retention": top}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
