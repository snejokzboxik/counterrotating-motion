"""Paired outcome comparison across C levels in the identical-particle grid."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent/"results"/"symmetric_coupling_grid_corrected"


def main() -> None:
    grouped = defaultdict(dict)
    for line in (ROOT/"trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cid = row["config_id"]
        # The config ID ends with _C<value>; all preceding fields define a pair.
        family, ctext = cid.rsplit("_C", 1)
        grouped[family].setdefault(ctext, {})[int(row["start_index"])] = row
    comparisons = []
    for family, levels in sorted(grouped.items()):
        cs = sorted(levels, key=float)
        for low, high in ((cs[0], cs[1]), (cs[0], cs[2])):
            transitions = 0
            category_counts = defaultdict(int)
            max_endpoint_difference = 0.0
            for i, a in levels[low].items():
                b = levels[high][i]
                ca, cb = a["category"], b["category"]
                category_counts[f"{ca} -> {cb}"] += 1
                transitions += ca != cb
                delta = max(abs(x-y) for x, y in zip(a["final_state"], b["final_state"]))
                max_endpoint_difference = max(max_endpoint_difference, delta)
            comparisons.append({"amplitude_frequency_family": family,
                "C_low": float(low), "C_high": float(high),
                "paired_starts": len(levels[low]), "category_transitions": transitions,
                "transition_counts": dict(sorted(category_counts.items())),
                "max_abs_final_state_difference": max_endpoint_difference})
    out = {"study": "paired C-level outcome comparison, identical-particle rest ensemble",
        "comparisons": comparisons,
        "interpretation": "Only the discrete category and maximum final-state differences are compared; no single trajectory is treated as a stability proof."}
    path = ROOT/"paired_coupling_comparison.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"pair_comparisons": len(comparisons),
        "category_transitions_total": sum(r["category_transitions"] for r in comparisons),
        "max_abs_final_state_difference": max(r["max_abs_final_state_difference"] for r in comparisons)},
        indent=2))


if __name__ == "__main__":
    main()
