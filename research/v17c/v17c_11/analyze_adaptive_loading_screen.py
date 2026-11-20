"""Summarize the fresh-start adaptive loading screen (early retention only)."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "adaptive_loading"
GROUPS = ("rest", "weak", "wide", "edge")


def main() -> None:
    protocols = json.loads((OUT / "adaptive_protocols.json").read_text(encoding="utf-8"))
    by = defaultdict(list)
    keys = set()
    with (OUT / "adaptive_runs.jsonl").open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["protocol_id"], row["ensemble"], row["start_id"])
            if key in keys:
                raise ValueError(f"duplicate run key at line {line_no}: {key}")
            keys.add(key)
            by[row["protocol_id"]].append(row)
    expected = len(protocols) * len(GROUPS) * 32
    if len(keys) != expected:
        raise ValueError(f"expected {expected} unique results, got {len(keys)}")

    summaries = []
    all_transition_complete = True
    for desc in protocols:
        pid = desc["id"]
        rr = by[pid]
        rates = {}
        losses = {}
        categories = Counter()
        for group in GROUPS:
            grp = [r for r in rr if r["ensemble"] == group]
            n = len(grp)
            retained = sum(bool(r["retained"]) for r in grp)
            rates[group] = retained / n
            losses[group] = Counter(r["category"] for r in grp if not r["retained"])
            categories.update(r["category"] for r in grp)
            if n != 32:
                raise ValueError(f"protocol {pid} {group} has N={n}, expected 32")
        complete = all(bool(r["transition_complete"]) for r in rr)
        all_transition_complete &= complete
        summaries.append({
            "protocol_id": pid,
            "family": desc["family"],
            "transition_cycles": desc["duration_cycles"],
            "transition_complete_by_48": complete,
            "rest_retained": sum(1 for r in rr if r["ensemble"] == "rest" and r["retained"]),
            "weak_retained": sum(1 for r in rr if r["ensemble"] == "weak" and r["retained"]),
            "wide_retained": sum(1 for r in rr if r["ensemble"] == "wide" and r["retained"]),
            "edge_retained": sum(1 for r in rr if r["ensemble"] == "edge" and r["retained"]),
            **{f"P_early_{g}": rates[g] for g in GROUPS},
            "J_standard_early": min(rates[g] for g in GROUPS[:3]),
            "J_robust_early": min(rates.values()),
            "categories": dict(categories),
            "nonretained_categories": {g: dict(losses[g]) for g in GROUPS},
        })

    for field, filename in (("J_standard_early", "ranked_standard_early.json"),
                            ("J_robust_early", "ranked_robust_early.json")):
        ranked = sorted(summaries, key=lambda x: (x[field], x["J_robust_early"],
                         x["P_early_edge"], -x["transition_cycles"]), reverse=True)
        (OUT / filename).write_text(json.dumps(ranked, indent=2) + "\n", encoding="utf-8")
    (OUT / "adaptive_summary.json").write_text(json.dumps({
        "protocols": len(protocols), "unique_runs": len(keys), "N_per_ensemble": 32,
        "all_transitions_complete": all_transition_complete,
        "metric_scope": "retention through 48 primary RF cycles only; not long-term counter-rotation probability",
        "protocols_summary": summaries,
    }, indent=2) + "\n", encoding="utf-8")
    with (OUT / "adaptive_summary.csv").open("w", newline="", encoding="utf-8") as f:
        cols = [k for k, v in summaries[0].items() if not isinstance(v, (dict, list))]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for row in summaries:
            w.writerow({k: row[k] for k in cols})

    ranked = sorted(summaries, key=lambda x: (x["J_robust_early"], x["J_standard_early"],
                     x["P_early_edge"], -x["transition_cycles"]), reverse=True)
    top = ranked[:16]
    mat = np.asarray([[x[f"P_early_{g}"] for g in GROUPS] for x in top])
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(mat, aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(4), GROUPS)
    ax.set_yticks(range(len(top)), [f'#{x["protocol_id"]} {x["family"]}' for x in top])
    ax.set_title("V17C.11 adaptive screen: early retention through 48 RF cycles (N=32/group)")
    ax.set_xlabel("frozen-start ensemble"); ax.set_ylabel("top schedules by min retention over all groups")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]*100:.1f}%", ha="center", va="center",
                    color="white" if mat[i,j] < .78 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="retained fraction")
    fig.tight_layout(); fig.savefig(OUT / "adaptive_early_retention_top16.png", dpi=180); plt.close(fig)
    best = ranked[0]
    print(f"validated {len(keys)} unique trajectories; transitions complete={all_transition_complete}")
    print(f"robust early-retention leader #{best['protocol_id']} {best['family']}: "
          f"rest={best['rest_retained']}/32 weak={best['weak_retained']}/32 "
          f"wide={best['wide_retained']}/32 edge={best['edge_retained']}/32; "
          f"Jstd={best['J_standard_early']:.4f}, Jrobust={best['J_robust_early']:.4f}")
    for row in sorted(summaries, key=lambda x: (x["J_standard_early"], x["J_robust_early"]), reverse=True)[:5]:
        print(row["protocol_id"], row["family"], row["rest_retained"], row["weak_retained"],
              row["wide_retained"], row["edge_retained"], row["J_standard_early"], row["J_robust_early"])


if __name__ == "__main__":
    main()
