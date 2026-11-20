"""Validate and rank the 1024-protocol early-retention screen."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "adaptive_loading_v2"
GROUPS = ("rest", "weak", "wide", "edge")
N = 32


def main():
    protocols = json.loads((OUT / "adaptive_protocols.json").read_text(encoding="utf-8"))
    by, keys = defaultdict(list), set()
    with (OUT / "adaptive_runs.jsonl").open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["protocol_id"], r["ensemble"], r["start_id"])
            if key in keys:
                raise ValueError(f"duplicate key at line {line_no}: {key}")
            keys.add(key); by[r["protocol_id"]].append(r)
    expected = len(protocols) * len(GROUPS) * N
    if len(keys) != expected:
        raise ValueError(f"expected {expected} unique rows, found {len(keys)}")
    summaries = []
    for d in protocols:
        rr = by[d["id"]]
        if len(rr) != N * 4:
            raise ValueError(f"protocol {d['id']} row count {len(rr)} != {N*4}")
        counts, causes = {}, {}
        cat = Counter()
        for g in GROUPS:
            grp = [r for r in rr if r["ensemble"] == g]
            if len(grp) != N:
                raise ValueError(f"protocol {d['id']} {g}: N={len(grp)}")
            counts[g] = sum(bool(r["retained"]) for r in grp)
            causes[g] = dict(Counter(r["category"] for r in grp if not r["retained"]))
            cat.update(r["category"] for r in grp)
        rate = {g: counts[g] / N for g in GROUPS}
        complete = all(bool(r["transition_complete"]) for r in rr)
        if not complete:
            raise ValueError(f"protocol {d['id']} transition did not complete by horizon")
        summaries.append({
            "protocol_id": d["id"], "family": d["family"],
            "transition_cycles": d["duration_cycles"],
            **{f"retained_{g}": counts[g] for g in GROUPS},
            **{f"P_early_{g}": rate[g] for g in GROUPS},
            "J_standard_early": min(rate[g] for g in GROUPS[:3]),
            "J_robust_early": min(rate.values()),
            "all_categories": dict(cat), "loss_causes": causes,
        })
    standard = sorted(summaries, key=lambda x: (x["J_standard_early"], x["J_robust_early"],
                      x["P_early_edge"], -x["transition_cycles"]), reverse=True)
    robust = sorted(summaries, key=lambda x: (x["J_robust_early"], x["J_standard_early"],
                    x["P_early_edge"], -x["transition_cycles"]), reverse=True)
    (OUT / "ranked_standard_early.json").write_text(json.dumps(standard, indent=2) + "\n", encoding="utf-8")
    (OUT / "ranked_robust_early.json").write_text(json.dumps(robust, indent=2) + "\n", encoding="utf-8")
    (OUT / "adaptive_summary.json").write_text(json.dumps({
        "protocols": len(protocols), "unique_runs": len(keys), "N_per_ensemble": N,
        "all_transitions_complete": True,
        "metric_scope": "48-cycle early retention only; not strict long-term P_CR",
        "protocol_summaries": summaries,
    }, indent=2) + "\n", encoding="utf-8")
    with (OUT / "adaptive_summary.csv").open("w", newline="", encoding="utf-8") as f:
        cols = [k for k, v in summaries[0].items() if not isinstance(v, (dict, list))]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in summaries:
            w.writerow({k: r[k] for k in cols})
    top = robust[:20]
    mat = np.asarray([[x[f"P_early_{g}"] for g in GROUPS] for x in top])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, aspect="auto", vmin=.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(4), GROUPS)
    ax.set_yticks(range(len(top)), [f'#{x["protocol_id"]} {x["family"]}' for x in top])
    ax.set_title("V17C.11 1024-protocol screen: early retention (N=32/group)")
    ax.set_xlabel("frozen-start ensemble"); ax.set_ylabel("top schedules by robust early retention")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]*100:.1f}%", ha="center", va="center",
                    color="white" if mat[i,j] < .78 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="retained fraction"); fig.tight_layout()
    fig.savefig(OUT / "adaptive_early_retention_top20.png", dpi=180); plt.close(fig)
    md = json.loads((OUT / "adaptive_metadata.json").read_text(encoding="utf-8"))
    md["status"] = "complete"
    md["unique_rows_verified"] = len(keys)
    (OUT / "adaptive_metadata.json").write_text(json.dumps(md, indent=2) + "\n", encoding="utf-8")
    print(f"verified {len(keys)} unique rows; all protocols transition by 48 cycles")
    for label, order, score in (("J_standard", standard, "J_standard_early"),
                                ("J_robust", robust, "J_robust_early")):
        print(label)
        for r in order[:8]:
            print(r["protocol_id"], r["family"], r["retained_rest"], r["retained_weak"],
                  r["retained_wide"], r["retained_edge"], round(r[score], 4))


if __name__ == "__main__":
    main()
