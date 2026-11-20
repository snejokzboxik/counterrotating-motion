"""Score the independent strict-loading holdout with unconditional denominators."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "strict_loading_holdout"
GROUPS = ("rest", "weak", "wide", "edge")


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n; den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="strict_loading_holdout",
                        help="holdout output directory under v17c_11")
    args = parser.parse_args()
    OUT = HERE / args.output
    protocols = json.loads((OUT / "holdout_protocols.json").read_text(encoding="utf-8"))
    metadata = json.loads((OUT / "holdout_metadata.json").read_text(encoding="utf-8"))
    n = int(metadata["N_per_ensemble"])
    by = defaultdict(list); keys = set()
    with (OUT / "holdout_runs.jsonl").open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["protocol_id"], r["ensemble"], r["start_id"])
            if key in keys:
                raise ValueError(f"duplicate key on line {line_no}: {key}")
            keys.add(key); by[r["protocol_id"]].append(r)
    expected = len(protocols) * len(GROUPS) * n
    if len(keys) != expected:
        raise ValueError(f"expected {expected} rows, found {len(keys)}")

    summaries = []
    for desc in protocols:
        rr = by[desc["id"]]
        if len(rr) != len(GROUPS) * n:
            raise ValueError(f"protocol {desc['id']} has {len(rr)} rows")
        item = {"protocol_id": desc["id"], "label": desc["label"],
                "transition_cycles": desc["duration_cycles"]}
        rates = {}
        for group in GROUPS:
            grp = [x for x in rr if x["ensemble"] == group]
            if len(grp) != n:
                raise ValueError(f"{desc['label']} {group} has N={len(grp)}")
            success = sum(x["category"] == "CR_screen" for x in grp)
            retained = sum(bool(x["retained"]) for x in grp)
            rate = success / n; rates[group] = rate
            cats = Counter(x["category"] for x in grp)
            item[f"cr_{group}"] = success
            item[f"pcr_{group}"] = rate
            item[f"wilson95_{group}"] = wilson(success, n)
            item[f"retained_{group}"] = retained
            item[f"retention_{group}"] = retained / n
            item[f"categories_{group}"] = dict(cats)
            cr = [x for x in grp if x["category"] == "CR_screen"]
            if cr:
                item[f"mean_frequency_hz_{group}"] = np.mean([x["frequency_Hz"] for x in cr], axis=0).tolist()
                item[f"mean_turns_{group}"] = np.mean([x["turns"] for x in cr], axis=0).tolist()
                item[f"mean_rmin_{group}"] = np.mean([x["r_min"] for x in cr], axis=0).tolist()
                item[f"mean_min_separation_{group}"] = float(np.mean([x["minimum_separation"] for x in cr]))
            else:
                item[f"mean_frequency_hz_{group}"] = None
                item[f"mean_turns_{group}"] = None
                item[f"mean_rmin_{group}"] = None
                item[f"mean_min_separation_{group}"] = None
        item["J_standard_PCR"] = min(rates[g] for g in GROUPS[:3])
        item["J_robust_PCR"] = min(rates.values())
        item["overall_PCR_unconditional"] = sum(item[f"cr_{g}"] for g in GROUPS) / (n * len(GROUPS))
        summaries.append(item)

    result = {
        "protocols": len(protocols), "unique_runs": len(keys),
        "N_per_ensemble": n, "strict_success": "category == CR_screen",
        "denominator": "all starts in each ensemble; includes wall loss, contact, axis ambiguity, localization and inconclusive outcomes as failures",
        "measurement": "15,000 periods after envelope completion and 200-period relaxation",
        "summaries": summaries,
    }
    (OUT / "holdout_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    flat = []
    for x in summaries:
        row = {k: v for k, v in x.items() if not isinstance(v, (dict, list))}
        flat.append(row)
    with (OUT / "holdout_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)

    xloc = np.arange(len(protocols)); width = .18
    fig, ax = plt.subplots(figsize=(12, 6))
    for j, group in enumerate(GROUPS):
        vals = [s[f"pcr_{group}"] for s in summaries]
        ax.bar(xloc + (j - 1.5) * width, vals, width, label=group)
        for i, v in enumerate(vals):
            lo, hi = summaries[i][f"wilson95_{group}"]
            ax.errorbar(xloc[i] + (j - 1.5) * width, v,
                        # Clamp round-off at p=1: Wilson's upper limit can be
                        # 1-1 ulp, which otherwise creates a negative y-error.
                        yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
                        fmt="none", ecolor="black", capsize=2, lw=.8)
    ax.axhline(.95, color="tab:red", ls="--", lw=1, label="95% target")
    ax.set_xticks(xloc, [x["label"] for x in protocols], rotation=15)
    ax.set_ylim(0, 1.02); ax.set_ylabel("strict CR / all starts")
    ax.set_title("Independent holdout: strict long-term counter-rotation probability")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, -0.13)); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "strict_holdout_comparison.png", dpi=180); plt.close(fig)

    for x in summaries:
        print(x["label"], "P_CR", {g: f"{x[f'cr_{g}']}/{n}" for g in GROUPS},
              f"J={x['J_standard_PCR']:.4f}", f"Jrobust={x['J_robust_PCR']:.4f}")
        for g in GROUPS:
            print(" ", g, x[f"categories_{g}"])


if __name__ == "__main__":
    main()
