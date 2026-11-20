"""Summarize the early-retention screen without treating it as strict CR."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import loading_dynamics as ld

HERE = Path(__file__).resolve().parent
DATA = HERE / "loading_screen"
GROUPS = ("rest", "weak", "wide", "edge")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [float(center-half), float(center+half)]


def main():
    rows = [json.loads(s) for s in (DATA / "early_screen.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    desc = json.loads((DATA / "protocols.json").read_text(encoding="utf-8"))
    seen = [(r["protocol_id"], r["ensemble"], r["start_id"]) for r in rows]
    if len(seen) != len(set(seen)):
        raise RuntimeError("duplicate protocol/ensemble/start keys in checkpoint")
    by = defaultdict(list)
    for r in rows:
        by[(r["protocol_id"], r["ensemble"])].append(r)

    per_protocol = []
    family_values = defaultdict(lambda: defaultdict(list))
    for d in desc:
        pid = int(d["id"])
        group = {g: by.get((pid, g), []) for g in GROUPS}
        if not all(group[g] for g in GROUPS):
            continue
        rates, intervals, failures, categories, opposite_proxy = {}, {}, {}, {}, {}
        for g in GROUPS:
            a = group[g]
            k = sum(bool(r["retained"]) for r in a)
            rates[g] = k / len(a)
            intervals[g] = wilson(k, len(a))
            failures[g] = dict(Counter(r["category"] for r in a))
            proxy = 0
            for r in a:
                w = np.asarray(r["window_turns"], dtype=float)
                if w.shape == (3, 2) and np.all(w[:, 0] * w[:, 1] < 0):
                    proxy += 1
            opposite_proxy[g] = proxy
            family_values[d["family"]][g].append(rates[g])
        transition_complete = all(r["transition_completed"] for g in GROUPS for r in group[g])
        score_std = min(rates[g] for g in GROUPS[:3])
        score_robust = min(rates.values())
        per_protocol.append({
            "protocol_id": pid, "family": d["family"],
            "transition_cycles": float(d.get("duration_cycles", 0.0)),
            "transition_complete_at_48": transition_complete,
            "N_per_ensemble": min(len(group[g]) for g in GROUPS),
            **{f"retained_{g}": rates[g] for g in GROUPS},
            **{f"retained_count_{g}": sum(bool(r["retained"]) for r in group[g]) for g in GROUPS},
            **{f"wilson95_{g}_low": intervals[g][0] for g in GROUPS},
            **{f"wilson95_{g}_high": intervals[g][1] for g in GROUPS},
            **{f"opposite_window_proxy_{g}": opposite_proxy[g] for g in GROUPS},
            "min_rest_weak_wide": score_std, "min_all_four": score_robust,
            "failures_json": json.dumps(failures, separators=(",", ":")),
        })

    per_protocol.sort(key=lambda x: (x["transition_complete_at_48"], x["min_rest_weak_wide"], x["min_all_four"]), reverse=True)
    with (DATA / "p2_protocol_retention.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_protocol[0]))
        w.writeheader(); w.writerows(per_protocol)

    family_summary = []
    for fam, groups in sorted(family_values.items()):
        n_proto = len(groups["rest"])
        family_summary.append({"family": fam, "protocols": n_proto,
                               **{f"mean_retained_{g}": float(np.mean(groups[g])) for g in GROUPS}})
    with (DATA / "p2_family_retention.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(family_summary[0]))
        w.writeheader(); w.writerows(family_summary)

    complete = [x for x in per_protocol if x["transition_complete_at_48"]]
    top = sorted(complete, key=lambda x: (x["min_rest_weak_wide"], x["min_all_four"],
                                           sum(x[f"retained_{g}"] for g in GROUPS)), reverse=True)[:12]
    raw_counts = {g: dict(Counter(r["category"] for r in rows if r["protocol_id"] == 0 and r["ensemble"] == g)) for g in GROUPS}
    summary = {
        "status": "complete_early_retention_screen_not_strict_counterrotation",
        "protocol_count": len(desc), "runs": len(rows), "unique_run_keys": len(set(seen)),
        "N_per_ensemble": 16, "ensembles": list(GROUPS),
        "horizon_primary_RF_periods": 48, "steps_per_period": 96,
        "workers": 4,
        "baseline_local15_retained": {
            g: sum(bool(r["retained"]) for r in by[(0, g)]) for g in GROUPS},
        "baseline_local15_categories": raw_counts,
        "protocols_fully_transitioned_by_48": len(complete),
        "top_training_screen_candidates": top,
        "scoring_note": "P2 is a low-N training screen based only on retention through a short horizon. It is not P_CR, not a holdout, and not proof of autonomous late counterrotation. Opposite-window counts are diagnostic proxies only.",
    }
    (DATA / "p2_screen_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Plot retention for baseline and leaders; every ensemble remains separate.
    chosen = [0] + [int(x["protocol_id"]) for x in top[:7]]
    labels = ["local15 constant"] + [f"#{x['protocol_id']} {x['family']}" for x in top[:7]]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(chosen)); width = .18
    colors = ["#3465a4", "#4e9a06", "#c17d11", "#a40000"]
    for gi, g in enumerate(GROUPS):
        ys = []
        for pid in chosen:
            a = by[(pid, g)]
            ys.append(sum(bool(r["retained"]) for r in a) / len(a))
        ax.bar(x + (gi-1.5)*width, ys, width, label=g, color=colors[gi])
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylim(0, 1.04); ax.set_ylabel("Retained through 48 primary RF periods")
    ax.set_title("V17C.11 P2: matched-start early-retention screen (N=16/group)")
    ax.grid(axis="y", alpha=.25); ax.legend(ncol=4)
    fig.tight_layout(); fig.savefig(DATA / "p2_retention_leaders.png", dpi=180); plt.close(fig)

    # Display actual continuous common envelopes for the leading training protocol.
    pid = int(top[0]["protocol_id"])
    protocol = np.asarray(desc[pid]["protocol"], dtype=np.float64)
    total = max(4.0, ld.protocol_duration_cycles(protocol))
    cycles = np.linspace(0, total, 1600)
    env = np.asarray([ld._envelopes(float(2*np.pi*t), protocol) for t in cycles])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, label in enumerate(("RF A", "RF B", "DC")):
        ax.plot(cycles, env[:, i], label=label)
    ax.axvline(ld.protocol_duration_cycles(protocol), color="black", ls="--", lw=1, label="terminal local15")
    ax.set(xlabel="Primary RF periods", ylabel="Envelope multiplier", ylim=(-.03, None),
           title=f"Preliminary leader #{pid}: {desc[pid]['family']} (training screen only)")
    ax.grid(alpha=.25); ax.legend(ncol=4)
    fig.tight_layout(); fig.savefig(DATA / "p2_leader_envelopes.png", dpi=180); plt.close(fig)
    print(f"summary: {len(rows)} runs; {len(complete)} fully transitioned; leader #{top[0]['protocol_id']}")


if __name__ == "__main__":
    main()
