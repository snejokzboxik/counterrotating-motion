"""Integrity, success-criterion, and paired-start audit for the N=256 holdout."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
OUT = HERE / "strict_loading_holdout_n256"
GROUPS = ("rest", "weak", "wide", "edge")
PROTOCOLS = (1, 2, 4)


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return [float(center - half), float(center + half)]


def main() -> None:
    meta = json.loads((OUT / "holdout_metadata.json").read_text(encoding="utf-8"))
    protocol_rows = json.loads((OUT / "holdout_protocols.json").read_text(encoding="utf-8"))
    n = int(meta["N_per_ensemble"])
    expected = len(PROTOCOLS) * len(GROUPS) * n
    records: dict[tuple[int, str, int], dict] = {}
    category_counts: dict[str, dict[str, int]] = defaultdict(dict)
    with (OUT / "holdout_runs.jsonl").open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (int(r["protocol_id"]), r["ensemble"], int(r["start_id"]))
            if key in records:
                raise ValueError(f"duplicate result key at line {line_no}: {key}")
            records[key] = r
    if len(records) != expected:
        raise ValueError(f"expected {expected} results, got {len(records)}")

    starts = np.load(OUT / "holdout_starts.npz")
    hash_audit = {}
    for group in GROUPS:
        initial = starts[f"{group}_initial"]
        phase = starts[f"{group}_phase"]
        expected_hash = meta["start_hashes"][group]
        got_initial = sha256_array(initial)
        got_phase = sha256_array(phase)
        if got_initial != expected_hash["initial_sha256"] or got_phase != expected_hash["phase_sha256"]:
            raise ValueError(f"frozen start checksum mismatch in {group}")
        hash_audit[group] = {"initial_sha256": got_initial, "phase_sha256": got_phase}

    # Verify identical starts were used for each protocol, and the stored rows
    # are complete and finite. Successful rows must meet the signed-turn,
    # three-window, retention, and finite-body-separation criteria.
    for group in GROUPS:
        for sid in range(n):
            base = records[(PROTOCOLS[0], group, sid)]
            expected_initial = starts[f"{group}_initial"][sid].tolist()
            expected_phase = starts[f"{group}_phase"][sid].tolist()
            if base["initial_state"] != expected_initial or base["initial_rf_phase"] != expected_phase:
                raise ValueError(f"journal does not match frozen NPZ start: {group}/{sid}")
            for pid in PROTOCOLS[1:]:
                other = records[(pid, group, sid)]
                if (other["initial_state"] != base["initial_state"] or
                        other["initial_rf_phase"] != base["initial_rf_phase"]):
                    raise ValueError(f"protocols do not share frozen start: {group}/{sid}")

    particle_radius_sum = float(sum(meta["particle_radii_over_R"])) if "particle_radii_over_R" in meta else 0.00209
    allowed_categories = {"CR_screen", "finite_particle_wall_contact", "axis_ambiguous"}
    for key, r in records.items():
        pid, group, _ = key
        if pid not in PROTOCOLS or group not in GROUPS:
            raise ValueError(f"unexpected row key {key}")
        if bool(r["strict_success"]) != (r["category"] == "CR_screen"):
            raise ValueError(f"success flag/category mismatch: {key}")
        if r["category"] not in allowed_categories:
            raise ValueError(f"unexpected failure category {r['category']}: {key}")
        duration = float(r["transition_cycles"])
        expected_periods = int(np.ceil(duration)) + int(r["relax_periods"]) + int(r["measured_periods"])
        completed = int(r["completed_periods"])
        if (completed < 0 or completed > expected_periods or
                (r["category"] == "CR_screen" and completed != expected_periods) or
                int(r["measured_periods"]) != 15000 or int(r["steps_per_period"]) != 96):
            raise ValueError(f"period/step count mismatch: {key}")
        for field in ("initial_state", "initial_rf_phase", "final_state", "turns", "window_turns",
                      "frequency_Hz", "r_min", "r_max", "max_abs_z", "minimum_separation"):
            if not np.all(np.isfinite(np.asarray(r[field], dtype=np.float64))):
                raise ValueError(f"non-finite {field} in {key}")
        category_counts[str(pid)].setdefault(group, Counter())
        category_counts[str(pid)][group][r["category"]] += 1
        if r["category"] == "CR_screen":
            freq = np.asarray(r["frequency_Hz"], dtype=float)
            turns = np.asarray(r["turns"], dtype=float)
            windows = np.asarray(r["window_turns"], dtype=float)
            if not r["retained"] or freq[0] * freq[1] >= 0 or turns[0] * turns[1] >= 0:
                raise ValueError(f"success does not have retained opposite rotations: {key}")
            if np.min(np.abs(turns)) < 10 or windows.shape != (3, 2) or np.any(windows[:, 0] * windows[:, 1] >= 0):
                raise ValueError(f"success misses turn or late-window criteria: {key}")
            if float(r["minimum_separation"]) <= particle_radius_sum:
                raise ValueError(f"successful trajectory approaches finite-body contact: {key}")

    summaries = []
    for pid in PROTOCOLS:
        per_group = {}
        total = 0
        retained_total = 0
        direction_counts = Counter()
        for group in GROUPS:
            group_rows = [records[(pid, group, sid)] for sid in range(n)]
            success = sum(r["category"] == "CR_screen" for r in group_rows)
            retained = sum(bool(r["retained"]) for r in group_rows)
            total += success
            retained_total += retained
            cats = Counter(r["category"] for r in group_rows)
            for r in group_rows:
                if r["category"] == "CR_screen":
                    direction_counts[tuple("+" if f > 0 else "-" for f in r["frequency_Hz"])] += 1
            per_group[group] = {
                "n": n, "strict_counterrotations": success, "p_cr_unconditional": success / n,
                "wilson95": wilson(success, n), "retained": retained,
                "retention_fraction": retained / n, "outcomes": dict(cats),
            }
        summaries.append({
            "protocol_id": pid,
            "label": next(p["label"] for p in protocol_rows if int(p["id"]) == pid),
            "n_total": n * len(GROUPS), "strict_counterrotations": total,
            "p_cr_unconditional": total / (n * len(GROUPS)),
            "wilson95_overall": wilson(total, n * len(GROUPS)),
            "retained_total": retained_total,
            "retention_total_fraction": retained_total / (n * len(GROUPS)),
            "J_standard": min(per_group[g]["p_cr_unconditional"] for g in GROUPS[:3]),
            "J_robust": min(per_group[g]["p_cr_unconditional"] for g in GROUPS),
            "direction_counts_among_strict_successes": {"".join(k): v for k, v in direction_counts.items()},
            "groups": per_group,
        })

    paired = []
    for group in GROUPS:
        for a, b in ((1, 2), (1, 4), (2, 4)):
            only_a = only_b = both = neither = 0
            for sid in range(n):
                sa = records[(a, group, sid)]["category"] == "CR_screen"
                sb = records[(b, group, sid)]["category"] == "CR_screen"
                if sa and sb: both += 1
                elif sa: only_a += 1
                elif sb: only_b += 1
                else: neither += 1
            discordant = only_a + only_b
            p = float(binomtest(min(only_a, only_b), discordant, 0.5).pvalue) if discordant else 1.0
            paired.append({"ensemble": group, "protocol_a": a, "protocol_b": b,
                           "both_success": both, "a_only_success": only_a,
                           "b_only_success": only_b, "both_failure": neither,
                           "mcnemar_exact_two_sided_p": p})

    result = {
        "status": "passed",
        "rows": len(records), "expected_rows": expected, "unique_keys": len(records),
        "n_per_protocol_ensemble": n, "frozen_starts_identical_across_protocols": True,
        "frozen_start_checksums": hash_audit,
        "integrator_steps_per_primary_RF_period": 96,
        "relax_periods": 200, "measured_periods_after_load": 15000,
        "frequency_orientation_among_strict_CR": "validated in each row; aggregates below",
        "protocol_summaries": summaries,
        "paired_same_start_comparisons": paired,
    }
    (OUT / "holdout_integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(records)} unique rows; frozen starts and checksums agree; all strict success criteria rechecked.")
    for s in summaries:
        rates = ", ".join(f"{g} {s['groups'][g]['strict_counterrotations']}/{n}" for g in GROUPS)
        print(f"{s['label']}: {rates}; J={s['J_standard']:.4f}; Jrobust={s['J_robust']:.4f}; overall={s['p_cr_unconditional']:.4f}")


if __name__ == "__main__":
    main()
