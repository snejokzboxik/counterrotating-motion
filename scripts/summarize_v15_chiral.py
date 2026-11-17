"""Reconstruct V15 chirality and torque summaries from checkpoint rows."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v15"
CHECKPOINT = OUT / "ideal_chiral_checkpoint.jsonl"


def ci95(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    half = float(student_t.ppf(0.975, len(vals)-1) * np.std(vals, ddof=1) / math.sqrt(len(vals)))
    return mean-half, mean+half


def write_csv(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    rows = []
    keys = set()
    run_hashes = set()
    with CHECKPOINT.open(encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["key"] in keys:
                raise RuntimeError(f"duplicate row key {r['key']}")
            keys.add(r["key"]); run_hashes.add(r["run_hash"]); rows.append(r)
    if len(run_hashes) != 1:
        raise RuntimeError(f"expected one protocol hash, found {len(run_hashes)}")

    grouped = defaultdict(dict)
    for r in rows:
        gkey = (r["point_id"], float(r["mu"]), float(r["delta_rad"]), float(r["tau0_rad"]), r["config"])
        pkey = (int(r["replica"]), int(r["pair_index"]), int(r["Lz_sign"]))
        if pkey in grouped[gkey]:
            raise RuntimeError(f"duplicate subgroup row {gkey} {pkey}")
        grouped[gkey][pkey] = r

    summary = []
    replicate_metrics = {}
    exact_so_mismatch = 0
    exact_q_mismatch = 0
    exact_ro_reflection_mismatch = 0
    incomplete_groups = []
    for gkey, records in sorted(grouped.items()):
        point, mu, delta, tau0, config = gkey
        reps = sorted({p[0] for p in records})
        rep_rows = []
        for rep in reps:
            plus = [records[(rep, i, 1)] for i in range(16) if (rep, i, 1) in records]
            minus = [records[(rep, i, -1)] for i in range(16) if (rep, i, -1) in records]
            if len(plus) != 16 or len(minus) != 16:
                incomplete_groups.append([*gkey, rep, len(plus), len(minus)])
                continue
            rp = np.mean([r["status"] == "retained" for r in plus])
            rm = np.mean([r["status"] == "retained" for r in minus])
            re_total = 0.5 * (rp + rm)
            dchi = rp-rm
            plus_exits = [float(r["exit_rf_step_estimate"]) for r in plus if r["status"] == "escaped"]
            minus_exits = [float(r["exit_rf_step_estimate"]) for r in minus if r["status"] == "escaped"]
            all_exit = plus_exits + minus_exits
            all_dell = [float(r["Lz_end_dimensionless"])-float(r["Lz0_dimensionless"]) for r in plus+minus]
            surv_dell = [float(r["Lz_end_dimensionless"])-float(r["Lz0_dimensionless"]) for r in plus+minus if r["status"] == "retained"]
            esc_dell = [float(r["Lz_end_dimensionless"])-float(r["Lz0_dimensionless"]) for r in plus+minus if r["status"] == "escaped"]
            torq_res = [abs(float(r["torque_identity_residual"])) for r in plus+minus]
            rep_rows.append(dict(replica=rep, P_plus=float(rp), P_minus=float(rm), P_total=float(re_total),
                                 delta_chi=float(dchi), median_exit_rf=float(np.median(all_exit)) if all_exit else None,
                                 mean_delta_ell=float(np.mean(all_dell)),
                                 mean_delta_ell_survivors=float(np.mean(surv_dell)) if surv_dell else None,
                                 mean_delta_ell_escaped=float(np.mean(esc_dell)) if esc_dell else None,
                                 max_abs_torque_residual=float(max(torq_res))))
        if len(rep_rows) < 4:
            incomplete_groups.append([*gkey, "replicas", len(rep_rows)])
            continue
        for model_name, values in (("P_plus", [x["P_plus"] for x in rep_rows]),
                                   ("P_minus", [x["P_minus"] for x in rep_rows]),
                                   ("P_total", [x["P_total"] for x in rep_rows]),
                                   ("delta_chi", [x["delta_chi"] for x in rep_rows])):
            lo, hi = ci95(values)
            summary.append(dict(point_id=point, mu=mu, delta_rad=delta, tau0_rad=tau0,
                                config=config, metric=model_name, mean=float(np.mean(values)),
                                ci95_low=lo, ci95_high=hi,
                                replicate_values=json.dumps(values, separators=(",", ":")),
                                retained_plus=int(round(sum(x["P_plus"]*16 for x in rep_rows))),
                                retained_minus=int(round(sum(x["P_minus"]*16 for x in rep_rows))),
                                total_pairs=16*len(rep_rows), unresolved=sum(r["status"] == "unresolved" for r in records.values())))
        denom = float(np.mean([x["P_plus"]+x["P_minus"] for x in rep_rows]))
        dchi = float(np.mean([x["delta_chi"] for x in rep_rows]))
        summary.append(dict(point_id=point, mu=mu, delta_rad=delta, tau0_rad=tau0,
                            config=config, metric="C_chi" if denom > 0.05 else "C_chi_undefined_low_denominator",
                            mean=dchi/denom if denom > 0.05 else None,
                            ci95_low=None, ci95_high=None,
                            replicate_values=json.dumps([x["delta_chi"] for x in rep_rows], separators=(",", ":")),
                            retained_plus=int(round(sum(x["P_plus"]*16 for x in rep_rows))),
                            retained_minus=int(round(sum(x["P_minus"]*16 for x in rep_rows))),
                            total_pairs=16*len(rep_rows), unresolved=sum(r["status"] == "unresolved" for r in records.values())))
        replicate_metrics[gkey] = rep_rows

    # Exact equivariance checks on raw pairwise classifications.
    basekeys = {(r["point_id"], r["mu"], r["delta_rad"], r["tau0_rad"], r["replica"], r["pair_index"])
                for r in rows}
    lookup = {(r["point_id"], float(r["mu"]), float(r["delta_rad"]), float(r["tau0_rad"]),
               int(r["replica"]), int(r["pair_index"]), r["config"], int(r["Lz_sign"])): r for r in rows}
    for point, mu, delta, tau0, rep, pair in basekeys:
        for model in ("Q", "SO_equal_harmonic", "SO_equal_rms"):
            a = lookup.get((point, float(mu), float(delta), float(tau0), int(rep), int(pair), model, 1))
            b = lookup.get((point, float(mu), float(delta), float(tau0), int(rep), int(pair), model, -1))
            if a and b and a["status"] != b["status"]:
                if model == "Q": exact_q_mismatch += 1
                else: exact_so_mismatch += 1
        rp = lookup.get((point, float(mu), float(delta), float(tau0), int(rep), int(pair), "RO_PLUS", 1))
        rm = lookup.get((point, float(mu), float(delta), float(tau0), int(rep), int(pair), "RO_MINUS", -1))
        rp_m = lookup.get((point, float(mu), float(delta), float(tau0), int(rep), int(pair), "RO_PLUS", -1))
        rm_p = lookup.get((point, float(mu), float(delta), float(tau0), int(rep), int(pair), "RO_MINUS", 1))
        if rp and rm and rp["status"] != rm["status"]: exact_ro_reflection_mismatch += 1
        if rp_m and rm_p and rp_m["status"] != rm_p["status"]: exact_ro_reflection_mismatch += 1

    write_csv(OUT / "ideal_chiral_summary.csv", summary)
    audit = dict(rows=len(rows), unique_keys=len(keys), protocol_hash=next(iter(run_hashes)),
                 unresolved=sum(r["status"] == "unresolved" for r in rows),
                 over_250V=sum(not bool(r["within_250V"]) for r in rows),
                 max_torque_identity_residual=max((abs(float(r["torque_identity_residual"])) for r in rows), default=0.0),
                 exact_Q_mirror_status_mismatches=exact_q_mismatch,
                 exact_SO_mirror_status_mismatches=exact_so_mismatch,
                 RO_reflection_pair_status_mismatches=exact_ro_reflection_mismatch,
                 incomplete_groups=incomplete_groups)
    (OUT / "ideal_chiral_symmetry_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
