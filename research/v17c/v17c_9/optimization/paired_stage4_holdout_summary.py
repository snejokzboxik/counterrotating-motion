"""Paired finite-sphere holdout comparison: refined config52 versus candidate48."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
BASELINE_RUNS = HERE / "stage3" / "runs"
REFINED_RUNS = HERE / "stage4_candidate48" / "holdout_runs"
TAG = "pairedholdout20261150"
A1, A2 = 0.0002, 0.00189
N = 128


def wilson(k, n, z=1.959963984540054):
    p = k/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0., center-half), min(1., center+half)]


def read_rows(path):
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    result = {int(r["index"]): r for r in rows}
    if len(rows) != N or len(result) != N or set(result) != set(range(N)):
        raise ValueError(f"expected exactly {N} unique rows in {path}, found {len(rows)}")
    return result


def summarize(rows):
    cr = [r for r in rows.values() if r["category"] == "CR_screen"]
    retained = [r for r in rows.values() if r["retained"]]
    return {"N_total": N, "category_counts": {c: sum(r["category"] == c for r in rows.values())
             for c in sorted({r["category"] for r in rows.values()})},
        "N_CR": len(cr), "P_CR": len(cr)/N, "Wilson95_CR": wilson(len(cr), N),
        "N_retained": len(retained), "P_retained": len(retained)/N,
        "Wilson95_retained": wilson(len(retained), N),
        "P_CR_given_retention": len(cr)/len(retained) if retained else None,
        "median_frequency_Hz_CR": np.median([r["frequency_Hz"] for r in cr], axis=0).tolist() if cr else None,
        "median_minimum_separation_R_CR": float(np.median([r["minimum_separation"] for r in cr])) if cr else None}


def main():
    out = {"study": "fresh paired N=128 finite-sphere holdout",
        "parameters": {"refined_config": 52, "baseline_config": 48,
            "periods": 15000, "steps_per_period": 64, "sphere_radii_over_R": [A1, A2],
            "shared_holdout_seed_tag": TAG}, "ensembles": {}}
    for group in ("rest", "weak", "wide"):
        suffix = f"matched_holdout_none_chi0-0_a{A1:g}-{A2:g}_{TAG}_{N}_15000p_64s.jsonl"
        b = read_rows(BASELINE_RUNS / f"cfg48_{group}_{suffix}")
        c = read_rows(REFINED_RUNS / f"cfg52_{group}_{suffix}")
        for i in range(N):
            if not np.array_equal(np.asarray(b[i]["initial"]), np.asarray(c[i]["initial"])) or \
               not np.array_equal(np.asarray(b[i]["phase"]), np.asarray(c[i]["phase"])):
                raise ValueError(f"initial condition or RF phase mismatch: {group}:{i}")
        base_cr = {i for i, r in b.items() if r["category"] == "CR_screen"}
        cand_cr = {i for i, r in c.items() if r["category"] == "CR_screen"}
        candidate_only = len(cand_cr-base_cr)
        baseline_only = len(base_cr-cand_cr)
        discordant = candidate_only+baseline_only
        mcnemar = binomtest(candidate_only, discordant, .5, alternative="two-sided").pvalue if discordant else 1.0
        out["ensembles"][group] = {"candidate48": summarize(b), "config52": summarize(c),
            "paired_CR_change": {"config52_only_CR": candidate_only,
                "candidate48_only_CR": baseline_only,
                "same_CR": len(cand_cr & base_cr),
                "neither_CR": N-len(cand_cr | base_cr),
                "exact_McNemar_p_two_sided": float(mcnemar)}}
    dest = HERE / "stage4_candidate48" / "paired_holdout_summary.json"
    dest.write_text(json.dumps(out, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
