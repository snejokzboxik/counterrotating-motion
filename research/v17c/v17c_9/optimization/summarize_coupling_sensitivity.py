"""Summarize paired candidate52 holdouts across Coulomb-coupling scales."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ROOT = HERE / "stage4_candidate48"
N = 128
A1, A2 = 0.0002, 0.00189
BASE_TAG = "pairedholdout20261150"


def wilson(k, n=N, z=1.959963984540054):
    p = k/n
    den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [max(0., c-h), min(1., c+h)]


def read(path):
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    data = {int(r["index"]): r for r in rows}
    if len(rows) != N or len(data) != N or set(data) != set(range(N)):
        raise ValueError(f"expected {N} unique trials: {path}")
    return data


def summarize(rows):
    cr = [r for r in rows.values() if r["category"] == "CR_screen"]
    retained = [r for r in rows.values() if r["retained"]]
    return {
        "N_total": N,
        "category_counts": {c: sum(r["category"] == c for r in rows.values())
                            for c in sorted({r["category"] for r in rows.values()})},
        "N_CR": len(cr), "P_CR": len(cr)/N, "Wilson95_CR": wilson(len(cr)),
        "N_retained": len(retained), "P_retained": len(retained)/N,
        "Wilson95_retained": wilson(len(retained)),
        "P_CR_given_retention": len(cr)/len(retained) if retained else None,
        "median_frequency_Hz_CR": np.median([r["frequency_Hz"] for r in cr], axis=0).tolist() if cr else None,
    }


def main():
    out = {"study": "paired candidate52 Coulomb sensitivity N=128",
           "parameters": {"config_id": 52, "scales": [0, 1, 2], "periods": 15000,
                          "steps": 64, "sphere_radii_over_R": [A1, A2],
                          "matched_holdout_seed_tag": BASE_TAG}, "ensembles": {}}
    runs = ROOT / "holdout_runs"
    for group in ("rest", "weak", "wide"):
        suffix = f"matched_holdout_none_chi0-0_a{A1:g}-{A2:g}_{{tag}}_128_15000p_64s.jsonl"
        files = {
            "zero": runs / f"cfg52_{group}_{suffix.format(tag='Cscale0_pairedCscale0_20261150')}",
            "nominal": runs / f"cfg52_{group}_{suffix.format(tag=BASE_TAG)}",
            "double": runs / f"cfg52_{group}_{suffix.format(tag='Cscale2_pairedCscale2_20261150')}",
        }
        rows = {key: read(path) for key, path in files.items()}
        nominal = rows["nominal"]
        for key in ("zero", "double"):
            for i in range(N):
                if not np.array_equal(nominal[i]["initial"], rows[key][i]["initial"]) or \
                   not np.array_equal(nominal[i]["phase"], rows[key][i]["phase"]):
                    raise ValueError(f"initial conditions mismatch: {group}:{key}:{i}")
        group_out = {}
        for key, label in (("zero", "C=0"), ("nominal", "C=C0"), ("double", "C=2C0")):
            group_out[label] = summarize(rows[key])
        nominal_cr = {i for i,r in nominal.items() if r["category"] == "CR_screen"}
        group_out["paired_vs_nominal"] = {}
        for key,label in (("zero","C=0"),("double","C=2C0")):
            cr={i for i,r in rows[key].items() if r["category"] == "CR_screen"}
            c_only=len(cr-nominal_cr); n_only=len(nominal_cr-cr); discord=c_only+n_only
            p=binomtest(c_only,discord,.5,alternative="two-sided").pvalue if discord else 1.0
            group_out["paired_vs_nominal"][label]={"control_only_CR":c_only,
                "nominal_only_CR":n_only,"both_CR":len(cr&nominal_cr),
                "neither_CR":N-len(cr|nominal_cr),"exact_McNemar_p_two_sided":float(p)}
        out["ensembles"][group]=group_out
    dest=ROOT/"coupling_sensitivity_holdout_summary.json"
    dest.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
