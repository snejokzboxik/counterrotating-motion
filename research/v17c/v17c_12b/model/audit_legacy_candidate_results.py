"""Extract comparable V17C.10 strict-count tables into the V17C.12B record."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V17C = HERE.parents[1]
V10 = V17C/"v17c_10"
OUT = HERE.parent/"results"/"legacy_candidate_v17c10_comparison.json"


def main() -> None:
    local = json.loads((V10/"local15_refinement_confirmation_N256"/
        "paired_validation_summary.json").read_text(encoding="utf-8"))
    refine = json.loads((V10/"refine12_neighborhood_validation"/
        "paired_validation_summary.json").read_text(encoding="utf-8"))
    rows = {}
    for name in ("candidate52", "local15"):
        rows[name] = {"source": "local15_refinement_confirmation_N256/paired_validation_summary.json",
            "protocol": local["protocol"], "ensembles": local["by_candidate_and_ensemble"][name]}
    for name in ("refine12", "refine12_sobol17"):
        rows[name] = {"source": "refine12_neighborhood_validation/paired_validation_summary.json",
            "protocol": refine["protocol"], "ensembles": refine["by_candidate_and_ensemble"][name]}
    out = {"study": "V17C.10 strict CR outcomes, legacy virtual-particle parameters",
        "interpretation": [
            "These are the archived V17C.10 outcomes under its original drag-free, point-dynamics setup with finite-radius collision tests.",
            "They count the single CR orientation accepted by that protocol and do not establish the reverse orientation or bistability.",
            "The V17C.10 common-medium Stokes mapping is inconsistent at its archived radii; see PHYSICAL_PARAMETERIZATION_RU.md.",
            "Denominators remain stratified by the four start ensembles and must not be pooled as a universal probability."],
        "candidates": rows,
        "all_checks_pass_in_source_summaries": {
            "local15_refinement_confirmation_N256": local["all_checks_pass"],
            "refine12_neighborhood_validation": refine["all_checks_pass"]}}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({name: {e: {"n": v["n"], "strict_cr": v["strict_cr"],
        "strict_fraction": v["strict_fraction"], "category_counts": v["category_counts"]}
        for e, v in d["ensembles"].items()} for name, d in rows.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
