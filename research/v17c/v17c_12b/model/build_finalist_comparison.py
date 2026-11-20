"""Assemble the final V17C.12B candidate comparison from audited campaign artifacts."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    best_path = ROOT / "best_physical_configuration.json"
    best = read(best_path)
    best["status"] = (
        "no_confirmed_physically_realizable_configuration; 32 conditionally selected starts exceeded 100 orbits, "
        "but the full basin and electrical/material feasibility remain unqualified"
    )
    best["best_discovery_candidate"]["status"] = (
        "best strict fresh-start CR frequency among current mapped candidates; 32 selected starts passed 240000-period "
        "and 100-orbit checks, but this is conditional persistence rather than basin-wide stability or physical qualification"
    )
    screen = read(RESULTS / "physical_parameter_sobol_screen" / "campaign_meta.json")
    screen_configs = {item["id"]: item for item in screen["configurations"]}
    short = read(RESULTS / "updated_objective_validation" / "summary.json")
    scaled = read(RESULTS / "rescaled_legacy_candidate_validation" / "summary.json")
    fresh = read(RESULTS / "additional_physical_candidate_validation" / "summary.json")
    fresh_audit = read(RESULTS / "additional_physical_candidate_validation" / "independent_audit.json")
    field_audit = read(RESULTS / "rescaled_legacy_candidate_validation" / "field_independent_audit.json")
    if not fresh_audit["integrity_pass"] or not field_audit["integrity_pass"]:
        raise SystemExit("refusing to write best-configuration summary from a failed independent audit")

    legacy_names = {
        "candidate52": "legacy_candidate52",
        "local15": "legacy_local15",
        "refine12_sobol17": "legacy_refine12_sobol17",
    }
    literal_legacy = {name: short["configuration_results"][key] for name, key in legacy_names.items()}
    rescaled_names = {
        "candidate52": "scaled_candidate52_f45hz",
        "local15": "scaled_local15_f50hz",
        "refine12_sobol17": "scaled_refine12_sobol17_f50hz",
    }
    frequency_rescaled_legacy = {name: scaled["configuration_results"][key] for name, key in rescaled_names.items()}

    range_supported = short["configuration_results"]["phys0570"]
    best["final_comparative_validation"] = {
        "objective": "Any strict sustained counterrotation direction (+−) or (−+) is success; no balance between directions required. Co-rotation, wall loss, axial loss, axis ambiguity, and unresolved motion remain separate outcomes.",
        "classification": "Fresh-start RK4/64 for 15000 RF periods; strict direction requires consistent signs in three late windows and at least 10 net turns per particle. Four fixed ensembles have 128 starts each; these are stratified sample frequencies, not IID probabilities.",
        "literal_10mm_air_legacy_mappings": literal_legacy,
        "frequency_rescaled_legacy_candidates": frequency_rescaled_legacy,
        "fresh_physical_screen_leader": {
            "configuration": screen_configs["phys0570"],
            "validation": range_supported,
            "interpretation": "Best fresh-start result among the screened pairs whose two particle diameters lie within the cited Al2O3/hollow-glass experimental size ranges; still below 80% and not a jointly measured particle pair.",
        },
        "additional_physical_finalists": {
            "campaign": "results/additional_physical_candidate_validation",
            "trajectory_sha256": fresh["trajectory_sha256"],
            "independent_audit_trajectory_sha256": fresh_audit["trajectory_sha256"],
            "independent_audit_file_sha256": sha(RESULTS / "additional_physical_candidate_validation" / "independent_audit.json"),
            "independent_audit_script_sha256": fresh_audit["audit_script_sha256"],
            "audit_integrity_pass": fresh_audit["integrity_pass"],
            "candidates": {
                cid: {"configuration": screen_configs[cid], "validation": result}
                for cid, result in fresh["configuration_results"].items()
            },
            "interpretation": "All three pairs fall within cited particle-diameter ranges, but each fresh-start strict CR frequency is below the 80% target.",
        },
        "field_validation_scope": {
            "candidate": "scaled_refine12_sobol17_f50hz",
            "independent_MFS_FEM_audit_pass": field_audit["integrity_pass"],
            "MFS_RF_only_max_V_m": field_audit["mfs_refine12_max_V_m"],
            "FEM_refinement_RF_only_max_V_m": field_audit["fem_refine12_refinement_V_m"],
            "interpretation": "Field cross-check is for the ideal 2-D electrode cross-section. It is not a gas-breakdown, corona, vacuum-emission, or finite-length device qualification.",
        },
        "aggregate_record_counts": {
            "audited_saved_trajectory_rows_across_campaigns": 155300,
            "campaign_configuration_evaluations_including_repeats": 2516,
            "discarded_integrations_from_attempt_02_not_in_saved_row_count": 128,
            "attempt_01_completed_rows": 0,
        },
        "scientific_status": "No physically confirmed configuration is established. The scaled refine12 point clears 80% only as a dimensionless-preserving physical mapping; its 214 um hollow-sphere diameter is outside cited tested distributions and its electrode-field/breakdown qualification is unresolved. The better size-range-supported point phys0570 reaches 71.875%.",
    }
    best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"updated": str(best_path),
        "best_dynamic": best["best_discovery_candidate"]["configuration"],
        "size_range_supported_CR": range_supported["strict_CR_any_direction_of_512"],
        "additional_audit_pass": fresh_audit["integrity_pass"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
