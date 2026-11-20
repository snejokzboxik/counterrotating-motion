"""Independent structural checks for the MFS/FEM field audit artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rescaled_legacy_candidate_validation"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    mfs = json.loads((OUT / "mfs_field_audit.json").read_text(encoding="utf-8"))
    fem = json.loads((OUT / "fem_field_audit.json").read_text(encoding="utf-8"))
    meta = json.loads((OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    candidate_ids = {c["id"] for c in meta["candidates"]}
    mfs_ids = set(mfs["results"])
    fem_ids = set(fem["refinement"][-1]["candidate_fields"])
    p32 = next(x for x in mfs["results"]["scaled_refine12_sobol17_f50hz"]["mfs_convergence"] if x["p_collocation_per_rod"] == 32)
    p24 = next(x for x in mfs["results"]["scaled_refine12_sobol17_f50hz"]["mfs_convergence"] if x["p_collocation_per_rod"] == 24)
    fem_best = [row["candidate_fields"]["scaled_refine12_sobol17_f50hz"]["rf_only_surface_max_V_m"] for row in fem["refinement"]]
    mfs_best = p32["rf_only_max_surface_field_V_m_all"]
    checks = {
        "all_three_candidates_present_in_MFS_and_FEM": candidate_ids == mfs_ids == fem_ids,
        "MFS_refine12_order_converges_24_to_32": abs(p32["rf_only_max_surface_field_V_m_all"] / p24["rf_only_max_surface_field_V_m_all"] - 1.0) < 1e-3,
        "MFS_direct_phase_torus_check_within_1e-4": p32["support_vs_direct_torus_max_rel_diff"] < 1e-4,
        "FEM_refine12_monotonic_mesh_convergence": fem_best[0] < fem_best[1] < fem_best[2] < mfs_best,
        "FEM_fine_mesh_within_2pct_of_MFS": abs(fem_best[-1] / mfs_best - 1.0) < 0.02,
        "MFS_source_hash_matches": mfs["mfs_source_sha256"] == sha(ROOT / "model" / "mfs_breakdown.py"),
        "FEM_source_hash_matches": fem["source_hashes"]["fem"] == sha(ROOT / "model" / "fem_crosscheck.py"),
        "field_model_hashes_match": mfs["source_meta_sha256"] == fem["source_hashes"]["campaign_meta"] == sha(OUT / "campaign_meta.json"),
    }
    report = {"audit": "rescaled_candidate_field_audit", "integrity_pass": all(checks.values()),
              "checks": checks, "mfs_refine12_max_V_m": mfs_best,
              "fem_refine12_refinement_V_m": fem_best,
              "fem_fine_relative_to_mfs": fem_best[-1] / mfs_best - 1.0,
              "scope": "Independent numerical cross-check of 2-D RF-only conductor-boundary maxima; not a discharge/breakdown assessment.",
              "mfs_artifact_sha256": sha(OUT / "mfs_field_audit.json"),
              "fem_artifact_sha256": sha(OUT / "fem_field_audit.json"),
              "audit_script_sha256": sha(Path(__file__))}
    (OUT / "field_independent_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
