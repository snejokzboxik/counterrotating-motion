"""Independent audit of the three additional physical finalists."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "additional_physical_candidate_validation"
SCREEN = ROOT / "results" / "physical_parameter_sobol_screen"
GROUPS = ("rest", "weak", "wide", "edge")
STRICT_TURNS = 10.0


def label(raw: str, turns: np.ndarray, windows: np.ndarray, retained: bool, axis: bool, threshold: float) -> str:
    if raw == "finite_particle_contact": return "PARTICLE_CONTACT"
    if raw == "finite_particle_wall_contact": return "WALL_OR_AXIAL_ESCAPE"
    if raw == "lost_retention": return "ESCAPE"
    if raw == "point_singularity_unresolved": return "UNRESOLVED"
    if axis: return "AXIS_AMBIGUOUS"
    signs = np.zeros((3, 2), dtype=np.int8)
    for wi in range(3):
        for pi in range(2):
            if abs(windows[wi, pi]) >= 0.05:
                signs[wi, pi] = 1 if windows[wi, pi] > 0 else -1
    if retained and np.all(signs != 0) and np.all(np.abs(turns) >= threshold):
        if np.all(signs[:, 0] * signs[:, 1] < 0):
            if threshold == STRICT_TURNS:
                return "STRICT_CR_PLUS_MINUS" if turns[0] > 0 else "STRICT_CR_MINUS_PLUS"
            return "PROVISIONAL_CR_PLUS_MINUS" if turns[0] > 0 else "PROVISIONAL_CR_MINUS_PLUS"
        if np.all(signs[:, 0] * signs[:, 1] > 0):
            if threshold == STRICT_TURNS:
                return "STRICT_CO_PLUS_PLUS" if turns[0] > 0 else "STRICT_CO_MINUS_MINUS"
            return "PROVISIONAL_CO_PLUS_PLUS" if turns[0] > 0 else "PROVISIONAL_CO_MINUS_MINUS"
    if retained and np.all(np.abs(turns) < 0.25): return "NONROTATING"
    return "UNRESOLVED_LATE_WINDOW" if threshold == STRICT_TURNS else "UNRESOLVED_SHORT"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    meta = json.loads((OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    codes = json.loads((OUT / "campaign_code_sha256.json").read_text(encoding="utf-8"))
    frozen = np.load(OUT / "frozen_starts.npz", allow_pickle=True)
    starts, phases = frozen["initial"], frozen["phases"]
    rows = [json.loads(line) for line in (OUT / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    configs = {c["id"]: c for c in meta["candidates"]}
    groups = meta["ensembles"]
    n = meta["starts_per_ensemble"]
    expected = {(cid, g, i) for cid in configs for g in groups for i in range(n)}
    keys = [(r["configuration"], r["ensemble"], int(r["start_index"])) for r in rows]
    unique_complete = len(keys) == len(set(keys)) and set(keys) == expected
    starts_match = True; horizon = True; strict_labels_match = True; provisional_labels_match = True; reconstructed = {}
    loss_modes = defaultdict(Counter)
    for row in rows:
        cid, group, i = row["configuration"], row["ensemble"], int(row["start_index"])
        idx = groups.index(group) * n + i
        starts_match &= np.array_equal(np.asarray(row["initial"]), starts[idx])
        starts_match &= np.array_equal(np.asarray(row["phase"]), phases[idx])
        horizon &= (bool(row["retained"]) and row["completed_periods"] == meta["periods"]) or (not bool(row["retained"]) and row["completed_periods"] < meta["periods"])
        axis = bool(min(row["r_min_over_R"]) < 0.02)
        turns = np.asarray(row["turns"], dtype=float)
        windows = np.asarray(row["window_turns"], dtype=float)
        strict_labels_match &= label(row["raw_category"], turns, windows, bool(row["retained"]), axis, STRICT_TURNS) == row["strict_late_category"]
        provisional_labels_match &= label(row["raw_category"], turns, windows, bool(row["retained"]), axis, 1.0) == row["provisional_category"]
        if row["strict_late_category"] == "WALL_OR_AXIAL_ESCAPE":
            ph = configs[cid]["physical"]
            radii = (ph["a1_over_R"], ph["a2_over_R"])
            rmax = row["r_max_over_R"]
            zmax = row["z_max_over_R"]
            radial = any(rmax[j] + radii[j] >= 1.0 for j in range(2))
            axial = any(zmax[j] + radii[j] >= 2.0 for j in range(2))
            mode = "radial_only" if radial and not axial else "axial_only" if axial and not radial else "radial_and_axial" if radial and axial else "unresolved_geometry"
            loss_modes[cid][mode] += 1
            if row["completed_periods"] == 0:
                loss_modes[cid]["before_one_completed_RF_period"] += 1
    global_outcomes = Counter()
    for cid in configs:
        rr = [r for r in rows if r["configuration"] == cid]
        cc = Counter(r["strict_late_category"] for r in rr)
        global_outcomes.update(cc)
        groups_out = {}
        for group in GROUPS:
            gr = [r for r in rr if r["ensemble"] == group]
            c = Counter(r["strict_late_category"] for r in gr)
            groups_out[group] = {"n": len(gr), "strict_late": dict(sorted(c.items())),
                "counterrotation_any_direction": c["STRICT_CR_PLUS_MINUS"] + c["STRICT_CR_MINUS_PLUS"],
                "CR_plus_minus": c["STRICT_CR_PLUS_MINUS"], "CR_minus_plus": c["STRICT_CR_MINUS_PLUS"],
                "co_rotation": c["STRICT_CO_PLUS_PLUS"] + c["STRICT_CO_MINUS_MINUS"],
                "escape": c["WALL_OR_AXIAL_ESCAPE"]}
        cr = cc["STRICT_CR_PLUS_MINUS"] + cc["STRICT_CR_MINUS_PLUS"]
        reconstructed[cid] = {"rows": len(rr), "strict_CR_any_direction_of_512": cr,
            "strict_CR_fraction_of_512": cr / 512.0,
            "co_rotation_total": cc["STRICT_CO_PLUS_PLUS"] + cc["STRICT_CO_MINUS_MINUS"],
            "escape_total": cc["WALL_OR_AXIAL_ESCAPE"], "outcome_counts": dict(sorted(cc.items())),
            "per_ensemble": groups_out}
    source_meta = json.loads((SCREEN / "campaign_meta.json").read_text(encoding="utf-8"))
    source_cfgs = {c["id"]: c for c in source_meta["configurations"]}
    candidate_cfg_match = all(configs[cid] == source_cfgs[cid] for cid in configs)
    attempt1_codes = json.loads((OUT / "attempt_01_campaign_code_sha256.json").read_text(encoding="utf-8"))
    attempt2_codes = json.loads((OUT / "attempt_02_campaign_code_sha256.json").read_text(encoding="utf-8"))
    attempt1_diagnostic = json.loads((OUT / "attempt_01_diagnostic.json").read_text(encoding="utf-8"))
    attempt2_diagnostic = json.loads((OUT / "attempt_02_diagnostic.json").read_text(encoding="utf-8"))
    attempt_archives_match = (
        sha(OUT / "campaign_runner_source_attempt01.py") == attempt1_codes["runner_sha256"] == attempt1_codes["saved_runner_sha256"]
        and sha(OUT / "campaign_runner_source_attempt02.py") == attempt2_codes["runner_sha256"] == attempt2_codes["saved_runner_sha256"]
        and attempt1_diagnostic["completed_trajectory_rows"] == attempt1_diagnostic["trajectory_file_bytes_at_stop"] == 0
        and attempt2_diagnostic["saved_trajectory_rows"] == 0
        and attempt2_diagnostic["integrated_calls_without_saved_rows"] == 128
    )
    radial_only_losses = all(
        loss_modes[cid]["radial_only"] == reconstructed[cid]["escape_total"]
        and loss_modes[cid]["axial_only"] == 0
        and loss_modes[cid]["radial_and_axial"] == 0
        and loss_modes[cid]["unresolved_geometry"] == 0
        for cid in configs
    )
    checks = {
        "three_expected_candidates": set(configs) == {"phys0352", "phys0109", "phys0526"},
        "expected_complete_unique_trajectory_keys": len(rows) == len(expected) and unique_complete,
        "frozen_starts_and_phases_match": bool(starts_match),
        "full_retained_horizons_and_early_losses": bool(horizon),
        "strict_and_provisional_labels_recomputed": bool(strict_labels_match and provisional_labels_match),
        "all_wall_losses_reconstruct_as_radial_only": bool(radial_only_losses),
        "candidate_configurations_match_audited_screen": candidate_cfg_match,
        "summary_reconstructed_from_rows": reconstructed == summary["configuration_results"],
        "trajectory_hash_matches_summary": sha(OUT / "trajectories.jsonl") == summary["trajectory_sha256"],
        "campaign_meta_sources_match": meta["source_screen_meta_sha256"] == sha(SCREEN / "campaign_meta.json"),
        "ranking_source_hash_matches": meta["source_screen_ranking_sha256"] == sha(SCREEN / "updated_objective_ranking.json"),
        "runner_hashes_match": sha(ROOT / "additional_physical_candidate_validation.py") == codes["runner_sha256"] == codes["saved_runner_sha256"],
        "integrator_hash_matches": sha(ROOT / "model" / "basin_scan.py") == codes["integrator_sha256"] == codes["saved_integrator_sha256"],
        "field_hash_matches": sha(ROOT / "model" / "field_model.json") == codes["field_model_sha256"],
        "frozen_starts_hash_matches": sha(OUT / "frozen_starts.npz") == codes["frozen_starts_sha256"] == meta["frozen_starts_sha256"],
        "interrupted_attempt_sources_and_diagnostics_preserved": bool(attempt_archives_match),
    }
    report = {"audit": "additional_physical_candidate_validation", "integrity_pass": all(checks.values()),
        "checks": checks, "configurations": len(configs), "rows": len(rows), "unique_keys": len(set(keys)),
        "configuration_results": reconstructed, "trajectory_sha256": sha(OUT / "trajectories.jsonl"),
        "loss_mechanisms": {cid: dict(sorted(counts.items())) for cid, counts in loss_modes.items()},
        "audit_script_sha256": sha(Path(__file__))}
    (OUT / "independent_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "configuration_results"}, indent=2, ensure_ascii=False))
    if not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
