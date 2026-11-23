"""Independent bookkeeping audit for the continuous-rod endcap FEM results."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "continuous_rods_endcap_fem.json"
OUTPUT = RESULTS / "endcap_geometry_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    levels = sorted(data["mesh_levels"], key=lambda row: row["mesh_hmax_mm"], reverse=True)
    assert [round(row["mesh_hmax_mm"], 2) for row in levels] == [.60, .45, .34, .27, .24]
    assert all(row["unclassified_boundary_nodes"] == 0 for row in levels)
    assert len(levels[-1]["linear_solver"]["basis_solves"]) == 6
    residuals = [solve["relative_residual"] for solve in levels[-1]["linear_solver"]["basis_solves"]]
    assert max(residuals) < 1e-9

    intervals = []
    for coarse, fine in zip(levels[:-1], levels[1:]):
        rf_changes = []
        for mode in (2, -4):
            a = next(x["transfer_magnitude_at_0p9R"] for x in coarse["RF_transfer"] if x["m"] == mode)
            b = next(x["transfer_magnitude_at_0p9R"] for x in fine["RF_transfer"] if x["m"] == mode)
            rf_changes.append(abs(b - a) / abs(b))
        za = coarse["DC_endcap_screen"]["particle_force_balance"]
        zb = fine["DC_endcap_screen"]["particle_force_balance"]
        z_changes = [abs(zb[i]["z_eq_mm"] - za[i]["z_eq_mm"]) for i in range(2)]
        passed = max(rf_changes) < .02 and max(z_changes) < .1
        intervals.append({"coarse_hmax_mm": coarse["mesh_hmax_mm"],
                          "fine_hmax_mm": fine["mesh_hmax_mm"],
                          "rf_relative_changes": rf_changes,
                          "z_absolute_changes_mm": z_changes,
                          "per_interval_pass": passed})
    convergence_pass = len(intervals) >= 2 and all(x["per_interval_pass"] for x in intervals[-2:])
    assert convergence_pass == data["convergence_gate"]["passed"]

    attempted = data["incomplete_mesh_attempts"][0]
    archive = RESULTS / attempted["compressed_mesh_archive"]
    assert sha256_file(archive) == attempted["compressed_archive_sha256"]
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        assert len(members) == 1
        digest = hashlib.sha256()
        with zf.open(members[0]) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    assert digest.hexdigest().upper() == attempted["mesh_sha256"]

    finest = levels[-1]
    dc = finest["DC_endcap_screen"]
    voltage = float(dc["endcap_voltage_for_particle1_balance_V"])
    gap_m = .75e-3
    v_over_d = voltage / gap_m
    report = {
        "geometry_definition": data["geometry_definition"],
        "solved_mesh_levels": [{"hmax_mm": x["mesh_hmax_mm"], "vertices": x["vertices"],
                                "tetrahedra": x["tetrahedra"],
                                "unclassified_boundary_nodes": x["unclassified_boundary_nodes"]}
                               for x in levels],
        "finest_solved_linear_residual_max": max(residuals),
        "interval_convergence_audit": intervals,
        "two_consecutive_refinement_gate_passed": convergence_pass,
        "incomplete_finer_mesh": {"hmax_mm": attempted["mesh_hmax_mm"],
                                   "vertices": attempted["vertices"],
                                   "tetrahedra": attempted["tetrahedra"],
                                   "field_solved": False,
                                   "archive_sha256": attempted["compressed_archive_sha256"],
                                   "mesh_sha256": attempted["mesh_sha256"]},
        "endcap_force_balance_screen": {
            "particle1_target_z_mm": dc["particle1_target_z_mm"],
            "equilibrium_z_mm": [x["z_eq_mm"] for x in dc["particle_force_balance"]],
            "predicted_separation_mm": dc["predicted_separation_mm"],
            "equal_endcap_voltage_V": voltage,
            "estimated_nearest_gap_mm": gap_m * 1e3,
            "crude_voltage_over_gap_V_m": v_over_d,
            "screening_only_not_peak_field_or_breakdown_solution": True},
        "final_trajectories_run": False,
        "audit_passed": True,
        "limitations": ["P1 finite-element potential only; full-field derivatives and sharp-edge surface maxima are not convergence-certified",
                        "last-two mesh gate applies only to scalar RF transfers and force-balance roots",
                        "no particle dynamics; endcap voltage screen rejects this placement"]
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_passed": True, "solved_mesh_levels": len(levels),
                      "two_consecutive_refinement_gate_passed": convergence_pass,
                      "maximum_linear_residual": max(residuals),
                      "endcap_voltage_kV": voltage / 1e3,
                      "crude_V_over_d_MV_m": v_over_d / 1e6,
                      "report": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
