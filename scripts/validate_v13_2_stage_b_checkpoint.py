"""Validate and summarize a resumable V13.2 Stage B JSONL checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability/stage_b_checkpoint.jsonl"
OUT = DEFAULT_CHECKPOINT.parent


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def result_key(row: dict) -> tuple:
    return (round(float(row["alpha"]), 9), round(float(row["beta"]), 9),
            round(float(row["phase_rad"]), 12), row["model"],
            round(float(row["eta"]), 8), int(row["index"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected", type=int, default=22_800)
    args = parser.parse_args()

    rows = []
    for line_no, line in enumerate(args.checkpoint.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at line {line_no}: {exc}") from exc
        rows.append(row)

    key_counts = Counter(
        (
            round(float(row["alpha"]), 9),
            round(float(row["beta"]), 9),
            round(float(row["phase_rad"]), 12),
            row["model"],
            round(float(row["eta"]), 8),
            int(row["index"]),
        )
        for row in rows
    )
    duplicates = sum(count - 1 for count in key_counts.values() if count > 1)
    if duplicates:
        raise SystemExit(f"Checkpoint contains {duplicates} duplicate task keys")

    status_counts = Counter(row.get("status_100", "missing") for row in rows)
    bad = sum(n for status, n in status_counts.items() if status not in {"retained", "escaped"})
    output = {
        "checkpoint": str(args.checkpoint.relative_to(ROOT)) if args.checkpoint.is_relative_to(ROOT) else str(args.checkpoint),
        "completed_unique_tasks": len(rows),
        "expected_total_tasks": args.expected,
        "remaining_tasks": max(0, args.expected - len(rows)),
        "duplicate_task_keys": duplicates,
        "invalid_or_unresolved_statuses": bad,
        "status_counts": dict(sorted(status_counts.items())),
        "integrity": "ok" if len(rows) <= args.expected and bad == 0 else "needs_review",
    }
    checks = {}
    if len(rows) == args.expected and args.checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve():
        expected_outputs = {
            "stage_b_trajectory_results.csv": 22_800,
            "stage_b_nonlinear_map_summary.csv": 2_850,
            "stage_b_floquet_grid.csv": 285,
            "stage_b_floquet_boundaries.csv": 90,
            "stage_b_voltage_feasibility.csv": 1_425,
        }
        for name, expected_rows in expected_outputs.items():
            table = csv_rows(OUT / name)
            checks[name] = {"rows": len(table), "expected_rows": expected_rows,
                            "count_ok": len(table) == expected_rows}
        final_rows = csv_rows(OUT / "stage_b_trajectory_results.csv")
        final_keys = [result_key(row) for row in final_rows]
        checks["trajectory_csv_keys_match_checkpoint"] = set(final_keys) == set(key_counts) and len(set(final_keys)) == len(final_keys)
        checks["all_Floquet_boundaries_found"] = all(row["status"] == "root_found" for row in csv_rows(OUT / "stage_b_floquet_boundaries.csv"))
        checks["no_unresolved_summary_cells"] = all(int(row["n_unresolved"]) == 0 for row in csv_rows(OUT / "stage_b_nonlinear_map_summary.csv"))
        local = OUT / "local_candidate_refinement"
        if (local / "retention_summary.csv").exists():
            local_summary = csv_rows(local / "retention_summary.csv")
            checks["local_refinement"] = {
                "rows": len(csv_rows(local / "trajectory_results.csv")),
                "summary_rows": len(local_summary),
                "unresolved": sum(int(row["status_100"] not in {"retained", "escaped"})
                                   for row in csv_rows(local / "trajectory_results.csv")),
                "RO_over_SO_groups": sum(float(row["paired_RO_minus_SO"]) > 0 for row in local_summary),
            }
        step_summary = local / "step_halving" / "validation_summary.json"
        if step_summary.exists():
            checks["step_halving"] = json.loads(step_summary.read_text(encoding="utf-8"))
        fem = OUT / "stage_b_full_fem_h50_p4_controls"
        if (fem / "trajectory_results.csv").exists():
            fem_rows = csv_rows(fem / "trajectory_results.csv")
            checks["selected_full_FEM"] = {
                "rows": len(fem_rows),
                "unresolved": sum(row["status_100"] not in {"retained", "escaped"} for row in fem_rows),
            }
        count_checks = [v["count_ok"] for v in checks.values() if isinstance(v, dict) and "count_ok" in v]
        no_unresolved = checks.get("no_unresolved_summary_cells", False)
        output["final_output_checks"] = checks
        bool_checks = [checks.get("trajectory_csv_keys_match_checkpoint", False),
                       checks.get("all_Floquet_boundaries_found", False), no_unresolved]
        if "local_refinement" in checks:
            local_check = checks["local_refinement"]
            bool_checks.append(local_check["rows"] == 1_728 and local_check["summary_rows"] == 18
                               and local_check["unresolved"] == 0 and local_check["RO_over_SO_groups"] == 0)
        if "step_halving" in checks:
            step_check = checks["step_halving"]
            bool_checks.append(step_check["n"] == 384 and step_check["classification_changes"] == 0)
        if "selected_full_FEM" in checks:
            fem_check = checks["selected_full_FEM"]
            bool_checks.append(fem_check["rows"] == 384 and fem_check["unresolved"] == 0)
        output["integrity"] = "ok" if output["integrity"] == "ok" and all(count_checks) and all(bool_checks) else "needs_review"
        (OUT / "stage_b_integrity.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
