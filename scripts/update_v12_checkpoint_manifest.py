#!/usr/bin/env python3
"""Refresh the V12 terminal-checkpoint manifest and emit safe Git pathspecs.

Only immutable, complete trajectory JSON files are written to the NUL-delimited
path list. Live journals, run state, temporary files, and partial checkpoints
are deliberately excluded so the list can be passed to ``git add
--pathspec-from-file=... --pathspec-file-nul`` while queues continue running.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
V12 = REPO / "outputs/octupole_comparison_home/dimensionless_acceptance_v12"
CONFIG_HASH = "e6bf3a33795e4a5354a8"


def load_rows(paths: list[Path]):
    for path in paths:
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-list", required=True, type=Path,
                        help="write NUL-separated repo-relative terminal checkpoint paths here")
    args = parser.parse_args()

    broad_root = V12 / "runs/e6bf3a33795e4a5354a8_equal_broad_lambda_h50_4x256/checkpoints"
    replay_root = V12 / "validation/v11_map_replay_v12/69f9b2c5a9e00ed1"
    broad_paths = sorted(broad_root.rglob("*.json"))
    replay_paths = sorted(replay_root.rglob("*.json"))

    broad_terminal: list[tuple[Path, dict]] = []
    broad_partial: Counter[str] = Counter()
    broad_errors = 0
    for path, row in load_rows(broad_paths):
        if row.get("config_hash") != CONFIG_HASH:
            continue
        status = row.get("status")
        if row.get("complete") is True and status in {"retained", "escaped"}:
            broad_terminal.append((path, row))
        elif status == "numerical_error":
            broad_errors += 1
        else:
            broad_partial[row.get("model", "unknown")] += 1

    replay_terminal: list[tuple[Path, dict]] = []
    replay_partial = 0
    replay_errors = 0
    replay_mismatches = 0
    replay_by_phase_model: Counter[str] = Counter()
    replay_exit_diffs: list[float] = []
    for path, row in load_rows(replay_paths):
        if row.get("run_status") == "numerical_error":
            replay_errors += 1
            continue
        if row.get("run_status") != "complete":
            replay_partial += 1
            continue
        replay_terminal.append((path, row))
        key = f"phase{int(row['phase_deg']):02d}/{row['model']}"
        replay_by_phase_model[key] += 1
        if row.get("status_match") is False:
            replay_mismatches += 1
        diff = row.get("exit_time_difference_RF")
        if isinstance(diff, (int, float)):
            replay_exit_diffs.append(float(diff))

    terminal = broad_terminal + replay_terminal
    relative_paths = sorted(path.relative_to(REPO).as_posix() for path, _ in terminal)
    args.path_list.parent.mkdir(parents=True, exist_ok=True)
    args.path_list.write_bytes(b"\0".join(path.encode("utf-8") for path in relative_paths) + b"\0")

    manifest_path = V12 / "validation/active_checkpoint_snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["snapshot_is_final"] = False
    active = manifest["active_runs"]
    active[0].update({
        "terminal_retained_or_escaped_checkpoints": len(broad_terminal),
        "terminal_by_model_scale": dict(sorted(Counter(row["model"] for _, row in broad_terminal).items())),
        "partial_or_nonterminal_checkpoints_excluded": sum(broad_partial.values()),
        "partial_by_model_scale": dict(sorted(broad_partial.items())),
        "numerical_error_checkpoints": broad_errors,
        "journal_lines_on_disk": sum(1 for _ in (V12 / "runs/e6bf3a33795e4a5354a8_equal_broad_lambda_h50_4x256/completed_jobs.jsonl").open(encoding="utf-8")),
    })
    active[1].update({
        "terminal_replay_checkpoints": len(replay_terminal),
        "terminal_by_phase_model": dict(sorted(replay_by_phase_model.items())),
        "partial_replay_checkpoints_excluded": replay_partial,
        "status_mismatches_at_snapshot": replay_mismatches,
        "numerical_error_checkpoints": replay_errors,
        "exit_time_comparison_count": len(replay_exit_diffs),
        "maximum_exit_time_difference_RF": max(replay_exit_diffs, default=None),
    })
    manifest["terminal_checkpoint_files"] = len(terminal)
    manifest["terminal_checkpoint_bytes"] = sum(path.stat().st_size for path, _ in terminal)
    manifest["temporary_files_excluded"] = True
    manifest["partial_checkpoints_excluded_from_snapshot_commit_diff"] = True
    manifest["partial_checkpoints_excluded_from_git_tree"] = False
    manifest["legacy_partial_checkpoint_files_already_in_git_tree"] = 6
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    base_counts = {model: sum(row["model"] == f"{model}_h50_scale1" for _, row in broad_terminal)
                   for model in ("A", "B", "C")}
    print(json.dumps({
        "snapshot_utc": manifest["snapshot_utc"],
        "broad_terminal": len(broad_terminal),
        "baseline_A_B_C": base_counts,
        "broad_partial": sum(broad_partial.values()),
        "broad_errors": broad_errors,
        "replay_terminal": len(replay_terminal),
        "replay_partial": replay_partial,
        "replay_status_mismatches": replay_mismatches,
        "replay_exit_comparisons": len(replay_exit_diffs),
        "replay_max_exit_difference_RF": max(replay_exit_diffs, default=None),
        "total_terminal_files": len(terminal),
        "total_terminal_bytes": manifest["terminal_checkpoint_bytes"],
        "path_list": str(args.path_list),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
