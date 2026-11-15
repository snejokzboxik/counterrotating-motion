"""Validate and summarize a resumable V13.2 Stage B JSONL checkpoint."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_2_nonlinear_stability/stage_b_checkpoint.jsonl"


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
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
