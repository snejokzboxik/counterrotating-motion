"""Independent integrity and outcome audit for a saved parameter-grid JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    root = args.directory
    meta_path = root / "campaign_meta.json"
    data_path = root / "trajectories.jsonl"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = []
    seen: set[tuple[int, int]] = set()
    duplicates = []
    malformed = []
    for line_no, line in enumerate(data_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            key = (int(row["config_index"]), int(row["start_index"]))
            if key in seen:
                duplicates.append({"line": line_no, "key": list(key)})
            seen.add(key)
            rows.append(row)
        except Exception as exc:  # preserve line-level evidence in audit
            malformed.append({"line": line_no, "error": repr(exc)})

    expected_cfgs = meta["configs"]
    expected_n = int(meta["n_starts_per_config"])
    expected = {(i, j) for i in range(len(expected_cfgs)) for j in range(expected_n)}
    missing = sorted(expected-seen)
    unexpected = sorted(seen-expected)
    counts = Counter(str(r.get("category", "MISSING")) for r in rows)
    by_config: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        ci = int(row["config_index"])
        by_config[ci][str(row.get("category", "MISSING"))] += 1

    configs = []
    for i, cfg in enumerate(expected_cfgs):
        p = cfg["p"]
        c = by_config[i]
        cr = c.get("CR_PLUS_MINUS", 0)+c.get("CR_MINUS_PLUS", 0)
        configs.append({"config_index": i, "config_id": cfg["id"],
            "A": p[0], "B": p[1], "nu": p[2], "beta": p[3],
            "C": p[5], "lambda1": p[6], "lambda2": p[7],
            "outcomes": dict(sorted(c.items())), "n": sum(c.values()),
            "cr_count": cr, "cr_fraction": cr/expected_n,
            "one_sided_95_upper_if_zero_CR": (1-0.05**(1/expected_n) if cr == 0 else None)})

    audit = {"study": meta.get("study"), "integrity": {
            "expected_configurations": len(expected_cfgs), "starts_per_configuration": expected_n,
            "expected_rows": len(expected), "actual_rows": len(rows),
            "unique_keys": len(seen), "missing_keys": missing, "unexpected_keys": unexpected,
            "duplicate_keys": duplicates, "malformed_lines": malformed,
            "complete_and_unique": len(rows) == len(expected) and not missing and not unexpected
                and not duplicates and not malformed},
        "outcome_counts": dict(sorted(counts.items())),
        "configuration_results": configs,
        "limits": ["Rest-start discovery screen only; not four-ensemble basin probabilities.",
            "A zero-event upper bound is per configuration and does not pool the shared paired starts."],
        "sha256": {"campaign_meta.json": sha256(meta_path), "trajectories.jsonl": sha256(data_path)}}
    out = root/"independent_audit.json"
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "complete_and_unique": audit["integrity"]["complete_and_unique"],
        "actual_rows": len(rows), "outcome_counts": audit["outcome_counts"], "sha256": audit["sha256"]},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
