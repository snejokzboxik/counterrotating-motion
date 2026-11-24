"""Verify V17C.9 Sobol checkpoint compatibility before resuming any work."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy
import numba

HERE = Path(__file__).resolve().parent
V17C9 = HERE.parent / "v17c_9"
OPT = V17C9 / "optimization"
STAGE4 = OPT / "stage4_candidate48"
STAGE5 = OPT / "stage5_candidate52"
sys.path.insert(0, str(V17C9 / "baseline"))
sys.path.insert(0, str(OPT))
import basin_scan
import refine_candidate52_screen as scan


def array_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.shape == b.shape and np.array_equal(a, b))


def main():
    base_file = np.load(STAGE4 / "configurations.npz")
    base = np.asarray(base_file["p"][scan.BASE_ID], dtype=float)
    base_qratio = float(base_file["qratio"][scan.BASE_ID])
    configs = scan.make_configs(base, base_qratio)
    expected_p = np.asarray([c["p"] for c in configs])
    expected_q = np.asarray([c["qratio"] for c in configs])

    saved = np.load(STAGE5 / "configurations.npz")
    checks = {
        "config_parameters_exact": array_equal(saved["p"], expected_p),
        "charge_ratios_exact": array_equal(saved["qratio"], expected_q),
    }
    starts, phases = {}, {}
    for group, seed in scan.START_SEEDS.items():
        starts[group], phases[group] = basin_scan.make_initials(
            group, scan.N_PER_ENSEMBLE, seed
        )
        checks[f"{group}_initials_exact"] = array_equal(
            saved[f"initial_{group}"], starts[group]
        )
        checks[f"{group}_phases_exact"] = array_equal(
            saved[f"phase_{group}"], phases[group]
        )

    expected_meta = {
        "n_configs": scan.NCFG + 1,
        "N_per_ensemble": scan.N_PER_ENSEMBLE,
        "ensembles": ["rest", "weak", "wide"],
        "periods": scan.PERIODS,
        "steps": scan.STEPS,
        "transient_cycles": scan.SKIP,
        "config_seed": scan.CONFIG_SEED,
        "start_seeds": scan.START_SEEDS,
        "finite_spheres_a_over_R": [scan.A1, scan.A2],
        "objective": "maximize minimum CR proxy across rest/weak/wide while retaining all trials in denominators",
        "parameter_ranges": {
            "rf_scale_vs_candidate52": [0.85, 1.15],
            "B_over_A": [3.2, 4.4],
            "nu": [2.5, 3.3],
            "dc_scale_vs_rf_squared": [0.8, 1.4],
            "mass_ratio": [6.0, 12.0],
            "charge_ratio": [2.8, 4.4],
            "lambda1": [0.12, 0.26],
            "lambda2": [0.07, 0.16],
            "C_eff": [6.5e-6, 1.9e-5],
        },
        "anchor_candidate_id": scan.BASE_ID,
        "note": "Short local screen only; coupling is bounded away from zero. Keep a fresh untouched holdout for the winner.",
    }
    saved_meta = json.loads((STAGE5 / "meta.json").read_text(encoding="utf-8"))
    checks["metadata_exact"] = saved_meta == expected_meta

    rows = [
        json.loads(line)
        for line in (STAGE5 / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [
        (r["config_id"], r["ensemble"], r["start_id"])
        for r in rows
    ]
    checks["row_count_849"] = len(rows) == 849
    checks["unique_task_keys"] = len(set(keys)) == len(keys)
    checks["all_ids_in_range"] = all(
        0 <= r["config_id"] <= scan.NCFG
        and r["ensemble"] in starts
        and 0 <= r["start_id"] < scan.N_PER_ENSEMBLE
        for r in rows
    )
    checks["run_config_ids_match"] = all(
        0 <= r["config_id"] < len(saved["p"])
        and array_equal(saved["p"][r["config_id"]], expected_p[r["config_id"]])
        and saved["qratio"][r["config_id"]] == expected_q[r["config_id"]]
        for r in rows
    )
    checks["recorded_starts_match"] = all(
        array_equal(r["initial"], saved[f"initial_{r['ensemble']}"][r["start_id"]])
        and array_equal(r["phase"], saved[f"phase_{r['ensemble']}"][r["start_id"]])
        and r["seed"] == scan.START_SEEDS[r["ensemble"]]
        for r in rows
    )
    checks["classifier_exact"] = all(
        r["cr_proxy"]
        == scan.cr_proxy(
            r["retained"],
            np.asarray(r["turns"]),
            np.asarray(r["window_turns"]),
            np.asarray(r["r_min"]),
        )
        for r in rows
    )
    from collections import Counter

    counts = Counter((r["config_id"], r["ensemble"]) for r in rows)
    expected_complete = all(
        counts[(i, group)] == scan.N_PER_ENSEMBLE
        for i in range(17)
        for group in starts
    )
    expected_partial = {
        group: counts[(17, group)] for group in starts
    } == {"rest": 16, "weak": 16, "wide": 1}
    checks["checkpoint_shape_matches_handoff"] = expected_complete and expected_partial

    manifest = json.loads((STAGE5 / "MANIFEST.json").read_text(encoding="utf-8"))
    expected_hash = next(
        entry["sha256"] for entry in manifest["files"]
        if entry["path"].endswith("/runs.jsonl")
    )
    actual_hash = hashlib.sha256((STAGE5 / "runs.jsonl").read_bytes()).hexdigest()
    checks["runs_sha256_matches_manifest"] = actual_hash == expected_hash

    result = {
        "branch_head": "cc41be9082c62e9b907eb459f453ac51001d4267",
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "rows": len(rows),
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "config_rows_0_16": expected_complete,
        "config17_rows": {g: counts[(17, g)] for g in starts},
        "sha256_runs": actual_hash,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "model_config_sha256": hashlib.sha256(
            basin_scan.SOURCE_CFG.read_bytes()
        ).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
