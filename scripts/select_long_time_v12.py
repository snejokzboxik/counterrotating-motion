"""Pre-register an outcome-blind long-time subset from a completed input design."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/octupole_comparison_home/dimensionless_acceptance_v12/runs"
DEFAULT_MODELS = [f"{kind}_h50_scale{scale:g}" for kind in ("A", "B", "C") for scale in (0.5, 1, 2)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="input run directory name under V12 runs/")
    parser.add_argument("--per-replicate", type=int, default=16)
    parser.add_argument("--seed", type=int, default=121216)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = RUNS / args.source_run
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if args.per_replicate < 1:
        parser.error("per-replicate must be positive")
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    missing = sorted(set(model_names) - {f"{kind}_h{mesh}_scale{scale:g}"
                                         for kind in cfg["models"] for mesh in cfg["meshes_um"]
                                         for scale in cfg["scales"]
                                         if not kind.startswith("ideal") or mesh == min(cfg["meshes_um"])})
    if missing:
        parser.error(f"requested models are not present in source design: {missing}")

    rng = np.random.default_rng(args.seed)
    rows = []
    rep_counts = {}
    for rep in range(cfg["replicates"]):
        path = source / f"initial_states_r{rep:02d}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            candidates = list(csv.DictReader(stream))
        if args.per_replicate > len(candidates):
            parser.error(f"requested {args.per_replicate} points from replicate {rep} with only {len(candidates)}")
        chosen = np.sort(rng.choice(len(candidates), size=args.per_replicate, replace=False))
        rep_counts[str(rep)] = {"input_points": len(candidates), "selected_points": len(chosen)}
        for idx in chosen:
            row = candidates[int(idx)]
            sample_id = f"r{int(row['replicate']):02d}_{int(row['sample_index']):06d}"
            rows.append({
                "sample_id": sample_id, "replicate": int(row["replicate"]),
                "sample_index": int(row["sample_index"]), "x0_m": float(row["x0_m"]),
                "y0_m": float(row["y0_m"]), "vx0_m_s": float(row["vx0_m_s"]),
                "vy0_m_s": float(row["vy0_m_s"]), "E0_eV": float(row["E0_eV"]),
                "theta0_rad": float(row["theta0_rad"]), "beta_rad": float(row["beta_rad"]),
                "phase_rad": float(row["phase_rad"]),
                "inclusion_probability": args.per_replicate / len(candidates),
                "inverse_probability_weight": len(candidates) / args.per_replicate,
            })
    rows.sort(key=lambda row: (row["replicate"], row["sample_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    ids_path = args.output.with_suffix(".txt")
    ids_path.write_text("\n".join(row["sample_id"] for row in rows) + "\n", encoding="utf-8")
    manifest = {
        "selection": "simple random sample without replacement within each independent Sobol scramble; selected using initial states only, never outcome/status files",
        "source_run": args.source_run, "source_config_hash": cfg["config_hash"],
        "source_horizon_RF": cfg["horizon_RF"], "seed": args.seed,
        "per_replicate": args.per_replicate, "replicate_counts": rep_counts,
        "sample_count": len(rows), "models_for_long_time": model_names,
        "outcome_blind": True,
        "sha256_csv": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "sha256_sample_ids": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
        "usage": "For S(N), use equal weights within each replicate; the inverse-probability weight documents the finite-source-sample representation. Keep outcome-enriched boundary examples separate and do not pool them unweighted into S(N).",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(args.output), "sample_ids": str(ids_path),
                      "samples": len(rows), "models": model_names,
                      "outcome_blind": True}, indent=2))


if __name__ == "__main__":
    main()
