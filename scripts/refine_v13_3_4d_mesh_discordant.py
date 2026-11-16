"""Tight-step re-integration of h50/h35-discordant 4D launches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
RTOL = 1e-11
ATOL = 1e-13
MAX_STEP = 2 * math.pi / 128


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def init_worker(mesh_um: int) -> None:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_v13_3_fem_controls as fem
    fem.RTOL, fem.ATOL, fem.MAX_STEP = RTOL, ATOL, MAX_STEP
    fem.initialize(mesh_um)


def integrate(task: tuple) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_v13_3_fem_controls as fem
    row, point, mesh_um, initial = task
    result = fem.simulate_segment(point, row["model"], float(row["mu"]),
                                  math.radians(float(row["phase_deg"])), 0., int(row["state_i"]),
                                  initial, initial, 0, int(row["horizon_RF"]), 0.)
    return {"mesh_um": mesh_um, "point_id": row["point_id"], "model": row["model"],
            "mu": float(row["mu"]), "phase_deg": float(row["phase_deg"]),
            "replica": int(row["replica"]), "state_i": int(row["state_i"]),
            "base_h50_status": row["reference_status"], "base_h35_status": row["comparison_status"],
            "tight_status": result["status"], "tight_exit_RF": result["exit_periods"],
            "tight_rho_max": result["rho_max_cumulative"], "nfev": result["nfev"],
            "integrator": "DOP853", "rtol": RTOL, "atol": ATOL, "max_step_tau": MAX_STEP}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_key(row: dict, mesh_um: int | None = None) -> str:
    mesh = int(mesh_um if mesh_um is not None else row["mesh_um"])
    return "|".join((str(mesh), row["point_id"], row["model"], format(float(row["mu"]), ".12g"),
                      format(float(row["phase_deg"]), ".12g"), str(int(row["replica"])),
                      str(int(row["state_i"]))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", default="acceptance_4d_fem_comparison_N100")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if Path(args.output_directory).name != args.output_directory:
        parser.error("output directory must be a direct child of the atlas")
    if args.workers != 1:
        parser.error("use exactly one FEM worker to bound memory")
    out = BASE / args.output_directory
    source_path = out / "mesh_pair_trajectories.csv"
    source_rows = read_csv(source_path)
    discordant = [r for r in source_rows if r["status_changed"].lower() == "true"]
    if len(discordant) != 10:
        raise ValueError(f"expected ten h50/h35 status discordances, found {len(discordant)}")
    points = {r["candidate_id"]: r for r in read_csv(BASE / "acceptance_points.csv")}
    initial_rows = read_csv(BASE / "acceptance_4d_fem_h50_selected_N100/initial_conditions.csv")
    initials = {(float(r["mu"]), int(r["replica"]), int(r["state_i"])):
                [float(r[k]) for k in ("x0_rho", "y0_rho", "vx0_rho_tau", "vy0_rho_tau")]
                for r in initial_rows}
    check_path = out / "mesh_discordant_refinement_checkpoint.jsonl"
    done: dict[str, dict] = {}
    if check_path.exists():
        for line in check_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                # A prior interrupted run may have appended a replay after reading
                # phase values from CSV as integers. Canonical numeric formatting
                # collapses those equivalent keys; the latest complete row wins.
                done[checkpoint_key(row)] = row
    manifest_h50 = json.loads((BASE / "acceptance_4d_fem_h50_selected_N100/manifest.json").read_text(encoding="utf-8"))
    manifest_h35 = json.loads((BASE / "acceptance_4d_fem_h35_selected_N100/manifest.json").read_text(encoding="utf-8"))
    fem_script = ROOT / "scripts/run_v13_3_fem_controls.py"
    tasks_by_mesh = {35: [], 50: []}
    for row in discordant:
        point = points[row["point_id"]]
        initial = initials[(float(row["mu"]), int(row["replica"]), int(row["state_i"]))]
        for mesh_um in (35, 50):
            key = checkpoint_key(row, mesh_um)
            if key not in done:
                tasks_by_mesh[mesh_um].append((row, point, mesh_um, initial))
    for mesh_um in (50, 35):
        tasks = tasks_by_mesh[mesh_um]
        if not tasks:
            continue
        with ProcessPoolExecutor(max_workers=1, initializer=init_worker, initargs=(mesh_um,)) as pool, \
                check_path.open("a", encoding="utf-8") as stream:
            futures = [pool.submit(integrate, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                key = checkpoint_key(result)
                result["checkpoint_key"] = key
                stream.write(json.dumps(result, separators=(",", ":")) + "\n")
                stream.flush()
                done[key] = result
    rows = list(done.values())
    # Compact the append-only checkpoint by canonical key after a completed pass,
    # so resume can validate one row per physical launch.
    temporary = check_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for key in sorted(done):
            stream.write(json.dumps(done[key], separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, check_path)
    write_csv(out / "mesh_discordant_tight_results.csv", rows)
    by = {}
    for row in rows:
        key = f"{row['point_id']}|{row['model']}|{row['mu']}|{row['phase_deg']}|{row['replica']}|{row['state_i']}"
        by.setdefault(key, {})[int(row["mesh_um"])] = row
    if len(by) != len(discordant) or any(set(v) != {35, 50} for v in by.values()):
        raise ValueError("incomplete h50/h35 refined status pairs")
    pair_rows = []
    for key, values in sorted(by.items()):
        pair_rows.append({"pair_key": key, "base_h50_status": values[50]["base_h50_status"],
                          "base_h35_status": values[35]["base_h35_status"],
                          "tight_h50_status": values[50]["tight_status"],
                          "tight_h35_status": values[35]["tight_status"],
                          "tight_mesh_status_changed": values[50]["tight_status"] != values[35]["tight_status"],
                          "tight_h50_rho_max": values[50]["tight_rho_max"],
                          "tight_h35_rho_max": values[35]["tight_rho_max"]})
    write_csv(out / "mesh_discordant_tight_pairs.csv", pair_rows)
    metadata = {"discordant_launch_pairs": len(discordant), "refined_mesh_trajectories": len(rows),
                "solver": {"method": "DOP853", "rtol": RTOL, "atol": ATOL, "max_step_tau": MAX_STEP},
                "source_mesh_pair_sha256": sha256(source_path),
                "fem_source_hashes": {"h50": manifest_h50["source_hashes"], "h35": manifest_h35["source_hashes"]},
                "note": "only the ten default-step h50/h35 status-discordant paired launches were refined"}
    (out / "mesh_discordant_tight_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"discordant_pairs": len(discordant),
                      "refined_trajectories": len(rows),
                      "tight_mesh_discordances": sum(x["tight_mesh_status_changed"] for x in pair_rows),
                      "output_directory": str(out)}, indent=2))


if __name__ == "__main__":
    main()
