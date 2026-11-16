"""Halve the time step for h35 FEM controls with discordant or late-exit outcomes."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
OUT = BASE / "fem_controls_h35"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def init_worker():
    import run_v13_3_fem_controls as fem
    fem.initialize(mesh_um=35)
    fem.RTOL, fem.ATOL = 1e-10, 1e-12
    fem.MAX_STEP = 2 * math.pi / 128


def worker(task):
    import run_v13_3_fem_controls as fem
    point, model, phase, state_i, initial = task
    return fem.simulate_segment(point, model, 1., phase, 0., state_i, initial,
                                initial, 0, 100, 0.)


def main():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    pairs = read_csv(OUT / "ideal_vs_fem_pairs.csv")
    points = {r["candidate_id"]: r for r in read_csv(BASE / "fem_control_points.csv")}
    fem_rows = read_csv(OUT / "trajectory_results.csv")
    fem_index = {(r["candidate_id"], r["model"], round(float(r["tau0_rad"]), 12), int(r["state_i"])): r
                 for r in fem_rows if int(r["horizon"]) == 100}
    selected = []
    for row in pairs:
        dexit = math.nan
        try:
            if row["ideal_exit_RF"] not in ("", "nan") and row["FEM_exit_RF"] not in ("", "nan"):
                dexit = abs(float(row["ideal_exit_RF"]) - float(row["FEM_exit_RF"]))
        except ValueError:
            pass
        if row["status_agree"] != "True" or (math.isfinite(dexit) and dexit >= 3.):
            selected.append((row, dexit))
    # Dedupe defensively and keep every discordant status plus the large-exit-delay controls.
    tasks, provenance = [], {}
    for row, dexit in selected:
        cid = row["candidate_id"]
        p = points[cid]
        point = {**p, "alpha": float(p["alpha"]), "beta": float(p["beta"]), "kappa": float(p["kappa"])}
        model, phase, state_i = row["model"], float(row["tau0_deg"]) * math.pi / 180., int(row["state_i"])
        key = (cid, model, phase, state_i)
        if key in provenance:
            continue
        source = fem_index[(cid, model, round(phase, 12), state_i)]
        initial = [float(source[k]) for k in ("x0_rho", "y0_rho", "vx0_rho_tau", "vy0_rho_tau")]
        tasks.append((point, model, phase, state_i, initial))
        provenance[key] = {"candidate_id": cid, "model": model, "phase_rad": phase,
                           "state_i": state_i, "ideal_status": row["ideal_status"],
                           "coarse_FEM_status": row["FEM_status"],
                           "ideal_exit_RF": row["ideal_exit_RF"], "coarse_FEM_exit_RF": row["FEM_exit_RF"],
                           "coarse_exit_abs_difference_RF": dexit}
    ck = OUT / "step_halving_checkpoint.jsonl"
    completed = {}
    if ck.exists():
        for line in ck.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["key"] in completed:
                raise RuntimeError(f"duplicate step-refinement key {row['key']}")
            completed[row["key"]] = row
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    basis_paths = [ROOT / "outputs/abc_validation/direct_fem_validation/v2/h35_P4/sector_solution.npz",
                   ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas/fem/quadrupole_h35_P4_sector_solution.npz",
                   ROOT / "outputs/meshes/gmsh_35um/trap.msh"]
    manifest = {"script_sha256": script_hash,
                "field_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in basis_paths},
                "selected_task_count": len(tasks), "integrator": {"method": "DOP853", "rtol": 1e-10,
                    "atol": 1e-12, "max_step_tau": 2*math.pi/128},
                "horizon_RF": 100, "aperture_rho": 1., "tau0": "paired per original control",
                "selection": "all ideal/FEM status disagreements plus same-status escapes with |delta exit| >= 3 RF cycles"}
    (OUT / "step_halving_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pending = [t for t in tasks if "|".join((t[0]["candidate_id"], "1.0", t[1], str(t[2]), str(t[3]), "100")) not in completed]
    if pending:
        n = 0
        with ProcessPoolExecutor(max_workers=2, initializer=init_worker) as pool, ck.open("a", encoding="utf-8") as stream:
            futures = [pool.submit(worker, task) for task in pending]
            for future in as_completed(futures):
                row = future.result()
                row["step_refined"] = True
                row["key"] = "|".join((row["candidate_id"], str(row["mu"]), row["model"],
                                        str(row["tau0_rad"]), str(row["state_i"]), "100"))
                row["task_key"] = row["key"]
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                completed[row["key"]] = row
                n += 1
                if n % 4 == 0:
                    stream.flush(); os.fsync(stream.fileno())
            stream.flush(); os.fsync(stream.fileno())
    # Write matched, inspectable step-refinement rows.
    out = []
    by_candidate = {r["candidate_id"]: r for r in read_csv(BASE / "fem_control_points.csv")}
    for key, row in completed.items():
        cid, model, phase, state_i = row["candidate_id"], row["model"], float(row["tau0_rad"]), int(row["state_i"])
        p = by_candidate[cid]
        want = next((v for k, v in provenance.items() if k == (cid, model, phase, state_i)), None)
        out.append({"candidate_id": cid, "alpha": p["alpha"], "beta": p["beta"], "kappa": p["kappa"],
                    "model": model, "tau0_rad": phase, "state_i": state_i,
                    "ideal_status": want["ideal_status"] if want else "",
                    "coarse_FEM_status": want["coarse_FEM_status"] if want else "",
                    "step_refined_status": row["status"],
                    "ideal_exit_RF": want["ideal_exit_RF"] if want else "",
                    "coarse_FEM_exit_RF": want["coarse_FEM_exit_RF"] if want else "",
                    "step_refined_exit_RF": row["exit_periods"],
                    "rho_max": row["rho_max_cumulative"],
                    "integrator": "DOP853 rtol=1e-10 atol=1e-12 max_step=2pi/128"})
    with (OUT / "step_halving_results.csv").open("w", newline="", encoding="utf-8") as stream:
        if out:
            writer = csv.DictWriter(stream, fieldnames=list(out[0]))
            writer.writeheader(); writer.writerows(out)
    print(json.dumps({**manifest, "completed": len(out), "pending": len(pending)}, indent=2))


if __name__ == "__main__":
    main()
