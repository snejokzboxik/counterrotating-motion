"""Fresh-start validation of three additional finalists from the physical Sobol screen."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
RESULTS = HERE / "results"
SCREEN = RESULTS / "physical_parameter_sobol_screen"
PARENT = RESULTS / "updated_objective_validation"
OUT = RESULTS / "additional_physical_candidate_validation"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODEL))
import physical_mixed_pair_search as base  # noqa: E402
import updated_objective_validation as classify  # noqa: E402

IDS = ("phys0352", "phys0109", "phys0526")
GROUPS = ("rest", "weak", "wide", "edge")
N_PER_GROUP = 128
PERIODS = 15_000
STEPS = 64
SKIP = 200
CHUNK = 128
WORKERS = min(11, os.cpu_count() or 1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_configurations() -> tuple[list[dict], np.ndarray, np.ndarray, list[str]]:
    screen_meta = json.loads((SCREEN / "campaign_meta.json").read_text(encoding="utf-8"))
    all_configs = {c["id"]: c for c in screen_meta["configurations"]}
    configs = [all_configs[cid] for cid in IDS]
    frozen = np.load(PARENT / "frozen_starts.npz", allow_pickle=True)
    starts, phases = frozen["initial"], frozen["phases"]
    groups = frozen["ensemble"].astype(str).tolist()
    return configs, starts, phases, groups


def main() -> None:
    configs, starts, phases, start_groups = load_configurations()
    source_frozen = PARENT / "frozen_starts.npz"
    frozen_path = OUT / "frozen_starts.npz"
    if frozen_path.exists() and sha(frozen_path) != sha(source_frozen):
        raise RuntimeError("existing additional-candidate frozen starts differ from parent panel")
    if not frozen_path.exists():
        shutil.copy2(source_frozen, frozen_path)
    ranking = json.loads((SCREEN / "updated_objective_ranking.json").read_text(encoding="utf-8"))
    screen_summary = {x["configuration"]: x for x in ranking["all_configurations"]}
    meta = {"study": "V17C.12B additional physical finalists fresh-start validation",
        "objective": "any stable CR direction counts; suppress co-rotation and losses",
        "candidate_ids": list(IDS), "candidates": configs,
        "discovery_rank_and_count": {cid: {"rank": screen_summary[cid]["rank_updated_objective"],
            "CR_count_of_128": screen_summary[cid]["preliminary_CR_count_of_128"]} for cid in IDS},
        "source_screen_meta_sha256": sha(SCREEN / "campaign_meta.json"),
        "source_screen_ranking_sha256": sha(SCREEN / "updated_objective_ranking.json"),
        "parent_validation_sha256": sha(PARENT / "campaign_meta.json"),
        "frozen_starts_sha256": sha(frozen_path), "ensembles": list(GROUPS),
        "starts_per_ensemble": N_PER_GROUP, "seeds_and_starts_source": "updated_objective_validation frozen panel; 128 each of rest/weak/wide/edge",
        "periods": PERIODS, "steps_per_rf_period": STEPS, "skip_periods": SKIP,
        "strict_turn_threshold_per_particle": 10.0, "workers": WORKERS,
        "integrator_sha256": sha(MODEL / "basin_scan.py"), "field_model_sha256": sha(MODEL / "field_model.json"),
        "runner_sha256": sha(Path(__file__))}
    meta_path = OUT / "campaign_meta.json"
    if meta_path.exists():
        previous_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous_meta != meta:
            previous_core = {k: v for k, v in previous_meta.items() if k != "runner_sha256"}
            current_core = {k: v for k, v in meta.items() if k != "runner_sha256"}
            if previous_core != current_core:
                raise RuntimeError("checkpoint metadata differs from this fixed-candidate campaign")
            previous_meta_path = OUT / "attempt_01_campaign_meta.json"
            if not previous_meta_path.exists():
                previous_meta_path.write_text(json.dumps(previous_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    runner_copy = OUT / "campaign_runner_source.py"
    if runner_copy.exists() and sha(runner_copy) != sha(Path(__file__)):
        prior_runner = OUT / "campaign_runner_source_attempt01.py"
        if not prior_runner.exists(): shutil.copy2(runner_copy, prior_runner)
        shutil.copy2(Path(__file__), runner_copy)
    elif not runner_copy.exists():
        shutil.copy2(Path(__file__), runner_copy)
    integrator_copy = OUT / "campaign_integrator_source.py"
    if not integrator_copy.exists(): shutil.copy2(MODEL / "basin_scan.py", integrator_copy)
    code = {"runner_sha256": sha(Path(__file__)), "saved_runner_sha256": sha(runner_copy),
        "integrator_sha256": sha(MODEL / "basin_scan.py"), "saved_integrator_sha256": sha(integrator_copy),
        "field_model_sha256": sha(MODEL / "field_model.json"), "frozen_starts_sha256": sha(frozen_path)}
    code_path = OUT / "campaign_code_sha256.json"
    if code_path.exists():
        previous_code = json.loads(code_path.read_text(encoding="utf-8"))
        if previous_code != code:
            prior_code = OUT / "attempt_01_campaign_code_sha256.json"
            if not prior_code.exists():
                prior_code.write_text(json.dumps(previous_code, indent=2) + "\n", encoding="utf-8")
    code_path.write_text(json.dumps(code, indent=2) + "\n", encoding="utf-8")
    offsets = {g: i * N_PER_GROUP for i, g in enumerate(GROUPS)}
    model_p, family, k, coeff = base.sim.load_model()
    rows_path = OUT / "trajectories.jsonl"
    existing = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); existing[(row["configuration"], row["ensemble"], int(row["start_index"]))] = row
    cfg = {x["id"]: x for x in configs}
    tasks = [(cid, group, i) for cid in IDS for group in GROUPS for i in range(N_PER_GROUP)
             if (cid, group, i) not in existing]
    print(json.dumps({"configurations": IDS, "todo": len(tasks), "workers": WORKERS, "periods": PERIODS, "steps": STEPS}), flush=True)

    def run(task: tuple[str, str, int]) -> dict:
        cid, group, si = task; x = cfg[cid]; ph = x["physical"]; idx = offsets[group] + si
        cc = 0.5 * (ph["particle1"]["Cunningham"] + ph["particle2"]["Cunningham"])
        raw = base.sim.integrate_one(starts[idx], phases[idx], np.asarray(x["p"], dtype=np.float64),
            family, k, coeff, PERIODS, STEPS, SKIP, qratio=x["qratio"],
            oseen_Re_per_speed=ph["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2, a1_over_R=ph["a1_over_R"], a2_over_R=ph["a2_over_R"],
            gravity_z=ph["gravity_z_dimensionless"], drag_mode=2, slip_correction=cc)
        end, turns, windows, rmin, rmax, zmax, dtheta, dmin, completed, retained, escaped, singular, rawcat, freq, vmax = raw
        axis = bool(min(rmin) < base.sim.RMIN_CLASS)
        provisional = classify.provisional(str(rawcat), turns, windows, bool(retained), axis)
        strict = classify.strict_late_label(str(rawcat), turns, windows, bool(retained), axis)
        return {"configuration": cid, "ensemble": group, "start_index": si,
            "seed": int(json.loads((PARENT / "campaign_meta.json").read_text(encoding="utf-8"))["seed_records"][GROUPS.index(group)]["seed"]),
            "initial": starts[idx].tolist(), "phase": phases[idx].tolist(), "final_state": end.tolist(),
            "raw_category": str(rawcat), "provisional_category": provisional, "strict_late_category": strict,
            "turns": turns.tolist(), "window_turns": windows.tolist(), "frequency_over_rf_Hz": freq.tolist(),
            "r_min_over_R": rmin.tolist(), "r_max_over_R": rmax.tolist(), "z_max_over_R": zmax.tolist(),
            "max_Re_by_particle": (vmax * np.asarray([ph["particle1"]["Re_per_dimensionless_speed"], ph["particle2"]["Re_per_dimensionless_speed"]])).tolist(),
            "distance_min_over_R": float(dmin), "retained": bool(retained), "escaped": bool(escaped),
            "singular_unresolved": bool(singular), "completed_periods": int(completed)}

    if tasks:
        # Compile the Numba specialization serially before starting the worker pool.
        cid, group, si = tasks[0]; x = cfg[cid]; ph = x["physical"]; idx = offsets[group] + si
        cc = 0.5 * (ph["particle1"]["Cunningham"] + ph["particle2"]["Cunningham"])
        base.sim.integrate_one(starts[idx], phases[idx], np.asarray(x["p"], dtype=np.float64),
            family, k, coeff, 1, STEPS, 0, qratio=x["qratio"],
            oseen_Re_per_speed=ph["particle1"]["Re_per_dimensionless_speed"],
            oseen_Re2_per_speed=ph["particle2"]["Re_per_dimensionless_speed"],
            hydro_mode=2, a1_over_R=ph["a1_over_R"], a2_over_R=ph["a2_over_R"],
            gravity_z=ph["gravity_z_dimensionless"], drag_mode=2, slip_correction=cc)
    started = time.perf_counter()
    with rows_path.open("a", encoding="utf-8", buffering=1) as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for offset in range(0, len(tasks), CHUNK):
            batch = tasks[offset:offset + CHUNK]
            futures = [pool.submit(run, task) for task in batch]
            for future in as_completed(futures):
                row = future.result(); f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                existing[(row["configuration"], row["ensemble"], row["start_index"])] = row
            progress = {"completed_rows": len(existing), "expected_rows": len(configs) * len(GROUPS) * N_PER_GROUP,
                "last_checkpoint_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "elapsed_seconds_this_invocation": time.perf_counter() - started}
            (OUT / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
            print(f"checkpoint {progress['completed_rows']}/{progress['expected_rows']}; {progress['elapsed_seconds_this_invocation']:.1f}s", flush=True)
    rows = list(existing.values()); results = {}
    for cid in IDS:
        rr = [r for r in rows if r["configuration"] == cid]; by_group = {}
        for group in GROUPS:
            gr = [r for r in rr if r["ensemble"] == group]; counts = Counter(r["strict_late_category"] for r in gr)
            by_group[group] = {"n": len(gr), "strict_late": dict(sorted(counts.items())),
                "counterrotation_any_direction": counts["STRICT_CR_PLUS_MINUS"] + counts["STRICT_CR_MINUS_PLUS"],
                "CR_plus_minus": counts["STRICT_CR_PLUS_MINUS"], "CR_minus_plus": counts["STRICT_CR_MINUS_PLUS"],
                "co_rotation": counts["STRICT_CO_PLUS_PLUS"] + counts["STRICT_CO_MINUS_MINUS"],
                "escape": counts["WALL_OR_AXIAL_ESCAPE"]}
        counts = Counter(r["strict_late_category"] for r in rr)
        cr = counts["STRICT_CR_PLUS_MINUS"] + counts["STRICT_CR_MINUS_PLUS"]
        results[cid] = {"rows": len(rr), "strict_CR_any_direction_of_512": cr,
            "strict_CR_fraction_of_512": cr / 512.0,
            "co_rotation_total": counts["STRICT_CO_PLUS_PLUS"] + counts["STRICT_CO_MINUS_MINUS"],
            "escape_total": counts["WALL_OR_AXIAL_ESCAPE"], "outcome_counts": dict(sorted(counts.items())),
            "per_ensemble": by_group}
    summary = {"study": meta["study"], "rows": len(rows), "expected_rows": len(configs)*len(GROUPS)*N_PER_GROUP,
        "trajectory_sha256": sha(rows_path), "configuration_results": results,
        "classification": "Fresh-start 15000-period RK4/64; strict late label requires consistent signs in three late windows and at least 10 net turns per particle; discovery frequency only.",
        "wall_seconds": time.perf_counter()-started}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "configuration_results"}, indent=2, ensure_ascii=False))
    for cid, value in results.items(): print(json.dumps({"configuration": cid, **value}, ensure_ascii=False))


if __name__ == "__main__":
    main()
