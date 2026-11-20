"""Long-horizon check of representative CR starts from the rescaled refine12 point."""
from __future__ import annotations

import hashlib
import json
import os
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
SOURCE = RESULTS / "rescaled_legacy_candidate_validation"
OUT = RESULTS / "long_horizon_refine12_validation"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MODEL))
import physical_mixed_pair_search as base  # noqa: E402
import updated_objective_validation as labels  # noqa: E402

PERIODS = 120_000
STEPS = 64
SKIP = 60_000
N_PER_ENSEMBLE = 8
GROUPS = ("rest", "weak", "wide", "edge")
WORKERS = min(11, os.cpu_count() or 1)
CONFIG_ID = "scaled_refine12_sobol17_f50hz"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_rows() -> tuple[list[dict], np.ndarray, np.ndarray, dict, dict]:
    campaign = json.loads((SOURCE / "campaign_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    cfg = next(c for c in campaign["candidates"] if c["id"] == CONFIG_ID)
    frozen = np.load(SOURCE / "frozen_starts.npz", allow_pickle=True)
    starts, phases = frozen["initial"], frozen["phases"]
    offsets = {g: i * 128 for i, g in enumerate(GROUPS)}
    rows = [json.loads(line) for line in (SOURCE / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    for group in GROUPS:
        eligible = sorted((r for r in rows if r["configuration"] == CONFIG_ID and r["ensemble"] == group
                           and r["strict_late_category"].startswith("STRICT_CR_")), key=lambda r: int(r["start_index"]))
        if len(eligible) < N_PER_ENSEMBLE:
            raise RuntimeError(f"not enough 15k CR starts in ensemble {group}")
        positions = np.rint(np.linspace(0, len(eligible) - 1, N_PER_ENSEMBLE)).astype(int)
        for position in positions:
            source = eligible[int(position)]
            si = int(source["start_index"])
            idx = offsets[group] + si
            selected.append({"ensemble": group, "start_index": si, "source_strict_label": source["strict_late_category"],
                             "source_turns": source["turns"], "initial": starts[idx].tolist(), "phase": phases[idx].tolist(),
                             "seed": int(source["seed"])})
    return selected, starts, phases, cfg, campaign


def main() -> None:
    selected, _, _, cfg, campaign = select_rows()
    meta = {"study": "V17C.12B 120000-period representative persistence check",
            "configuration": CONFIG_ID, "source_campaign": "rescaled_legacy_candidate_validation",
            "source_summary_sha256": sha(SOURCE / "summary.json"),
            "source_trajectory_sha256": json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))["trajectory_sha256"],
            "seed_selection": "8 evenly spaced start indices among strict 15000-period CR results in each fixed ensemble",
            "selected_starts": selected, "periods": PERIODS, "steps_per_rf_period": STEPS,
            "skip_before_late_windows": SKIP, "workers": WORKERS, "target_turns_per_particle": 100,
            "integrator_sha256": sha(MODEL / "basin_scan.py"), "field_model_sha256": sha(MODEL / "field_model.json"),
            "runner_sha256": sha(Path(__file__))}
    (OUT / "campaign_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil_path = OUT / "campaign_integrator_source.py"
    if not shutil_path.exists():
        shutil_path.write_bytes((MODEL / "basin_scan.py").read_bytes())
    rows_path = OUT / "trajectories.jsonl"
    if rows_path.exists():
        existing = {json.loads(line)["key"] for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    else:
        existing = set()

    _, family, k, coeff = base.sim.load_model()
    p = np.asarray(cfg["p"], dtype=np.float64)
    ph = cfg["physical"]
    re1 = ph["particle1"]["Re_per_dimensionless_speed"]
    re2 = ph["particle2"]["Re_per_dimensionless_speed"]
    cc = 0.5 * (ph["particle1"]["Cunningham"] + ph["particle2"]["Cunningham"])
    todo = [(i, x) for i, x in enumerate(selected) if f"{x['ensemble']}:{x['start_index']}" not in existing]
    print(json.dumps({"selected": len(selected), "todo": len(todo), "workers": WORKERS, "periods": PERIODS, "steps": STEPS}), flush=True)

    def run(item: tuple[int, dict]) -> dict:
        index, x = item
        raw = base.sim.integrate_one(np.asarray(x["initial"], dtype=np.float64), np.asarray(x["phase"], dtype=np.float64),
            p, family, k, coeff, PERIODS, STEPS, SKIP, qratio=cfg["qratio"], oseen_Re_per_speed=re1,
            oseen_Re2_per_speed=re2, hydro_mode=2, a1_over_R=ph["a1_over_R"], a2_over_R=ph["a2_over_R"],
            gravity_z=ph["gravity_z_dimensionless"], drag_mode=2, slip_correction=cc)
        end, turns, windows, rmin, rmax, zmax, dtheta, dmin, completed, retained, escaped, singular, category, freq, vmax = raw
        axis = bool(min(rmin) < base.sim.RMIN_CLASS)
        strict = labels.strict_late_label(str(category), turns, windows, bool(retained), axis)
        return {"key": f"{x['ensemble']}:{x['start_index']}", "ensemble": x["ensemble"], "start_index": x["start_index"],
            "seed": x["seed"], "initial": x["initial"], "phase": x["phase"], "source_strict_label": x["source_strict_label"],
            "source_turns": x["source_turns"], "raw_category": str(category), "strict_late_category": strict,
            "turns": turns.tolist(), "window_turns": windows.tolist(), "frequency_over_rf_Hz": freq.tolist(),
            "max_Re_by_particle": (vmax * np.asarray([re1, re2])).tolist(), "min_r_over_R": rmin.tolist(),
            "max_r_over_R": rmax.tolist(), "max_abs_z_over_R": zmax.tolist(), "min_pair_distance_over_R": float(dmin),
            "retained": bool(retained), "escaped": bool(escaped), "completed_periods": int(completed)}

    started = time.perf_counter()
    with rows_path.open("a", encoding="utf-8", buffering=1) as f, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run, item) for item in todo]
        for done, future in enumerate(as_completed(futures), 1):
            f.write(json.dumps(future.result(), ensure_ascii=False, separators=(",", ":")) + "\n")
            if done % 8 == 0 or done == len(todo):
                progress = {"completed_rows": len(existing) + done, "expected_rows": len(selected),
                    "last_checkpoint_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "elapsed_seconds": time.perf_counter() - started}
                (OUT / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
                print(f"checkpoint {progress['completed_rows']}/{len(selected)}; {progress['elapsed_seconds']:.1f}s", flush=True)
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {g: Counter(r["strict_late_category"] for r in rows if r["ensemble"] == g) for g in GROUPS}
    long_cr = [r for r in rows if r["strict_late_category"].startswith("STRICT_CR_")
               and min(abs(x) for x in r["turns"]) >= 100.0]
    summary = {"study": meta["study"], "rows": len(rows), "expected_rows": len(selected),
        "trajectory_sha256": sha(rows_path), "outcomes_by_ensemble": {g: dict(groups[g]) for g in GROUPS},
        "long_horizon_CR_count": len(long_cr), "long_horizon_CR_fraction_of_selected": len(long_cr) / max(1, len(rows)),
        "at_least_100_turns_per_particle_count": sum(min(abs(t) for t in r["turns"]) >= 100.0 for r in rows),
        "retained_full_horizon_count": sum(bool(r["retained"]) and r["completed_periods"] == PERIODS for r in rows),
        "median_min_abs_turns": float(np.median([min(abs(t) for t in r["turns"]) for r in rows])),
        "wall_seconds": time.perf_counter() - started,
        "interpretation": "Selected trajectories conditional on a 15000-period strict CR result; persistence check, not basin probability.",
        "runner_sha256": sha(Path(__file__)), "integrator_sha256": sha(MODEL / "basin_scan.py")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
