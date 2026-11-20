"""Reproducible symmetric-particle basin campaign for V17C.12B.

The default run uses four independent initial-condition ensembles, 128 starts
per ensemble, RK4/64, and a 15,000-period horizon. Use --pilot first to measure
throughput; completed rows are append-only checkpoints and safely resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist, qmc

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
RESULTS = HERE / "results"
sys.path.insert(0, str(MODEL))
import basin_scan as integrator  # noqa: E402

REAL_SEEDS = {"rest": 20261201, "weak": 20261202, "wide": 20261203, "edge": 20261204}
GROUPS = ("rest", "weak", "wide", "edge")
RADIUS = 17e-6
RHO_PARTICLE = 3990.0
RHO_AIR = 1.177
ETA_AIR = 1.81e-5
MEAN_FREE_PATH = 65.1e-9
R = 0.010
KB = 1.380649e-23
TEMP = 300.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def memory_snapshot() -> dict[str, int | float]:
    """Read total/free physical memory with the Windows GlobalMemoryStatusEx API."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        return {"available": False}
    return {
        "available": True,
        "load_percent": int(s.dwMemoryLoad),
        "total_bytes": int(s.ullTotalPhys),
        "available_bytes": int(s.ullAvailPhys),
        "total_GiB": round(s.ullTotalPhys / 2**30, 3),
        "available_GiB": round(s.ullAvailPhys / 2**30, 3),
    }


def edge_initials(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    u = qmc.Sobol(d=14, scramble=True, seed=seed).random_base2(int(math.log2(count)))
    out = np.zeros((count, 12), dtype=np.float64)
    for particle, rcol, tcol, zcol, vcol in (
        (0, 0, 1, 4, 8), (1, 2, 3, 5, 11),
    ):
        radius = np.sqrt(0.72**2 + u[:, rcol] * (0.90**2 - 0.72**2))
        theta = 2.0 * math.pi * u[:, tcol]
        out[:, 6*particle] = radius * np.cos(theta)
        out[:, 6*particle+1] = radius * np.sin(theta)
        out[:, 6*particle+2] = 0.8 * (2.0*u[:, zcol] - 1.0)
    velocity_columns = np.clip(u[:, [6, 7, 8, 9, 10, 11]], 1e-7, 1.0-1e-7)
    # Moderate initial velocity dispersion in the radially edge-biased ensemble.
    from scipy.special import ndtri
    out[:, [3, 4, 5, 9, 10, 11]] = 0.05 * ndtri(velocity_columns)
    phases = 2.0 * math.pi * u[:, 12:14]
    return out, phases


def make_inputs(n: int, seeds: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_y, all_phi, labels = [], [], []
    for group in GROUPS:
        if group == "edge":
            y, phi = edge_initials(n, seeds[group])
        else:
            y, phi = integrator.make_initials(group, n, seeds[group])
        all_y.append(y)
        all_phi.append(phi)
        labels.extend([group] * n)
    return np.vstack(all_y), np.vstack(all_phi), np.asarray(labels, dtype="U8")


def category_label(raw: str, turns: np.ndarray, retained: bool,
                   escaped: bool, singular: bool, axis_ambiguous: bool) -> str:
    if singular:
        return "UNRESOLVED"
    if raw == "finite_particle_contact":
        return "PARTICLE_CONTACT"
    if raw == "finite_particle_wall_contact":
        return "WALL_CONTACT"
    if escaped or raw == "lost_retention":
        return "ESCAPE"
    if axis_ambiguous or raw == "axis_ambiguous":
        return "AXIS_AMBIGUOUS"
    if raw == "CR_screen":
        return "CR_PLUS_MINUS" if turns[0] > 0 else "CR_MINUS_PLUS"
    if raw == "same_direction":
        return "CO_PLUS_PLUS" if turns[0] > 0 else "CO_MINUS_MINUS"
    if raw == "localized_or_slow":
        return "NONROTATING"
    if retained:
        return "UNRESOLVED"
    return "UNRESOLVED"


def wilson_interval(success: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = success / total
    den = 1.0 + z*z/total
    mid = (p + z*z/(2.0*total)) / den
    half = z * math.sqrt(p*(1.0-p)/total + z*z/(4.0*total*total)) / den
    return [max(0.0, mid-half), min(1.0, mid+half)]


def exact_upper_zero(success: int, total: int, alpha: float = 0.05) -> float | None:
    if success != 0 or total == 0:
        return None
    return float(beta_dist.ppf(1.0-alpha, 1.0, total))


def build_summary(rows: list[dict], n: int) -> dict:
    summary = {"sample_size_per_ensemble": n, "categories": {}, "ensembles": {}}
    all_counts = {}
    for group in GROUPS:
        rr = [x for x in rows if x["ensemble"] == group]
        counts = {}
        for row in rr:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        total = len(rr)
        crpm = counts.get("CR_PLUS_MINUS", 0)
        crmp = counts.get("CR_MINUS_PLUS", 0)
        co = counts.get("CO_PLUS_PLUS", 0) + counts.get("CO_MINUS_MINUS", 0)
        retained = sum(int(x["retained"]) for x in rr)
        summary["ensembles"][group] = {
            "n": total,
            "counts": counts,
            "P_CR_PLUS_MINUS": crpm / total if total else None,
            "P_CR_MINUS_PLUS": crmp / total if total else None,
            "P_CR": (crpm + crmp) / total if total else None,
            "P_CR_Wilson95": wilson_interval(crpm + crmp, total),
            "P_CO": co / total if total else None,
            "P_CO_Wilson95": wilson_interval(co, total),
            "P_retained": retained / total if total else None,
            "Q_CR_given_retained": (crpm + crmp) / retained if retained else None,
            "P_CO_zero_one_sided95_upper": exact_upper_zero(co, total),
            "rotation_balance": min(crpm, crmp) / max(crpm + crmp, 1),
        }
        for cat, count in counts.items():
            all_counts[cat] = all_counts.get(cat, 0) + count
    total = sum(x["n"] for x in summary["ensembles"].values())
    crpm = all_counts.get("CR_PLUS_MINUS", 0)
    crmp = all_counts.get("CR_MINUS_PLUS", 0)
    co = all_counts.get("CO_PLUS_PLUS", 0) + all_counts.get("CO_MINUS_MINUS", 0)
    retained = sum(int(x["retained"]) for x in rows)
    summary["categories"] = all_counts
    summary["overall_unweighted_strata_pooled"] = {
        "n": total,
        "P_CR_PLUS_MINUS": crpm/total if total else None,
        "P_CR_MINUS_PLUS": crmp/total if total else None,
        "P_CR": (crpm+crmp)/total if total else None,
        "P_CR_Wilson95": wilson_interval(crpm+crmp, total),
        "P_CO": co/total if total else None,
        "P_CO_Wilson95": wilson_interval(co, total),
        "P_retained": retained/total if total else None,
        "Q_CR_given_retained": (crpm+crmp)/retained if retained else None,
        "P_CO_zero_one_sided95_upper": exact_upper_zero(co, total),
        "rotation_balance": min(crpm, crmp) / max(crpm+crmp, 1),
    }
    return summary


def read_rows(path: Path) -> dict[int, dict]:
    found = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                index = int(row["global_index"])
                if index in found:
                    raise RuntimeError(f"duplicate global_index {index} at line {line_no}")
                found[index] = row
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=min(11, os.cpu_count() or 1))
    ap.add_argument("--pilot", action="store_true", help="32 starts/group, 500 periods for a throughput pilot")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--periods", type=int, default=15000)
    ap.add_argument("--transient", type=int, default=200)
    args = ap.parse_args()
    n = 32 if args.pilot else 128
    periods = 500 if args.pilot else args.periods
    steps = args.steps
    seeds = {key: (20261210+i if args.pilot else REAL_SEEDS[key]) for i, key in enumerate(GROUPS)}
    if n & (n-1):
        raise ValueError("n must be a power of two")
    if not 1 <= args.threads <= min(11, os.cpu_count() or 1):
        raise ValueError(f"threads must be in [1, {min(11, os.cpu_count() or 1)}]")
    out = RESULTS / ("pilot" if args.pilot else "identical_alumina_screen_corrected")
    out.mkdir(parents=True, exist_ok=True)
    model_json = json.loads((RESULTS / "physical_seed.json").read_text(encoding="utf-8"))
    p = np.asarray(model_json["p"], dtype=np.float64)
    qratio = float(model_json["qratio"])
    initial, phases, labels = make_inputs(n, seeds)
    starts_path = out / "frozen_starts.npz"
    if starts_path.exists():
        saved = np.load(starts_path)
        if not (np.array_equal(saved["initial"], initial) and np.array_equal(saved["phases"], phases)
                and np.array_equal(saved["labels"], labels)):
            raise RuntimeError("frozen-start file differs from deterministic regenerated inputs")
    else:
        np.savez_compressed(starts_path, initial=initial, phases=phases, labels=labels,
                            seeds=json.dumps(seeds), p=p, qratio=qratio, n=n,
                            periods=periods, steps=steps)
    p_hash = sha256(RESULTS / "physical_seed.json")
    meta = {
        "study": "V17C.12B symmetric Al2O3, gravity, continuum drag screen",
        "pilot": args.pilot, "n_per_ensemble": n, "groups": list(GROUPS), "seeds": seeds,
        "parameters": p.tolist(), "qratio": qratio, "periods": periods,
        "steps_per_RF_period": steps, "transient_periods": args.transient,
        "particle_radius_m": RADIUS, "particle_density_kg_m3": RHO_PARTICLE,
        "particle_hard_contact_a_over_R": RADIUS/R, "retention_R_m": R,
        "gravity_axis_z_m_s2": -9.80665,
        "drag_model": "Schiller-Naumann F/F_Stokes=1+0.15 Re^0.687, plus Cunningham slip; continuum, Re<1000",
        "hydrodynamics": "pair Rotne-Prager-like far-field mode 2; finite-size contact truncation; lubrication not modeled",
        "slip_correction": 1.0 + (MEAN_FREE_PATH/RADIUS) * (1.257 + 0.4*math.exp(-1.1*RADIUS/MEAN_FREE_PATH)),
        "Re_per_dimensionless_speed": 2.0*RHO_AIR*RADIUS*R*model_json["trap"]["omega1_rad_s"]/ETA_AIR,
        "MFS_field_model_sha256": sha256(MODEL / "field_model.json"),
        "physical_parameterization_sha256": p_hash,
        "starts_sha256": sha256(starts_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "logical_processors": os.cpu_count(), "workers": args.threads,
        "trajectory_parallelism": "ThreadPoolExecutor over independent Numba nogil RK4 trajectories",
        "memory_start": memory_snapshot(),
        "denominators": "every requested start; failures remain in unconditional denominator",
        "classification": "CR_PLUS_MINUS, CR_MINUS_PLUS, CO_PLUS_PLUS, CO_MINUS_MINUS, NONROTATING, AXIS_AMBIGUOUS, WALL_CONTACT, PARTICLE_CONTACT, ESCAPE, UNRESOLVED",
    }
    meta_path = out / "campaign_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise RuntimeError("campaign metadata mismatch; refusing to append to a different run")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result_path = out / "trajectories.jsonl"
    existing = read_rows(result_path)
    total = n * len(GROUPS)
    if any(index < 0 or index >= total for index in existing):
        raise RuntimeError("checkpoint has an out-of-range trajectory index")
    todo = [i for i in range(total) if i not in existing]
    family, k, coeff = integrator.load_model()[1:]
    gravity_z = -9.80665 / (R * model_json["trap"]["omega1_rad_s"]**2)
    re_per_speed = meta["Re_per_dimensionless_speed"]
    slip_correction = meta["slip_correction"]
    print(json.dumps({"n": n, "periods": periods, "steps": steps, "workers": args.threads,
                      "done": len(existing), "todo": len(todo), "memory": meta["memory_start"]}, ensure_ascii=False), flush=True)

    if todo:
        first = todo[0]
        integrator.integrate_one(
            initial[first], phases[first], p, family, k, coeff,
            min(periods, 2), steps, 0, qratio=qratio,
            hydro_mode=2, a1_over_R=RADIUS/R, a2_over_R=RADIUS/R,
            oseen_Re_per_speed=re_per_speed, oseen_Re2_per_speed=re_per_speed,
            gravity_z=gravity_z, drag_mode=2, slip_correction=slip_correction,
        )

    def run_one(global_index: int) -> dict:
        group_index, start_index = divmod(global_index, n)
        y = initial[global_index]
        phase = phases[global_index]
        raw = integrator.integrate_one(
            y, phase, p, family, k, coeff, periods, steps, args.transient,
            qratio=qratio, hydro_mode=2,
            a1_over_R=RADIUS/R, a2_over_R=RADIUS/R,
            oseen_Re_per_speed=re_per_speed, oseen_Re2_per_speed=re_per_speed,
            gravity_z=gravity_z, drag_mode=2, slip_correction=slip_correction,
        )
        end, turns, windows, rmin, rmax, zmax, dtheta, dmin, completed, retained, escaped, singular, cat, freq, vmax = raw
        c = category_label(cat, turns, retained, escaped, singular,
                           bool(min(rmin) < integrator.RMIN_CLASS))
        m_particle = float(model_json["particle"]["mass_kg"])
        q_thermal = KB*TEMP/(m_particle*(R*model_json["trap"]["omega1_rad_s"])**2)
        Re = re_per_speed * vmax
        return {
            "global_index": global_index, "ensemble": str(labels[global_index]),
            "start_index": start_index, "seed": seeds[str(labels[global_index])],
            "initial": y.tolist(), "phase": phase.tolist(), "final_state": end.tolist(),
            "turns": turns.tolist(), "window_turns": windows.tolist(),
            "frequency_Hz": (turns * model_json["trap"]["f1_Hz"] /
                             max(1, completed-args.transient)).tolist(),
            "frequency_Hz_integrator_legacy": freq.tolist(), "r_min_over_R": rmin.tolist(),
            "r_max_over_R": rmax.tolist(), "abs_z_max_over_R": zmax.tolist(),
            "distance_min_over_R": float(dmin),
            "surface_gap_min_m": float((dmin - 2.0*RADIUS/R)*R),
            "completed_periods": int(completed), "retained": bool(retained),
            "escaped": bool(escaped), "singular": bool(singular), "raw_category": str(cat),
            "category": c, "max_speed_dimensionless": vmax.tolist(),
            "max_Re_by_particle": Re.tolist(),
            "thermal_energy_over_mR2Omega2": q_thermal,
        }

    tic = time.perf_counter()
    cpu0 = time.process_time()
    with result_path.open("a", encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {pool.submit(run_one, i): i for i in todo}
            for done_count, future in enumerate(as_completed(futures), 1):
                row = future.result()
                existing[row["global_index"]] = row
                fout.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                if done_count % 16 == 0 or done_count == len(todo):
                    fout.flush()
                    os.fsync(fout.fileno())
                    elapsed = time.perf_counter() - tic
                    throughput = done_count / max(elapsed, 1e-12)
                    print(f"checkpoint {len(existing)}/{total}; {elapsed:.1f}s; {throughput:.3f} trajectories/s", flush=True)

    ordered = [existing[i] for i in sorted(existing)]
    summary = build_summary(ordered, n)
    summary.update({
        "campaign_meta_sha256": sha256(meta_path),
        "trajectory_rows": len(ordered),
        "trajectory_file_sha256": sha256(result_path),
        "elapsed_s_this_invocation": time.perf_counter() - tic,
        "process_cpu_s_this_invocation": time.process_time() - cpu0,
        "memory_end": memory_snapshot(),
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
