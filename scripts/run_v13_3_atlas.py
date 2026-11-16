"""Resumable ideal-field three-parameter retention atlas for V13.3."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
V13_1 = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1"

ALPHAS = np.linspace(-0.05, 0.05, 17)
BETAS = np.linspace(0.05, 0.30, 21)
ETAS = (-0.20, -0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10, 0.20)
PHASES = (0.0, math.pi / 2)
N_INITIAL = 8
N_RF = 100
RTOL, ATOL = 2.0e-10, 2.0e-12
MAX_STEP = 2.0 * math.pi / 64.0
APERTURE = 1.0


def acceleration(x: float, y: float, tau: float, kappa: float, beta: float,
                 alpha: float, delta: float = 0.0, tau0: float = 0.0) -> tuple[float, float]:
    """Acceleration from V13.3 ideal Q+RO potential; delta and tau0 are distinct."""
    z = complex(x, y)
    phase = tau + tau0 - delta
    w = 4.0 * kappa * z**3 * np.exp(-1j * phase)
    w += 2.0 * (beta * math.cos(tau + tau0) + alpha) * z
    return -float(w.real), float(w.imag)


def _integrate(task: tuple) -> dict:
    (alpha_i, beta_i, eta_i, alpha, beta, kappa, model, phase_i,
     phase, state_i, state) = task
    x0, y0, vx0, vy0 = map(float, state)

    def rhs(tau: float, y: np.ndarray) -> np.ndarray:
        ax, ay = acceleration(y[0], y[1], tau, kappa, beta, alpha,
                              delta=0.0, tau0=phase)
        return np.array([y[2], y[3], ax, ay])

    def aperture_event(tau: float, y: np.ndarray) -> float:
        return float(math.hypot(y[0], y[1]) - APERTURE)

    aperture_event.terminal = True
    aperture_event.direction = 1.0
    row = {
        "alpha_i": alpha_i, "beta_i": beta_i, "eta_i": eta_i,
        "alpha": alpha, "beta": beta, "kappa": kappa,
        "model": model, "phase_i": phase_i,
        "tau0_rad": phase, "tau0_deg": math.degrees(phase),
        "state_i": state_i, "x0_rho": x0, "y0_rho": y0,
        "vx0_rho_tau": vx0, "vy0_rho_tau": vy0,
        "N_RF": N_RF, "aperture_rho": APERTURE,
        "integrator": "DOP853", "rtol": RTOL, "atol": ATOL,
        "max_step_tau": MAX_STEP,
    }
    try:
        sol = solve_ivp(rhs, (0.0, 2.0 * math.pi * N_RF),
                        (x0, y0, vx0, vy0), method="DOP853",
                        rtol=RTOL, atol=ATOL, max_step=MAX_STEP,
                        events=aperture_event, dense_output=True)
        if not sol.success:
            status, exit_periods = "unresolved", math.nan
        elif len(sol.t_events[0]):
            status = "escaped"
            exit_periods = float(sol.t_events[0][0] / (2.0 * math.pi))
        else:
            status, exit_periods = "retained", math.nan
        radii = np.hypot(sol.y[0], sol.y[1])
        rho_max = float(np.max(radii))
        if len(sol.t) > 1:
            mids = (sol.t[:-1] + sol.t[1:]) / 2.0
            midstates = sol.sol(mids)
            rho_max = max(rho_max, float(np.max(np.hypot(midstates[0], midstates[1]))))
        if status == "escaped":
            rho_max = max(APERTURE, rho_max)
        row.update({
            "status": status, "exit_periods": exit_periods,
            "rho_max": rho_max, "nfev": int(sol.nfev),
            "final_state": [float(v) for v in sol.y[:, -1]],
            "solver_message": str(sol.message),
        })
    except Exception as exc:  # unresolved cases are never silently counted as losses
        row.update({"status": "unresolved", "exit_periods": math.nan,
                    "rho_max": math.nan, "nfev": math.nan,
                    "final_state": None,
                    "solver_message": f"{type(exc).__name__}: {exc}"})
    row["task_key"] = task_key(row)
    return row


def task_key(row: dict) -> str:
    return "|".join(str(v) for v in (
        row["alpha_i"], row["beta_i"], row["eta_i"], row["phase_i"],
        row["state_i"], row["model"], row["N_RF"]))


def load_initials() -> np.ndarray:
    file = V13_1 / "ideal_pilot" / "initial_conditions.csv"
    with file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    names = ("x0_rho", "y0_rho", "vx0_rho_per_tau", "vy0_rho_per_tau")
    if len(rows) < N_INITIAL:
        raise ValueError(f"Expected at least {N_INITIAL} initial states in {file}")
    return np.array([[float(row[k]) for k in names] for row in rows[:N_INITIAL]])


def run_hash(initials: np.ndarray) -> tuple[str, str]:
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {
        "script_sha256": script_hash, "alphas": ALPHAS.tolist(),
        "betas": BETAS.tolist(), "etas": ETAS, "phases": PHASES,
        "initials_sha256": hashlib.sha256(initials.tobytes()).hexdigest(),
        "N_RF": N_RF, "aperture": APERTURE,
        "integrator": ["DOP853", RTOL, ATOL, MAX_STEP], "delta": 0.0,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, script_hash


def read_checkpoint(path: Path, expected_hash: str) -> tuple[dict[str, dict], dict]:
    rows: dict[str, dict] = {}
    manifest_path = path.with_name("run_manifest.json")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("run_hash") != expected_hash:
            raise RuntimeError("Checkpoint manifest does not match this grid/protocol/code; use a new output directory.")
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("run_hash") != expected_hash:
                    raise RuntimeError(f"Checkpoint run hash mismatch on line {line_no}")
                key = task_key(row)
                if key in rows:
                    raise RuntimeError(f"Duplicate checkpoint task key: {key}")
                rows[key] = row
    return rows, json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":"))
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})


def wilson(successes: int, trials: int) -> tuple[float, float]:
    if not trials:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / trials
    den = 1.0 + z*z/trials
    center = (p + z*z/(2*trials)) / den
    half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / den
    return max(0.0, center-half), min(1.0, center+half)


def summarize(rows: list[dict]) -> list[dict]:
    by: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["alpha_i"], row["beta_i"], row["eta_i"], row["alpha"],
               row["beta"], row["kappa"], row["model"], row["phase_i"], row["tau0_deg"])
        by.setdefault(key, []).append(row)
    indexed = {key: {r["state_i"]: r for r in group} for key, group in by.items()}
    out = []
    for key, group in sorted(by.items()):
        ai, bi, ei, alpha, beta, kappa, model, pi, phase_deg = key
        resolved = [r for r in group if r["status"] in ("retained", "escaped")]
        kept = sum(r["status"] == "retained" for r in resolved)
        lo, hi = wilson(kept, len(resolved))
        exits = [r["exit_periods"] for r in resolved if r["status"] == "escaped"]
        reference_key = (ai, bi, 4, alpha, beta, 0.0, "Q", pi, phase_deg)
        reference = indexed.get(reference_key, {})
        paired = [(int(r["status"] == "retained"),
                   int(reference[r["state_i"]]["status"] == "retained"))
                  for r in resolved if r["state_i"] in reference and
                  reference[r["state_i"]]["status"] in ("retained", "escaped")]
        diffs = np.array([a-b for a, b in paired], dtype=float)
        # With only eight starts this is a descriptive paired difference, not a population-level claim.
        dmean = float(np.mean(diffs)) if len(diffs) else math.nan
        dse = float(np.std(diffs, ddof=1)/math.sqrt(len(diffs))) if len(diffs) > 1 else math.nan
        out.append({
            "alpha_i": ai, "beta_i": bi, "eta_i": ei, "alpha": alpha,
            "beta": beta, "kappa": kappa, "model": model,
            "phase_i": pi, "tau0_deg": phase_deg, "N_RF": N_RF,
            "n_total": len(group), "n_resolved": len(resolved),
            "n_unresolved": len(group)-len(resolved), "n_retained": kept,
            "P_ret": kept/len(resolved) if resolved else math.nan,
            "wilson95_low": lo, "wilson95_high": hi,
            "P_bounds_low_all": kept/len(group) if group else math.nan,
            "P_bounds_high_all": (kept+len(group)-len(resolved))/len(group) if group else math.nan,
            "paired_n_vs_Q": len(diffs), "paired_diff_QplusRO_minus_Q": dmean,
            "paired_diff_se_descriptive": dse,
            "paired_diff_t95_low_descriptive": dmean-2.365*dse if np.isfinite(dse) else math.nan,
            "paired_diff_t95_high_descriptive": dmean+2.365*dse if np.isfinite(dse) else math.nan,
            "n_discordant_RO_ret_Q_escape": sum(a == 1 and b == 0 for a, b in paired),
            "n_discordant_RO_escape_Q_ret": sum(a == 0 and b == 1 for a, b in paired),
            "median_exit_periods": float(np.median(exits)) if exits else math.nan,
            "max_exit_periods": float(max(exits)) if exits else math.nan,
            "median_rho_max": float(np.median([r["rho_max"] for r in group
                                                if np.isfinite(r["rho_max"])])) if group else math.nan,
        })
    return out


def make_tasks(initials: np.ndarray, eta_i: int | None,
               todo_keys: set[str]) -> list[tuple]:
    tasks = []
    eta_indices = range(len(ETAS)) if eta_i is None else (eta_i,)
    for ai, alpha in enumerate(ALPHAS):
        for bi, beta in enumerate(BETAS):
            for ei in eta_indices:
                eta = ETAS[ei]
                models = (("Q", 0.0, 4),) if eta == 0.0 else (("RO", eta, ei),)
                for model, actual_eta, task_eta_i in models:
                    kappa = actual_eta * beta
                    for pi, phase in enumerate(PHASES):
                        for si, state in enumerate(initials):
                            row = {"alpha_i": ai, "beta_i": bi, "eta_i": task_eta_i,
                                   "phase_i": pi, "state_i": si, "model": model,
                                   "N_RF": N_RF}
                            key = task_key(row)
                            if key not in todo_keys:
                                continue
                            tasks.append((ai, bi, task_eta_i, float(alpha), float(beta),
                                          float(kappa), model, pi, float(phase), si,
                                          tuple(map(float, state))))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel process count; default leaves CPU/RAM reserve.")
    parser.add_argument("--eta-index", type=int,
                        help="Run only this eta slice (0..8), or all slices in order.")
    parser.add_argument("--pilot", action="store_true",
                        help="Run a 2x2 parameter subset for code/performance validation only.")
    parser.add_argument("--eta-batch", action="store_true",
                        help="When all eta are requested, stop after the next unfinished eta batch.")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    OUT.mkdir(parents=True, exist_ok=True)
    initials = load_initials()
    digest, script_hash = run_hash(initials)
    checkpoint = OUT / "atlas_checkpoint.jsonl"
    manifest_path = OUT / "run_manifest.json"
    existing, old_manifest = read_checkpoint(checkpoint, digest)
    manifest = {
        "version": "V13.3_atlas_v1", "run_hash": digest,
        "script_sha256": script_hash, "model": "ideal_Q_and_QplusRO",
        "delta_rad": 0.0, "alphas": ALPHAS.tolist(), "betas": BETAS.tolist(),
        "etas": list(ETAS), "phases_rad": list(PHASES),
        "initial_states_file": str((V13_1 / "ideal_pilot" / "initial_conditions.csv").relative_to(ROOT)),
        "initial_states_used": initials.tolist(), "n_initial": N_INITIAL,
        "N_RF": N_RF, "aperture_rho": APERTURE,
        "integrator": {"method": "DOP853", "rtol": RTOL, "atol": ATOL,
                       "max_step_tau": MAX_STEP},
        "field": "ideal leading-harmonic field; full FEM controls are a separate validation stage",
        "task_counts": {"Q": len(ALPHAS)*len(BETAS)*len(PHASES)*N_INITIAL,
                        "RO": len(ALPHAS)*len(BETAS)*8*len(PHASES)*N_INITIAL},
        "workers_requested": args.workers,
        "resume": bool(existing),
    }
    if old_manifest:
        manifest["created_utc"] = old_manifest.get("created_utc")
    else:
        from datetime import datetime, timezone
        manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    eta_selection = [args.eta_index] if args.eta_index is not None else [4, 0, 1, 2, 3, 5, 6, 7, 8]
    if args.eta_index is not None and not 0 <= args.eta_index < len(ETAS):
        parser.error("--eta-index must be in 0..8")
    if args.pilot:
        # Four corners, eta=0 and the central eta=-0.05, 2 states and both phase slices.
        initials = initials[:2]
        alphas, betas = (0, len(ALPHAS)-1), (0, len(BETAS)-1)
        eta_selection = [4, 2]
        tasks = []
        for ai in alphas:
            for bi in betas:
                for ei in eta_selection:
                    eta = ETAS[ei]
                    model = "Q" if eta == 0 else "RO"
                    actual_eta = 0. if eta == 0 else eta
                    for pi, phase in enumerate(PHASES):
                        for si, state in enumerate(initials):
                            tasks.append((ai, bi, ei, float(ALPHAS[ai]), float(BETAS[bi]),
                                          float(actual_eta*BETAS[bi]), model, pi, float(phase),
                                          si, tuple(map(float, state))))
        # The pilot uses its own output/checkpoint, preventing protocol mixing with the atlas.
        pilot_path = OUT / "pilot_checkpoint.jsonl"
        done = set()
        if pilot_path.exists():
            done = {task_key(json.loads(line)) for line in pilot_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        tasks = [task for task in tasks if "|".join(map(str, (task[0],task[1],task[2],task[7],task[9],task[6],N_RF))) not in done]
        with ProcessPoolExecutor(max_workers=args.workers) as pool, pilot_path.open("a", encoding="utf-8") as stream:
            futures = [pool.submit(_integrate, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                row["run_hash"] = hashlib.sha256((digest+"pilot").encode()).hexdigest()
                stream.write(json.dumps(row, separators=(",", ":"))+"\n")
                stream.flush()
        print(f"pilot_done={len(tasks)} output={pilot_path}")
        return

    for ei in eta_selection:
        eta = ETAS[ei]
        # Every nonzero eta batch uses the same single Q reference at eta-index 4.
        q_needed = eta == 0.0 and not any(row["model"] == "Q" for row in existing.values())
        needed = set()
        for ai in range(len(ALPHAS)):
            for bi in range(len(BETAS)):
                model_eta = 4 if eta == 0 else ei
                if eta == 0:
                    for pi in range(len(PHASES)):
                        for si in range(N_INITIAL):
                            needed.add("|".join(map(str, (ai, bi, model_eta, pi, si, "Q", N_RF))))
                else:
                    for pi in range(len(PHASES)):
                        for si in range(N_INITIAL):
                            needed.add("|".join(map(str, (ai, bi, model_eta, pi, si, "RO", N_RF))))
        if eta != 0.0 and not any(row["model"] == "Q" for row in existing.values()):
            raise RuntimeError("Run the eta=0/Q batch first so all RO results have a paired baseline.")
        needed -= set(existing)
        if not needed:
            print(f"eta_index={ei} eta={eta:+.3f}: already complete")
            continue
        tasks = make_tasks(load_initials(), ei, needed)
        if args.pilot:
            continue
        print(f"starting eta_index={ei} eta={eta:+.3f}: pending={len(tasks)} workers={args.workers}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool, checkpoint.open("a", encoding="utf-8") as stream:
            futures = [pool.submit(_integrate, task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                row = future.result()
                row["run_hash"] = digest
                stream.write(json.dumps(row, separators=(",", ":"))+"\n")
                completed += 1
                if completed % 100 == 0:
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(f"eta={eta:+.3f} completed={completed}/{len(tasks)}", flush=True)
                existing[row["task_key"]] = row
            stream.flush()
            os.fsync(stream.fileno())
        all_rows = list(existing.values())
        write_csv(OUT / "atlas_trajectory_results.csv", all_rows)
        summary = summarize(all_rows)
        write_csv(OUT / "atlas_retention_summary.csv", summary)
        manifest["completed_tasks"] = len(existing)
        manifest["unresolved_tasks"] = sum(r["status"] == "unresolved" for r in existing.values())
        manifest["last_completed_eta_index"] = ei
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"finished eta_index={ei} total_completed={len(existing)} summaries={len(summary)}", flush=True)
        if args.eta_batch:
            break


if __name__ == "__main__":
    main()
