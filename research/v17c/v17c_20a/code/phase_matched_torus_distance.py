from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research" / "v17c" / "v17c_20a"
sys.path.insert(0, str(BASE / "code"))
import run_long_qr as qr  # noqa: E402
import direct_perturbation_tests as perturb  # noqa: E402


def wrapped_diff(a, b):
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def project_to_reference(targets, target_phase, reference, reference_phase, scales, tol):
    projected = np.full(len(targets), np.nan)
    candidate_counts = np.zeros(len(targets), dtype=int)
    for k, (x, ph) in enumerate(zip(targets, target_phase)):
        delta = np.abs(wrapped_diff(reference_phase, ph))
        mask = delta <= tol
        count = int(mask.sum())
        candidate_counts[k] = count
        if count:
            d = (reference[mask] - x) / scales
            projected[k] = np.sqrt(np.min(np.einsum("ij,ij->i", d, d)))
    return projected, candidate_counts


def main():
    candidate = "refine12_sobol17"
    cid = "scaled_refine12_sobol17_f50hz"
    _, case, p, qratio, y0, phase0, source, trajectory = qr.load_late_seed(cid, 240000)
    summary = perturb.load_nominal_frequency(candidate)
    scales, vscales = perturb.metric_scales(case, summary)
    ref_file = BASE / "results" / "stroboscopic" / "refine12_sobol17_240000p_seed_100p_transient_20000p_every_rf1.npz"
    ref = np.load(ref_file)
    reference = np.asarray(ref["states"], dtype=float)
    ref_phase = np.asarray(ref["phase2_rad"], dtype=float)
    # The reference is a finite sample of the baseline orbit at matching RF1 phase.
    field = qr.a.load_field()
    directions = perturb.direction_vectors(y0, vscales)
    selection = [("azimuth_position_p1", 1e-6), ("azimuth_position_p2", 1e-6),
                 ("radial_position_p1", 1e-6), ("tangential_velocity_p1", 1e-6),
                 ("random_full_state", 1e-6), ("radial_position_p1", 1e-4)]
    tolerance = 0.04
    rows = []
    curves = []
    epsdirs = []
    for direction, eps in selection:
        d = directions[direction].copy()
        d /= np.linalg.norm(d / scales)
        initial = y0 + eps * d
        periods, steps = 5000, 64
        raw = qr.a.PAIR._integrate_trace(
            initial, phase0, p, field["family"], field["k"], field["coeff"], qratio,
            case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"],
            case["re2_per_speed"], case["gravity_z"], periods, steps, 0, steps)
        states = np.asarray(raw[1], dtype=float)
        idx = np.asarray(raw[2], dtype=float) / steps
        phase2 = (phase0[1] + 2.0 * np.pi * float(p[2]) * idx) % (2.0 * np.pi)
        dist, counts = project_to_reference(states, phase2, reference, ref_phase, scales, tolerance)
        # Exclude the exact startup state from summary windows only if it has no reference matches.
        usable = np.isfinite(dist)
        period_bins = np.arange(0, periods + 1, 500)
        for left, right in zip(period_bins[:-1], period_bins[1:]):
            m = usable & (idx >= left) & (idx < right)
            rows.append({"direction": direction, "epsilon_metric": eps,
                         "window_rf1_periods": [int(left), int(right)], "n_targets": int(m.sum()),
                         "median_distance_to_phase_matched_reference": float(np.median(dist[m])) if m.any() else None,
                         "p90_distance_to_phase_matched_reference": float(np.quantile(dist[m], 0.9)) if m.any() else None,
                         "median_reference_candidates_per_target": float(np.median(counts[m])) if m.any() else None,
                         "phase_tolerance_rad": tolerance, "status": int(raw[-1]),
                         "completed_periods": int(raw[-2])})
        curves.append(dist)
        epsdirs.append((direction, eps))
    result = {
        "candidate": candidate, "base_period": source,
        "reference_file": str(ref_file.relative_to(ROOT)).replace("\\", "/"),
        "reference_samples": len(reference), "reference_horizon_rf1_periods": 20000,
        "perturbed_horizon_rf1_periods": 5000, "steps_per_rf1_period": 64,
        "same_rf1_strobe": True,
        "matching_rule": "For each target, nearest baseline state only among samples with the same RF1 stroboscopic phase and |wrapped delta RF2 phase| <= 0.04 rad. No unrestricted time-shift minimization.",
        "distance_metric": "Euclidean distance in dimensionless mechanical coordinates after dividing velocity differences by secular orbital speed scales.",
        "phase_tolerance_rad": tolerance, "rows": rows,
        "limitation": "This compares to a finite sampled trajectory segment, not a mathematically reconstructed invariant torus. Nearest-set distances are biased upward by finite reference coverage; a decrease is evidence only when stable to longer reference segments and phase tolerances.",
    }
    outdir = BASE / "results" / "torus_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "phase_matched_set_distance.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(outdir / "phase_matched_set_distance_curves.npz",
                        period_index=np.asarray(ref["period_index"][:5001] if False else np.arange(5001)),
                        curves=np.asarray(curves), directions=np.asarray([x[0] for x in epsdirs]),
                        epsilons=np.asarray([x[1] for x in epsdirs]))
    print(json.dumps({"rows": [{k: r[k] for k in ("direction", "epsilon_metric", "window_rf1_periods", "median_distance_to_phase_matched_reference", "p90_distance_to_phase_matched_reference")}
                              for r in rows], "reference_samples": len(reference), "tolerance_rad": tolerance}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    colors = {"azimuth_position_p1": "#0072B2", "azimuth_position_p2": "#56B4E9",
              "radial_position_p1": "#D55E00", "tangential_velocity_p1": "#009E73", "random_full_state": "#CC79A7"}
    for curve, (name, eps) in zip(curves, epsdirs):
        idx = np.arange(len(curve))
        mask = np.isfinite(curve)
        # Plot binned medians so finite sampling fluctuations are visible but not smoothed.
        centers=[]; medians=[]; lows=[]; highs=[]
        for left, right in zip(range(0, 5000, 250), range(250, 5250, 250)):
            sel = mask & (idx >= left) & (idx < right)
            if sel.sum() > 10:
                centers.append((left+right)/2); medians.append(np.median(curve[sel]))
                lows.append(np.quantile(curve[sel], 0.25)); highs.append(np.quantile(curve[sel], 0.75))
        ax.plot(centers, medians, marker="o", markersize=2.5, linewidth=0.8,
                color=colors[name], alpha=0.9, label=f"{name.replace('_',' ')}, ε={eps:g}")
        ax.fill_between(centers, lows, highs, color=colors[name], alpha=0.12, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("elapsed RF1 periods")
    ax.set_ylabel("distance to baseline sample at matched RF phases")
    ax.set_title("Finite phase-conditioned set-distance diagnostic")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(fontsize=7, ncol=2)
    figdir = BASE / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figdir / f"FIG04_phase_matched_set_distance.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
