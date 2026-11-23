from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
V20B = ROOT / "research" / "v17c" / "v17c_20b"
BASE = ROOT / "research" / "v17c" / "v17c_20a"
sys.path.insert(0, str(BASE / "code"))
import run_long_qr as qr  # noqa: E402


def load_nominal_frequency(candidate):
    fn = {"candidate52": "candidate52_50000_common_corrected_summary.json",
          "local15": "local15_50000_common_corrected_summary.json",
          "refine12_sobol17": "refine12_240k_corrected_drag_summary.json"}[candidate]
    return json.loads((V20B / "results" / fn).read_text(encoding="utf-8"))


def direction_vectors(y, vscales):
    dirs = {}
    for particle in (0, 1):
        j = 6 * particle
        r = y[j:j+2]
        r = r / max(np.linalg.norm(r), 1e-15)
        radial = np.zeros(12); radial[j:j+2] = r
        tangent = np.zeros(12); tangent[j:j+2] = [-r[1], r[0]]
        vz = np.zeros(12); vz[j+2] = 1.0
        vtan = np.zeros(12); vtan[j+3:j+5] = vscales[particle] * np.array([-r[1], r[0]])
        vrad = np.zeros(12); vrad[j+3:j+5] = vscales[particle] * r
        dirs[f"radial_position_p{particle+1}"] = radial
        dirs[f"azimuth_position_p{particle+1}"] = tangent
        dirs[f"axial_position_p{particle+1}"] = vz
        dirs[f"radial_velocity_p{particle+1}"] = vrad
        dirs[f"tangential_velocity_p{particle+1}"] = vtan
    random = np.random.default_rng(20261123).normal(size=12)
    dirs["random_full_state"] = random
    return dirs


def metric_scales(case, summary):
    radii = np.asarray(summary["late_mean_radius_R"], dtype=float)
    frequencies = np.asarray(summary["late_frequency_Hz"], dtype=float)
    omega_dim = 2.0 * math.pi * np.abs(frequencies) / float(case["f1_Hz"])
    vscales = np.maximum(radii * omega_dim, 1e-5)
    scales = np.ones(12)
    scales[3:6] = vscales[0]
    scales[9:12] = vscales[1]
    return scales, vscales


def strobe(candidate, case, p, qratio, initial, phase, periods, steps, skip_periods=100):
    field = qr.a.load_field()
    raw = qr.a.PAIR._integrate_trace(
        initial, phase, p, field["family"], field["k"], field["coeff"], qratio,
        case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"],
        case["re2_per_speed"], case["gravity_z"], periods, steps, 0,
        steps * skip_periods)
    return raw[1], raw[2], int(raw[-2]), int(raw[-1])


def observables(states, indices, steps, f1_hz):
    nperiod = np.asarray(indices, dtype=float) / steps
    out = []
    for j in (0, 6):
        r = np.hypot(states[:, j], states[:, j+1])
        th = np.unwrap(np.arctan2(states[:, j+1], states[:, j]))
        pfit = np.polyfit(nperiod, th, 1)
        out.append({"mean_radius_R": float(r.mean()), "radius_std_R": float(r.std()),
                    "frequency_Hz": float(pfit[0] * f1_hz / (2.0 * math.pi)),
                    "turns": float((th[-1] - th[0]) / (2.0 * math.pi))})
    return out


def run_candidate(candidate, periods=5000, steps=64):
    cid = {"candidate52": "scaled_candidate52_f45hz", "local15": "scaled_local15_f50hz",
           "refine12_sobol17": "scaled_refine12_sobol17_f50hz"}[candidate]
    _, case, p, qratio, y0, phase, source, trajectory = qr.load_late_seed(cid)
    old_summary = load_nominal_frequency(candidate)
    scales, vscales = metric_scales(case, old_summary)
    vectors = direction_vectors(y0, vscales)
    if candidate == "refine12_sobol17":
        epsilons = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
        directions = ("azimuth_position_p1", "azimuth_position_p2", "radial_position_p1",
                      "tangential_velocity_p1", "random_full_state")
    else:
        epsilons = (1e-6, 1e-4)
        directions = ("azimuth_position_p1", "azimuth_position_p2", "radial_position_p1",
                      "tangential_velocity_p1")
    base_states, base_idx, base_done, base_status = strobe(candidate, case, p, qratio, y0, phase, periods, steps)
    if base_done != periods or base_status != 0:
        raise RuntimeError(f"nominal trajectory failed for {candidate}: {base_done}/{periods}, {base_status}")
    base_obs = observables(base_states, base_idx, steps, case["f1_Hz"])
    rows = []
    distance_archive = []
    for direction in directions:
        d = vectors[direction].copy()
        metric_norm = np.linalg.norm(d / scales)
        d /= metric_norm
        for epsilon in epsilons:
            perturbed = y0 + epsilon * d
            states, idx, done, status = strobe(candidate, case, p, qratio, perturbed, phase, periods, steps)
            if done != periods:
                status = status or 99
            n = min(len(states), len(base_states))
            dx = states[:n] - base_states[:n]
            dist = np.linalg.norm(dx / scales, axis=1)
            tperiod = np.asarray(idx[:n], dtype=float) / steps
            finite = (dist > 0) & np.isfinite(dist)
            # Estimate finite-time separation rate only while the displacement
            # remains below 0.1 in the normalized metric.
            fit_mask = finite & (dist < 0.1)
            slope = None
            if fit_mask.sum() >= 3:
                slope = float(np.polyfit(tperiod[fit_mask] * 2.0 * math.pi, np.log(dist[fit_mask]), 1)[0])
            obs = observables(states[:n], idx[:n], steps, case["f1_Hz"]) if n >= 3 else None
            rows.append({"direction": direction, "epsilon_metric": epsilon, "status": status,
                         "completed_periods": done, "initial_distance_metric": float(dist[0]),
                         "final_distance_metric": float(dist[-1]), "max_distance_metric": float(dist.max()),
                         "finite_time_separation_rate_per_tau_below_0p1": slope,
                         "nominal_late_observables": base_obs, "perturbed_observables": obs,
                         "same_initial_RF_phases": True})
            distance_archive.append(dist)
    result = {
        "candidate": candidate, "source_period": source, "source_trajectory": str(trajectory.relative_to(ROOT)).replace("\\", "/"),
        "rf_phase_rad_fixed": phase.tolist(), "horizon_rf1_periods": periods, "steps_per_rf1_period": steps,
        "distance_metric": "sqrt(sum(delta_position/R)^2 + sum(delta_velocity/v_orbit_scale)^2); v_orbit_scale from V20B late mean angular rate and radius.",
        "velocity_scales_dimensionless": vscales.tolist(), "perturbation_directions": list(directions),
        "perturbation_amplitudes": list(epsilons), "rows": rows,
        "interpretation_limit": "Direct phase-locked distances can preserve internal phase offsets on an invariant torus; nonconvergence of pointwise states is therefore not a test against attraction to a set. Finite-time output is supplementary to QR and phase-conditioned diagnostics.",
    }
    outdir = BASE / "results" / "perturbations"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{candidate}_{source}p_seed_{periods}p_phase_locked"
    (outdir / f"{tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(outdir / f"{tag}.npz", period_index=np.asarray(base_idx, dtype=float)/steps,
                        base_states=base_states, distances=np.asarray(distance_archive),
                        directions=np.asarray(directions), epsilons=np.asarray(epsilons), scales=scales)
    return result, np.asarray(base_idx, dtype=float)/steps, np.asarray(distance_archive), directions, epsilons


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outputs = [run_candidate(c) for c in ("candidate52", "local15", "refine12_sobol17")]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), constrained_layout=True, sharey=True)
    colors = {1e-8: "#0072B2", 1e-7: "#56B4E9", 1e-6: "#009E73", 1e-5: "#E69F00", 1e-4: "#D55E00"}
    for ax, (result, t, distances, directions, epsilons) in zip(axes, outputs):
        for i, (direction, eps) in enumerate((d, e) for d in directions for e in epsilons):
            if i >= len(distances): break
            ax.plot(t, distances[i], color=colors[eps], alpha=0.75, linewidth=0.8,
                    label=f"{direction.replace('_',' ')}; ε={eps:g}" if direction == directions[0] else None)
        ax.set_yscale("log")
        ax.set_xlabel("elapsed RF1 periods")
        ax.set_title(result["candidate"])
        ax.grid(True, alpha=0.2, which="both")
    axes[0].set_ylabel("phase-locked separation (scaled norm)")
    handles, labels = axes[0].get_legend_handles_labels()
    # Compact legend for epsilon colors; direction is described in the caption.
    eps_handles = [plt.Line2D([0], [0], color=colors[e], lw=2, label=f"ε={e:g}") for e in colors]
    fig.legend(handles=eps_handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Direct perturbations under identical RF phases; pointwise convergence is not required on a torus")
    fig.subplots_adjust(bottom=0.20)
    figdir = BASE / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figdir / f"FIG02_phase_locked_perturbations.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(json.dumps({c[0]["candidate"]: {"runs": len(c[0]["rows"]), "final_d_min": min(r["final_distance_metric"] for r in c[0]["rows"]),
                                        "final_d_max": max(r["final_distance_metric"] for r in c[0]["rows"])} for c in outputs}, indent=2))


if __name__ == "__main__":
    main()
