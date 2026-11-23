from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[4]
V20B = ROOT / "research" / "v17c" / "v17c_20b"
BASE = ROOT / "research" / "v17c" / "v17c_20a"
sys.path.insert(0, str(BASE / "code"))
import run_long_qr as qr  # noqa: E402


def lowpass_positions(states: np.ndarray, cutoff_cycles_per_rf1: float = 0.03):
    sos = butter(4, cutoff_cycles_per_rf1 / 0.5, btype="lowpass", output="sos")
    slow = np.empty((len(states), 6), dtype=float)
    for k, col in enumerate((0, 1, 2, 6, 7, 8)):
        slow[:, k] = sosfiltfilt(sos, states[:, col])
    return slow


def estimate_slow_frequencies(slow: np.ndarray, f1_hz: float, periods: np.ndarray):
    output = []
    window = min(4000, max(1000, (periods[-1] - periods[0]) // 5))
    starts = np.arange(periods[0], periods[-1] - window + 1, max(500, window // 2), dtype=int)
    if len(starts) == 0:
        starts = np.asarray([periods[0]], dtype=int)
    for part in (0, 1):
        xy = slow[:, part * 3:part * 3 + 2]
        theta = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
        radii = np.hypot(xy[:, 0], xy[:, 1])
        for start in starts:
            mask = (periods >= start) & (periods <= start + window)
            if mask.sum() < 100:
                continue
            pfit = np.polyfit(periods[mask], theta[mask], 1)
            turns = float((theta[mask][-1] - theta[mask][0]) / (2.0 * math.pi))
            output.append({"particle": part + 1, "window_start_rf1_period": int(start),
                           "window_periods": int(periods[mask][-1] - periods[mask][0]),
                           "turns": turns, "frequency_hz": float(pfit[0] * f1_hz / (2.0 * math.pi)),
                           "mean_radius_R": float(np.mean(radii[mask])),
                           "radius_std_R": float(np.std(radii[mask]))})
    return output


def spectral_peaks(x: np.ndarray, max_peaks=8):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), d=1.0)
    spec[0] = 0.0
    peaks, _ = find_peaks(spec)
    peaks = sorted(peaks, key=lambda i: spec[i], reverse=True)[:max_peaks]
    return [{"cycles_per_rf1_period": float(freq[i]), "power_fraction": float(spec[i] / spec.sum())}
            for i in peaks]


def run_one(candidate, periods, source_period=None, steps=64, transition=100):
    candidate_id = {"candidate52": "scaled_candidate52_f45hz", "local15": "scaled_local15_f50hz",
                    "refine12_sobol17": "scaled_refine12_sobol17_f50hz"}[candidate]
    _, case, p, qratio, y, phase, source, trajectory = qr.load_late_seed(candidate_id, source_period)
    field = qr.a.load_field()
    y, phase = qr.transition(case, p, qratio, y, phase, transition, steps)
    raw = qr.a.PAIR._integrate_trace(
        y, phase, p, field["family"], field["k"], field["coeff"], qratio,
        case["a1_over_R"], case["a2_over_R"], case["re1_per_speed"],
        case["re2_per_speed"], case["gravity_z"], periods, steps, 0, steps)
    end, states, indices = raw[0], raw[1], raw[2]
    status = int(raw[-1]); completed = int(raw[-2])
    if status != 0 or completed != periods:
        raise RuntimeError(f"{candidate} sampling stopped: status={status}, completed={completed}")
    states = np.asarray(states, dtype=np.float64)
    period_index = np.asarray(indices, dtype=np.int64) / steps
    slow = lowpass_positions(states)
    start_phase = phase.copy()
    phases2 = (start_phase[1] + 2.0 * math.pi * float(p[2]) * period_index) % (2.0 * math.pi)
    theta = np.column_stack((np.unwrap(np.arctan2(slow[:, 1], slow[:, 0])),
                             np.unwrap(np.arctan2(slow[:, 4], slow[:, 3]))))
    relative = np.unwrap(theta[:, 1] - theta[:, 0])
    radial = np.column_stack((np.hypot(slow[:, 0], slow[:, 1]), np.hypot(slow[:, 3], slow[:, 4])))
    t = np.asarray(period_index, dtype=int)
    freqs = estimate_slow_frequencies(slow, case["f1_Hz"], t)
    phases_hist, bins = np.histogram(phases2, bins=36, range=(0.0, 2.0 * math.pi))
    section = []
    for b in range(36):
        m = (phases2 >= bins[b]) & (phases2 < bins[b + 1])
        if int(m.sum()) >= 10:
            section.append({"phase2_bin_rad": [float(bins[b]), float(bins[b + 1])], "n": int(m.sum()),
                            "mean_r1_R": float(radial[m, 0].mean()), "std_r1_R": float(radial[m, 0].std()),
                            "mean_r2_R": float(radial[m, 1].mean()), "std_r2_R": float(radial[m, 1].std()),
                            "mean_z1_R": float(slow[m, 2].mean()), "mean_z2_R": float(slow[m, 5].mean()),
                            "mean_relative_phase_rad": float(np.angle(np.mean(np.exp(1j * relative[m]))))})
    out_dir = BASE / "results" / "stroboscopic"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{candidate}_{source}p_seed_{transition}p_transient_{periods}p_every_rf1"
    np.savez_compressed(out_dir / f"{tag}.npz", states=states, slow_positions_R=slow,
                        period_index=period_index, phase2_rad=phases2,
                        relative_orbital_phase_rad=relative, radial_R=radial,
                        phase_at_sampling_start_rad=phase, p=p, qratio=qratio, steps=steps)
    result = {
        "candidate": candidate, "source_period": source, "source_trajectory": str(trajectory.relative_to(ROOT)).replace("\\", "/"),
        "trajectory_sampling": {"completed_rf1_periods": completed, "samples": len(states),
                                "steps_per_rf1_period": steps, "sample_every_rf1_period": True,
                                "phase1_fixed_to_sampling_start": float(start_phase[0]),
                                "phase2_advances_by_nu_per_rf1_period": float(p[2]),
                                "phase2_alias_cycles_per_rf1_period": float(abs(float(p[2]) - round(float(p[2])))),
                                "phase2_initial_at_sampling_start": float(start_phase[1]),
                                "lowpass_cutoff_cycles_per_rf1_period": 0.03},
        "late_frequency_windows": freqs,
        "phase_conditioned_section_bins": section,
        "phase2_bin_histogram_n": phases_hist.tolist(),
        "phase2_bins_edges_rad": bins.tolist(),
        "spectral_peaks_slow_x1": spectral_peaks(slow[:, 0]),
        "spectral_peaks_slow_x2": spectral_peaks(slow[:, 3]),
        "relative_phase_range_rad": [float(np.min(relative)), float(np.max(relative))],
        "mean_radius_R": radial.mean(axis=0).tolist(), "std_radius_R": radial.std(axis=0).tolist(),
        "z_range_slow_R": [[float(slow[:, 2].min()), float(slow[:, 2].max())], [float(slow[:, 5].min()), float(slow[:, 5].max())]],
        "phase_conditioning_warning": "Samples occur once per RF1 period but RF2 phase generally does not return. This is a quasiperiodic stroboscopic section, not a periodic Poincare map. The listed bins are descriptive; finite sampling does not establish torus dimension.",
        "lowpass_warning": "A digital low-pass filter applied to the RF1-stroboscopic sequence suppresses RF2 alias components. It is a diagnostic filter, not a derived averaging equation.",
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": candidate, "samples": len(states), "mean_radius_R": result["mean_radius_R"],
                      "frequency_windows": freqs, "phase2_coverage_bins": int(np.count_nonzero(phases_hist))}, indent=2))
    return result, states, slow, phases2, relative


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    results = []
    for candidate, n in (("candidate52", 5000), ("local15", 5000), ("refine12_sobol17", 20000)):
        results.append(run_one(candidate, n))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), constrained_layout=True)
    for col, (result, states, slow, phase2, relative) in enumerate(results):
        sc = axes[0, col].scatter(slow[:, 0], slow[:, 1], c=phase2, s=2.3, cmap="twilight", norm=Normalize(0, 2*math.pi), rasterized=True)
        axes[0, col].set_aspect("equal", adjustable="datalim")
        axes[0, col].set_xlabel(r"$x_1/R$")
        if col == 0: axes[0, col].set_ylabel(r"$y_1/R$")
        axes[0, col].set_title(result["candidate"] + " — RF1 strobe")
        # Phase-conditioned radius samples: discrete RF2-phase bins, not a fitted curve.
        rr = np.hypot(slow[:, 0], slow[:, 1])
        axes[1, col].scatter(phase2, rr, s=2.0, alpha=0.45, color=("#0072B2", "#D55E00", "#009E73")[col], rasterized=True)
        axes[1, col].set_xlim(0, 2*math.pi)
        axes[1, col].set_xlabel(r"RF2 phase $\psi_2$ (rad)")
        if col == 0: axes[1, col].set_ylabel(r"filtered $r_1/R$")
    cb = fig.colorbar(sc, ax=axes[0, :].tolist(), shrink=0.82, pad=0.02)
    cb.set_label(r"RF2 phase $\psi_2$ (rad)")
    fig.suptitle("RF1 stroboscopic sections retain the independent RF2 phase")
    fig_dir = BASE / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(fig_dir / f"FIG03_stroboscopic_phase_sections.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
