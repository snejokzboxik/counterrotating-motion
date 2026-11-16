"""Combine A9 full-FEM QMC mass nodes and estimate conditional A4 FWHM.

Only directly computed mass nodes enter the profile. Replicate-level linear
interpolation is used only between the two samples bracketing half height.
The reported R_A4 is a width descriptor of transverse acceptance, not a
mass-filter resolving power.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
DEFAULT_SOURCES = (
    "acceptance_4d_fem_h50_a9_mass_edges_N100",
    "acceptance_4d_fem_h50_a9_adaptive_N100",
    "acceptance_4d_fem_h50_selected_N100",
    "acceptance_4d_fem_h50_a9_width_refinement_N100",
)
M0_TH = 3200.0465226


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, data: list[dict]) -> None:
    if not data:
        raise ValueError(f"refusing to write empty result: {path}")
    columns = list(dict.fromkeys(k for row in data for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=float)
    mean = float(np.mean(a))
    if len(a) < 2:
        return mean, math.nan, math.nan
    half = float(student_t.ppf(.975, len(a) - 1) * np.std(a, ddof=1) / math.sqrt(len(a)))
    return mean, mean - half, mean + half


def half_height_crossings(x: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
    peak = float(np.max(y))
    if peak <= 0:
        return peak, math.nan, math.nan, math.nan, 0, False, math.nan
    peak_i = int(np.flatnonzero(y == peak)[0])
    half = peak / 2
    above = y >= half
    edges = np.diff(np.r_[False, above, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    components = list(zip(starts.tolist(), ends.tolist()))
    component = next(((a, b) for a, b in components if a <= peak_i <= b), None)
    if component is None:
        raise RuntimeError("sampled maximum not in above-half component")
    a, b = component

    def crossing(i: int, j: int) -> float:
        return float(x[i] + (half - y[i]) * (x[j] - x[i]) / (y[j] - y[i]))

    left = crossing(a - 1, a) if a > 0 else math.nan
    right = crossing(b, b + 1) if b < len(x) - 1 else math.nan
    width = right - left if math.isfinite(left) and math.isfinite(right) else math.nan
    local = np.diff(x[max(0, a - 1):min(len(x) - 1, b + 1) + 1])
    max_spacing = float(np.max(local)) if len(local) else math.nan
    resolved = math.isfinite(width) and width >= 3 * max_spacing and len(components) == 1
    peak_mu = float(np.mean(x[np.flatnonzero(y == peak)]))
    return peak, peak_mu, left, right, width, len(components), resolved, max_spacing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--output-directory", default="acceptance_4d_fem_mass_profile_h50_a9_refined_N100")
    args = parser.parse_args()
    out_name = Path(args.output_directory)
    if out_name.name != args.output_directory:
        parser.error("output directory must be a direct child of the V13.3 atlas")
    out = BASE / args.output_directory
    out.mkdir(parents=True, exist_ok=True)

    selected: dict[tuple[str, float, int], tuple[int, dict[str, str], str]] = {}
    seeds_by_source = {}
    hashes = {}
    for source in args.sources:
        if Path(source).name != source:
            parser.error("source directories must be direct children of the V13.3 atlas")
        root = BASE / source
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("field_mode") != "fem_h50_P4":
            raise ValueError(f"not a full h50 P4 source: {source}")
        seeds = tuple(manifest["replica_seeds"])
        if seeds_by_source and seeds not in seeds_by_source.values():
            raise ValueError("Sobol replicate seeds differ across mass nodes")
        seeds_by_source[source] = seeds
        hashes[source] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        nstate = int(manifest["states_per_replica"])
        for row in rows(root / "phase_averaged_replica_acceptance_N100.csv"):
            if row["point_id"] != "a9_b2_e0":
                continue
            key = (row["model"], float(row["mu"]), int(row["replica"]))
            old = selected.get(key)
            if old is not None and nstate < old[0]:
                continue
            if old is not None and nstate == old[0] and source != old[2]:
                raise ValueError(f"duplicate mass node with equal sample size: {key}")
            selected[key] = (nstate, row, source)

    models = sorted({key[0] for key in selected})
    replicas = sorted({key[2] for key in selected})
    masses = sorted({key[1] for key in selected})
    if models != ["Q", "RO"] or len(replicas) != 4:
        raise ValueError(f"incomplete model/replicate coverage: models={models}, reps={replicas}")
    profile = []
    by_model_rep: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for model in models:
        for mu in masses:
            for rep in replicas:
                key = (model, mu, rep)
                if key not in selected:
                    raise ValueError(f"missing QMC replicate row: {key}")
                nstate, row, source = selected[key]
                a4 = float(row["phase_average_A4_lower_m4_s2"])
                by_model_rep[(model, rep)].append((mu, a4))
                profile.append({"point_id": "a9_b2_e0", "model": model, "mu": mu,
                                "replica": rep, "sobol_seed": seeds_by_source[source][rep],
                                "states_per_replica": nstate, "A4_m4_s2": a4,
                                "source_directory": source})
    write(out / "fem_A4_mass_profile_replicates_N100.csv", profile)

    summary_rows = []
    for model in models:
        for mu in masses:
            vals = [selected[(model, mu, rep)][1]["phase_average_A4_lower_m4_s2"] for rep in replicas]
            m, lo, hi = mean_ci([float(v) for v in vals])
            summary_rows.append({"point_id": "a9_b2_e0", "model": model, "mu": mu,
                                 "M_over_Q_Th_per_e": mu * M0_TH,
                                 "A4_m4_s2_mean": m, "A4_t95_low": lo, "A4_t95_high": hi,
                                 "replicates": len(replicas), "uncertainty": "Student-t 95% across independent scrambled-Sobol replicate A4 values"})
    write(out / "fem_A4_mass_profile_summary_N100.csv", summary_rows)

    per_rep = []
    for model in models:
        for rep in replicas:
            series = sorted(by_model_rep[(model, rep)])
            x, y = np.asarray([v[0] for v in series]), np.asarray([v[1] for v in series])
            peak, peak_mu, left, right, width, n_lobes, resolved, spacing = half_height_crossings(x, y)
            per_rep.append({"model": model, "replica": rep, "peak_A4_m4_s2": peak,
                            "peak_mu": peak_mu, "left_halfheight_mu": left,
                            "right_halfheight_mu": right, "FWHM_mu": width,
                            "conditional_R_A4": peak_mu / width if resolved else math.nan,
                            "bounded_both_sides": math.isfinite(left) and math.isfinite(right),
                            "halfheight_lobes": n_lobes, "resolved_on_mass_grid": resolved,
                            "max_local_mass_step": spacing, "mass_nodes": len(x),
                            "threshold_rule": "replicate-specific half of sampled A4 maximum; linear interpolation only across the two bracketing computed nodes; require one bounded component and width >= 3 local grid steps"})
    write(out / "fem_A4_FWHM_per_replica_N100.csv", per_rep)

    fwhm_summary = []
    for model in models:
        rr = [row for row in per_rep if row["model"] == model]
        valid = [row for row in rr if row["bounded_both_sides"] and row["halfheight_lobes"] == 1 and row["resolved_on_mass_grid"]]
        if len(valid) == len(rr):
            peak, peak_lo, peak_hi = mean_ci([float(row["peak_mu"]) for row in rr])
            width, width_lo, width_hi = mean_ci([float(row["FWHM_mu"]) for row in rr])
            rmean, rlo, rhi = mean_ci([float(row["conditional_R_A4"]) for row in rr])
            status = "all four replicates bounded, single-lobed, and grid-resolved; conditional A4-profile width only"
        else:
            peak = peak_lo = peak_hi = width = width_lo = width_hi = rmean = rlo = rhi = math.nan
            status = "withheld: one or more replicates unbounded, multi-lobed, or under-resolved"
        fwhm_summary.append({"model": model, "replicates": len(rr), "valid_replicates": len(valid),
                             "peak_mu_mean": peak, "peak_mu_t95_low": peak_lo, "peak_mu_t95_high": peak_hi,
                             "FWHM_mu_mean": width, "FWHM_mu_t95_low": width_lo, "FWHM_mu_t95_high": width_hi,
                             "conditional_R_A4_mean": rmean, "conditional_R_A4_t95_low": rlo,
                             "conditional_R_A4_t95_high": rhi, "status": status,
                             "not_a_device_resolution": True})
    write(out / "fem_A4_FWHM_summary_N100.csv", fwhm_summary)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    colors = {"Q": "#1f77b4", "RO": "#d62728"}
    for model in models:
        rr = [r for r in summary_rows if r["model"] == model]
        xx = np.asarray([r["mu"] for r in rr])
        yy = np.asarray([r["A4_m4_s2_mean"] for r in rr])
        low = np.maximum(0., yy - np.asarray([r["A4_t95_low"] for r in rr]))
        high = np.maximum(0., np.asarray([r["A4_t95_high"] for r in rr]) - yy)
        ax.errorbar(xx, yy, yerr=np.vstack((low, high)), marker="o", capsize=2,
                    linestyle="none", color=colors[model], label=model)
    ro = next(row for row in fwhm_summary if row["model"] == "RO")
    if math.isfinite(ro["FWHM_mu_mean"]):
        ax.set_title(f"A9 h50 P4; conditional A4-profile FWHM μ={ro['FWHM_mu_mean']:.3g} (not device resolution)")
    else:
        ax.set_title("A9 h50 P4; measured mass nodes only")
    ax.set(xlabel="Normalized mass μ", ylabel=r"$A_4$ (m$^4$ s$^{-2}$)")
    ax.legend(frameon=False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fem_A4_profile_fwhm_N100.{ext}", dpi=220 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)

    meta = {"point_id": "a9_b2_e0", "field_mode": "fem_h50_P4", "horizon_RF": 100,
            "source_directories": list(args.sources), "manifest_sha256": hashes,
            "duplicate_policy": "for each model, mu, and replica choose the source with the largest states_per_replica; ties across sources are an error",
            "replica_seeds_must_match": True, "mass_nodes_used": masses,
            "FWHM_quantity": "phase-averaged physical 4D acceptance A4, not finite-filter transmission",
            "interpolation": "piecewise linear only between directly simulated nodes that straddle replicate-specific half-height",
            "resolution_metric": "conditional_R_A4 = peak_mu / FWHM_mu; not mass-filter resolving power"}
    (out / "fem_A4_FWHM_manifest_N100.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mass_nodes": masses, "fwhm_summary": fwhm_summary}, indent=2))


if __name__ == "__main__":
    main()
